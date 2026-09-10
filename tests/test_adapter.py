"""Tests for the resilient AlphaGenome adapter (model/alpha_genome_adapter.py).

Every test runs against a mocked ``dna_client``, so the suite is green
whether or not the optional extra is installed.  No test touches the network.

Covers three things:
  1. the adapter interface (config, padding, forward, n_tracks, cache);
  2. the resilience state machine (transient retry, quota wait-and-resume,
     INVALID_ARGUMENT diagnostic retries, fatal errors, client TTL rebuilds);
  3. the persistent SQLite cache (write-through, resume across "restarts",
     fingerprint isolation between configs).
"""

from __future__ import annotations

from contextlib import contextmanager
from enum import Enum
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from deepISA.model import alpha_genome_adapter as aga
from deepISA.model.alpha_genome_adapter import (
    AlphaGenomeAdapter,
    _classify_error,
    _pad_seqs,
    _tensor_to_seqs,
    load_config,
)
from deepISA.utils import one_hot_encode

try:
    from alphagenome.models.dna_output import OutputType
except ImportError:
    class OutputType(Enum):
        """Stub enum so the mocked tests run without the optional extra."""

        DNASE = "DNASE"
        CAGE = "CAGE"
        ATAC = "ATAC"
        RNA_SEQ = "RNA_SEQ"


# ── mock helpers ──────────────────────────────────────────────────────────────


@contextmanager
def _mocked_alphagenome():
    """Patch the adapter module's optional imports (they are None without the extra)."""
    with patch.object(aga, "dna_client") as mock_dc, patch.object(
        aga, "OutputType", OutputType
    ):
        yield mock_dc


class _FakeStatusCode:
    """Mimics grpc.StatusCode's repr ("StatusCode.UNAVAILABLE")."""

    def __init__(self, name):
        self._name = name

    def __str__(self):
        return f"StatusCode.{self._name}"


class _FakeRpcError(Exception):
    """Mimics grpc.RpcError: .code() returns a status-code-like object."""

    def __init__(self, code_name, msg="rpc error"):
        super().__init__(msg)
        self._code = _FakeStatusCode(code_name)

    def code(self):
        return self._code


def _fake_metadata(biosample: str, output_type: str) -> pd.DataFrame:
    """Return metadata with real OutputType enum objects, matching the live API."""
    return pd.DataFrame({
        "biosample_name": [biosample],
        "output_type":    [OutputType[output_type]],
        "ontology_curie": ["CL:0000000"],
    })


def _fake_track_output(n_positions: int, n_tracks: int, value: float,
                       biosample: str = "GM12878"):
    td = MagicMock()
    td.values = np.full((n_positions, n_tracks), value, dtype=np.float32)
    # metadata must be a real DataFrame so probe-call col-index logic works
    td.metadata = pd.DataFrame({"biosample_name": [biosample] * n_tracks})
    return td


def _fake_predict_output(value: float, output_attr: str = "dnase",
                         biosample: str = "GM12878"):
    out = MagicMock()
    setattr(out, output_attr, _fake_track_output(16384, 1, value, biosample))
    return out


def _make_adapter(tmp_path, biosample="GM12878", output_type="DNASE", mock_dc=None,
                  aggregation="sum", extra_cfg=None, cfg_name="cfg.yaml"):
    cfg = {"api_key": "k", "output_type": output_type, "biosample_name": biosample,
           "context_len": 16384, "seq_len": 600, "aggregation": aggregation}
    cfg.update(extra_cfg or {})
    (tmp_path / cfg_name).write_text(yaml.dump(cfg))
    mock_dc.create.return_value.output_metadata.return_value.concatenate.return_value = (
        _fake_metadata(biosample, output_type))
    return AlphaGenomeAdapter(str(tmp_path / cfg_name))


def _fast_retry_cfg():
    """Resilience knobs shrunk to milliseconds so tests never really sleep."""
    return {
        "retry": {
            "transient": {"initial_delay_seconds": 0.01, "max_delay_seconds": 0.01,
                          "multiplier": 2.0, "jitter": 0.0},
            "quota": {"wait_forever": True, "initial_delay_seconds": 0.01,
                      "max_delay_seconds": 0.02, "max_wait_seconds": 3600},
            "invalid_argument": {"max_attempts": 3, "delays_seconds": [0.01, 0.01, 0.01]},
        },
        "client": {"max_age_seconds": 3600, "max_calls": 5000, "rpc_timeout_seconds": 300},
    }


# ── load_config ───────────────────────────────────────────────────────────────


def test_load_config_reads_fields(tmp_path):
    cfg = {
        "api_key": "testkey",
        "output_type": "DNASE",
        "biosample_name": "GM12878",
        "context_len": 16384,
        "seq_len": 600,
        "aggregation": "sum",
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(cfg))

    loaded = load_config(str(p))
    assert loaded["api_key"] == "testkey"
    assert loaded["context_len"] == 16384
    assert loaded["aggregation"] == "sum"


def test_load_config_missing_required_key(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.dump({"output_type": "DNASE"}))

    with pytest.raises(KeyError):
        load_config(str(p))


def test_load_config_rejects_unknown_aggregation(tmp_path):
    """Regression: the documented aggregation knob is validated, not silently ignored."""
    cfg = {
        "api_key": "k",
        "output_type": "DNASE",
        "biosample_name": "GM12878",
        "aggregation": "median",
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.dump(cfg))

    with pytest.raises(ValueError, match="aggregation"):
        load_config(str(p))


def test_load_config_fills_resilience_defaults(tmp_path):
    """Old minimal YAMLs get the full resilience defaults merged in."""
    p = tmp_path / "min.yaml"
    p.write_text(yaml.dump({"api_key": "k", "output_type": "DNASE",
                            "biosample_name": "GM12878"}))
    loaded = load_config(str(p))
    assert loaded["cache"]["enabled"] is True
    assert loaded["retry"]["quota"]["wait_forever"] is True
    assert loaded["client"]["max_age_seconds"] == 3600


# ── vectorized sequence utilities ────────────────────────────────────────────


def test_tensor_to_seqs_roundtrip():
    """one_hot_encode → tensor → _tensor_to_seqs should recover original strings."""
    seqs = ["ACGT" * 150]                  # 600 bp
    x = torch.from_numpy(one_hot_encode(seqs))  # (1, 4, 600)
    assert _tensor_to_seqs(x) == seqs


def test_tensor_to_seqs_n_positions():
    x = torch.zeros(1, 4, 4)              # all-zero → 'N'
    assert _tensor_to_seqs(x)[0] == "NNNN"


def test_pad_seqs_total_length():
    padded = _pad_seqs(["ACGT" * 150], context_len=16384, seq_len=600)
    assert len(padded[0]) == 16384


def test_pad_seqs_centre_preserved():
    seq = "ACGT" * 150
    padded = _pad_seqs([seq], context_len=16384, seq_len=600)[0]
    pad_left = (16384 - 600) // 2
    assert padded[pad_left: pad_left + 600] == seq


def test_pad_seqs_flanks_are_n():
    padded = _pad_seqs(["A" * 600], context_len=16384, seq_len=600)[0]
    pad_left = (16384 - 600) // 2
    assert set(padded[:pad_left]) == {"N"}
    assert set(padded[pad_left + 600:]) == {"N"}


# ── error classification (unit) ──────────────────────────────────────────────


def test_classify_error_variants():
    assert _classify_error(_FakeRpcError("UNAVAILABLE")) == aga._TRANSIENT
    assert _classify_error(_FakeRpcError("DEADLINE_EXCEEDED")) == aga._TRANSIENT
    assert _classify_error(_FakeRpcError("RESOURCE_EXHAUSTED")) == aga._QUOTA
    # message-only fallbacks (REST-style errors)
    assert _classify_error(Exception("429 Too Many Requests")) == aga._QUOTA
    assert _classify_error(Exception("Request contains an invalid argument")) == aga._INVALID_ARGUMENT
    assert _classify_error(Exception("Invalid API key provided")) == aga._FATAL
    # api-core style: .code attribute instead of .code()
    err = Exception("Permission denied")
    err.code = _FakeStatusCode("PERMISSION_DENIED")
    assert _classify_error(err) == aga._FATAL
    # unknown → transient (unattended runs prefer waiting over dying)
    assert _classify_error(KeyError("weird failure")) == aga._TRANSIENT


# ── AlphaGenomeAdapter interface (all API calls mocked) ──────────────────────


def test_adapter_requires_extra_when_not_installed(tmp_path):
    """Without the alphagenome extra, construction raises a helpful ImportError."""
    cfg = {"api_key": "k", "output_type": "DNASE", "biosample_name": "GM12878"}
    (tmp_path / "cfg.yaml").write_text(yaml.dump(cfg))
    with patch.object(aga, "dna_client", None):
        with pytest.raises(ImportError, match=r"alphagenome is not installed"):
            AlphaGenomeAdapter(str(tmp_path / "cfg.yaml"))


def test_adapter_forward_returns_n_by_n_tracks(tmp_path):
    with _mocked_alphagenome() as mock_dc:
        mock_dc.create.return_value.predict_sequence.return_value = (
            _fake_predict_output(1.0))
        adapter = _make_adapter(tmp_path, mock_dc=mock_dc)

        x = torch.from_numpy(one_hot_encode(["ACGT" * 150]))  # (1, 4, 600)
        out = adapter(x)

        assert out.shape == (1, 1)   # 1 seq × 1 track (mock has 1 track)
        assert out.dtype == torch.float32


def test_adapter_default_sum_is_log1p_of_sum(tmp_path):
    """Default aggregation 'sum' reduces the window, then log1p — matching the
    hand-coded log1p(sum) behaviour the score pipeline was calibrated against."""
    signal_value = 0.5
    with _mocked_alphagenome() as mock_dc:
        mock_dc.create.return_value.predict_sequence.return_value = (
            _fake_predict_output(signal_value))
        adapter = _make_adapter(tmp_path, mock_dc=mock_dc)

        x = torch.from_numpy(one_hot_encode(["ACGT" * 150]))
        out = adapter(x)

        expected = np.log1p(signal_value * 600 * 1)
        assert float(out[0, 0]) == pytest.approx(expected)


def test_adapter_aggregation_mean(tmp_path):
    """Regression: aggregation='mean' averages the window instead of summing it."""
    with _mocked_alphagenome() as mock_dc:
        mock_dc.create.return_value.predict_sequence.return_value = (
            _fake_predict_output(2.0))
        adapter = _make_adapter(tmp_path, mock_dc=mock_dc, aggregation="mean")

        x = torch.from_numpy(one_hot_encode(["ACGT" * 150]))
        out = adapter(x)

        assert float(out[0, 0]) == pytest.approx(np.log1p(2.0))   # not log1p(2.0 * 600)


def test_adapter_aggregation_max(tmp_path):
    with _mocked_alphagenome() as mock_dc:
        mock_dc.create.return_value.predict_sequence.return_value = (
            _fake_predict_output(2.0))
        adapter = _make_adapter(tmp_path, mock_dc=mock_dc, aggregation="max")

        x = torch.from_numpy(one_hot_encode(["ACGT" * 150]))
        out = adapter(x)

        assert float(out[0, 0]) == pytest.approx(np.log1p(2.0))


def test_adapter_cache_deduplicates_api_calls(tmp_path):
    """Identical sequences must produce only one API call, not two."""
    with _mocked_alphagenome() as mock_dc:
        mock_dc.create.return_value.predict_sequence.return_value = (
            _fake_predict_output(1.0))
        adapter = _make_adapter(tmp_path, mock_dc=mock_dc)

        calls_after_init = mock_dc.create.return_value.predict_sequence.call_count

        x = torch.from_numpy(one_hot_encode(["ACGT" * 150]))

        adapter(x)   # first call  → API hit, stored in cache
        adapter(x)   # second call → cache hit, no API call

        assert mock_dc.create.return_value.predict_sequence.call_count == calls_after_init + 1
        assert adapter.cache_size == 1


def test_adapter_clear_cache_memory_and_disk(tmp_path):
    """clear_cache() drops the in-memory layer (the SQLite layer still serves);
    clear_cache(disk=True) wipes both so the next call hits the API again."""
    with _mocked_alphagenome() as mock_dc:
        predict = mock_dc.create.return_value.predict_sequence
        predict.return_value = _fake_predict_output(1.0)
        adapter = _make_adapter(tmp_path, mock_dc=mock_dc)
        calls_after_init = predict.call_count

        x = torch.from_numpy(one_hot_encode(["ACGT" * 150]))

        adapter(x)
        assert adapter.cache_size == 1

        adapter.clear_cache()               # memory only — disk still serves
        assert adapter.cache_size == 0
        adapter(x)
        assert predict.call_count == calls_after_init + 1   # served from disk

        adapter.clear_cache(disk=True)      # both layers
        adapter(x)
        assert predict.call_count == calls_after_init + 2   # back to the API


def test_adapter_bad_biosample_raises(tmp_path):
    with _mocked_alphagenome() as mock_dc:
        mock_dc.create.return_value.output_metadata.return_value.concatenate.return_value = (
            _fake_metadata("GM12878", "DNASE"))
        cfg = {"api_key": "k", "output_type": "DNASE", "biosample_name": "NonExistent",
               "context_len": 16384, "seq_len": 600, "aggregation": "sum"}
        (tmp_path / "cfg.yaml").write_text(yaml.dump(cfg))

        with pytest.raises(ValueError, match="not found"):
            AlphaGenomeAdapter(str(tmp_path / "cfg.yaml"))


# ── resilience: transient / quota / INVALID_ARGUMENT / fatal ─────────────────


def test_transient_unavailable_recovers(tmp_path, monkeypatch):
    """UNAVAILABLE twice, then success — retried with rebuilds, no user error."""
    slept = []
    monkeypatch.setattr(aga.time, "sleep", lambda s: slept.append(s))
    with _mocked_alphagenome() as mock_dc:
        predict = mock_dc.create.return_value.predict_sequence
        predict.side_effect = [
            _fake_predict_output(1.0),                      # probe
            _FakeRpcError("UNAVAILABLE"),
            _FakeRpcError("UNAVAILABLE"),
            _fake_predict_output(1.0),                      # recovery
        ]
        adapter = _make_adapter(tmp_path, mock_dc=mock_dc, extra_cfg=_fast_retry_cfg())

        out = adapter(torch.from_numpy(one_hot_encode(["ACGT" * 150])))
        assert float(out[0, 0]) == pytest.approx(np.log1p(600.0))
        assert predict.call_count == 4
        assert adapter.stats["retries"] == 2
        assert adapter.stats["client_rebuilds"] >= 2
        assert len(slept) == 2                                # backoff between attempts


def test_transient_deadline_exceeded_recovers(tmp_path, monkeypatch):
    monkeypatch.setattr(aga.time, "sleep", lambda s: None)
    with _mocked_alphagenome() as mock_dc:
        predict = mock_dc.create.return_value.predict_sequence
        predict.side_effect = [
            _fake_predict_output(1.0),
            _FakeRpcError("DEADLINE_EXCEEDED"),
            _fake_predict_output(1.0),
        ]
        adapter = _make_adapter(tmp_path, mock_dc=mock_dc, extra_cfg=_fast_retry_cfg())

        out = adapter(torch.from_numpy(one_hot_encode(["ACGT" * 150])))
        assert float(out[0, 0]) == pytest.approx(np.log1p(600.0))
        assert adapter.stats["retries"] == 1


def test_quota_waits_then_resumes(tmp_path, monkeypatch):
    """RESOURCE_EXHAUSTED → heartbeat wait → automatic resume, not a crash."""
    slept = []
    monkeypatch.setattr(aga.time, "sleep", lambda s: slept.append(s))
    with _mocked_alphagenome() as mock_dc:
        predict = mock_dc.create.return_value.predict_sequence
        predict.side_effect = [
            _fake_predict_output(1.0),
            _FakeRpcError("RESOURCE_EXHAUSTED"),
            _fake_predict_output(1.0),
        ]
        adapter = _make_adapter(tmp_path, mock_dc=mock_dc, extra_cfg=_fast_retry_cfg())

        out = adapter(torch.from_numpy(one_hot_encode(["ACGT" * 150])))
        assert float(out[0, 0]) == pytest.approx(np.log1p(600.0))
        assert adapter.stats["quota_waits"] == 1
        assert adapter.stats["quota_wait_seconds"] >= 0.01
        assert len(slept) >= 1


def test_quota_respects_max_wait_when_wait_forever_false(tmp_path, monkeypatch):
    """wait_forever: false + max_wait_seconds gives up with a resume hint."""
    monkeypatch.setattr(aga.time, "sleep", lambda s: None)
    cfg = _fast_retry_cfg()
    cfg["retry"]["quota"].update({"wait_forever": False, "max_wait_seconds": 0.05,
                                  "initial_delay_seconds": 0.01,
                                  "max_delay_seconds": 0.05})
    with _mocked_alphagenome() as mock_dc:
        predict = mock_dc.create.return_value.predict_sequence
        predict.side_effect = [
            _fake_predict_output(1.0),
            _FakeRpcError("RESOURCE_EXHAUSTED"),
            _FakeRpcError("RESOURCE_EXHAUSTED"),
            _FakeRpcError("RESOURCE_EXHAUSTED"),
        ]
        adapter = _make_adapter(tmp_path, mock_dc=mock_dc, extra_cfg=cfg)

        with pytest.raises(RuntimeError, match="resume from cache"):
            adapter(torch.from_numpy(one_hot_encode(["ACGT" * 150])))
        assert adapter.stats["quota_waits"] >= 1


def test_invalid_argument_recovers_after_rebuild(tmp_path, monkeypatch):
    """The 2-hour INVALID_ARGUMENT mystery: rebuild the client, retry, succeed."""
    monkeypatch.setattr(aga.time, "sleep", lambda s: None)
    with _mocked_alphagenome() as mock_dc:
        predict = mock_dc.create.return_value.predict_sequence
        predict.side_effect = [
            _fake_predict_output(1.0),
            _FakeRpcError("INVALID_ARGUMENT"),
            _fake_predict_output(1.0),
        ]
        adapter = _make_adapter(tmp_path, mock_dc=mock_dc, extra_cfg=_fast_retry_cfg())

        out = adapter(torch.from_numpy(one_hot_encode(["ACGT" * 150])))
        assert float(out[0, 0]) == pytest.approx(np.log1p(600.0))
        assert adapter.stats["invalid_argument_recoveries"] == 1
        assert adapter.stats["client_rebuilds"] >= 1


def test_invalid_argument_raises_after_max_attempts(tmp_path, monkeypatch):
    """Three consecutive INVALID_ARGUMENTs for the same request → give up
    with the fingerprint context instead of looping forever."""
    monkeypatch.setattr(aga.time, "sleep", lambda s: None)
    with _mocked_alphagenome() as mock_dc:
        predict = mock_dc.create.return_value.predict_sequence
        predict.side_effect = [
            _fake_predict_output(1.0),
            _FakeRpcError("INVALID_ARGUMENT"),
            _FakeRpcError("INVALID_ARGUMENT"),
            _FakeRpcError("INVALID_ARGUMENT"),
        ]
        adapter = _make_adapter(tmp_path, mock_dc=mock_dc, extra_cfg=_fast_retry_cfg())

        with pytest.raises(RuntimeError, match="INVALID_ARGUMENT persisted"):
            adapter(torch.from_numpy(one_hot_encode(["ACGT" * 150])))
        assert predict.call_count == 4            # probe + 3 attempts
        assert adapter.stats["retries"] == 2


def test_fatal_auth_error_not_retried(tmp_path, monkeypatch):
    """Auth/permission problems fail fast — retrying a bad key cannot help."""
    monkeypatch.setattr(aga.time, "sleep", lambda s: None)
    with _mocked_alphagenome() as mock_dc:
        predict = mock_dc.create.return_value.predict_sequence
        predict.side_effect = [
            _fake_predict_output(1.0),
            _FakeRpcError("UNAUTHENTICATED", "Invalid API key"),
        ]
        adapter = _make_adapter(tmp_path, mock_dc=mock_dc, extra_cfg=_fast_retry_cfg())

        with pytest.raises(_FakeRpcError):
            adapter(torch.from_numpy(one_hot_encode(["ACGT" * 150])))
        assert predict.call_count == 2           # probe + single failed attempt
        assert adapter.stats["retries"] == 0


def test_local_input_error_not_retried(tmp_path, monkeypatch):
    """Our own input validation raises before any API call is attempted."""
    with _mocked_alphagenome() as mock_dc:
        predict = mock_dc.create.return_value.predict_sequence
        predict.return_value = _fake_predict_output(1.0)
        adapter = _make_adapter(tmp_path, mock_dc=mock_dc)

        bad = torch.from_numpy(one_hot_encode(["ACGT" * 149]))   # 599 ≠ 600
        with pytest.raises(ValueError, match="input length"):
            adapter(bad)
        assert predict.call_count == 1           # only the construction probe


# ── resilience: client lifecycle ─────────────────────────────────────────────


def test_client_rebuilt_after_max_age(tmp_path):
    """max_age_seconds=0 → the client is recreated before every call."""
    with _mocked_alphagenome() as mock_dc:
        mock_dc.create.return_value.predict_sequence.return_value = (
            _fake_predict_output(1.0))
        cfg = _fast_retry_cfg()
        cfg["client"]["max_age_seconds"] = 0
        adapter = _make_adapter(tmp_path, mock_dc=mock_dc, extra_cfg=cfg)

        adapter(torch.from_numpy(one_hot_encode(["ACGT" * 150])))
        assert mock_dc.create.call_count >= 2    # initial + proactive rebuild
        assert adapter.stats["client_rebuilds"] >= 1


def test_client_rebuilt_after_max_calls(tmp_path):
    """max_calls=1 → the probe call alone triggers a rebuild for the next one."""
    with _mocked_alphagenome() as mock_dc:
        mock_dc.create.return_value.predict_sequence.return_value = (
            _fake_predict_output(1.0))
        cfg = _fast_retry_cfg()
        cfg["client"]["max_calls"] = 1
        adapter = _make_adapter(tmp_path, mock_dc=mock_dc, extra_cfg=cfg)

        adapter(torch.from_numpy(one_hot_encode(["ACGT" * 150])))
        assert mock_dc.create.call_count >= 2
        assert adapter.stats["client_rebuilds"] >= 1


# ── persistent cache: resume, isolation ──────────────────────────────────────


def test_disk_cache_survives_adapter_restart(tmp_path):
    """The 2-hour crash scenario: a fresh adapter (new 'process') serves the
    same sequence from SQLite with zero extra API calls."""
    x = torch.from_numpy(one_hot_encode(["ACGT" * 150]))

    with _mocked_alphagenome() as mock_dc:
        mock_dc.create.return_value.predict_sequence.side_effect = [
            _fake_predict_output(1.0),   # probe
            _fake_predict_output(1.0),   # the one real prediction
        ]
        adapter1 = _make_adapter(tmp_path, mock_dc=mock_dc)
        value1 = adapter1(x)
        assert adapter1.stats["api_calls"] == 2

    # "restart": brand-new adapter, and the API must not be asked again
    with _mocked_alphagenome() as mock_dc:
        def strict_predict(sequence=None, **kwargs):
            if set(sequence) <= {"N"}:
                return _fake_predict_output(1.0)   # construction probe is fine
            raise AssertionError("cached sequence must not hit the API")

        mock_dc.create.return_value.predict_sequence.side_effect = strict_predict
        adapter2 = _make_adapter(tmp_path, mock_dc=mock_dc)

        value2 = adapter2(x)
        assert float(value2[0, 0]) == pytest.approx(float(value1[0, 0]))
        assert adapter2.stats["api_calls"] == 1       # probe only
        assert adapter2.stats["cache_hits"] == 1      # served from SQLite


def test_cache_disabled_means_no_persistence(tmp_path):
    """cache.enabled: false keeps old RAM-only behaviour across instances."""
    x = torch.from_numpy(one_hot_encode(["ACGT" * 150]))
    cfg = {"cache": {"enabled": False, "path": None}}

    with _mocked_alphagenome() as mock_dc:
        mock_dc.create.return_value.predict_sequence.side_effect = \
            lambda *a, **k: _fake_predict_output(1.0)
        _make_adapter(tmp_path, mock_dc=mock_dc, extra_cfg=cfg)(x)
        adapter2 = _make_adapter(tmp_path, mock_dc=mock_dc, extra_cfg=cfg)
        adapter2(x)
        assert adapter2.stats["api_calls"] == 2       # probe + a fresh prediction
        assert adapter2.stats["cache_misses"] == 1


def test_cache_fingerprint_isolates_configs(tmp_path):
    """Same sequence + same cache file but different aggregation → different
    fingerprint → no false cache hit (cache keys hash the full request, not
    just the DNA)."""
    x = torch.from_numpy(one_hot_encode(["ACGT" * 150]))
    shared = str(tmp_path / "shared.cache.sqlite")
    cache_cfg = {"cache": {"enabled": True, "path": shared}}

    with _mocked_alphagenome() as mock_dc:
        mock_dc.create.return_value.predict_sequence.side_effect = \
            lambda *a, **k: _fake_predict_output(2.0)
        sum_ad = _make_adapter(tmp_path, mock_dc=mock_dc, aggregation="sum",
                               extra_cfg=cache_cfg, cfg_name="sum.yaml")
        mean_ad = _make_adapter(tmp_path, mock_dc=mock_dc, aggregation="mean",
                                extra_cfg=cache_cfg, cfg_name="mean.yaml")
        v_sum = sum_ad(x)
        v_mean = mean_ad(x)

        assert float(v_sum[0, 0]) == pytest.approx(np.log1p(1200.0))
        assert float(v_mean[0, 0]) == pytest.approx(np.log1p(2.0))
        assert mean_ad.stats["api_calls"] == 2       # probe + real call (no false hit)


def test_stats_counters_sane(tmp_path):
    with _mocked_alphagenome() as mock_dc:
        mock_dc.create.return_value.predict_sequence.return_value = (
            _fake_predict_output(1.0))
        adapter = _make_adapter(tmp_path, mock_dc=mock_dc)

        x = torch.from_numpy(one_hot_encode(["ACGT" * 150]))
        adapter(x)
        adapter(x)

        s = adapter.stats
        for key in ("api_calls", "api_successes", "cache_hits", "cache_misses",
                    "retries", "quota_waits", "quota_wait_seconds",
                    "client_rebuilds", "invalid_argument_recoveries",
                    "cache_entries"):
            assert key in s
        assert s["api_calls"] == 2                 # probe + one prediction
        assert s["cache_hits"] == 1                # second forward
        assert s["cache_misses"] == 1
        assert s["cache_entries"] == 1


# ── full-chain integration (compute_predictions, all mocked) ─────────────────


def test_full_chain_compute_predictions(tmp_path):
    """adapter works as model arg in deepISA's compute_predictions — zero ISA code changes."""
    from deepISA.model.predict import compute_predictions

    with _mocked_alphagenome() as mock_dc:
        mock_dc.create.return_value.output_metadata.return_value.concatenate.return_value = (
            _fake_metadata("GM12878", "DNASE"))
        mock_dc.create.return_value.predict_sequence.side_effect = [
            _fake_predict_output(1.0),   # probe in __init__
            _fake_predict_output(2.0),   # seq 1 original
            _fake_predict_output(1.0),   # seq 1 ablated
        ]
        adapter = _make_adapter(tmp_path, mock_dc=mock_dc)

        device = torch.device("cpu")
        seqs_orig  = ["ACGT" * 150]
        seqs_ablat = ["NNNN" * 150]

        preds_orig  = compute_predictions(adapter, seqs_orig,  device, batch_size=1)
        preds_ablat = compute_predictions(adapter, seqs_ablat, device, batch_size=1)

        isa = preds_orig[:, 0] - preds_ablat[:, 0]
        assert preds_orig.shape  == (1, 1)
        assert preds_ablat.shape == (1, 1)
        # log1p(sum) semantics: log1p(2.0*600) - log1p(1.0*600)
        assert float(isa[0]) == pytest.approx(np.log1p(1200.0) - np.log1p(600.0))


def test_run_single_isa_with_adapter_end_to_end(tmp_path):
    """Regression for the tutorial's core claim: swapping Conv -> adapter
    leaves the ISA pipeline unchanged. Runs the *real* calc_pred_orig +
    run_single_isa against a fully mocked AlphaGenome API — no network,
    works without the extra.

    The fake API scores each base (A=4, C=3, G=2, T=-1, N=0), so every value
    is exactly computable under the adapter's log1p(sum) semantics: the
    genome window sums to 1600 and ablating the 10 bp all-A motif at [0, 10)
    costs 36 ISA units (log1p(1600) - log1p(1560)).
    """
    from deepISA.score.single_isa import calc_pred_orig, run_single_isa
    from deepISA.utils import load_fasta

    genome = "A" * 100 + "C" * 100 + "G" * 100 + "T" * 100 + "A" * 200  # 600 bp
    with open(tmp_path / "genome.fa", "w") as f:
        f.write(">chr1\n")
        for i in range(0, 600, 60):
            f.write(genome[i:i + 60] + "\n")
    fasta = load_fasta(str(tmp_path / "genome.fa"))

    motif_locs = pd.DataFrame({
        "chrom": ["chr1"], "start": [0], "end": [10],
        "region": ["chr1:0-600"], "score": [7.5],
        "start_rel": [0], "end_rel": [10], "tf": ["MA0001.1"],
    })
    motif_locs.to_csv(tmp_path / "motif_locs.csv", index=False)

    with _mocked_alphagenome() as mock_dc:
        mock_dc.create.return_value.output_metadata.return_value.concatenate.return_value = (
            _fake_metadata("GM12878", "DNASE"))

        base_w = {"A": 4.0, "C": 3.0, "G": 2.0, "T": -1.0}

        def fake_predict(sequence=None, **kwargs):
            td = MagicMock()
            td.values = np.array(
                [base_w.get(ch, 0.0) for ch in sequence], dtype=np.float32
            )[:, None]  # (context_len, 1)
            td.metadata = pd.DataFrame({"biosample_name": ["GM12878"]})
            out = MagicMock()
            out.dnase = td
            return out

        mock_dc.create.return_value.predict_sequence.side_effect = fake_predict
        adapter = _make_adapter(tmp_path, mock_dc=mock_dc)  # probe call consumed

        pred_orig = str(tmp_path / "pred_orig_ag.csv")
        calc_pred_orig(
            model=adapter,
            fasta=fasta,
            motif_locs_path=str(tmp_path / "motif_locs.csv"),
            tracks=[0],
            outpath=pred_orig,
            device="cpu",
            pred_batch_size=4,
        )

        out_single = str(tmp_path / "motif_single_isa_ag.csv")
        run_single_isa(
            model=adapter,
            fasta=fasta,
            motif_locs_path=str(tmp_path / "motif_locs.csv"),
            pred_orig_path=pred_orig,
            outpath=out_single,
            device="cpu",
            tracks=[0],
            num_regions_per_batch=10,
            pred_batch_size=4,
        )

        df = pd.read_csv(out_single)
        assert len(df) == 1
        expected_isa = np.log1p(1600.0) - np.log1p(1560.0)
        # write_stream_csv rounds to 4 decimals — compare at that precision
        assert df["isa_t0"].iloc[0] == pytest.approx(expected_isa, abs=1e-4)
        # probe + region prediction + one ablated prediction = 3 API calls
        assert mock_dc.create.return_value.predict_sequence.call_count == 3
