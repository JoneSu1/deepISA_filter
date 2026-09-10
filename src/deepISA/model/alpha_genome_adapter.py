"""AlphaGenome adapter — resilient drop-in nn.Module backend for deepISA.

Config formats
--------------
Single track (backward-compatible):
    api_key: YOUR_KEY            # or ${ALPHAGENOME_API_KEY} (env var expanded)
    output_type: DNASE
    biosample_name: GM12878
    context_len: 16384   # optional, default 16384
    seq_len: 600          # optional, default 600
    aggregation: sum      # optional: sum | mean | max (default sum)

Multi-track:
    api_key: YOUR_KEY
    tracks:
      - output_type: DNASE
        biosample_name: GM12878
      - output_type: CAGE
        biosample_name: GM12878
      - output_type: ATAC
        biosample_name: K562
    context_len: 16384
    seq_len: 600
    aggregation: sum

Every sequence makes exactly ONE API call regardless of how many tracks are
configured.  Columns in the output tensor are ordered by the `tracks` list.
`aggregation` reduces the seq_len window per track and ``log1p`` is applied
afterwards, so the default ``sum`` yields ``log1p(sum)``.

Resilience (all optional keys with sensible defaults; old YAMLs work unchanged)
------------------------------------------------------------------------------
Long ISA runs issue tens of thousands of serial API calls and previously died
on the first transient error.  The adapter is now a small state machine:

    forward(seq) → cache lookup → HIT: return
                              └ MISS → [fresh client?] → API call
                                        ├ success → write-through cache → return
                                        └ error → classify:
                                             transient (UNAVAILABLE, DEADLINE_EXCEEDED,
                                               INTERNAL, UNKNOWN, ...) → backoff + rebuild,
                                               keep waiting (5 s → 300 s, forever)
                                             quota (RESOURCE_EXHAUSTED / 429) → wait and
                                               resume (1→2→5→10→15 min, forever) with
                                               heartbeat logs; cache keeps progress safe
                                             INVALID_ARGUMENT → local validation, client
                                               rebuild, 3 diagnostic attempts (5/30/120 s)
                                             fatal (auth / permission / local input) → raise

    cache:      enabled (default true), path (default <config>.cache.sqlite)
                — SQLite write-through: every successful prediction is committed
                before it is returned, so a crash / HPC walltime / reboot loses
                at most one in-flight call.  Re-running the same command resumes
                from the cache for free.
    retry:      transient backoff, quota wait-and-resume, INVALID_ARGUMENT policy
    client:     proactive rebuild after max_age_seconds / max_calls, reactive
                rebuild on transient errors, optional per-RPC timeout
    logging:    request-fingerprint warnings on every failure (call id, sequence
                hash, status, client age, attempt — never the full sequence),
                periodic stats, heartbeat while waiting for quota.

Interface (unchanged):
    adapter = AlphaGenomeAdapter("ag_config.yaml")
    adapter(x)                 # (N, 4, seq_len) → (N, n_tracks) float32
    adapter.n_tracks
    adapter.clear_cache()      # optional disk=True also wipes the SQLite cache
    adapter.stats              # live counters (calls, hits, retries, rebuilds...)
    adapter.diagnose()         # one probe call + full health report
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from loguru import logger

_DEFAULTS: dict[str, Any] = {
    "context_len": 16384,
    "seq_len": 600,
    "aggregation": "sum",
    "cache": {"enabled": True, "path": None},
    "retry": {
        "enabled": True,
        "transient": {
            "initial_delay_seconds": 5,
            "max_delay_seconds": 300,
            "multiplier": 2.0,
            "jitter": 0.2,
        },
        "quota": {
            "wait_forever": True,
            "initial_delay_seconds": 60,
            "max_delay_seconds": 900,
            "max_wait_seconds": 3600,
        },
        "invalid_argument": {
            "max_attempts": 3,
            "delays_seconds": [5, 30, 120],
        },
    },
    "client": {"max_age_seconds": 3600, "max_calls": 5000, "rpc_timeout_seconds": 300},
    "logging": {"level": "INFO", "log_stats_every": 1000},
}

_BASES = np.array(['A', 'C', 'G', 'T'], dtype='U1')

# Error kinds returned by _classify_error.
_TRANSIENT = "TRANSIENT"
_QUOTA = "QUOTA"
_INVALID_ARGUMENT = "INVALID_ARGUMENT"
_FATAL = "FATAL"

# Substring tokens matched against type name + message + gRPC/api-core status
# code (uppercased).  Checked in this order.
_IA_TOKENS = ("INVALID_ARGUMENT", "INVALID ARGUMENT")
_QUOTA_TOKENS = ("RESOURCE_EXHAUSTED", "429", "TOO MANY REQUESTS",
                 "RATE LIMIT", "RATE_LIMIT", "QUOTA")
_FATAL_TOKENS = ("UNAUTHENTICATED", "PERMISSION_DENIED", "UNAUTHORIZED",
                 "FORBIDDEN", "API KEY", "API_KEY", "401", "403")
_TRANSIENT_TOKENS = ("UNAVAILABLE", "DEADLINE_EXCEEDED", "INTERNAL", "UNKNOWN",
                     "ABORTED", "CONNECTION", "TIMEOUT", "TIMED OUT",
                     "TEMPORARILY", "RESET", "SSL", "500", "502", "503", "504")


def _classify_error(exc: Exception) -> str:
    """Classify an exception from the API layer into a retry policy.

    Reads the gRPC-style status code when available (``exc.code()`` as in
    ``grpc.RpcError``, or the ``exc.code`` attribute used by google-api-core),
    then falls back to message substrings.  Unknown errors default to
    TRANSIENT: for unattended multi-hour runs, waiting beats dying.
    """
    raw = None
    try:
        raw = exc.code()  # grpc.RpcError
    except Exception:
        raw = getattr(exc, "code", None)  # google.api_core / others
    text = " ".join([type(exc).__name__, str(exc), str(raw)]).upper()
    if any(t in text for t in _IA_TOKENS):
        return _INVALID_ARGUMENT
    if any(t in text for t in _QUOTA_TOKENS):
        return _QUOTA
    if any(t in text for t in _FATAL_TOKENS):
        return _FATAL
    return _TRANSIENT


def _canonical_json(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _fmt_duration(seconds: float) -> str:
    s = int(max(0, seconds))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        raise ValueError(f"alpha_genome config file is empty: {path}")
    if "api_key" not in cfg:
        raise KeyError("alpha_genome config missing required key: 'api_key'")
    # Normalise old single-track format → new tracks list
    if "tracks" not in cfg:
        for key in ("output_type", "biosample_name"):
            if key not in cfg:
                raise KeyError(f"alpha_genome config missing required key: '{key}'")
        cfg["tracks"] = [{"output_type": cfg["output_type"],
                          "biosample_name": cfg["biosample_name"]}]
    cfg = _deep_merge(_DEFAULTS, cfg)
    if cfg["aggregation"] not in ("sum", "mean", "max"):
        raise ValueError(
            f"aggregation must be one of 'sum' | 'mean' | 'max', "
            f"got {cfg['aggregation']!r}"
        )
    if cfg["seq_len"] > cfg["context_len"]:
        raise ValueError(
            f"seq_len ({cfg['seq_len']}) must not exceed context_len "
            f"({cfg['context_len']})"
        )
    cfg["api_key"] = os.path.expandvars(cfg["api_key"])
    return cfg


def _tensor_to_seqs(x: torch.Tensor) -> list[str]:
    """(N, 4, L) one-hot tensor → list[str]. Vectorized via argmax."""
    x_np     = x.cpu().numpy()
    idx      = x_np.argmax(axis=1)
    has_base = x_np.max(axis=1) > 0
    chars    = np.where(has_base, _BASES[idx], 'N')
    return [''.join(row) for row in chars]


def _pad_seqs(seqs: list[str], context_len: int, seq_len: int) -> list[str]:
    """Centre each seq in context_len of N padding."""
    pad_left  = (context_len - seq_len) // 2
    pad_right = context_len - seq_len - pad_left
    pre, suf  = 'N' * pad_left, 'N' * pad_right
    return [pre + s + suf for s in seqs]


# Optional dependency (``pip install -e ".[alphagenome]"``). Importing the
# adapter must not fail when the extra is absent — same policy as the lazy
# pyBigWig imports in utils/mapper. AlphaGenomeAdapter.__init__ raises a
# clear ImportError instead, and tests can mock dna_client / OutputType.
try:
    from alphagenome.models import dna_client
    from alphagenome.models.dna_output import OutputType
except ImportError:
    dna_client = None
    OutputType = None


class _SqliteCache:
    """Write-through SQLite cache: one row per request fingerprint.

    stdlib-only.  Every successful prediction is committed before it is
    returned, so the worst-case loss on a crash is the single in-flight call.
    """

    def __init__(self, path: str):
        self._path = path
        self._conn = sqlite3.connect(path, timeout=30)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS predictions ("
            "cache_key TEXT PRIMARY KEY, "
            "prediction BLOB NOT NULL, "
            "created_at TEXT NOT NULL, "
            "metadata TEXT)"
        )
        self._conn.commit()

    def get(self, key: str) -> Optional[list]:
        row = self._conn.execute(
            "SELECT prediction FROM predictions WHERE cache_key = ?", (key,)
        ).fetchone()
        return json.loads(row[0].decode()) if row else None

    def put(self, key: str, values: list) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO predictions VALUES (?, ?, ?, ?)",
            (key, json.dumps(values).encode("utf-8"), _utc_now(),
             _canonical_json({"adapter": "alphagenome"})),
        )
        self._conn.commit()  # durable before the caller sees the value

    def __len__(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM predictions"
        ).fetchone()[0]

    def clear(self) -> None:
        self._conn.execute("DELETE FROM predictions")
        self._conn.commit()

    @property
    def path(self) -> str:
        return self._path


class AlphaGenomeAdapter(nn.Module):
    """
    Drop-in nn.Module replacement for deepISA's Conv model, hardened for
    multi-hour unattended ISA runs (see module docstring for the state
    machine and config keys).

    Returns (N, n_tracks) float32 tensor compatible with
    calc_pred_orig / run_single_isa / run_combi_isa.  Use adapter.n_tracks
    to know the output width.
    """

    def __init__(self, config_path: str) -> None:
        super().__init__()
        cfg = load_config(config_path)
        self._cfg = cfg
        self._config_path = str(Path(config_path).resolve())

        if dna_client is None:
            raise ImportError(
                "alphagenome is not installed. Install the extra and retry: "
                "pip install -e '.[alphagenome]'"
            )

        # ── Layer 3 wiring: stats / logging knobs ─────────────────────────────
        self._stats = {
            "api_calls": 0,
            "api_successes": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "retries": 0,
            "quota_waits": 0,
            "quota_wait_seconds": 0.0,
            "client_rebuilds": 0,
            "invalid_argument_recoveries": 0,
            "last_error": None,
            "last_error_kind": None,
        }
        log_cfg = cfg["logging"]
        self._log_stats_every = max(1, int(log_cfg["log_stats_every"]))
        self._verbose = str(log_cfg["level"]).upper() in ("DEBUG", "INFO")

        retry_cfg = cfg["retry"]
        self._retry_enabled = bool(retry_cfg["enabled"])
        tr = retry_cfg["transient"]
        self._tr_initial = float(tr["initial_delay_seconds"])
        self._tr_max = float(tr["max_delay_seconds"])
        self._tr_multiplier = float(tr["multiplier"])
        self._tr_jitter = float(tr["jitter"])
        qt = retry_cfg["quota"]
        self._quota_wait_forever = bool(qt["wait_forever"])
        self._quota_initial = float(qt["initial_delay_seconds"])
        self._quota_max = float(qt["max_delay_seconds"])
        self._quota_max_wait = float(qt["max_wait_seconds"])
        ia = retry_cfg["invalid_argument"]
        self._ia_max_attempts = max(1, int(ia["max_attempts"]))
        self._ia_delays = [float(d) for d in ia["delays_seconds"]]
        cl = cfg["client"]
        self._client_max_age = float(cl["max_age_seconds"])
        self._client_max_calls = int(cl["max_calls"])
        self._rpc_timeout = float(cl["rpc_timeout_seconds"])

        # ── Layer 1 wiring: persistent cache ──────────────────────────────────
        self._mem: dict[str, list] = {}
        cache_cfg = cfg["cache"]
        if cache_cfg["enabled"]:
            cache_path = cache_cfg["path"] or str(
                Path(self._config_path).with_suffix(".cache.sqlite")
            )
            self._disk: Optional[_SqliteCache] = _SqliteCache(cache_path)
            if self._verbose:
                logger.info(
                    f"[AlphaGenome] persistent cache: {cache_path} "
                    f"({len(self._disk)} existing predictions will be reused)"
                )
        else:
            self._disk = None

        # ── Layer 2 wiring: client lifecycle ──────────────────────────────────
        # dna_client.create accepts a client-wide `timeout` (seconds) that acts
        # as the per-RPC deadline — a single call can never hang forever; the
        # retry state machine takes over when the deadline trips.
        self._dna_model = dna_client.create(cfg["api_key"], timeout=self._rpc_timeout)
        self._client_created_at = time.time()
        self._client_calls = 0

        meta = self._dna_model.output_metadata(
            dna_client.Organism.HOMO_SAPIENS
        ).concatenate()

        ctx = cfg["context_len"]
        sl = cfg["seq_len"]
        self._context_len = ctx
        self._seq_len     = sl
        self._start_idx   = (ctx - sl) // 2
        self._end_idx     = self._start_idx + sl

        # ── Resolve each (output_type, biosample) track ───────────────────────
        tracks_cfg = cfg["tracks"]
        all_terms: list[str] = []
        all_output_type_enums: list[OutputType] = []

        for track in tracks_cfg:
            ot_str = track["output_type"]
            bio    = track["biosample_name"]
            ot_enum = OutputType[ot_str]
            matched = meta[
                (meta["output_type"] == ot_enum) &
                (meta["biosample_name"] == bio)
            ]
            if matched.empty:
                available = sorted(
                    meta[meta["output_type"] == ot_enum]
                    ["biosample_name"].dropna().unique()
                )[:15]
                raise ValueError(
                    f"biosample_name='{bio}' not found for output_type='{ot_str}'.\n"
                    f"Available (first 15): {available}\n"
                    f"Browse notebooks/ag_biosample_reference.csv to find valid names."
                )
            terms = matched["ontology_curie"].dropna().unique().tolist()
            all_terms.extend(terms)
            if ot_enum not in all_output_type_enums:
                all_output_type_enums.append(ot_enum)

        self._all_output_type_enums: list[OutputType] = all_output_type_enums
        self._all_terms: list[str] = list(dict.fromkeys(all_terms))  # dedup, keep order

        # Keep _ontology_terms as alias for backward compatibility
        self._ontology_terms = self._all_terms

        # ── Probe call (resilient): learn column indices per track ────────────
        probe_out = self._resilient_call("N" * ctx, seq_label="probe")

        # _extraction_plan: ordered list of (attr_name, col_indices_array)
        self._extraction_plan: list[tuple[str, np.ndarray]] = []
        for track in tracks_cfg:
            ot_str = track["output_type"]
            bio    = track["biosample_name"]
            attr   = ot_str.lower()          # "DNASE" → "dnase", "RNA_SEQ" → "rna_seq"
            track_data = getattr(probe_out, attr)
            tmeta = track_data.metadata.reset_index(drop=True)
            col_idx = np.where(tmeta["biosample_name"] == bio)[0]
            if len(col_idx) == 0:
                raise ValueError(
                    f"Probe returned no columns for biosample='{bio}' in {ot_str}. "
                    f"Available in probe: {tmeta['biosample_name'].tolist()}"
                )
            self._extraction_plan.append((attr, col_idx))

        self._n_tracks: int = sum(len(idx) for _, idx in self._extraction_plan)

    # ── Public properties ─────────────────────────────────────────────────────

    @property
    def n_tracks(self) -> int:
        """Total number of output tracks across all configured (output_type, biosample) pairs."""
        return self._n_tracks

    @property
    def cache_size(self) -> int:
        return len(self._mem)

    @property
    def stats(self) -> dict:
        """Live counters (copy) — useful from notebooks to check a long run."""
        out = dict(self._stats)
        out["cache_entries"] = len(self._disk) if self._disk is not None else len(self._mem)
        return out

    def clear_cache(self, disk: bool = False) -> None:
        """Clear the in-memory cache; ``disk=True`` also wipes the SQLite cache."""
        self._mem.clear()
        if disk and self._disk is not None:
            n = len(self._disk)
            self._disk.clear()
            logger.info(
                f"[AlphaGenome] disk cache cleared ({n} entries) at {self._disk.path}"
            )

    # ── nn.Module interface ───────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x      : (N, 4, seq_len) one-hot tensor from compute_predictions
        returns: (N, n_tracks) float32 tensor
        """
        # _pad_seqs centres the sequence assuming exactly seq_len bases; a
        # mismatched input would silently mis-window the extraction below.
        if x.shape[-1] != self._seq_len:
            raise ValueError(
                f"input length {x.shape[-1]} != configured seq_len "
                f"{self._seq_len}; set seq_len in the AG config to match "
                "your region length"
            )
        seqs        = _tensor_to_seqs(x)
        seqs_padded = _pad_seqs(seqs, self._context_len, self._seq_len)
        rows        = [self._get_prediction(raw, padded)
                       for raw, padded in zip(seqs, seqs_padded)]
        return torch.tensor(np.asarray(rows, dtype=np.float32))

    def diagnose(self) -> dict:
        """One probe call + health report.  Returns the report as a dict."""
        t0 = time.time()
        status = "READY"
        try:
            self._ensure_fresh_client()
            self._call_api("N" * self._context_len)
        except Exception as exc:
            status = f"DEGRADED ({_classify_error(exc)})"
        report = {
            "api": "OK" if status == "READY" else status,
            "probe_latency_seconds": round(time.time() - t0, 2),
            "biosamples": [t["biosample_name"] for t in self._cfg["tracks"]],
            "output_types": [t["output_type"] for t in self._cfg["tracks"]],
            "n_tracks": self._n_tracks,
            "cache_entries": len(self._disk) if self._disk is not None else len(self._mem),
            "cache_path": self._disk.path if self._disk is not None else None,
            "client_age": _fmt_duration(time.time() - self._client_created_at),
            "client_calls": self._client_calls,
            "retries": self._stats["retries"],
            "client_rebuilds": self._stats["client_rebuilds"],
            "status": status,
        }
        logger.info("[AlphaGenome] adapter diagnostic:\n" + "\n".join(
            f"  {k:24s} {v}" for k, v in report.items()))
        return report

    # ── Layer 1: cache lookup / write-through ─────────────────────────────────

    def _cache_key(self, raw_seq: str) -> str:
        """Fingerprint everything that can change the prediction, not just DNA."""
        payload = {
            "adapter_cache_schema": 1,
            "model": "alphagenome-api",
            "sequence": raw_seq,
            "context_len": self._context_len,
            "seq_len": self._seq_len,
            "aggregation": self._cfg["aggregation"],
            "requested_outputs": [str(o) for o in self._all_output_type_enums],
            "ontology_terms": self._all_terms,
            "tracks": [
                {"output_type": t["output_type"], "biosample_name": t["biosample_name"]}
                for t in self._cfg["tracks"]
            ],
        }
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    def _get_prediction(self, raw: str, padded: str) -> list:
        if raw in self._mem:
            self._stats["cache_hits"] += 1
            return self._mem[raw]
        if self._disk is not None:
            hit = self._disk.get(self._cache_key(raw))
            if hit is not None:
                self._stats["cache_hits"] += 1
                self._mem[raw] = hit
                return hit
        self._stats["cache_misses"] += 1
        values = self._extract_values(self._resilient_call(padded, seq_label=raw))
        self._mem[raw] = values
        if self._disk is not None:
            self._disk.put(self._cache_key(raw), values)  # write-through
        return values

    def _extract_values(self, output) -> list:
        agg_fn = {"sum": np.sum, "mean": np.mean, "max": np.max}[
            self._cfg["aggregation"]
        ]
        parts = []
        for attr, col_idx in self._extraction_plan:
            track_data = getattr(output, attr)
            window = track_data.values[self._start_idx:self._end_idx, :]
            parts.append(np.log1p(agg_fn(window[:, col_idx], axis=0)))
        return np.concatenate(parts).tolist()

    # ── Layer 2: resilient API call ───────────────────────────────────────────

    def _ensure_fresh_client(self) -> None:
        age = time.time() - self._client_created_at
        if age > self._client_max_age or self._client_calls >= self._client_max_calls:
            reason = (f"age={_fmt_duration(age)}"
                      if age > self._client_max_age
                      else f"calls={self._client_calls}")
            self._rebuild_client(f"proactive ({reason})")

    def _rebuild_client(self, reason: str) -> None:
        for attempt in range(5):
            try:
                self._dna_model = dna_client.create(
                    self._cfg["api_key"], timeout=self._rpc_timeout
                )
                self._client_created_at = time.time()
                self._client_calls = 0
                self._stats["client_rebuilds"] += 1
                if self._verbose:
                    logger.info(f"[AlphaGenome] client rebuilt ({reason}).")
                return
            except Exception as exc:
                if _classify_error(exc) == _FATAL:
                    raise
                logger.warning(
                    f"[AlphaGenome] client rebuild failed ({exc}); "
                    f"retry {attempt + 1}/5 in 5 s"
                )
                time.sleep(5)
        raise RuntimeError(
            "[AlphaGenome] client could not be rebuilt after 5 attempts"
        )

    def _call_api(self, padded: str):
        return self._dna_model.predict_sequence(
            sequence=padded,
            requested_outputs=self._all_output_type_enums,
            ontology_terms=self._all_terms,
        )

    def _log_failure(self, call_id: int, seq_label: str, kind: str,
                     exc: Exception, attempt: int) -> None:
        seq_hash = (hashlib.sha256(seq_label.encode()).hexdigest()[:12]
                    if seq_label != "probe" else "probe")
        logger.warning(
            f"[AlphaGenome] RPC failed | call_id={call_id} seq_hash={seq_hash} "
            f"seq_len={self._context_len} "
            f"biosample={','.join(t['biosample_name'] for t in self._cfg['tracks'])} "
            f"status={kind} error={str(exc)[:200]!r} "
            f"client_age={_fmt_duration(time.time() - self._client_created_at)} "
            f"client_calls={self._client_calls} attempt={attempt}"
        )

    def _maybe_log_stats(self) -> None:
        if self._verbose and self._stats["api_calls"] % self._log_stats_every == 0:
            s = self.stats
            logger.info(
                f"[AlphaGenome] progress: api_calls={s['api_calls']} "
                f"cache_hits={s['cache_hits']} retries={s['retries']} "
                f"cache_entries={s['cache_entries']}"
            )

    def _next_quota_delay(self, n: int) -> float:
        """1 → 2 → 5 → 10 × initial, capped at max (default 1→2→5→10→15 min)."""
        scale = (1, 2, 5, 10)[n] if n < 4 else self._quota_max / self._quota_initial
        return float(min(self._quota_initial * scale, self._quota_max))

    def _sleep_with_heartbeat(self, delay: float, reason: str) -> None:
        logger.info(
            f"[AlphaGenome] {reason}. Waiting {_fmt_duration(delay)} before retry. "
            f"Cache contains {self.stats['cache_entries']} predictions. "
            f"ISA progress is safe."
        )
        elapsed = 0.0
        while elapsed < delay:
            step = min(60.0, delay - elapsed)
            time.sleep(step)
            elapsed += step
            if elapsed < delay and int(elapsed // 300) > int((elapsed - step) // 300):
                logger.info(
                    f"[AlphaGenome] Still waiting. Next retry in "
                    f"{_fmt_duration(delay - elapsed)}. "
                    f"Total waiting time: {_fmt_duration(elapsed)}."
                )

    def _validate_request(self, padded: str) -> None:
        """Local sanity check for the INVALID_ARGUMENT diagnostic path."""
        if len(padded) != self._context_len:
            raise ValueError(
                f"local request validation failed: padded length {len(padded)} "
                f"!= context_len {self._context_len}"
            )
        bad = set(padded) - set("ACGTN")
        if bad:
            raise ValueError(
                f"local request validation failed: unexpected characters {sorted(bad)}"
            )

    def _resilient_call(self, padded: str, seq_label: str = "probe"):
        """One API request with the full retry state machine around it."""
        transient_n = 0
        ia_failures = 0
        quota_n = 0
        quota_waited = 0.0
        quota_episode_start: Optional[float] = None

        while True:
            call_id = self._stats["api_calls"] + 1
            self._ensure_fresh_client()
            try:
                self._stats["api_calls"] += 1
                output = self._call_api(padded)
                self._client_calls += 1
                self._stats["api_successes"] += 1
                if quota_episode_start is not None and self._verbose:
                    logger.info(
                        "[AlphaGenome] API recovered after "
                        f"{_fmt_duration(time.time() - quota_episode_start)}. "
                        "Resuming predictions."
                    )
                if ia_failures > 0:
                    self._stats["invalid_argument_recoveries"] += 1
                self._maybe_log_stats()
                return output

            except Exception as exc:
                kind = _classify_error(exc)
                self._stats["last_error"] = str(exc)[:300]
                self._stats["last_error_kind"] = kind
                self._log_failure(call_id, seq_label, kind, exc,
                                  attempt=max(transient_n, ia_failures, quota_n))

                if not self._retry_enabled or kind == _FATAL:
                    raise

                if kind == _QUOTA:
                    delay = self._next_quota_delay(quota_n)
                    if (not self._quota_wait_forever
                            and quota_waited + delay > self._quota_max_wait):
                        raise RuntimeError(
                            f"[AlphaGenome] quota wait would exceed "
                            f"max_wait_seconds={self._quota_max_wait}. "
                            f"Re-run the same command to resume from cache "
                            f"({self.stats['cache_entries']} predictions kept)."
                        ) from exc
                    if quota_episode_start is None:
                        quota_episode_start = time.time()
                    self._sleep_with_heartbeat(delay, f"{_QUOTA} ({exc.__class__.__name__})")
                    quota_waited += delay
                    quota_n += 1
                    self._stats["quota_waits"] += 1
                    self._stats["quota_wait_seconds"] += delay
                    continue

                if kind == _INVALID_ARGUMENT:
                    ia_failures += 1
                    self._validate_request(padded)  # raises on a truly bad request
                    if ia_failures >= self._ia_max_attempts:
                        raise RuntimeError(
                            f"[AlphaGenome] INVALID_ARGUMENT persisted for "
                            f"{ia_failures} attempts (client rebuilt between "
                            "attempts) — the request itself is likely invalid; "
                            "see the fingerprint logs above."
                        ) from exc
                    delay = self._ia_delays[
                        min(ia_failures - 1, len(self._ia_delays) - 1)
                    ]
                    self._rebuild_client(f"INVALID_ARGUMENT attempt {ia_failures}")
                    self._stats["retries"] += 1
                    time.sleep(delay)
                    continue

                # TRANSIENT (and unknown, which defaults to transient)
                base = min(self._tr_initial * (self._tr_multiplier ** transient_n),
                           self._tr_max)
                delay = base * (1 + random.uniform(-self._tr_jitter, self._tr_jitter))
                transient_n += 1
                self._rebuild_client(f"transient {kind}")
                self._stats["retries"] += 1
                if self._verbose:
                    logger.info(
                        f"[AlphaGenome] transient {kind}; retrying in "
                        f"{delay:.1f} s (attempt {transient_n})"
                    )
                time.sleep(delay)
