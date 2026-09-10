import os
import pytest
import pandas as pd
import torch
from deepISA.model.cnn import Conv
from deepISA.utils import load_fasta
from deepISA.score.single_isa import calc_pred_orig, run_single_isa
from deepISA.score.aggregate_isa import calc_tf_importance


@pytest.fixture
def mock_setup(tmp_path):
    """Provides common resources for ISA testing."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Conv(mode='regression',
                 model_config={"seq_len": 600, "ks": [7], "cs": [32], "ds": [1],
                               "dropout": 0.0}).to(device)
    model.eval()

    fasta_path = tmp_path / "test.fa"
    with open(fasta_path, "w") as f:
        f.write(">chr1\n" + "A" * 2000)
    fasta = load_fasta(str(fasta_path))  # pre-loaded: skips the pysam path

    return model, fasta, device, tmp_path


def test_run_single_isa_incremental_io(mock_setup):
    """Verify that run_single_isa writes to disk in batches."""
    model, fasta, device, tmp_path = mock_setup
    outpath = tmp_path / "isa_results.csv"

    regions = ["chr1:100-700", "chr1:800-1400"]
    data = {
        "chrom": ["chr1"] * 10,
        "start": [110, 120, 130, 140, 150, 810, 820, 830, 840, 850],
        "end":   [115, 125, 135, 145, 155, 815, 825, 835, 845, 855],
        "tf": ["GATA1"] * 5 + ["CTCF"] * 5,
        "region": [regions[0]] * 5 + [regions[1]] * 5,
        "start_rel": [10, 20, 30, 40, 50] * 2,
        "end_rel":   [15, 25, 35, 45, 55] * 2,
    }
    motif_df = pd.DataFrame(data)
    motif_path = tmp_path / "motif_locs.csv"
    motif_df.to_csv(motif_path, index=False)

    pred_orig_path = tmp_path / "pred_orig.csv"
    calc_pred_orig(model=model, fasta=fasta, motif_locs_path=str(motif_path),
                   tracks=[0], outpath=str(pred_orig_path), device=device,
                   pred_batch_size=4)

    run_single_isa(model=model, fasta=fasta, motif_locs_path=str(motif_path),
                   pred_orig_path=str(pred_orig_path), outpath=str(outpath),
                   device=device, num_regions_per_batch=1, pred_batch_size=4)

    assert os.path.exists(str(outpath))
    results_df = pd.read_csv(outpath)
    assert len(results_df) == 10


def test_calc_tf_importance(tmp_path):
    isa_path = tmp_path / "mock_isa.csv"
    out_path = tmp_path / "tf_importance.csv"
    mock_data = pd.DataFrame({
        "tf": ["GATA1"] * 15 + ["CTCF"] * 15,
        "isa_t0": [0.5] * 15 + [0.1] * 15,
        "isa_t1": [0.8] * 15 + [0.2] * 15
    })
    mock_data.to_csv(isa_path, index=False)

    agg_df = calc_tf_importance(str(isa_path), str(out_path))

    assert "mean_isa_t0" in agg_df.columns
    assert agg_df.shape[0] == 2
    assert out_path.exists()


def test_empty_motif_df_handling(mock_setup):
    """Ensure the system handles empty inputs gracefully."""
    model, fasta, device, tmp_path = mock_setup
    motif_path = tmp_path / "empty_motifs.csv"
    pd.DataFrame(columns=["chrom", "start", "end", "region", "tf",
                          "start_rel", "end_rel"]).to_csv(motif_path, index=False)
    pred_orig_path = tmp_path / "empty_pred.csv"
    pd.DataFrame({"region": [], "pred_t0": []}).to_csv(pred_orig_path, index=False)

    res = run_single_isa(model=model, fasta=fasta, motif_locs_path=str(motif_path),
                         pred_orig_path=str(pred_orig_path),
                         outpath=str(tmp_path / "empty.csv"), device=device)
    assert res is None
