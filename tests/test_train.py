import os
import torch
import pandas as pd
import pytest
import numpy as np
from deepISA.model.cnn import Conv
from deepISA.model.preprocess import compile_training_data
from deepISA.model.train import train_model

@pytest.fixture
def fake_genomic_data(tmp_path):
    """Generates a fake FASTA and regions dataframe for testing."""
    # 1. Create Fake FASTA (chr1 and chr2)
    fasta_path = tmp_path / "fake.fa"
    with open(fasta_path, "w") as f:
        f.write(">chr1\n" + "ATGC" * 2500 + "\n") # 10kb
        f.write(">chr2\n" + "GCTA" * 2500 + "\n") # 10kb

    # 2. Create Fake Regions (enough chr1 rows that the 15% val split holds
    #    >2 samples with varied targets — pearson is NaN on 1 point and the
    #    trainer would then never save a best checkpoint)
    rng = np.random.RandomState(42)
    n_pos = 20
    data = {
        'chrom': ['chr1'] * n_pos + ['chr2'] * 2,
        'start': list(range(100, 100 + 400 * n_pos, 400)) + [100, 500],
        'end':   list(range(700, 700 + 400 * n_pos, 400)) + [700, 1100],
        'target_reg': list(rng.rand(n_pos + 2)),
        'target_class': ([1, 0] * 10) + [1, 1]
    }
    df = pd.DataFrame(data)

    return df, str(fasta_path)

def test_full_pipeline_workflow(tmp_path, fake_genomic_data, monkeypatch):
    df, fasta_path = fake_genomic_data
    processed_dir = tmp_path / "processed"
    model_dir = tmp_path / "models"

    # Define a background pool that ONLY uses chr1 to avoid KeyError: 'chr3'
    # (sized to cover all positives for the 1:1 balance)
    fake_bg = pd.DataFrame({
        'chrom': ['chr1'] * 24,
        'start': list(range(2000, 2000 + 200 * 24, 200)),
        'end':   list(range(2600, 2600 + 200 * 24, 200)),
    })

    # CRITICAL: Patch where bioframe is actually used in the preprocess module
    monkeypatch.setattr("deepISA.model.preprocess.bf.read_table", lambda *args, **kwargs: fake_bg)
    monkeypatch.setattr("deepISA.model.preprocess.get_data_resource", lambda x: "fake_path.bed")

    # Step 1: compile the training memmaps (chr2 holdout + 85/15 train/val)
    compile_training_data(df=df, fasta_path=fasta_path, out_dir=str(processed_dir),
                          seq_len=600, target_reg_col="target_reg",
                          target_class_col="target_class", rc_aug=False)

    # Step 2: train on the compiled splits
    model = Conv(mode='dual', model_config={"seq_len": 600, "ks": [7], "cs": [8],
                                            "ds": [1], "dropout": 0.0})
    train_model(model=model, device=torch.device('cpu'),
                train_dat_dir=str(processed_dir), model_dir=str(model_dir),
                trainer_config={"epochs": 1, "batch_size": 2, "patience": 1,
                                "learning_rate": 1e-3},
                mode='dual')

    # Verify artifacts
    assert (model_dir / "trainer_config.json").exists()
    ckpts = list(model_dir.glob("model_*.pt"))
    assert ckpts, f"No checkpoints written to {model_dir}"

    train_x = processed_dir / "train" / "X.npy"
    assert train_x.exists(), f"Memmap data not found at {train_x}"

    # CRITICAL: Cleanup memmap handles to avoid leaks/locks
    import gc
    gc.collect()
