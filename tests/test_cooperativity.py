import pytest
import pandas as pd
from deepISA.plot.cooperativity import (
    get_prefix,
    hist_coop_score,
    heatmap_coop_score,
    plot_motif_distance_by_category
)

# --- Fixtures ---

@pytest.fixture
def mock_tf_df():
    """
    Generates a dataframe that mimics post-processed TF data.
    Added 'ks_q' because the internal assign_cooperativity logic requires it.
    """
    return pd.DataFrame({
        "tf_pair": ["SOX2|OCT4", "SOX17|OCT4", "GATA1|GATA2", "KLF4|KLF4"],
        "tf": ["SOX2", "SOX17", "GATA1", "KLF4"],
        "coop_score": [0.8, 0.7, -0.2, 0.5],
        "ks_q": [0.001, 0.001, 0.01, 0.001], # Essential for assign_cooperativity
        "cooperativity": ["Synergistic", "Synergistic", "Redundant", "Intermediate"],
        "mean_distance": [20, 25, 50, 15],
        "median_distance": [20, 25, 50, 15],
    })
# --- Unit Tests for Label Logic ---

@pytest.mark.parametrize("input_name, expected", [
    ("SOX2", "SOX"),
    ("ESRRA", "ESRRA"), # Should grab all letters
    ("GATA1", "GATA"),
    ("123", "123"),     # Fallback for no letters
])
def test_get_prefix(input_name, expected):
    assert get_prefix(input_name) == expected

# --- Integration Tests for Plotting ---

def test_hist_coop_score(mock_tf_df, tmp_path):
    out = tmp_path / "hist.png"

    # Test with annotations and vlines
    hist_coop_score(
        mock_tf_df,
        outpath=str(out),
        vlines=[0, 0.5],
        annotations=[(0.7, 0.5, "High")]
    )

    assert out.exists()

def test_heatmap_coop_score(tmp_path):
    # Prepare data specifically for a pivot-able heatmap
    heatmap_data = pd.DataFrame({
        "tf_pair": ["A|B", "A|C", "B|C"],
        "coop_score": [0.5, 0.1, -0.2],
        "ks_q": [0, 0, 0], # Add this
        "cooperativity": ["Synergistic", "Intermediate", "Redundant"]
    })
    out = tmp_path / "heatmap.pdf"

    heatmap_coop_score(heatmap_data, outpath=str(out), figsize=(5, 5))

    assert out.exists()

def test_plot_motif_distance_by_category(mock_tf_df, tmp_path):
    out = tmp_path / "distance_by_category.png"

    plot_motif_distance_by_category(mock_tf_df, outpath=str(out))

    assert out.exists()
