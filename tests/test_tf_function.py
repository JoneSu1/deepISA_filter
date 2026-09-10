import pytest
import pandas as pd
from deepISA.explore.tf_function import plot_usf_pfs, plot_cell_specificity
import matplotlib.pyplot as plt


@pytest.fixture
def mock_tf_data():
    """
    Creates a mock dataset simulating a pre-processed TF dataframe.
    Includes 'ks_q' to satisfy assign_cooperativity requirements.
    """
    data = {
        "tf": ["SP1", "KLF4", "FOXA1", "GATA1", "CTCF", "MYC", "SOX2", "NANOG", "ATF6", "EGR1"],
        # Mocking the raw scores that assign_cooperativity uses
        "synergy_score": [0.1, 0.85, 0.9, 0.75, 0.05, 0.4, 0.6, 0.5, 0.2, 0.15],
        "independence_score": [0.9, 0.15, 0.1, 0.25, 0.95, 0.6, 0.4, 0.5, 0.8, 0.85],
        # The missing column causing the KeyError
        "ks_q": [0.01, 0.001, 0.001, 0.02, 0.5, 0.05, 0.01, 0.01, 0.1, 0.05],
        # Optionally mock coop_score if you want to bypass the internal calculation
        "coop_score": [0.1, 0.85, 0.9, 0.75, 0.05, 0.4, 0.6, 0.5, 0.2, 0.15]
    }
    return pd.DataFrame(data)

def test_plot_usf_pfs_mock(mock_tf_data, tmp_path):
    """Tests the Pioneer/Stripe Factor plot using tmp_path for auto-cleanup."""
    # Create a temporary directory path provided by pytest
    out_path = tmp_path / "test_usf_pf.pdf"
    # We no longer pass score_col as the function uses coop_score internally
    plot_usf_pfs(mock_tf_data, outpath=str(out_path), fig_size=(4, 3))
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_cell_specificity_mock(mock_tf_data, tmp_path):
    """Tests the Cell Specificity trend plot."""
    out_path = tmp_path / "test_specificity.pdf"
    # Testing with custom window_size and fig_size
    plot_cell_specificity(
        mock_tf_data, 
        window_size=2, 
        fig_size=(3.5, 3), 
        outpath=str(out_path)
    )
    
    assert out_path.exists()
    assert out_path.stat().st_size > 0

def test_plot_without_path_is_safe(mock_tf_data):
    """Without outpath the plot is shown (and closed) as a side effect."""
    assert plot_usf_pfs(mock_tf_data, outpath=None) is None