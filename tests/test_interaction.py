import pytest
import pandas as pd
import numpy as np
from deepISA.plot.interaction import plot_interaction_decay
from deepISA.plot.null import plot_tf_pair_against_null


@pytest.fixture
def sample_df():
    """Creates a dummy dataframe for testing."""
    np.random.seed(42)
    data = {
        "distance": np.random.randint(50, 300, 100),
        "interaction_t0": np.random.normal(0, 1, 100),
        "interaction_t1": np.random.normal(0, 1, 100),
        "tf1": np.random.choice(["A", "B", "C"], 100),
        "tf2": np.random.choice(["X", "Y", "Z"], 100),
    }
    return pd.DataFrame(data)

## --- Tests for plot_tf_pair_against_null ---

def test_plot_tf_pair_no_data(sample_df):
    # Search for a TF pair that doesn't exist in the random sample
    result = plot_tf_pair_against_null(sample_df, tf_pair=("Non", "Existent"))
    assert result is None

def test_plot_tf_pair_kde_vs_cdf(sample_df, tmp_path):
    # Get a valid pair from the dataframe
    pair = (sample_df.iloc[0]['tf1'], sample_df.iloc[0]['tf2'])

    out_kde = tmp_path / "pair_kde.png"
    plot_tf_pair_against_null(sample_df, tf_pair=pair, plot_type='kde',
                              outpath=str(out_kde))
    assert out_kde.exists()

    out_cdf = tmp_path / "pair_cdf.png"
    plot_tf_pair_against_null(sample_df, tf_pair=pair, plot_type='cdf',
                              outpath=str(out_cdf))
    assert out_cdf.exists()

## --- Tests for plot_interaction_decay ---

def test_plot_interaction_decay_multi_track(sample_df, tmp_path):
    # Test passing a list of tracks
    out = tmp_path / "decay_multi.png"
    plot_interaction_decay(sample_df, track_idx=[0, 1], mode='absolute',
                           outpath=str(out))
    assert out.exists()

def test_plot_interaction_decay_signed_logic(sample_df, tmp_path):
    out = tmp_path / "decay_signed.png"
    plot_interaction_decay(sample_df, track_idx=0, mode='signed',
                           outpath=str(out))
    assert out.exists()
