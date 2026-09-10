import torch
import pytest
from deepISA.model.cnn import Conv


def _cfg(seq_len, ks, cs, ds):
    return {"seq_len": seq_len, "ks": ks, "cs": cs, "ds": ds, "dropout": 0.1}


def test_conv_output_shape():
    # Test Dual Mode
    model = Conv(mode='dual', model_config=_cfg(600, [15, 9], [16, 32], [1, 2]))
    x = torch.randn(8, 4, 600)
    out = model(x)
    assert out.shape == (8, 2)  # [batch, reg + clf]


def test_regression_mode_shape():
    """
    Test that 'regression' mode returns (Batch, 1) and
    only initializes the regression head.
    """
    batch_size = 4
    model = Conv(mode='regression', model_config=_cfg(600, [15, 9, 9], [16, 32, 64], [1, 2, 4]))
    x = torch.randn(batch_size, 4, 600)
    out = model(x)
    # Check output dimensions
    assert out.shape == (batch_size, 1)
    # Check internal structure: regression head exists, classification doesn't
    assert model.regression_head is not None
    assert model.classification_head is None


def test_classification_mode_shape():
    """
    Non-dual modes return (Batch, 1) through the regression head
    (the classification head only exists in 'dual' mode).
    """
    batch_size = 4
    model = Conv(mode='classification', model_config=_cfg(600, [15, 9, 9], [16, 32, 64], [1, 2, 4]))
    x = torch.randn(batch_size, 4, 600)
    out = model(x)
    assert out.shape == (batch_size, 1)
    assert model.classification_head is None
    assert model.regression_head is not None


def test_receptive_field_calc():
    # RF = 1 + sum((k-1)*d)
    # 1 + (15-1)*1 + (9-1)*2 = 1 + 14 + 16 = 31
    model = Conv(mode='regression', model_config=_cfg(600, [15, 9], [16, 32], [1, 2]))
    assert model.rf == 31


def test_invalid_params():
    with pytest.raises(ValueError):
        # Mismatched lengths
        Conv(mode='regression', model_config=_cfg(600, [15], [16, 32], [1]))
