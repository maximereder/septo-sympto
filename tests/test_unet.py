import pytest
import torch

from septosympto.models.unet import UNet

KERAS_REPORTED_PARAMS = 31_055_297
N_BATCHNORM = 18


def test_layer_counts_match_the_keras_model():
    m = UNet()
    assert len(m.ordered_convs()) == 19
    assert len(m.ordered_deconvs()) == 4
    assert len(m.ordered_bns()) == N_BATCHNORM


def test_parameter_count_matches_the_keras_model():
    """Keras counts trainable weights plus BN moving statistics.

    PyTorch splits those into parameters and buffers, and adds one scalar
    ``num_batches_tracked`` buffer per BatchNorm that Keras has no equivalent of.
    """
    m = UNet()
    params = sum(p.numel() for p in m.parameters())
    buffers = sum(b.numel() for b in m.buffers())
    assert params + buffers - N_BATCHNORM == KERAS_REPORTED_PARAMS


def test_batchnorm_uses_the_keras_epsilon():
    m = UNet()
    assert all(bn.eps == pytest.approx(1e-3) for bn in m.ordered_bns())


@pytest.mark.parametrize("shape", [(1, 3, 304, 3072), (2, 3, 64, 128)])
def test_forward_preserves_spatial_size(shape):
    m = UNet().eval()
    with torch.inference_mode():
        y = m(torch.zeros(shape))
    assert y.shape == (shape[0], 1, shape[2], shape[3])


def test_forward_returns_logits_not_probabilities():
    m = UNet().eval()
    with torch.inference_mode():
        y = m(torch.randn(1, 3, 64, 128) * 50)
    assert y.min() < 0 or y.max() > 1


def test_predict_returns_probabilities():
    m = UNet().eval()
    y = m.predict(torch.randn(1, 3, 64, 128))
    assert torch.all((y >= 0) & (y <= 1))
