import numpy as np
import pytest
import torch

from septosympto.models import P2PNet, available_counters, build_counter, counter_class

VARIANTS = ["p2p", "p2p-resnet18", "p2p-resnet50", "p2p-convnext-t"]


def a_model() -> P2PNet:
    return P2PNet(pretrained=False, stride=4, row=2, line=2)


def test_all_backbone_variants_are_registered():
    for name in VARIANTS:
        assert name in available_counters()


@pytest.mark.parametrize("name", VARIANTS)
def test_each_variant_runs_the_full_contract(name):
    model = build_counter(name, pretrained=False)
    x = torch.randn(1, 3, 64, 256)
    out = model(x)
    n = (64 // 4) * (256 // 4) * 4
    assert out["pred_points"].shape == (1, n, 2)
    loss = model.loss(out, [torch.rand(10, 2) * torch.tensor([256.0, 64.0])])
    loss.backward()
    model.eval()
    assert isinstance(model.decode(out, 0.5)[0], np.ndarray)


def test_variants_share_the_stride_so_anchor_count_matches():
    x = torch.randn(1, 3, 64, 256)
    counts = {
        name: build_counter(name, pretrained=False).eval()(x)["pred_points"].shape[1]
        for name in VARIANTS
    }
    assert len(set(counts.values())) == 1


def test_resnet18_is_lighter_than_vgg():
    vgg = sum(p.numel() for p in build_counter("p2p", pretrained=False).parameters())
    r18 = sum(p.numel() for p in build_counter("p2p-resnet18", pretrained=False).parameters())
    assert r18 < vgg


def test_vgg_backbone_keeps_its_stage_names_for_checkpoint_compat():
    """The trained checkpoint keys are backbone.stage1..stage5; guard against renames."""
    model = build_counter("p2p", pretrained=False)
    names = {n for n, _ in model.backbone.named_children()}
    assert {"stage1", "stage2", "stage3", "stage4", "stage5"} <= names


def test_p2p_is_registered_beside_heatmap():
    assert "p2p" in available_counters()
    assert "heatmap" in available_counters()
    assert counter_class("p2p") is P2PNet
    assert isinstance(build_counter("p2p", pretrained=False), P2PNet)


def test_forward_emits_a_point_set_in_pixel_coordinates():
    model = a_model().eval()
    with torch.inference_mode():
        out = model(torch.randn(2, 3, 128, 256))
    n = (128 // 4) * (256 // 4) * 4
    assert out["pred_points"].shape == (2, n, 2)
    assert out["pred_logits"].shape == (2, n, 2)


def test_loss_takes_the_harness_point_list_and_backpropagates():
    """The loop passes list[Tensor]; the model wraps it for SetCriterion itself."""
    model = a_model()
    out = model(torch.randn(1, 3, 128, 256))
    points = [torch.rand(20, 2) * torch.tensor([256.0, 128.0])]
    loss = model.loss(out, points)
    assert loss.ndim == 0
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_loss_handles_a_leaf_with_no_points():
    model = a_model()
    out = model(torch.randn(1, 3, 128, 256))
    loss = model.loss(out, [torch.empty(0, 2)])
    assert torch.isfinite(loss)


def test_decode_returns_pixel_points_per_image():
    model = a_model().eval()
    out = model(torch.randn(2, 3, 128, 256))
    decoded = model.decode(out, threshold=0.5)
    assert len(decoded) == 2
    assert all(isinstance(d, np.ndarray) and d.ndim == 2 and d.shape[1] == 2 for d in decoded)


def test_decode_threshold_controls_how_many_points_survive():
    model = a_model().eval()
    out = model(torch.randn(1, 3, 128, 256))
    assert len(model.decode(out, threshold=0.01)[0]) >= len(model.decode(out, threshold=0.99)[0])


def test_satisfies_the_same_contract_as_the_heatmap_counter():
    for name in ("p2p", "heatmap"):
        model = build_counter(name, pretrained=False) if name == "p2p" else build_counter(name)
        assert hasattr(model, "loss")
        assert hasattr(model, "decode")
        assert getattr(type(model), "INPUT_DIVISOR", 1) == 4
