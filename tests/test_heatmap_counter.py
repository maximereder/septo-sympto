import torch

from septosympto.models import HeatmapCounter, available_counters, build_counter, counter_class


def test_heatmap_is_registered():
    assert "heatmap" in available_counters()
    assert counter_class("heatmap") is HeatmapCounter
    assert isinstance(build_counter("heatmap"), HeatmapCounter)


def test_forward_downsamples_by_the_stride():
    model = HeatmapCounter().eval()
    out = model(torch.zeros(2, 3, 64, 256))
    assert out.shape == (2, 1, 64 // 4, 256 // 4)


def test_loss_is_lower_when_the_heatmap_matches_the_points():
    model = HeatmapCounter().eval()
    points = [torch.tensor([[32.0, 32.0], [200.0, 40.0]])]
    logits = model(torch.zeros(1, 3, 64, 256))

    on_target = torch.full_like(logits, -6.0)
    for px, py in (points[0] / 4).tolist():
        on_target[0, 0, int(py), int(px)] = 6.0

    assert model.loss(on_target, points) < model.loss(torch.full_like(logits, -6.0), points)


def test_decode_recovers_planted_peaks_as_points():
    model = HeatmapCounter().eval()
    logits = torch.full((1, 1, 32, 128), -6.0)
    logits[0, 0, 10, 20] = 6.0
    logits[0, 0, 25, 90] = 6.0

    points = model.decode(logits, threshold=0.3)[0]
    assert len(points) == 2
    xs = sorted(int(round(x)) for x, _ in points)
    assert xs == [20 * 4 + 2, 90 * 4 + 2]


def test_decode_returns_nothing_below_threshold():
    model = HeatmapCounter().eval()
    logits = torch.full((1, 1, 16, 16), -6.0)
    assert len(model.decode(logits, threshold=0.3)[0]) == 0


def test_overfits_one_image_end_to_end():
    """A few steps on a single image should make the count roughly right."""
    torch.manual_seed(0)
    model = HeatmapCounter()
    image = torch.rand(1, 3, 64, 256)
    points = [torch.tensor([[40.0, 30.0], [120.0, 40.0], [200.0, 20.0]])]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    for _ in range(60):
        optimizer.zero_grad()
        loss = model.loss(model(image), points)
        loss.backward()
        optimizer.step()

    model.eval()
    predicted = model.decode(model(image), threshold=0.3)[0]
    assert 2 <= len(predicted) <= 4
