import torch


def test_time_aware_fusion_detector_forward_event_only():
    from event_defect.model import TimeAwareFusionDetector

    model = TimeAwareFusionDetector(event_channels=7, image_channels=0, num_classes=2)
    x = torch.randn(2, 7, 32, 40)

    out = model(x)

    assert out["heatmap"].shape == (2, 1, 32, 40)
    assert out["class_logits"].shape == (2, 2)
    assert out["anomaly_score"].shape == (2, 1)


def test_time_aware_fusion_detector_forward_with_image_branch():
    from event_defect.model import TimeAwareFusionDetector

    model = TimeAwareFusionDetector(event_channels=7, image_channels=1, num_classes=3)
    events = torch.randn(1, 7, 48, 48)
    image = torch.randn(1, 1, 48, 48)

    out = model(events, image)

    assert out["heatmap"].shape == (1, 1, 48, 48)
    assert out["class_logits"].shape == (1, 3)
    assert torch.all((out["anomaly_score"].sigmoid() >= 0) & (out["anomaly_score"].sigmoid() <= 1))
