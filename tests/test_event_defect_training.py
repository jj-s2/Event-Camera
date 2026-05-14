import torch


def test_boxes_to_heatmap_marks_defect_area():
    from event_defect.training import boxes_to_heatmap

    targets = [
        {"boxes": torch.tensor([[2.0, 1.0, 4.0, 3.0]])},
        {"boxes": torch.empty((0, 4))},
    ]

    heatmap = boxes_to_heatmap(targets, height=6, width=8)

    assert tuple(heatmap.shape) == (2, 1, 6, 8)
    assert heatmap[0, 0, 1:4, 2:5].sum().item() == 9
    assert heatmap[1].sum().item() == 0


def test_compute_detection_loss_accepts_model_outputs():
    from event_defect.training import compute_detection_loss

    outputs = {
        "heatmap": torch.zeros(2, 1, 6, 8),
        "class_logits": torch.zeros(2, 2),
        "anomaly_score": torch.zeros(2, 1),
    }
    targets = [
        {
            "boxes": torch.tensor([[2.0, 1.0, 4.0, 3.0]]),
            "is_defect": torch.tensor(1),
        },
        {
            "boxes": torch.empty((0, 4)),
            "is_defect": torch.tensor(0),
        },
    ]

    loss, parts = compute_detection_loss(outputs, targets)

    assert loss.item() > 0
    assert {"heatmap", "classification", "anomaly"} <= set(parts)
