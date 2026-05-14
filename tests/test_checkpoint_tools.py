import torch


def test_checkpoint_roundtrip_builds_model_with_saved_class_map(tmp_path):
    from event_defect.checkpoint import load_detector_checkpoint
    from event_defect.model import build_model

    model = build_model(event_channels=3, num_classes=2, base_channels=4)
    ckpt_path = tmp_path / "model.pth"
    torch.save(
        {
            "model_state": model.state_dict(),
            "cfg": {
                "event_channels": 3,
                "base_channels": 4,
                "class_to_idx": {"normal": 0, "scratch": 1},
            },
        },
        ckpt_path,
    )

    loaded, cfg = load_detector_checkpoint(ckpt_path, torch.device("cpu"))

    assert cfg["class_to_idx"] == {"normal": 0, "scratch": 1}
    assert loaded(torch.zeros(1, 3, 16, 16))["class_logits"].shape == (1, 2)
