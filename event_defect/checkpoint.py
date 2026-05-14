from __future__ import annotations

from pathlib import Path

import torch

from .model import build_model


def load_detector_checkpoint(path: str | Path, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt.get("cfg", {})
    class_to_idx = cfg.get("class_to_idx", {"normal": 0, "defect": 1})
    model = build_model(
        event_channels=int(cfg.get("event_channels", int(cfg.get("bins", 5)) * 2 + 1)),
        image_channels=int(cfg.get("image_channels", 0)),
        num_classes=len(class_to_idx),
        base_channels=int(cfg.get("base_channels", 32)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg
