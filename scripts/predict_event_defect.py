from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch

from event_defect.data import load_event_h5
from event_defect.model import build_model
from event_defect.representation import build_event_tensor


def load_checkpoint(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt.get("cfg", {})
    bins = int(cfg.get("bins", 5))
    height = int(cfg.get("height", 260))
    width = int(cfg.get("width", 346))
    model = build_model(
        event_channels=int(cfg.get("event_channels", bins * 2 + 1)),
        image_channels=0,
        num_classes=len(cfg.get("class_to_idx", {"normal": 0, "defect": 1})),
        base_channels=int(cfg.get("base_channels", 32)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, {
        "bins": bins,
        "height": height,
        "width": width,
        "class_to_idx": cfg.get("class_to_idx", {"normal": 0, "defect": 1}),
    }


@torch.no_grad()
def predict(ckpt_path: Path, event_path: Path, heatmap_out: Path | None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_checkpoint(ckpt_path, device)
    events = load_event_h5(event_path)
    x = build_event_tensor(events, cfg["height"], cfg["width"], cfg["bins"])
    tensor = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(device)
    outputs = model(tensor)
    probs = outputs["class_logits"].softmax(dim=1)[0].cpu()
    idx_to_class = {idx: name for name, idx in cfg["class_to_idx"].items()}
    pred_idx = int(probs.argmax().item())
    defect_prob = float(probs[1:].sum().item()) if len(probs) > 1 else 0.0
    anomaly = float(outputs["anomaly_score"].sigmoid()[0, 0].cpu())
    print(f"Predicted class:     {idx_to_class.get(pred_idx, str(pred_idx))}")
    print(f"Defect probability: {defect_prob:.4f}")
    print(f"Anomaly score:       {anomaly:.4f}")

    if heatmap_out is not None:
        heatmap = outputs["heatmap"].sigmoid()[0, 0].cpu().numpy()
        heatmap_img = np.uint8(np.clip(heatmap * 255.0, 0, 255))
        heatmap_out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(heatmap_out), heatmap_img)
        print(f"Heatmap saved to {heatmap_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict one h5 event defect sample.")
    parser.add_argument("event_path")
    parser.add_argument("--ckpt", default="checkpoints_event_defect/best.pth")
    parser.add_argument("--heatmap-out", default=None)
    args = parser.parse_args()

    predict(
        ckpt_path=Path(args.ckpt),
        event_path=Path(args.event_path),
        heatmap_out=Path(args.heatmap_out) if args.heatmap_out else None,
    )


if __name__ == "__main__":
    main()
