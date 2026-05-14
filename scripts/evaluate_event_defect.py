from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from event_defect.checkpoint import load_detector_checkpoint
from event_defect.dataset import EventDefectDataset
from event_defect.training import collate_detection_batch, evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an event defect checkpoint on a manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_detector_checkpoint(args.ckpt, device)
    ds = EventDefectDataset(
        args.manifest,
        height=int(cfg.get("height", 260)),
        width=int(cfg.get("width", 346)),
        bins=int(cfg.get("bins", 5)),
        class_to_idx=cfg.get("class_to_idx"),
        pseudo_box_quantile=float(cfg.get("pseudo_box_quantile", 0.85)),
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_detection_batch,
    )
    metrics = evaluate(model, loader, device)
    text = json.dumps(metrics, indent=2)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
