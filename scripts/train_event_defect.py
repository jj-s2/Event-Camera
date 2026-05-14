from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split

from event_defect.dataset import EventDefectDataset
from event_defect.model import build_model
from event_defect.training import collate_detection_batch, evaluate, train_one_epoch


def make_loaders(args):
    full_train_ds = EventDefectDataset(
        args.train_manifest,
        args.height,
        args.width,
        args.bins,
        pseudo_box_quantile=args.pseudo_box_quantile,
    )
    class_to_idx = full_train_ds.class_to_idx
    train_ds = full_train_ds
    if args.val_manifest:
        val_ds = EventDefectDataset(
            args.val_manifest,
            args.height,
            args.width,
            args.bins,
            class_to_idx=class_to_idx,
            pseudo_box_quantile=args.pseudo_box_quantile,
        )
    else:
        val_size = max(1, int(len(train_ds) * args.val_ratio))
        train_size = max(1, len(train_ds) - val_size)
        train_ds, val_ds = random_split(
            train_ds,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed),
        )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_detection_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_detection_batch,
    )
    return train_loader, val_loader, class_to_idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Train event-camera industrial defect detector.")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", default=None)
    parser.add_argument("--height", type=int, default=260)
    parser.add_argument("--width", type=int, default=346)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--pseudo-box-quantile", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", default="checkpoints_event_defect")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, class_to_idx = make_loaders(args)

    event_channels = args.bins * 2 + 1
    model = build_model(
        event_channels=event_channels,
        image_channels=0,
        num_classes=len(class_to_idx),
        base_channels=args.base_channels,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_score = -1.0
    history = []
    cfg = vars(args) | {"event_channels": event_channels, "class_to_idx": class_to_idx}
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        score = val_metrics["f1"] - val_metrics["loss"] * 0.01
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['class_accuracy']:.4f} "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "cfg": cfg,
            "val_metrics": val_metrics,
        }
        torch.save(ckpt, save_dir / "last.pth")
        if score >= best_score:
            best_score = score
            torch.save(ckpt, save_dir / "best.pth")

    with (save_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved checkpoints to {save_dir}")


if __name__ == "__main__":
    main()
