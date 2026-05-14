# -*- coding: utf-8 -*-
"""
Step 3: 断点续训
从任意 checkpoint 恢复训练，支持修改 epoch/lr
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import json, time, argparse

from model import build_model
from dataset_loader import get_loaders


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out  = model(x)
        loss = criterion(out, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct    += (out.argmax(1) == y).sum().item()
        total      += len(y)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    tp = fp = tn = fn = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out  = model(x)
        loss = criterion(out, y)
        pred = out.argmax(1)
        total_loss += loss.item() * len(y)
        correct    += (pred == y).sum().item()
        total      += len(y)
        tp += ((pred == 1) & (y == 1)).sum().item()
        fp += ((pred == 1) & (y == 0)).sum().item()
        tn += ((pred == 0) & (y == 0)).sum().item()
        fn += ((pred == 0) & (y == 1)).sum().item()
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-6)
    return (total_loss / total, correct / total,
            {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
             "precision": prec, "recall": rec, "f1": f1})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",    default="checkpoints/best.pth")
    parser.add_argument("--epochs",  type=int, default=10)
    parser.add_argument("--lr",      type=float, default=None)
    parser.add_argument("--data",    default="data")
    parser.add_argument("--save",    default="checkpoints")
    parser.add_argument("--batch",   type=int, default=16)
    args = parser.parse_args()

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save)
    save_dir.mkdir(exist_ok=True)

    # 加载 checkpoint
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"Checkpoint 不存在: {ckpt_path}")
        return

    ckpt       = torch.load(str(ckpt_path), map_location=device)
    start_epoch= ckpt.get("epoch", 0) + 1
    best_f1    = ckpt.get("best_f1", 0.0)
    cfg        = ckpt.get("cfg", {})
    lr         = args.lr or cfg.get("lr", 1e-4)

    print(f"Resume from epoch {start_epoch}  best_f1={best_f1:.4f}  lr={lr}")

    train_loader, val_loader, _ = get_loaders(
        args.data, args.batch, 0)

    model = build_model(pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 恢复 optimizer 状态（如果有）
    if "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
            # 覆盖 lr
            for pg in optimizer.param_groups:
                pg["lr"] = lr
        except Exception:
            pass

    history = []
    end_epoch = start_epoch + args.epochs - 1

    for epoch in range(start_epoch, end_epoch + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, metrics = evaluate(
            model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch:03d}/{end_epoch}  "
              f"tr_acc={tr_acc:.4f}  val_acc={val_acc:.4f}  "
              f"F1={metrics['f1']:.4f}  {time.time()-t0:.1f}s")

        history.append({"epoch": epoch, "val_acc": val_acc,
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1": metrics["f1"],
                        "tp": metrics["tp"], "fp": metrics["fp"],
                        "tn": metrics["tn"], "fn": metrics["fn"]})

        if metrics["f1"] >= best_f1:
            best_f1 = metrics["f1"]
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "scheduler":   scheduler.state_dict(),
                "best_f1":     best_f1,
                "cfg":         cfg,
            }, save_dir / "best.pth")
            print(f"  ✓ Updated best.pth  F1={best_f1:.4f}")

        torch.save({
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "optimizer":   optimizer.state_dict(),
            "best_f1":     best_f1,
        }, save_dir / "last.pth")

    # 追加历史
    hist_path = save_dir / "history.json"
    if hist_path.exists():
        with open(hist_path) as f:
            existing = json.load(f)
    else:
        existing = []
    with open(hist_path, "w") as f:
        json.dump(existing + history, f, indent=2)

    print(f"\n续训完成，最优 F1={best_f1:.4f}")


if __name__ == "__main__":
    main()
