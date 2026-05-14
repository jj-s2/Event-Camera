# -*- coding: utf-8 -*-
"""
Step 1 Low: 低学习率微调（全参数，小 lr）
适合在已有 checkpoint 基础上精细调整
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from pathlib import Path
import json, time, sys

from model import build_model
from dataset_loader import get_loaders

CFG = {
    "data_dir":    "data",
    "batch_size":  8,
    "epochs":      10,
    "lr":          1e-4,          # 低学习率
    "weight_decay":1e-5,
    "save_dir":    "checkpoints_low",
    "num_workers": 0,
    "device":      "cuda" if torch.cuda.is_available() else "cpu",
    # 从哪个 checkpoint 继续（可选）
    "resume":      "checkpoints/best.pth",
}


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
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
        scheduler.step()
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
    save_dir = Path(CFG["save_dir"])
    save_dir.mkdir(exist_ok=True)
    device = torch.device(CFG["device"])

    train_loader, val_loader, _ = get_loaders(
        CFG["data_dir"], CFG["batch_size"], CFG["num_workers"])

    if len(train_loader.dataset) == 0:
        print("数据集为空，请先运行 Step_0_Convert_Dataset.py")
        return

    model = build_model(pretrained=False).to(device)

    # 加载已有 checkpoint
    resume_path = Path(CFG["resume"])
    if resume_path.exists():
        ckpt = torch.load(str(resume_path), map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"加载 checkpoint: {resume_path}  (epoch={ckpt.get('epoch','?')})")
    else:
        print(f"未找到 {resume_path}，从头开始低学习率训练")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=CFG["lr"],
                      weight_decay=CFG["weight_decay"])
    scheduler = OneCycleLR(
        optimizer, max_lr=CFG["lr"],
        steps_per_epoch=len(train_loader),
        epochs=CFG["epochs"],
    )

    best_f1 = 0.0
    history = []

    for epoch in range(1, CFG["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device)
        val_loss, val_acc, metrics = evaluate(
            model, val_loader, criterion, device)

        print(f"Epoch {epoch:03d}/{CFG['epochs']}  "
              f"tr_acc={tr_acc:.4f}  val_acc={val_acc:.4f}  "
              f"F1={metrics['f1']:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}  "
              f"{time.time()-t0:.1f}s")

        history.append({"epoch": epoch, "val_acc": val_acc,
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1": metrics["f1"],
                        "tp": metrics["tp"], "fp": metrics["fp"],
                        "tn": metrics["tn"], "fn": metrics["fn"]})

        if metrics["f1"] >= best_f1:
            best_f1 = metrics["f1"]
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "best_f1": best_f1, "cfg": CFG},
                       save_dir / "best_low.pth")
            print(f"  ✓ Saved best_low.pth  F1={best_f1:.4f}")

    with open(save_dir / "history_low.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n低学习率微调完成，最优 F1={best_f1:.4f}")


if __name__ == "__main__":
    main()
