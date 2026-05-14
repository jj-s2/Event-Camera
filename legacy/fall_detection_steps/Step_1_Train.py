# -*- coding: utf-8 -*-
"""
Step 1: 全参数训练
训练双流 RGB+事件流 跌倒检测模型（所有层均参与训练）
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import json, time

from model import build_model
from dataset_loader import get_loaders

# ── 配置 ──────────────────────────────────────────────
CFG = {
    "data_dir":    "data",
    "batch_size":  16,
    "epochs":      30,
    "lr":          1e-3,
    "weight_decay":1e-4,
    "save_dir":    "checkpoints",
    "num_workers": 0,
    "device":      "cuda" if torch.cuda.is_available() else "cpu",
}


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out  = model(x)
        loss = criterion(out, y)
        loss.backward()
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
    save_dir = Path(CFG["save_dir"])
    save_dir.mkdir(exist_ok=True)
    device = torch.device(CFG["device"])
    print(f"Device: {device}")

    train_loader, val_loader, _ = get_loaders(
        CFG["data_dir"], CFG["batch_size"], CFG["num_workers"])

    if len(train_loader.dataset) == 0:
        print("数据集为空，请先运行 Step_0_Convert_Dataset.py")
        return

    model     = build_model(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=CFG["lr"],
                      weight_decay=CFG["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=CFG["epochs"])

    best_f1   = 0.0
    history   = []

    for epoch in range(1, CFG["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, metrics = evaluate(
            model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch:03d}/{CFG['epochs']}  "
              f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  "
              f"F1={metrics['f1']:.4f}  {elapsed:.1f}s")

        history.append({"epoch": epoch, "tr_loss": tr_loss,
                        "val_loss": val_loss, "val_acc": val_acc,
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1": metrics["f1"],
                        "tp": metrics["tp"], "fp": metrics["fp"],
                        "tn": metrics["tn"], "fn": metrics["fn"]})

        # 保存最优模型
        if metrics["f1"] >= best_f1:
            best_f1 = metrics["f1"]
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "scheduler":  scheduler.state_dict(),
                "best_f1":    best_f1,
                "cfg":        CFG,
            }, save_dir / "best.pth")
            print(f"  ✓ Saved best.pth  F1={best_f1:.4f}")

        # 每5轮保存一次
        if epoch % 5 == 0:
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "scheduler":  scheduler.state_dict(),
                "cfg":        CFG,
            }, save_dir / f"epoch_{epoch:03d}.pth")

    # 保存训练历史
    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n训练完成，最优 F1={best_f1:.4f}，模型保存在 {save_dir}/")


if __name__ == "__main__":
    main()
