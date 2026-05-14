# -*- coding: utf-8 -*-
"""
Step 4: 训练过程可视化 + 混淆矩阵 + 特征图可视化
"""

import torch
import numpy as np
import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # 无 GUI 环境也能保存图片
import matplotlib.pyplot as plt

from model import build_model
from dataset_loader import get_loaders, FallEventDataset


# ── 1. 训练曲线 ────────────────────────────────────────
def plot_history(history_path: str, out_dir: str):
    with open(history_path) as f:
        hist = json.load(f)

    epochs   = [h["epoch"]   for h in hist]
    val_acc  = [h["val_acc"] for h in hist]
    f1       = [h["f1"]      for h in hist]
    tr_loss  = [h.get("tr_loss", 0) for h in hist]
    val_loss = [h.get("val_loss", 0) for h in hist]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, tr_loss,  label="Train Loss")
    axes[0].plot(epochs, val_loss, label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].set_xlabel("Epoch")

    axes[1].plot(epochs, val_acc, color="green")
    axes[1].set_title("Val Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1)

    axes[2].plot(epochs, f1, color="orange")
    axes[2].set_title("F1 Score")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylim(0, 1)

    plt.tight_layout()
    out = Path(out_dir) / "training_curves.png"
    plt.savefig(str(out), dpi=120)
    plt.close()
    print(f"训练曲线 -> {out}")


# ── 2. 混淆矩阵 ────────────────────────────────────────
def plot_confusion_matrix(ckpt_path: str, data_dir: str, out_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model(pretrained=False).to(device)
    ckpt   = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    from torch.utils.data import DataLoader
    ds     = FallEventDataset(data_dir, "test", augment=False)
    loader = DataLoader(ds, batch_size=16, shuffle=False)

    tp = fp = tn = fn = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            tp += ((pred == 1) & (y == 1)).sum().item()
            fp += ((pred == 1) & (y == 0)).sum().item()
            tn += ((pred == 0) & (y == 0)).sum().item()
            fn += ((pred == 0) & (y == 1)).sum().item()

    cm = np.array([[tn, fp], [fn, tp]])
    labels = ["No Fall", "Fall"]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14)
    plt.colorbar(im)
    plt.tight_layout()

    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-6)
    acc  = (tp + tn) / max(tp + fp + tn + fn, 1)
    print(f"Test  Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")

    out = Path(out_dir) / "confusion_matrix.png"
    plt.savefig(str(out), dpi=120)
    plt.close()
    print(f"混淆矩阵 -> {out}")


# ── 3. 样本可视化（RGB + 事件帧 + 预测结果）─────────────
def visualize_samples(ckpt_path: str, data_dir: str, out_dir: str,
                      n_samples: int = 8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model(pretrained=False).to(device)
    ckpt   = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds = FallEventDataset(data_dir, "test", augment=False)
    indices = list(range(min(n_samples, len(ds))))

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, n_samples * 3))
    # 统一为二维数组，无论 n_samples 是否为 1
    if n_samples == 1:
        axes = np.array([axes])

    with torch.no_grad():
        for row, idx in enumerate(indices):
            tensor, label = ds[idx]
            out   = model(tensor.unsqueeze(0).to(device))
            prob  = torch.softmax(out, dim=1)[0, 1].item()
            pred  = int(out.argmax(1).item())

            rgb_img = tensor[:3].permute(1, 2, 0).numpy()
            ev_img  = tensor[3:].permute(1, 2, 0).numpy()
            diff    = np.abs(rgb_img - ev_img)

            axes[row][0].imshow(np.clip(rgb_img, 0, 1))
            axes[row][0].set_title("RGB")
            axes[row][0].axis("off")

            axes[row][1].imshow(np.clip(ev_img, 0, 1))
            axes[row][1].set_title("Event Frame")
            axes[row][1].axis("off")

            color = "red" if pred == 1 else "green"
            axes[row][2].imshow(np.clip(diff, 0, 1))
            axes[row][2].set_title(
                f"GT={'Fall' if label else 'Normal'}  "
                f"Pred={'Fall' if pred else 'Normal'}  "
                f"p={prob:.2f}", color=color)
            axes[row][2].axis("off")

    plt.tight_layout()
    out = Path(out_dir) / "sample_predictions.png"
    plt.savefig(str(out), dpi=100)
    plt.close()
    print(f"样本预测图 -> {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",    default="checkpoints/best.pth")
    parser.add_argument("--data",    default="data")
    parser.add_argument("--history", default="checkpoints/history.json")
    parser.add_argument("--out",     default="visualizations")
    args = parser.parse_args()

    Path(args.out).mkdir(exist_ok=True)

    if Path(args.history).exists():
        plot_history(args.history, args.out)

    if Path(args.ckpt).exists():
        plot_confusion_matrix(args.ckpt, args.data, args.out)
        visualize_samples(args.ckpt, args.data, args.out)
    else:
        print(f"Checkpoint 不存在: {args.ckpt}")


if __name__ == "__main__":
    main()
