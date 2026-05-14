# -*- coding: utf-8 -*-
"""
Step 5: 单图/批量图片预测
输入：RGB 图片（自动生成事件帧占位）或 .npy 双通道文件
"""

import torch
import numpy as np
import cv2
import argparse
from pathlib import Path

from model import build_model

FRAME_H, FRAME_W = 260, 346
LABELS = {0: "No Fall", 1: "FALL"}
COLORS = {0: (0, 200, 0), 1: (0, 0, 255)}


def load_model(ckpt_path: str, device):
    model = build_model(pretrained=False).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def preprocess_image(img_path: str) -> torch.Tensor:
    """RGB 图片 → (1,6,H,W) tensor（事件帧置零）"""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
    rgb = cv2.resize(img, (FRAME_W, FRAME_H)).astype(np.float32) / 255.0
    ev  = np.zeros_like(rgb)
    dual = np.concatenate([rgb, ev], axis=2)   # (H,W,6)
    return torch.from_numpy(dual.transpose(2, 0, 1)).unsqueeze(0)


def preprocess_npy(npy_path: str) -> torch.Tensor:
    """加载 .npy 双通道文件 → (1,6,H,W)"""
    dual = np.load(npy_path).astype(np.float32)   # (H,W,6)
    return torch.from_numpy(dual.transpose(2, 0, 1)).unsqueeze(0)


@torch.no_grad()
def predict_one(model, tensor: torch.Tensor, device) -> tuple:
    tensor = tensor.to(device)
    out    = model(tensor)
    prob   = torch.softmax(out, dim=1)[0]
    pred   = int(out.argmax(1).item())
    return pred, float(prob[1].item())


def predict_and_show(model, path: str, device):
    """预测并显示结果"""
    if path.endswith(".npy"):
        tensor = preprocess_npy(path)
        dual   = np.load(path).astype(np.float32)
        rgb_show = (dual[:, :, :3] * 255).astype(np.uint8)
        ev_show  = (dual[:, :, 3:] * 255).astype(np.uint8)
    else:
        tensor   = preprocess_image(path)
        img      = cv2.imread(path)
        rgb_show = cv2.resize(img, (FRAME_W, FRAME_H))
        ev_show  = np.zeros_like(rgb_show)

    pred, prob = predict_one(model, tensor, device)
    color = COLORS[pred]
    label = LABELS[pred]

    # 标注
    cv2.putText(rgb_show, f"{label} ({prob:.2f})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.rectangle(rgb_show, (0, 0),
                  (FRAME_W - 1, FRAME_H - 1), color, 3)

    display = np.hstack([rgb_show, ev_show])
    cv2.imshow(f"Prediction: {Path(path).name}", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return pred, prob


def batch_predict(model, folder: str, device, ext=(".jpg", ".npy")):
    """批量预测文件夹"""
    folder = Path(folder)
    files  = [f for f in folder.rglob("*") if f.suffix in ext]
    print(f"批量预测 {len(files)} 个文件...")

    results = []
    for fp in files:
        try:
            if fp.suffix == ".npy":
                tensor = preprocess_npy(str(fp))
            else:
                tensor = preprocess_image(str(fp))
            pred, prob = predict_one(model, tensor, device)
            results.append((str(fp), pred, prob))
            print(f"  {fp.name}: {LABELS[pred]} ({prob:.3f})")
        except Exception as e:
            print(f"  [error] {fp.name}: {e}")

    fall_n = sum(1 for _, p, _ in results if p == 1)
    print(f"\n结果: {fall_n}/{len(results)} 预测为跌倒")
    return results


def main():
    parser = argparse.ArgumentParser(description="跌倒检测单图/批量预测")
    parser.add_argument("input", help="图片路径、.npy 路径或文件夹")
    parser.add_argument("--ckpt", default="checkpoints/best.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(args.ckpt, device)
    print(f"模型加载完成: {args.ckpt}")

    p = Path(args.input)
    if p.is_dir():
        batch_predict(model, str(p), device)
    else:
        pred, prob = predict_and_show(model, str(p), device)
        print(f"预测结果: {LABELS[pred]}  置信度: {prob:.4f}")


if __name__ == "__main__":
    main()
