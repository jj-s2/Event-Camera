# -*- coding: utf-8 -*-
"""
Step 0: 数据集转换
将 RGB 视频转换为 RGB+事件流 双通道图像
输出格式：(H, W, 6) = RGB(3) + EventFrame(3) 拼接保存为 .npy
目录结构：
  data/
    train/fall/*.npy   data/train/no_fall/*.npy
    val/fall/*.npy     data/val/no_fall/*.npy
    test/fall/*.npy    data/test/no_fall/*.npy
"""

import cv2
import numpy as np
import os
import random
from pathlib import Path

# ── 配置 ──────────────────────────────────────────────
FRAME_H, FRAME_W = 260, 346          # DAVIS346 分辨率
EV_THRESHOLD     = 0.12              # 事件触发阈值
SAMPLE_INTERVAL  = 3                 # 每隔几帧取一帧
SPLIT_RATIO      = (0.7, 0.15, 0.15) # train/val/test
SEED             = 42
OUT_DIR          = Path("data")

# 输入视频目录（按标签分文件夹）
# 格式：{"fall": ["video1.mp4", ...], "no_fall": [...]}
# 可修改为你的实际路径
VIDEO_DIRS = {
    "fall":    Path("GMDCSA24") if Path("GMDCSA24").exists() else None,
    "no_fall": Path("GMDCSA24") if Path("GMDCSA24").exists() else None,
}


# ── 事件流模拟 ─────────────────────────────────────────
def simulate_events(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
    """从两帧灰度图生成事件流 (N,4): [t, x, y, p]"""
    pl   = np.log1p(prev_gray.astype(np.float32))
    cl   = np.log1p(curr_gray.astype(np.float32))
    diff = cl - pl
    events = []
    for pol, mask in [(1, diff > EV_THRESHOLD), (-1, diff < -EV_THRESHOLD)]:
        ys, xs = np.where(mask)
        if len(xs):
            ts = np.zeros(len(xs), np.float32)
            ps = np.full(len(xs), pol, np.float32)
            events.append(np.stack([ts, xs.astype(np.float32),
                                    ys.astype(np.float32), ps], axis=1))
    return np.concatenate(events) if events else np.zeros((0, 4), np.float32)


def events_to_frame(events: np.ndarray, h: int, w: int) -> np.ndarray:
    """事件流 → 3通道图像 (H,W,3) uint8"""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    if len(events) == 0:
        return frame
    xs = np.clip(events[:, 1].astype(int), 0, w - 1)
    ys = np.clip(events[:, 2].astype(int), 0, h - 1)
    ps = events[:, 3]
    frame[ys[ps > 0], xs[ps > 0]] = [0,   80, 255]   # 正极性 蓝
    frame[ys[ps < 0], xs[ps < 0]] = [255, 80, 0  ]   # 负极性 红
    return frame


def make_dual_frame(rgb: np.ndarray, ev_frame: np.ndarray) -> np.ndarray:
    """拼接 RGB + 事件帧 → (H, W, 6) float32 归一化"""
    rgb_f = rgb.astype(np.float32) / 255.0
    ev_f  = ev_frame.astype(np.float32) / 255.0
    return np.concatenate([rgb_f, ev_f], axis=2)   # (H,W,6)


# ── 视频 → 样本列表 ────────────────────────────────────
def video_to_samples(video_path: str, label: int) -> list:
    """从视频抽帧，返回 [(dual_frame, label), ...]"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [skip] {video_path}")
        return []

    samples   = []
    frame_idx = 0
    prev_gray = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % SAMPLE_INTERVAL == 0:
            rgb  = cv2.resize(frame, (FRAME_W, FRAME_H))
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                events   = simulate_events(prev_gray, gray)
                ev_frame = events_to_frame(events, FRAME_H, FRAME_W)
                dual     = make_dual_frame(rgb, ev_frame)
                samples.append((dual, label))
            prev_gray = gray
        frame_idx += 1

    cap.release()
    return samples


# ── 扫描视频目录 ───────────────────────────────────────
def collect_videos(video_root: Path, label: int) -> list:
    """递归找视频，按标签判断"""
    videos = []
    if video_root is None or not video_root.exists():
        return videos
    for ext in ("*.mp4", "*.avi", "*.mov"):
        for vp in video_root.rglob(ext):
            name = str(vp).lower()
            if label == 1:
                if "fall" in name and "adl" not in name and "no_fall" not in name:
                    videos.append(str(vp))
            else:
                if "adl" in name or "no_fall" in name or "normal" in name:
                    videos.append(str(vp))
    return videos


# ── 保存数据集 ─────────────────────────────────────────
def save_split(samples: list, split: str):
    for dual, label in samples:
        lbl_dir = OUT_DIR / split / ("fall" if label == 1 else "no_fall")
        lbl_dir.mkdir(parents=True, exist_ok=True)
        idx  = len(list(lbl_dir.glob("*.npy")))
        path = lbl_dir / f"{idx:06d}.npy"
        np.save(str(path), dual.astype(np.float32))


# ── 主流程 ─────────────────────────────────────────────
def main():
    random.seed(SEED)
    all_samples = []

    for label, (lbl_name, video_root) in [
            (0, ("no_fall", VIDEO_DIRS["no_fall"])),
            (1, ("fall",    VIDEO_DIRS["fall"]))]:
        videos = collect_videos(video_root, label)
        print(f"[{lbl_name}] 找到 {len(videos)} 个视频")
        for vp in videos:
            s = video_to_samples(vp, label)
            all_samples.extend(s)
            print(f"  {Path(vp).name}: {len(s)} 帧")

    if not all_samples:
        print("\n未找到视频，请先下载数据集（运行 download_dataset.py 选项3）")
        print("或手动设置 VIDEO_DIRS 指向你的视频目录")
        return

    random.shuffle(all_samples)
    n     = len(all_samples)
    n_tr  = int(n * SPLIT_RATIO[0])
    n_val = int(n * SPLIT_RATIO[1])

    train_s = all_samples[:n_tr]
    val_s   = all_samples[n_tr:n_tr + n_val]
    test_s  = all_samples[n_tr + n_val:]

    print(f"\n总样本: {n}  train={len(train_s)} val={len(val_s)} test={len(test_s)}")

    for split, samples in [("train", train_s), ("val", val_s), ("test", test_s)]:
        for s in samples:
            save_split([s], split)
        fall_n = sum(1 for _, l in samples if l == 1)
        print(f"  {split}: fall={fall_n} no_fall={len(samples)-fall_n}")

    print(f"\n数据集已保存到 {OUT_DIR}/")


if __name__ == "__main__":
    main()
