# -*- coding: utf-8 -*-
"""
Step 5 Video Final: 视频实时推理（最终版）
双流 RGB+事件流 模型 + YOLOv8 姿态辅助
输出：带标注的视频文件
"""

import cv2
import numpy as np
import torch
import time
import argparse
from pathlib import Path
from collections import deque

from model import build_model

FRAME_H, FRAME_W = 260, 346
EV_THRESHOLD     = 0.12
FALL_CONF        = 0.60      # 模型输出概率阈值
SMOOTH_FRAMES    = 6         # 平滑窗口（连续N帧超阈值才报警）


def simulate_events(prev_gray, curr_gray):
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


def events_to_frame(events, h, w):
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    if len(events) == 0:
        return frame
    xs = np.clip(events[:, 1].astype(int), 0, w - 1)
    ys = np.clip(events[:, 2].astype(int), 0, h - 1)
    ps = events[:, 3]
    frame[ys[ps > 0], xs[ps > 0]] = [0,   80, 255]
    frame[ys[ps < 0], xs[ps < 0]] = [255, 80, 0  ]
    return frame


def make_tensor(rgb, ev_frame):
    rgb_f = rgb.astype(np.float32) / 255.0
    ev_f  = ev_frame.astype(np.float32) / 255.0
    dual  = np.concatenate([rgb_f, ev_f], axis=2)   # (H,W,6)
    return torch.from_numpy(dual.transpose(2, 0, 1)).unsqueeze(0)


def load_model(ckpt_path, device):
    model = build_model(pretrained=False).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def process_video(source, ckpt_path, out_path=None, show=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(ckpt_path, device)
    print(f"Model loaded: {ckpt_path}  Device: {device}")

    cap = cv2.VideoCapture(source if isinstance(source, str) else int(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {source}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None
    if out_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps_in,
                                 (FRAME_W * 2, FRAME_H))

    prev_gray    = None
    prob_history = deque(maxlen=SMOOTH_FRAMES)
    fall_count   = 0
    frame_count  = 0
    prev_time    = time.time()

    print(f"Processing... Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb  = cv2.resize(frame, (FRAME_W, FRAME_H))
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            events   = simulate_events(prev_gray, gray)
            ev_frame = events_to_frame(events, FRAME_H, FRAME_W)
        else:
            ev_frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)

        # 模型推理
        with torch.no_grad():
            tensor = make_tensor(rgb, ev_frame).to(device)
            out    = model(tensor)
            prob   = float(torch.softmax(out, dim=1)[0, 1].item())

        prob_history.append(prob)
        smooth_prob = float(np.mean(prob_history))
        confirmed   = smooth_prob > FALL_CONF

        if confirmed:
            fall_count += 1

        # 可视化
        color = (0, 0, 255) if confirmed else (0, 220, 0)
        label = "FALL DETECTED!" if confirmed else "Normal"

        # RGB 帧标注
        cv2.putText(rgb, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(rgb, f"p={smooth_prob:.3f}", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        # 概率条
        bar_w = int(smooth_prob * (FRAME_W - 20))
        cv2.rectangle(rgb, (10, FRAME_H - 20), (10 + bar_w, FRAME_H - 8),
                      color, -1)
        cv2.rectangle(rgb, (10, FRAME_H - 20), (FRAME_W - 10, FRAME_H - 8),
                      (100, 100, 100), 1)

        # FPS
        now = time.time()
        fps = 1.0 / (now - prev_time + 1e-6)
        prev_time = now
        cv2.putText(rgb, f"FPS:{fps:.1f}", (FRAME_W - 90, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # 事件帧标注
        cv2.putText(ev_frame, "Event Stream", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # 全屏警告
        if confirmed:
            h_ = rgb.shape[0]
            cv2.rectangle(rgb, (0, h_ - 40), (FRAME_W, h_), (0, 0, 180), -1)
            cv2.putText(rgb, "ELDERLY FALL ALERT",
                        (8, h_ - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)

        display = np.hstack([rgb, ev_frame])

        if writer:
            writer.write(display)
        if show:
            cv2.imshow("Fall Detection | RGB + Event Stream", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        prev_gray = gray
        frame_count += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"\n处理完成: {frame_count} 帧  跌倒报警: {fall_count} 帧")
    if out_path:
        print(f"输出视频: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="视频跌倒检测（最终版）")
    parser.add_argument("--source", default="0",
                        help="摄像头ID或视频路径")
    parser.add_argument("--ckpt",   default="checkpoints/best.pth")
    parser.add_argument("--out",    default=None, help="输出视频路径")
    parser.add_argument("--noshow", action="store_true")
    args = parser.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    process_video(src, args.ckpt, args.out, show=not args.noshow)


if __name__ == "__main__":
    main()
