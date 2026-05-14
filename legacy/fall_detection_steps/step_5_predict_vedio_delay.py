# -*- coding: utf-8 -*-
"""
Step 5 Video Delay: 视频推理（带延迟缓冲）
模拟真实部署场景：事件流有时间延迟，用滑动窗口缓冲帧
支持：延迟补偿 + 时序 LSTM 平滑（可选）
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
FALL_CONF        = 0.58
BUFFER_SIZE      = 8        # 帧缓冲大小（模拟延迟）
TEMPORAL_WINDOW  = 10       # 时序平滑窗口


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


def load_model(ckpt_path, device):
    model = build_model(pretrained=False).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


class DelayBuffer:
    """模拟事件流延迟：RGB 帧先入缓冲，延迟 N 帧后才与事件帧配对"""
    def __init__(self, delay: int = 4):
        self.delay    = delay
        self.rgb_buf  = deque(maxlen=delay + 1)
        self.ev_buf   = deque(maxlen=delay + 1)

    def push(self, rgb, ev_frame):
        self.rgb_buf.append(rgb)
        self.ev_buf.append(ev_frame)

    def get_delayed_pair(self):
        """返回延迟后的 (rgb, ev_frame) 或 None"""
        if len(self.rgb_buf) < self.delay + 1:
            return None, None
        # RGB 用最新帧，事件帧用延迟帧
        rgb_now = self.rgb_buf[-1]
        ev_old  = self.ev_buf[0]
        return rgb_now, ev_old


def process_video_delay(source, ckpt_path, delay=4, out_path=None, show=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(ckpt_path, device)
    print(f"Model: {ckpt_path}  Delay: {delay} frames  Device: {device}")

    cap = cv2.VideoCapture(source if isinstance(source, str) else int(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {source}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None
    if out_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps_in,
                                 (FRAME_W * 2, FRAME_H + 60))

    buffer       = DelayBuffer(delay=delay)
    prob_history = deque(maxlen=TEMPORAL_WINDOW)
    prev_gray    = None
    fall_count   = 0
    frame_count  = 0
    prev_time    = time.time()

    # 延迟统计
    latency_ms_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t_start = time.time()
        rgb  = cv2.resize(frame, (FRAME_W, FRAME_H))
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            events   = simulate_events(prev_gray, gray)
            ev_frame = events_to_frame(events, FRAME_H, FRAME_W)
        else:
            ev_frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)

        buffer.push(rgb.copy(), ev_frame.copy())
        rgb_d, ev_d = buffer.get_delayed_pair()

        prob = 0.0
        if rgb_d is not None:
            rgb_f = rgb_d.astype(np.float32) / 255.0
            ev_f  = ev_d.astype(np.float32)  / 255.0
            dual  = np.concatenate([rgb_f, ev_f], axis=2)
            tensor = torch.from_numpy(
                dual.transpose(2, 0, 1)).unsqueeze(0).to(device)
            with torch.no_grad():
                out  = model(tensor)
                prob = float(torch.softmax(out, dim=1)[0, 1].item())

        prob_history.append(prob)
        smooth_prob = float(np.mean(prob_history))
        confirmed   = smooth_prob > FALL_CONF

        latency_ms = (time.time() - t_start) * 1000
        latency_ms_list.append(latency_ms)

        if confirmed:
            fall_count += 1

        # 可视化
        color = (0, 0, 255) if confirmed else (0, 220, 0)
        label = "FALL!" if confirmed else "Normal"

        rgb_show = rgb.copy()
        cv2.putText(rgb_show, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(rgb_show, f"p={smooth_prob:.3f}", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        now = time.time()
        fps = 1.0 / (now - prev_time + 1e-6)
        prev_time = now
        cv2.putText(rgb_show, f"FPS:{fps:.1f} Delay:{delay}f",
                    (FRAME_W - 140, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (200, 200, 200), 1)

        # 概率条
        bar_w = int(smooth_prob * (FRAME_W - 20))
        cv2.rectangle(rgb_show, (10, FRAME_H - 20),
                      (10 + bar_w, FRAME_H - 8), color, -1)

        if confirmed:
            h_ = rgb_show.shape[0]
            cv2.rectangle(rgb_show, (0, h_ - 40), (FRAME_W, h_),
                          (0, 0, 180), -1)
            cv2.putText(rgb_show, "ELDERLY FALL ALERT",
                        (8, h_ - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)

        ev_show = ev_frame.copy()
        cv2.putText(ev_show, f"Event (delay={delay}f)", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        display = np.hstack([rgb_show, ev_show])

        # 底部延迟信息栏
        info_bar = np.zeros((60, FRAME_W * 2, 3), dtype=np.uint8)
        avg_lat  = np.mean(latency_ms_list[-30:]) if latency_ms_list else 0
        cv2.putText(info_bar,
                    f"Latency: {avg_lat:.1f}ms  "
                    f"Buffer: {len(buffer.rgb_buf)}/{BUFFER_SIZE}  "
                    f"Fall frames: {fall_count}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (200, 200, 200), 1)
        display = np.vstack([display, info_bar])

        if writer:
            writer.write(display)
        if show:
            cv2.imshow("Fall Detection (Delay Mode)", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        prev_gray = gray
        frame_count += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    avg_lat = np.mean(latency_ms_list) if latency_ms_list else 0
    print(f"\n处理完成: {frame_count} 帧  跌倒报警: {fall_count} 帧")
    print(f"平均推理延迟: {avg_lat:.2f}ms")
    if out_path:
        print(f"输出视频: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="视频跌倒检测（延迟缓冲版）")
    parser.add_argument("--source", default="0")
    parser.add_argument("--ckpt",   default="checkpoints/best.pth")
    parser.add_argument("--delay",  type=int, default=4,
                        help="事件流延迟帧数")
    parser.add_argument("--out",    default=None)
    parser.add_argument("--noshow", action="store_true")
    args = parser.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    process_video_delay(src, args.ckpt, args.delay,
                        args.out, show=not args.noshow)


if __name__ == "__main__":
    main()
