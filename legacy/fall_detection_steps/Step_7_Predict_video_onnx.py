# -*- coding: utf-8 -*-
"""
Step 7: ONNX 模型视频推理
不依赖 PyTorch，只需 onnxruntime + opencv
适合部署到边缘设备
"""

import cv2
import numpy as np
import time
import argparse
from pathlib import Path
from collections import deque

FRAME_H, FRAME_W = 260, 346
EV_THRESHOLD     = 0.12
FALL_CONF        = 0.58
SMOOTH_FRAMES    = 8


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


def load_onnx_session(onnx_path: str):
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("请安装: pip install onnxruntime")

    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if ort.get_device() == "GPU"
                 else ["CPUExecutionProvider"])
    sess = ort.InferenceSession(onnx_path, providers=providers)
    print(f"ONNX 模型加载: {onnx_path}")
    print(f"  输入: {sess.get_inputs()[0].name}  "
          f"shape={sess.get_inputs()[0].shape}")
    print(f"  Provider: {sess.get_providers()}")
    return sess


def infer_onnx(sess, rgb: np.ndarray, ev_frame: np.ndarray) -> float:
    """返回跌倒概率 [0,1]"""
    rgb_f = rgb.astype(np.float32) / 255.0
    ev_f  = ev_frame.astype(np.float32) / 255.0
    dual  = np.concatenate([rgb_f, ev_f], axis=2)          # (H,W,6)
    inp   = dual.transpose(2, 0, 1)[np.newaxis, ...]        # (1,6,H,W)
    out   = sess.run(None, {"input": inp})[0]               # (1,2)
    # softmax
    exp   = np.exp(out - out.max(axis=1, keepdims=True))
    prob  = exp / exp.sum(axis=1, keepdims=True)
    return float(prob[0, 1])


def process_video(source, onnx_path, out_path=None, show=True):
    sess = load_onnx_session(onnx_path)

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
    latencies    = []

    print("ONNX 推理中... 按 Q 退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0   = time.time()
        rgb  = cv2.resize(frame, (FRAME_W, FRAME_H))
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            events   = simulate_events(prev_gray, gray)
            ev_frame = events_to_frame(events, FRAME_H, FRAME_W)
        else:
            ev_frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)

        prob = infer_onnx(sess, rgb, ev_frame)
        latencies.append((time.time() - t0) * 1000)

        prob_history.append(prob)
        smooth_prob = float(np.mean(prob_history))
        confirmed   = smooth_prob > FALL_CONF

        if confirmed:
            fall_count += 1

        color = (0, 0, 255) if confirmed else (0, 220, 0)
        label = "FALL!" if confirmed else "Normal"

        rgb_show = rgb.copy()
        cv2.putText(rgb_show, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(rgb_show, f"p={smooth_prob:.3f}", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        cv2.putText(rgb_show, "ONNX Runtime", (FRAME_W - 110, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 0), 1)

        now = time.time()
        fps = 1.0 / (now - prev_time + 1e-6)
        prev_time = now
        cv2.putText(rgb_show, f"FPS:{fps:.1f}", (FRAME_W - 80, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # 概率条
        bar_w = int(smooth_prob * (FRAME_W - 20))
        cv2.rectangle(rgb_show, (10, FRAME_H - 20),
                      (10 + bar_w, FRAME_H - 8), color, -1)
        cv2.rectangle(rgb_show, (10, FRAME_H - 20),
                      (FRAME_W - 10, FRAME_H - 8), (80, 80, 80), 1)

        if confirmed:
            h_ = rgb_show.shape[0]
            cv2.rectangle(rgb_show, (0, h_ - 40), (FRAME_W, h_),
                          (0, 0, 180), -1)
            cv2.putText(rgb_show, "ELDERLY FALL ALERT",
                        (8, h_ - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)

        ev_show = ev_frame.copy()
        cv2.putText(ev_show, "Event Stream", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        display = np.hstack([rgb_show, ev_show])

        if writer:
            writer.write(display)
        if show:
            cv2.imshow("Fall Detection ONNX | RGB + Event", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        prev_gray = gray
        frame_count += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    avg_lat = np.mean(latencies) if latencies else 0
    print(f"\n完成: {frame_count} 帧  跌倒报警: {fall_count} 帧")
    print(f"平均推理延迟: {avg_lat:.2f}ms  ({1000/max(avg_lat,1):.1f} FPS)")
    if out_path:
        print(f"输出: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="ONNX 视频跌倒检测")
    parser.add_argument("--source", default="0")
    parser.add_argument("--onnx",   default="exported/fall_detector.onnx")
    parser.add_argument("--out",    default=None)
    parser.add_argument("--noshow", action="store_true")
    args = parser.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    process_video(src, args.onnx, args.out, show=not args.noshow)


if __name__ == "__main__":
    main()
