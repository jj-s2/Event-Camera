from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from event_defect.data import load_event_h5
from event_defect.dataset import normalize_label
from event_defect.manifest import read_manifest
from event_defect.representation import build_event_tensor


def row_label(row: dict[str, str]) -> int:
    return 0 if normalize_label(row.get("label", "normal")) == "normal" else 1


def event_features(row: dict[str, str], height: int, width: int, bins: int, size: int = 24) -> np.ndarray:
    events = load_event_h5(row["event_path"])
    tensor = build_event_tensor(events, height, width, bins).astype(np.float32)
    maps = [cv2.resize(channel, (size, size), interpolation=cv2.INTER_AREA).ravel() for channel in tensor]
    flat = np.concatenate(maps)
    channel_pixels = tensor.reshape(tensor.shape[0], -1)
    stats = np.concatenate(
        [
            channel_pixels.mean(axis=1),
            channel_pixels.std(axis=1),
            channel_pixels.max(axis=1),
            np.quantile(channel_pixels, [0.25, 0.5, 0.75, 0.9, 0.95, 0.99], axis=1).ravel(),
        ]
    )
    if events.size:
        event_stats = np.array(
            [
                len(events),
                events[:, 0].mean(),
                events[:, 1].mean(),
                events[:, 0].std(),
                events[:, 1].std(),
                (events[:, 3] > 0).mean(),
            ],
            dtype=np.float32,
        )
    else:
        event_stats = np.zeros(6, dtype=np.float32)
    return np.concatenate([flat, stats.astype(np.float32), event_stats]).astype(np.float32)


def image_features(row: dict[str, str]) -> np.ndarray:
    image_path = row.get("image_path")
    if not image_path:
        raise ValueError("Manifest row is missing image_path; regenerate events with simulate_events_from_images.py")
    image = read_image(image_path)
    if image is None:
        raise FileNotFoundError(image_path)
    image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 5)
    residual = cv2.absdiff(gray, blur)
    edges = cv2.Canny(gray, 40, 120)

    gray_small = cv2.resize(gray, (40, 40), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() / 255.0
    residual_small = cv2.resize(residual, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() / 255.0
    edge_small = cv2.resize(edges, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() / 255.0
    color_small = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() / 255.0

    histograms: list[float] = []
    color_spaces = [image, cv2.cvtColor(image, cv2.COLOR_BGR2HSV), cv2.cvtColor(image, cv2.COLOR_BGR2LAB)]
    for color_space in color_spaces:
        for channel in cv2.split(color_space):
            hist = cv2.calcHist([channel], [0], None, [32], [0, 256]).ravel().astype(np.float32)
            histograms.extend((hist / max(float(hist.sum()), 1.0)).tolist())

    rings: list[float] = []
    yy, xx = np.mgrid[0:128, 0:128]
    radius = np.sqrt((xx - 63.5) ** 2 + (yy - 63.5) ** 2)
    for lo, hi in [(0, 24), (24, 38), (38, 52), (52, 70), (70, 100)]:
        pixels = gray[(radius >= lo) & (radius < hi)]
        rings.extend(
            [
                float(pixels.mean()) / 255.0,
                float(pixels.std()) / 255.0,
                float(np.quantile(pixels, 0.1)) / 255.0,
                float(np.quantile(pixels, 0.9)) / 255.0,
            ]
        )

    return np.concatenate(
        [
            gray_small,
            residual_small,
            edge_small,
            color_small,
            np.array(histograms, dtype=np.float32),
            np.array(rings, dtype=np.float32),
        ]
    ).astype(np.float32)


def read_image(path: str) -> np.ndarray | None:
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def make_xy(rows: list[dict[str, str]], mode: str, height: int, width: int, bins: int) -> tuple[np.ndarray, np.ndarray]:
    features = []
    for row in rows:
        if mode == "event":
            feature = event_features(row, height, width, bins)
        elif mode == "image":
            feature = image_features(row)
        elif mode == "fusion":
            feature = np.concatenate([event_features(row, height, width, bins), image_features(row)])
        else:
            raise ValueError(f"Unknown mode: {mode}")
        features.append(feature)
    labels = np.array([row_label(row) for row in rows], dtype=np.int64)
    return np.stack(features), labels


def candidate_models() -> list[tuple[str, Any]]:
    models: list[tuple[str, Any]] = []
    for c_value in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
        models.append(
            (
                f"logreg_C{c_value:g}",
                make_pipeline(
                    StandardScaler(),
                    LogisticRegression(C=c_value, class_weight="balanced", max_iter=10000),
                ),
            )
        )
    models.extend(
        [
            (
                "gb_depth2_lr005",
                GradientBoostingClassifier(random_state=42, n_estimators=120, learning_rate=0.05, max_depth=2),
            ),
            (
                "gb_depth2_lr008",
                GradientBoostingClassifier(random_state=43, n_estimators=80, learning_rate=0.08, max_depth=2),
            ),
            (
                "hist_gb_l2",
                HistGradientBoostingClassifier(random_state=42, l2_regularization=0.01, max_iter=200, learning_rate=0.04),
            ),
        ]
    )
    return models


def scores(estimator: Any, x_values: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(x_values)[:, 1]
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(x_values).astype(float)
    return estimator.predict(x_values).astype(float)


def select_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, dict[str, float | int]]:
    unique = np.unique(y_score)
    candidates = np.r_[unique.min() - 1e-9, (unique[:-1] + unique[1:]) / 2.0, unique.max() + 1e-9]
    best_threshold = float(candidates[0])
    best_summary: dict[str, float | int] = {}
    best_key = (-1.0, -1.0)
    for threshold in candidates:
        pred = (y_score >= threshold).astype(np.int64)
        summary = summarize_predictions(y_true, pred)
        key = ((float(summary["accuracy"]) + float(summary["f1"])) / 2.0, float(summary["recall"]))
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_summary = summary
    return best_threshold, best_summary


def summarize_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train calibrated feature ensemble for event-camera defect detection.")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--mode", choices=["event", "image", "fusion"], default="fusion")
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--bins", type=int, default=4)
    parser.add_argument("--out-dir", default="experiments/event_defect_feature_ensemble")
    args = parser.parse_args()

    train_rows = read_manifest(args.train_manifest)
    val_rows = read_manifest(args.val_manifest)
    test_rows = read_manifest(args.test_manifest)

    x_train, y_train = make_xy(train_rows, args.mode, args.height, args.width, args.bins)
    x_val, y_val = make_xy(val_rows, args.mode, args.height, args.width, args.bins)
    x_test, y_test = make_xy(test_rows, args.mode, args.height, args.width, args.bins)

    leaderboard = []
    best: tuple[float, str, Any, float, dict[str, float | int]] | None = None
    for name, estimator in candidate_models():
        estimator.fit(x_train, y_train)
        threshold, val_metrics = select_threshold(y_val, scores(estimator, x_val))
        test_pred = (scores(estimator, x_test) >= threshold).astype(np.int64)
        test_metrics = summarize_predictions(y_test, test_pred)
        rank_score = (float(val_metrics["accuracy"]) + float(val_metrics["f1"])) / 2.0
        leaderboard.append(
            {
                "model": name,
                "threshold": threshold,
                "val": val_metrics,
                "test": test_metrics,
            }
        )
        if best is None or rank_score > best[0]:
            best = (rank_score, name, estimator, threshold, test_metrics)

    assert best is not None
    _, best_name, best_estimator, best_threshold, best_test = best
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "mode": args.mode,
            "height": args.height,
            "width": args.width,
            "bins": args.bins,
            "model_name": best_name,
            "threshold": best_threshold,
            "estimator": best_estimator,
        },
        out_dir / "feature_ensemble.joblib",
    )
    report = {
        "mode": args.mode,
        "feature_shape": {
            "train": list(x_train.shape),
            "val": list(x_val.shape),
            "test": list(x_test.shape),
        },
        "best_model": best_name,
        "best_threshold": best_threshold,
        "best_test": best_test,
        "leaderboard": sorted(
            leaderboard,
            key=lambda item: (
                float(item["val"]["accuracy"]) + float(item["val"]["f1"]),
                float(item["test"]["accuracy"]),
            ),
            reverse=True,
        ),
    }
    text = json.dumps(report, indent=2)
    (out_dir / "metrics.json").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
