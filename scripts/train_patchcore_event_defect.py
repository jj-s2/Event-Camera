from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import cv2
import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.neighbors import NearestNeighbors
from torchvision.models import ResNet18_Weights, resnet18

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from event_defect.data import load_event_h5
from event_defect.dataset import normalize_label
from event_defect.manifest import read_manifest
from event_defect.representation import build_event_tensor


def read_image(path: str) -> np.ndarray | None:
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def row_label(row: dict[str, str]) -> int:
    return 0 if normalize_label(row.get("label", "normal")) == "normal" else 1


def event_rgb(row: dict[str, str], height: int, width: int, bins: int) -> np.ndarray:
    events = load_event_h5(row["event_path"])
    tensor = build_event_tensor(events, height, width, bins)
    positive = tensor[:bins].sum(axis=0)
    negative = tensor[bins : bins * 2].sum(axis=0)
    density = tensor[bins * 2]
    image = np.stack([positive, negative, density], axis=2).astype(np.float32)
    for channel_idx in range(image.shape[2]):
        channel = image[:, :, channel_idx]
        high = float(np.percentile(channel, 99)) if np.any(channel) else 1.0
        image[:, :, channel_idx] = np.clip(channel / max(high, 1e-6), 0.0, 1.0)
    return (image * 255.0).astype(np.uint8)


def row_rgb_image(row: dict[str, str], input_mode: str, height: int, width: int, bins: int) -> np.ndarray:
    if input_mode == "event":
        return event_rgb(row, height=height, width=width, bins=bins)

    image_path = row.get("image_path")
    if not image_path:
        raise ValueError("Manifest row is missing image_path; regenerate events with simulate_events_from_images.py")
    image = read_image(image_path)
    if image is None:
        raise FileNotFoundError(image_path)
    image = image[:, :, ::-1].copy()
    if input_mode == "aps":
        return image

    if input_mode == "fusion":
        event_image = cv2.resize(event_rgb(row, height=height, width=width, bins=bins), (image.shape[1], image.shape[0]))
        return cv2.addWeighted(image, 0.7, event_image, 0.3, 0.0)

    raise ValueError(f"Unknown input mode: {input_mode}")


class ResNetPatchExtractor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)
        self.transforms = weights.transforms()
        self.layer2_path = torch.nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
        )
        self.layer3 = model.layer3

    @torch.no_grad()
    def forward(self, image: np.ndarray, device: torch.device) -> np.ndarray:
        tensor = self.transforms(torch.from_numpy(image).permute(2, 0, 1)).unsqueeze(0).to(device)
        layer2_features = self.layer2_path(tensor)
        layer3_features = self.layer3(layer2_features)
        layer3_features = F.interpolate(
            layer3_features,
            size=layer2_features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        features = torch.cat([layer2_features, layer3_features], dim=1)
        patches = features[0].permute(1, 2, 0).reshape(-1, features.shape[1]).detach().cpu().numpy()
        patches = patches.astype(np.float32, copy=False)
        patches /= np.maximum(np.linalg.norm(patches, axis=1, keepdims=True), 1e-6)
        return patches


def labels(rows: list[dict[str, str]]) -> np.ndarray:
    return np.array([row_label(row) for row in rows], dtype=np.int64)


def extract_patch_maps(
    rows: list[dict[str, str]],
    extractor: ResNetPatchExtractor,
    device: torch.device,
    input_mode: str,
    height: int,
    width: int,
    bins: int,
) -> list[np.ndarray]:
    extractor.eval().to(device)
    patch_maps = []
    for row in rows:
        image = row_rgb_image(row, input_mode=input_mode, height=height, width=width, bins=bins)
        patch_maps.append(extractor(image, device=device))
    return patch_maps


def make_memory_bank(
    patch_maps: list[np.ndarray],
    rows: list[dict[str, str]],
    memory_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    normal_maps = [patches for patches, row in zip(patch_maps, rows) if row_label(row) == 0]
    if not normal_maps:
        raise ValueError("Training manifest must include at least one normal sample")
    memory = np.concatenate(normal_maps, axis=0)
    if memory_size < len(memory):
        indices = rng.choice(len(memory), size=memory_size, replace=False)
        memory = memory[indices]
    return memory.astype(np.float32, copy=False)


def score_patch_maps(
    patch_maps: list[np.ndarray],
    memory_bank: np.ndarray,
    score_name: str,
) -> np.ndarray:
    neighbor_index = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(memory_bank)
    scores = []
    for patches in patch_maps:
        distances = neighbor_index.kneighbors(patches, return_distance=True)[0].ravel()
        scores.append(patch_score(distances, score_name))
    return np.array(scores, dtype=np.float64)


def patch_score(distances: np.ndarray, score_name: str) -> float:
    if score_name == "max":
        return float(np.max(distances))
    if score_name.startswith("top"):
        count = int(score_name.removeprefix("top"))
        count = max(1, min(count, len(distances)))
        return float(np.mean(np.sort(distances)[-count:]))
    if score_name.startswith("q"):
        quantile = float(score_name.removeprefix("q")) / 100.0
        return float(np.quantile(distances, quantile))
    raise ValueError(f"Unknown score name: {score_name}")


def select_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, dict[str, float | int]]:
    unique = np.unique(y_score)
    candidates = np.r_[unique.min() - 1e-9, (unique[:-1] + unique[1:]) / 2.0, unique.max() + 1e-9]
    best_threshold = float(candidates[0])
    best_metrics: dict[str, float | int] = {}
    best_key = (-1.0, -1.0)
    for threshold in candidates:
        pred = (y_score >= threshold).astype(np.int64)
        metrics = summarize_predictions(y_true, pred)
        key = ((float(metrics["accuracy"]) + float(metrics["f1"])) / 2.0, float(metrics["recall"]))
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


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


def parse_int_list(raw: str) -> list[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def parse_str_list(raw: str) -> list[str]:
    return [value.strip() for value in raw.split(",") if value.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PatchCore-style normal-memory defect detector.")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--input-mode", choices=["aps", "event", "fusion"], default="aps")
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--bins", type=int, default=4)
    parser.add_argument("--memory-sizes", default="5000,10000,20000,40000,80000")
    parser.add_argument("--score-names", default="max,top5,top10,q99,q95")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="experiments/event_defect_patchcore")
    args = parser.parse_args()

    if "TORCH_HOME" not in os.environ:
        local_torch_home = Path.cwd() / ".torch"
        if local_torch_home.exists():
            os.environ["TORCH_HOME"] = str(local_torch_home)

    train_rows = read_manifest(args.train_manifest)
    val_rows = read_manifest(args.val_manifest)
    test_rows = read_manifest(args.test_manifest)
    y_val = labels(val_rows)
    y_test = labels(test_rows)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = ResNetPatchExtractor()
    train_patch_maps = extract_patch_maps(train_rows, extractor, device, args.input_mode, args.height, args.width, args.bins)
    val_patch_maps = extract_patch_maps(val_rows, extractor, device, args.input_mode, args.height, args.width, args.bins)
    test_patch_maps = extract_patch_maps(test_rows, extractor, device, args.input_mode, args.height, args.width, args.bins)

    rng = np.random.default_rng(args.seed)
    leaderboard = []
    best: dict[str, Any] | None = None
    best_memory_bank: np.ndarray | None = None
    for memory_size in parse_int_list(args.memory_sizes):
        memory_bank = make_memory_bank(train_patch_maps, train_rows, memory_size, rng)
        for score_name in parse_str_list(args.score_names):
            val_scores = score_patch_maps(val_patch_maps, memory_bank, score_name)
            threshold, val_metrics = select_threshold(y_val, val_scores)
            test_scores = score_patch_maps(test_patch_maps, memory_bank, score_name)
            test_pred = (test_scores >= threshold).astype(np.int64)
            test_metrics = summarize_predictions(y_test, test_pred)
            entry = {
                "memory_size": int(len(memory_bank)),
                "score_name": score_name,
                "threshold": threshold,
                "val": val_metrics,
                "test": test_metrics,
            }
            leaderboard.append(entry)
            val_key = (float(val_metrics["accuracy"]) + float(val_metrics["f1"])) / 2.0
            if best is None or val_key > float(best["val_selection_score"]):
                best = entry | {"val_selection_score": val_key}
                best_memory_bank = memory_bank.copy()

    assert best is not None and best_memory_bank is not None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "input_mode": args.input_mode,
            "height": args.height,
            "width": args.width,
            "bins": args.bins,
            "threshold": best["threshold"],
            "score_name": best["score_name"],
            "memory_bank": best_memory_bank,
        },
        out_dir / "patchcore_detector.joblib",
    )
    report = {
        "input_mode": args.input_mode,
        "device": str(device),
        "best": best,
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
