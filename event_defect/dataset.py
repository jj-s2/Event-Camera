from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .data import load_event_h5
from .labels import label_to_binary, mask_to_boxes
from .representation import build_event_tensor


class EventDefectDataset(Dataset):
    """Dataset for event-camera industrial defect samples.

    The manifest CSV must contain ``sample_id``, ``event_path`` and ``label``.
    ``mask_path`` is optional but recommended for localization training.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        height: int,
        width: int,
        bins: int = 5,
        min_mask_area: int = 8,
        class_to_idx: dict[str, int] | None = None,
        pseudo_box_quantile: float | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.height = int(height)
        self.width = int(width)
        self.bins = int(bins)
        self.min_mask_area = int(min_mask_area)
        self.pseudo_box_quantile = pseudo_box_quantile
        self.rows = _read_manifest(self.manifest_path)
        self.class_to_idx = class_to_idx or _build_class_map(self.rows)
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        self.num_classes = len(self.class_to_idx)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        row = self.rows[index]
        events = load_event_h5(row["event_path"])
        tensor = build_event_tensor(events, self.height, self.width, self.bins)
        x = torch.from_numpy(tensor.astype(np.float32, copy=False))

        class_name = normalize_label(row.get("label", "normal"))
        class_id = self.class_to_idx[class_name]
        is_defect = label_to_binary(class_name)
        boxes = _load_boxes(row.get("mask_path", ""), self.min_mask_area)
        if is_defect and len(boxes) == 0 and self.pseudo_box_quantile is not None:
            boxes = pseudo_boxes_from_events(
                events,
                height=self.height,
                width=self.width,
                quantile=self.pseudo_box_quantile,
                min_area=self.min_mask_area,
            )
        target = {
            "boxes": _boxes_tensor(boxes),
            "labels": torch.tensor([class_id] * len(boxes), dtype=torch.int64),
            "class_id": torch.tensor(class_id, dtype=torch.int64),
            "is_defect": torch.tensor(is_defect, dtype=torch.int64),
            "sample_id": row.get("sample_id", Path(row["event_path"]).stem),
        }
        if is_defect and len(boxes) == 0:
            target["labels"] = torch.tensor([class_id], dtype=torch.int64)
        return x, target


def _read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    required = {"sample_id", "event_path", "label"}
    if not rows:
        return []
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Manifest {path} missing columns: {sorted(missing)}")
    base = path.parent
    for row in rows:
        row["event_path"] = _resolve_path(base, row["event_path"])
        if row.get("mask_path"):
            row["mask_path"] = _resolve_path(base, row["mask_path"])
    return rows


def _resolve_path(base: Path, raw: str) -> str:
    path = Path(raw)
    if path.is_absolute():
        return str(path)
    return str((base / path).resolve())


def _load_boxes(mask_path: str, min_area: int) -> list[tuple[int, int, int, int]]:
    if not mask_path:
        return []
    path = Path(mask_path)
    if not path.exists():
        return []
    mask = np.asarray(Image.open(path))
    return mask_to_boxes(mask, min_area=min_area)


def normalize_label(label: str | int) -> str:
    if isinstance(label, int):
        return "normal" if label == 0 else "defect"
    normalized = str(label).strip().lower()
    if normalized in {"", "0", "normal", "ok", "good", "no_defect", "none", "background"}:
        return "normal"
    return normalized.replace(" ", "_").replace("-", "_")


def _build_class_map(rows: list[dict[str, str]]) -> dict[str, int]:
    labels = sorted({normalize_label(row.get("label", "normal")) for row in rows})
    ordered = ["normal"]
    ordered.extend(label for label in labels if label != "normal")
    return {label: idx for idx, label in enumerate(ordered)}


def pseudo_boxes_from_events(
    events: np.ndarray,
    height: int,
    width: int,
    quantile: float = 0.85,
    min_area: int = 8,
) -> list[tuple[int, int, int, int]]:
    arr = np.asarray(events, dtype=np.float32)
    if arr.size == 0:
        return []
    x = arr[:, 0].astype(np.int64)
    y = arr[:, 1].astype(np.int64)
    keep = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    if not np.any(keep):
        return []
    density = np.zeros((height, width), dtype=np.float32)
    np.add.at(density, (y[keep], x[keep]), 1.0)
    active = density[density > 0]
    if active.size == 0:
        return []
    threshold = float(np.quantile(active, quantile))
    mask = (density >= max(threshold, 1.0)).astype(np.uint8)
    effective_min_area = min(min_area, int(active.size))
    return mask_to_boxes(mask, min_area=max(1, effective_min_area))


def _boxes_tensor(boxes: list[tuple[int, int, int, int]]) -> torch.Tensor:
    if not boxes:
        return torch.empty((0, 4), dtype=torch.float32)
    return torch.tensor(boxes, dtype=torch.float32)
