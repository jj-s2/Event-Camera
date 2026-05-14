from __future__ import annotations

import cv2
import numpy as np


def mask_to_boxes(mask: np.ndarray, min_area: int = 8) -> list[tuple[int, int, int, int]]:
    """Convert a binary defect mask to inclusive xyxy boxes."""
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr.max(axis=2)
    binary = (arr > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    boxes: list[tuple[int, int, int, int]] = []
    for component in range(1, count):
        x, y, w, h, area = stats[component]
        if int(area) < min_area:
            continue
        boxes.append((int(x), int(y), int(x + w - 1), int(y + h - 1)))
    boxes.sort(key=lambda box: (box[1], box[0], box[3], box[2]))
    return boxes


def label_to_binary(label: str | int) -> int:
    """Map labels to binary normal/defect IDs."""
    if isinstance(label, int):
        return 0 if label == 0 else 1
    normalized = str(label).strip().lower()
    if normalized in {"0", "normal", "ok", "good", "no_defect", "none", "background"}:
        return 0
    return 1
