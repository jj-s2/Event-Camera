from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path


FIELDS = ["sample_id", "event_path", "mask_path", "label"]


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_manifest(path: str | Path, rows: list[dict[str, str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extra_fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in FIELDS and field not in extra_fields:
                extra_fields.append(field)
    fieldnames = FIELDS + extra_fields
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def stratified_split(
    rows: list[dict[str, str]],
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be non-negative and sum to less than 1")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("label", "normal")].append(row)

    rng = random.Random(seed)
    train: list[dict[str, str]] = []
    val: list[dict[str, str]] = []
    test: list[dict[str, str]] = []

    for label_rows in grouped.values():
        label_rows = list(label_rows)
        rng.shuffle(label_rows)
        n = len(label_rows)
        n_test = _split_count(n, test_ratio)
        n_val = _split_count(n - n_test, val_ratio / max(1.0 - test_ratio, 1e-12))
        test.extend(label_rows[:n_test])
        val.extend(label_rows[n_test : n_test + n_val])
        train.extend(label_rows[n_test + n_val :])

    for split in (train, val, test):
        rng.shuffle(split)
    return train, val, test


def _split_count(total: int, ratio: float) -> int:
    if total <= 0 or ratio <= 0:
        return 0
    count = int(round(total * ratio))
    if total >= 3:
        count = max(1, count)
    return min(count, max(total - 1, 0))
