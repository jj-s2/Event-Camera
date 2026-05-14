import csv
from pathlib import Path

import h5py
import numpy as np
import torch


def _write_event_file(path: Path, events: np.ndarray) -> None:
    with h5py.File(path, "w") as h5:
        h5.create_dataset("events", data=events.astype(np.float32))


def test_dataset_builds_stable_multiclass_label_map(tmp_path: Path):
    from event_defect.dataset import EventDefectDataset

    event_dir = tmp_path / "events"
    event_dir.mkdir()
    for name in ("a", "b", "c"):
        _write_event_file(event_dir / f"{name}.h5", np.zeros((0, 4), dtype=np.float32))

    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["sample_id", "event_path", "mask_path", "label"]
        )
        writer.writeheader()
        writer.writerow({"sample_id": "a", "event_path": "events/a.h5", "mask_path": "", "label": "scratch"})
        writer.writerow({"sample_id": "b", "event_path": "events/b.h5", "mask_path": "", "label": "normal"})
        writer.writerow({"sample_id": "c", "event_path": "events/c.h5", "mask_path": "", "label": "spot"})

    ds = EventDefectDataset(manifest, height=16, width=16, bins=2)

    assert ds.class_to_idx == {"normal": 0, "scratch": 1, "spot": 2}
    assert ds.num_classes == 3
    assert ds[0][1]["class_id"].item() == 1
    assert ds[2][1]["is_defect"].item() == 1


def test_dataset_creates_pseudo_box_from_event_density_when_mask_missing(tmp_path: Path):
    from event_defect.dataset import EventDefectDataset

    event_path = tmp_path / "scratch.h5"
    events = np.array(
        [[x, y, float(i), 1.0] for i, (x, y) in enumerate([(4, 4), (5, 4), (4, 5), (5, 5), (6, 5)])],
        dtype=np.float32,
    )
    _write_event_file(event_path, events)

    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["sample_id", "event_path", "mask_path", "label"]
        )
        writer.writeheader()
        writer.writerow({"sample_id": "scratch", "event_path": str(event_path), "mask_path": "", "label": "scratch"})

    ds = EventDefectDataset(manifest, height=12, width=12, bins=2, pseudo_box_quantile=0.4)
    _, target = ds[0]

    assert target["boxes"].shape == (1, 4)
    assert target["boxes"][0].tolist() == [4.0, 4.0, 6.0, 5.0]


def test_binary_metrics_report_precision_recall_f1():
    from event_defect.metrics import binary_metrics

    logits = torch.tensor([[0.0, 2.0], [2.0, 0.0], [0.1, 0.9], [0.9, 0.1]])
    labels = torch.tensor([1, 1, 0, 0])

    metrics = binary_metrics(logits, labels)

    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fn"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
