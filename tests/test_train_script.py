import csv
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np


def test_make_loaders_preserves_class_mapping_after_random_split(tmp_path: Path):
    from scripts.train_event_defect import make_loaders

    event_dir = tmp_path / "events"
    event_dir.mkdir()
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["sample_id", "event_path", "mask_path", "label"]
        )
        writer.writeheader()
        for idx, label in enumerate(["normal", "scratch", "spot", "stain"]):
            path = event_dir / f"{idx}.h5"
            with h5py.File(path, "w") as h5:
                h5.create_dataset("events", data=np.zeros((0, 4), dtype=np.float32))
            writer.writerow(
                {
                    "sample_id": str(idx),
                    "event_path": str(path),
                    "mask_path": "",
                    "label": label,
                }
            )

    args = SimpleNamespace(
        train_manifest=str(manifest),
        val_manifest=None,
        height=8,
        width=8,
        bins=2,
        pseudo_box_quantile=0.85,
        val_ratio=0.25,
        seed=1,
        batch_size=2,
        num_workers=0,
    )

    _, _, class_to_idx = make_loaders(args)

    assert class_to_idx == {"normal": 0, "scratch": 1, "spot": 2, "stain": 3}
