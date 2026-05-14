import csv
from pathlib import Path

import h5py
import numpy as np
import pytest


def test_load_event_h5_accepts_column_datasets(tmp_path: Path):
    from event_defect.data import load_event_h5

    path = tmp_path / "sample.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("x", data=np.array([1, 2, 3], dtype=np.int16))
        h5.create_dataset("y", data=np.array([4, 5, 6], dtype=np.int16))
        h5.create_dataset("t", data=np.array([0.1, 0.2, 0.3], dtype=np.float32))
        h5.create_dataset("p", data=np.array([1, -1, 1], dtype=np.int8))

    events = load_event_h5(path)

    assert events.shape == (3, 4)
    assert events.dtype == np.float32
    assert events[:, 0].tolist() == [1, 2, 3]
    assert events[:, 1].tolist() == [4, 5, 6]
    assert events[:, 3].tolist() == [1, -1, 1]


def test_load_event_h5_accepts_event_g_dataset_name(tmp_path: Path):
    from event_defect.data import load_event_h5

    path = tmp_path / "jcde_style.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("event_g", data=np.array([11, 12], dtype=np.uint16))
        h5.create_dataset("t", data=np.array([0.0, 1.0], dtype=np.float32))
        h5.create_dataset("x", data=np.array([7, 8], dtype=np.int16))
        h5.create_dataset("y", data=np.array([3, 4], dtype=np.int16))

    events = load_event_h5(path)

    assert events.tolist() == [[7.0, 3.0, 0.0, 11.0], [8.0, 4.0, 1.0, 12.0]]


def test_events_to_voxel_grid_keeps_polarity_and_time_surface():
    from event_defect.representation import events_to_voxel_grid, events_to_time_surface

    events = np.array(
        [
            [1, 1, 0.0, 1],
            [1, 1, 0.2, 1],
            [2, 1, 0.4, -1],
            [3, 2, 0.9, -1],
        ],
        dtype=np.float32,
    )

    voxel = events_to_voxel_grid(events, height=4, width=5, bins=3)
    time_surface = events_to_time_surface(events, height=4, width=5)

    assert voxel.shape == (6, 4, 5)
    assert voxel[0, 1, 1] > 0
    assert voxel[3 + 1, 1, 2] > 0
    assert voxel[3 + 2, 2, 3] > 0
    assert time_surface.shape == (1, 4, 5)
    assert time_surface[0, 2, 3] == pytest.approx(1.0)
    assert 0.0 <= float(voxel.min()) <= float(voxel.max()) <= 1.0


def test_mask_to_boxes_extracts_connected_components():
    from event_defect.labels import mask_to_boxes

    mask = np.zeros((8, 10), dtype=np.uint8)
    mask[1:4, 2:5] = 255
    mask[5:7, 7:9] = 255

    boxes = mask_to_boxes(mask, min_area=2)

    assert boxes == [(2, 1, 4, 3), (7, 5, 8, 6)]


def test_event_defect_dataset_returns_tensor_and_target(tmp_path: Path):
    from event_defect.dataset import EventDefectDataset

    sample_dir = tmp_path / "raw"
    sample_dir.mkdir()
    h5_path = sample_dir / "part_001.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset(
            "events",
            data=np.array(
                [
                    [2, 3, 0.0, 1],
                    [2, 3, 0.5, -1],
                    [5, 6, 1.0, 1],
                ],
                dtype=np.float32,
            ),
        )

    mask_path = sample_dir / "part_001_mask.png"
    from PIL import Image

    mask = np.zeros((12, 16), dtype=np.uint8)
    mask[2:5, 3:7] = 255
    Image.fromarray(mask).save(mask_path)

    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["sample_id", "event_path", "mask_path", "label"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "part_001",
                "event_path": str(h5_path),
                "mask_path": str(mask_path),
                "label": "scratch",
            }
        )

    ds = EventDefectDataset(manifest, height=12, width=16, bins=4)
    x, target = ds[0]

    assert tuple(x.shape) == (9, 12, 16)
    assert target["labels"].tolist() == [1]
    assert target["boxes"].tolist() == [[3.0, 2.0, 6.0, 4.0]]
    assert target["sample_id"] == "part_001"
