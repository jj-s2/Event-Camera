import csv
from pathlib import Path


def _write_manifest(path: Path, labels: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["sample_id", "event_path", "mask_path", "label"]
        )
        writer.writeheader()
        for idx, label in enumerate(labels):
            writer.writerow(
                {
                    "sample_id": f"{label}_{idx}",
                    "event_path": f"events/{label}_{idx}.h5",
                    "mask_path": "",
                    "label": label,
                }
            )


def test_stratified_split_keeps_every_label_in_each_split_when_possible(tmp_path: Path):
    from event_defect.manifest import read_manifest, stratified_split

    manifest = tmp_path / "manifest.csv"
    labels = ["normal"] * 10 + ["scratch"] * 10 + ["bent"] * 10
    _write_manifest(manifest, labels)

    train, val, test = stratified_split(read_manifest(manifest), val_ratio=0.2, test_ratio=0.2, seed=7)

    assert len(train) == 18
    assert len(val) == 6
    assert len(test) == 6
    for split in (train, val, test):
        assert {row["label"] for row in split} == {"normal", "scratch", "bent"}


def test_write_manifest_round_trips_rows(tmp_path: Path):
    from event_defect.manifest import read_manifest, write_manifest

    rows = [
        {"sample_id": "a", "event_path": "a.h5", "mask_path": "", "label": "normal"},
        {"sample_id": "b", "event_path": "b.h5", "mask_path": "b.png", "label": "scratch"},
    ]
    out = tmp_path / "out.csv"

    write_manifest(out, rows)

    assert read_manifest(out) == rows


def test_write_manifest_preserves_extra_columns(tmp_path: Path):
    from event_defect.manifest import read_manifest, write_manifest

    rows = [
        {
            "sample_id": "a",
            "event_path": "a.h5",
            "mask_path": "",
            "label": "normal",
            "image_path": "a.png",
        }
    ]
    out = tmp_path / "out.csv"

    write_manifest(out, rows)

    assert read_manifest(out) == rows
