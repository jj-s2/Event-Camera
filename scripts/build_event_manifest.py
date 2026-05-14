from __future__ import annotations

import argparse
import csv
from pathlib import Path


EVENT_EXTENSIONS = {".h5", ".hdf5"}
MASK_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def infer_label(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    if len(rel.parts) >= 2:
        parent = rel.parts[-2].lower()
        if parent not in {"events", "event", "raw", "h5"}:
            return parent
    name = path.stem.lower()
    for token in ("spot", "scratch", "stain", "crack", "pit", "normal", "good"):
        if token in name:
            return "normal" if token == "good" else token
    return "normal"


def find_mask(event_path: Path, mask_root: Path | None, mask_suffix: str) -> Path | None:
    search_roots = [event_path.parent]
    if mask_root is not None:
        search_roots.insert(0, mask_root)
    for root in search_roots:
        for ext in MASK_EXTENSIONS:
            candidate = root / f"{event_path.stem}{mask_suffix}{ext}"
            if candidate.exists():
                return candidate
            candidate = root / f"{event_path.stem}{ext}"
            if candidate.exists() and candidate != event_path:
                return candidate
    return None


def build_manifest(root: Path, out: Path, mask_root: Path | None, mask_suffix: str) -> int:
    event_files = sorted(p for p in root.rglob("*") if p.suffix.lower() in EVENT_EXTENSIONS)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["sample_id", "event_path", "mask_path", "label"]
        )
        writer.writeheader()
        for path in event_files:
            mask = find_mask(path, mask_root, mask_suffix)
            writer.writerow(
                {
                    "sample_id": path.stem,
                    "event_path": str(path.resolve()),
                    "mask_path": str(mask.resolve()) if mask else "",
                    "label": infer_label(path, root),
                }
            )
    return len(event_files)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a CSV manifest for event defect h5 files.")
    parser.add_argument("--root", required=True, help="Root directory containing .h5/.hdf5 files.")
    parser.add_argument("--out", default="data/event_defect_manifest.csv")
    parser.add_argument("--mask-root", default=None)
    parser.add_argument("--mask-suffix", default="_mask")
    args = parser.parse_args()

    count = build_manifest(
        root=Path(args.root),
        out=Path(args.out),
        mask_root=Path(args.mask_root) if args.mask_root else None,
        mask_suffix=args.mask_suffix,
    )
    print(f"Wrote {count} samples to {args.out}")


if __name__ == "__main__":
    main()
