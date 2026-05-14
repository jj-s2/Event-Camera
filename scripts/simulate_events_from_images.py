from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def simulate_events_from_image(
    image: np.ndarray,
    steps: int,
    shift_pixels: int,
    threshold: float,
) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    frames = []
    for step in range(steps):
        dx = int(round(step * shift_pixels / max(steps - 1, 1)))
        matrix = np.float32([[1, 0, dx], [0, 1, 0]])
        shifted = cv2.warpAffine(gray, matrix, (gray.shape[1], gray.shape[0]), borderMode=cv2.BORDER_REFLECT)
        frames.append(shifted)

    all_events = []
    for step in range(1, len(frames)):
        prev = np.log1p(frames[step - 1])
        curr = np.log1p(frames[step])
        diff = curr - prev
        t = np.full(diff.shape, step / max(len(frames) - 1, 1), dtype=np.float32)
        for polarity, mask in ((1, diff > threshold), (-1, diff < -threshold)):
            ys, xs = np.where(mask)
            if len(xs) == 0:
                continue
            ps = np.full(len(xs), polarity, dtype=np.float32)
            ts = t[ys, xs].astype(np.float32)
            all_events.append(np.stack([xs.astype(np.float32), ys.astype(np.float32), ts, ps], axis=1))
    if not all_events:
        return np.zeros((0, 4), dtype=np.float32)
    return np.concatenate(all_events, axis=0).astype(np.float32)


def infer_label(image_path: Path) -> str:
    parent = image_path.parent.name.lower()
    if parent in {"good", "normal", "ok", "no_defect"}:
        return "normal"
    if parent in {"images", "img", "train", "test"}:
        return "normal"
    return parent


def find_mvtec_mask(image_path: Path, dataset_root: Path) -> Path | None:
    parts = image_path.relative_to(dataset_root).parts
    if len(parts) < 4 or parts[-3] not in {"train", "test"}:
        return None
    category, split, defect_class, file_name = parts[-4], parts[-3], parts[-2], parts[-1]
    if defect_class == "good":
        return None
    mask = dataset_root / category / "ground_truth" / defect_class / f"{Path(file_name).stem}_mask.png"
    return mask if mask.exists() else None


def collect_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS and "ground_truth" not in p.parts)


def convert_dataset(
    root: Path,
    out_dir: Path,
    height: int,
    width: int,
    steps: int,
    shift_pixels: int,
    threshold: float,
    limit: int | None,
) -> int:
    event_dir = out_dir / "events"
    mask_dir = out_dir / "masks"
    event_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    images = collect_images(root)
    if limit is not None:
        images = images[:limit]

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["sample_id", "event_path", "mask_path", "label", "image_path"]
        )
        writer.writeheader()
        for idx, image_path in enumerate(images):
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            image = cv2.resize(image, (width, height))
            events = simulate_events_from_image(image, steps, shift_pixels, threshold)
            sample_id = f"{idx:06d}_{image_path.stem}"
            event_path = event_dir / f"{sample_id}.h5"
            with h5py.File(event_path, "w") as h5:
                h5.create_dataset("events", data=events, compression="gzip")

            mask_path = find_mvtec_mask(image_path, root)
            copied_mask = ""
            if mask_path is not None:
                copied = mask_dir / f"{sample_id}_mask.png"
                shutil.copyfile(mask_path, copied)
                copied_mask = str(copied.resolve())

            writer.writerow(
                {
                    "sample_id": sample_id,
                    "event_path": str(event_path.resolve()),
                    "mask_path": copied_mask,
                    "label": infer_label(image_path),
                    "image_path": str(image_path.resolve()),
                }
            )
    return len(images)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert industrial image defects into synthetic event streams.")
    parser.add_argument("--root", required=True, help="MVTec/NEU/image dataset root.")
    parser.add_argument("--out", default="data/simulated_event_defects")
    parser.add_argument("--height", type=int, default=260)
    parser.add_argument("--width", type=int, default=346)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--shift-pixels", type=int, default=12)
    parser.add_argument("--threshold", type=float, default=0.12)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    count = convert_dataset(
        root=Path(args.root),
        out_dir=Path(args.out),
        height=args.height,
        width=args.width,
        steps=args.steps,
        shift_pixels=args.shift_pixels,
        threshold=args.threshold,
        limit=args.limit,
    )
    print(f"Converted {count} images. Manifest: {Path(args.out) / 'manifest.csv'}")


if __name__ == "__main__":
    main()
