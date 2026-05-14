from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys

import cv2
import joblib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_patchcore_event_defect import (
    ResNetPatchExtractor,
    event_rgb,
    patch_score,
    row_rgb_image,
)


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def select_row(rows: list[dict[str, str]], sample_id: str | None) -> dict[str, str]:
    if sample_id is None:
        return rows[0]
    for row in rows:
        if row.get("sample_id") == sample_id:
            return row
    raise ValueError(f"Sample {sample_id!r} not found")


def read_mask(path: str) -> np.ndarray | None:
    if not path:
        return None
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def make_panel(
    row: dict[str, str],
    model_blob: dict,
    out_path: Path,
) -> dict[str, float | str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = ResNetPatchExtractor().eval().to(device)
    height = int(model_blob["height"])
    width = int(model_blob["width"])
    bins = int(model_blob["bins"])
    input_mode = str(model_blob["input_mode"])
    threshold = float(model_blob["threshold"])
    score_name = str(model_blob["score_name"])

    aps_image = row_rgb_image(row, input_mode="aps", height=height, width=width, bins=bins)
    detector_image = row_rgb_image(row, input_mode=input_mode, height=height, width=width, bins=bins)
    patches = extractor(detector_image, device=device)
    neighbor_index = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(model_blob["memory_bank"])
    distances = neighbor_index.kneighbors(patches, return_distance=True)[0].ravel()
    score = patch_score(distances, score_name)
    prediction = "defect" if score >= threshold else "normal"

    side = int(round(np.sqrt(len(distances))))
    anomaly = distances.reshape(side, side)
    anomaly = cv2.resize(anomaly, (224, 224), interpolation=cv2.INTER_CUBIC)
    heat_norm = np.clip(anomaly / max(threshold, 1e-6), 0.0, 1.0)
    heat = cv2.applyColorMap(np.uint8(heat_norm * 255.0), cv2.COLORMAP_JET)[:, :, ::-1]
    original = cv2.resize(aps_image, (224, 224), interpolation=cv2.INTER_AREA)
    event_image = cv2.resize(event_rgb(row, height=height, width=width, bins=bins), (224, 224), interpolation=cv2.INTER_NEAREST)
    overlay = cv2.addWeighted(original, 0.62, heat, 0.38, 0.0)

    mask = read_mask(row.get("mask_path", ""))
    if mask is not None:
        mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)

    canvas = Image.new("RGB", (224 * 3, 296), (245, 247, 250))
    for index, panel in enumerate([original, event_image, overlay]):
        canvas.paste(Image.fromarray(panel), (224 * index, 48))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
        font_bold = ImageFont.truetype("arialbd.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
        font_bold = font

    for index, title in enumerate(["Original APS", "Event RGB", "Anomaly Overlay"]):
        draw.text((224 * index + 12, 14), title, fill=(25, 35, 45), font=font_bold)
    status_color = (0, 130, 55) if prediction == row.get("label") else (190, 30, 30)
    draw.rectangle((0, 272, 224 * 3, 296), fill=(235, 239, 244))
    draw.text(
        (12, 274),
        f"sample={row.get('sample_id')}  GT={row.get('label')}  pred={prediction}  score={score:.4f}  threshold={threshold:.4f}",
        fill=status_color,
        font=font,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return {
        "sample_id": row.get("sample_id", ""),
        "label": row.get("label", ""),
        "prediction": prediction,
        "score": float(score),
        "threshold": threshold,
        "output": str(out_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Render one PatchCore detection demo panel.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if "TORCH_HOME" not in os.environ:
        local_torch_home = Path.cwd() / ".torch"
        if local_torch_home.exists():
            os.environ["TORCH_HOME"] = str(local_torch_home)

    rows = read_manifest(args.manifest)
    row = select_row(rows, args.sample_id)
    model_blob = joblib.load(args.model)
    result = make_panel(row, model_blob, Path(args.out))
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
