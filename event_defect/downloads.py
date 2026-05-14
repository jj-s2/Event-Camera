from __future__ import annotations

from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def validate_download_file(path: str | Path, min_size_bytes: int = 1_000_000) -> tuple[bool, str]:
    path = Path(path)
    if not path.exists():
        return False, f"{path} does not exist"
    size = path.stat().st_size
    prefix = path.read_bytes()[:256]
    lowered = prefix.lower()
    if b"<!doctype html" in lowered or b"<html" in lowered:
        return False, f"{path} is HTML, not a dataset archive"
    if size < min_size_bytes:
        return False, f"{path} is too small ({size} bytes)"
    return True, f"{path} looks valid ({size} bytes)"


def count_image_files(root: str | Path) -> int:
    root = Path(root)
    if not root.exists():
        return 0
    return sum(1 for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
