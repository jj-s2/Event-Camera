from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from event_defect.dataset import normalize_label
from event_defect.manifest import read_manifest, write_manifest


def to_binary_label(label: str) -> str:
    return "normal" if normalize_label(label) == "normal" else "defect"


def binarize_manifest(in_path: str | Path, out_path: str | Path) -> int:
    rows = read_manifest(in_path)
    converted = []
    for row in rows:
        item = dict(row)
        item["label"] = to_binary_label(row.get("label", "normal"))
        converted.append(item)
    write_manifest(out_path, converted)
    return len(converted)


def main() -> None:
    parser = argparse.ArgumentParser(description="Map multiclass event defect manifests to normal/defect labels.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    count = binarize_manifest(args.manifest, args.out)
    print(f"Wrote {count} rows -> {args.out}")


if __name__ == "__main__":
    main()
