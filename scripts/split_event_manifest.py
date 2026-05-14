from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from event_defect.manifest import read_manifest, stratified_split, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Create stratified train/val/test manifests.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    train, val, test = stratified_split(rows, args.val_ratio, args.test_ratio, args.seed)
    out_dir = Path(args.out_dir)
    write_manifest(out_dir / "train.csv", train)
    write_manifest(out_dir / "val.csv", val)
    write_manifest(out_dir / "test.csv", test)
    print(f"train={len(train)} val={len(val)} test={len(test)} -> {out_dir}")


if __name__ == "__main__":
    main()
