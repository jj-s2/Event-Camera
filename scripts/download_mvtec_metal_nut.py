from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from event_defect.downloads import count_image_files


def run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def clone_repo(target: Path, repo_url: str, include: str) -> None:
    if not target.exists():
        env = os.environ.copy()
        env["GIT_LFS_SKIP_SMUDGE"] = "1"
        run(["git", "clone", "--depth", "1", repo_url, str(target)], env=env)
    run(["git", "lfs", "pull", "-I", include], cwd=target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MVTec AD metal_nut from a Hugging Face mirror.")
    parser.add_argument("--target", default="external/mvtec-ad-metal-nut-real")
    parser.add_argument("--repo-url", default="https://hf-mirror.com/datasets/MSherbinii/mvtec-ad-metal-nut")
    parser.add_argument(
        "--include",
        default="metal_nut/train/good/**,metal_nut/test/**,metal_nut/ground_truth/**",
        help="Git LFS include glob.",
    )
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    target = Path(args.target)
    if args.reset and target.exists():
        shutil.rmtree(target)
    clone_repo(target, args.repo_url, args.include)
    count = count_image_files(target / "metal_nut")
    if count == 0:
        raise RuntimeError(
            "No image files were downloaded. Check network access to Hugging Face/hf-mirror and Git LFS."
        )
    print(f"Downloaded image files: {count}")


if __name__ == "__main__":
    main()
