# -*- coding: utf-8 -*-
"""
Step 2: 修改 Checkpoint
用途：合并多个 checkpoint、修改配置、裁剪权重、重命名 key
"""

import torch
import argparse
from pathlib import Path
from model import build_model


def inspect(ckpt_path: str):
    """查看 checkpoint 内容"""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    print(f"\n=== {ckpt_path} ===")
    for k, v in ckpt.items():
        if k == "model_state":
            print(f"  model_state: {len(v)} keys")
            for name, param in list(v.items())[:5]:
                print(f"    {name}: {param.shape}")
            print("    ...")
        else:
            print(f"  {k}: {v}")


def merge_checkpoints(ckpt_a: str, ckpt_b: str, alpha: float, out: str):
    """
    线性插值合并两个 checkpoint 的权重
    merged = alpha * A + (1-alpha) * B
    """
    a = torch.load(ckpt_a, map_location="cpu")["model_state"]
    b = torch.load(ckpt_b, map_location="cpu")["model_state"]
    merged = {k: alpha * a[k] + (1 - alpha) * b[k] for k in a}
    torch.save({"model_state": merged, "merged_from": [ckpt_a, ckpt_b],
                "alpha": alpha}, out)
    print(f"Merged -> {out}  (alpha={alpha})")


def strip_optimizer(ckpt_path: str, out: str):
    """去掉 optimizer/scheduler 状态，减小文件体积"""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    slim = {"model_state": ckpt["model_state"],
            "epoch":       ckpt.get("epoch"),
            "best_f1":     ckpt.get("best_f1")}
    torch.save(slim, out)
    orig_mb = Path(ckpt_path).stat().st_size / 1024 / 1024
    new_mb  = Path(out).stat().st_size / 1024 / 1024
    print(f"Stripped: {orig_mb:.1f}MB -> {new_mb:.1f}MB  -> {out}")


def verify(ckpt_path: str):
    """验证 checkpoint 能正常加载到模型"""
    model = build_model(pretrained=False)
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(
        ckpt["model_state"], strict=False)
    print(f"Verify {ckpt_path}:")
    print(f"  Missing keys:    {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")
    if not missing and not unexpected:
        print("  ✓ Perfect match")


def main():
    parser = argparse.ArgumentParser(description="Checkpoint 工具")
    sub = parser.add_subparsers(dest="cmd")

    p_inspect = sub.add_parser("inspect", help="查看 checkpoint")
    p_inspect.add_argument("path")

    p_merge = sub.add_parser("merge", help="合并两个 checkpoint")
    p_merge.add_argument("a")
    p_merge.add_argument("b")
    p_merge.add_argument("--alpha", type=float, default=0.5)
    p_merge.add_argument("--out",   default="checkpoints/merged.pth")

    p_strip = sub.add_parser("strip", help="去掉 optimizer 状态")
    p_strip.add_argument("path")
    p_strip.add_argument("--out", default=None)

    p_verify = sub.add_parser("verify", help="验证 checkpoint")
    p_verify.add_argument("path")

    args = parser.parse_args()

    if args.cmd == "inspect":
        inspect(args.path)
    elif args.cmd == "merge":
        merge_checkpoints(args.a, args.b, args.alpha, args.out)
    elif args.cmd == "strip":
        out = args.out or args.path.replace(".pth", "_slim.pth")
        strip_optimizer(args.path, out)
    elif args.cmd == "verify":
        verify(args.path)
    else:
        # 默认：检查所有 checkpoint
        for p in Path("checkpoints").glob("*.pth"):
            inspect(str(p))
            verify(str(p))


if __name__ == "__main__":
    main()
