# -*- coding: utf-8 -*-
"""
Step 6: 导出模型
支持导出为 ONNX / TorchScript / TorchScript Mobile
"""

import torch
import argparse
from pathlib import Path

from model import build_model

FRAME_H, FRAME_W = 260, 346


def export_onnx(ckpt_path: str, out_path: str, opset: int = 12):
    device = torch.device("cpu")
    model  = build_model(pretrained=False).to(device)
    ckpt   = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dummy = torch.randn(1, 6, FRAME_H, FRAME_W)

    torch.onnx.export(
        model, dummy, out_path,
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        do_constant_folding=True,
    )
    size_mb = Path(out_path).stat().st_size / 1024 / 1024
    print(f"ONNX 导出完成: {out_path}  ({size_mb:.2f} MB)")

    # 验证
    try:
        import onnx
        model_onnx = onnx.load(out_path)
        onnx.checker.check_model(model_onnx)
        print("ONNX 验证通过")
    except ImportError:
        print("提示: pip install onnx 可验证模型")


def export_torchscript(ckpt_path: str, out_path: str):
    device = torch.device("cpu")
    model  = build_model(pretrained=False).to(device)
    ckpt   = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dummy    = torch.randn(1, 6, FRAME_H, FRAME_W)
    scripted = torch.jit.trace(model, dummy)
    scripted.save(out_path)
    size_mb  = Path(out_path).stat().st_size / 1024 / 1024
    print(f"TorchScript 导出完成: {out_path}  ({size_mb:.2f} MB)")


def export_torchscript_mobile(ckpt_path: str, out_path: str):
    try:
        from torch.utils.mobile_optimizer import optimize_for_mobile
    except ImportError:
        print("mobile_optimizer 不可用，跳过移动端优化")
        export_torchscript(ckpt_path, out_path)
        return

    device = torch.device("cpu")
    model  = build_model(pretrained=False).to(device)
    ckpt   = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dummy    = torch.randn(1, 6, FRAME_H, FRAME_W)
    scripted = torch.jit.trace(model, dummy)
    optimized = optimize_for_mobile(scripted)
    optimized._save_for_lite_interpreter(out_path)
    size_mb = Path(out_path).stat().st_size / 1024 / 1024
    print(f"Mobile TorchScript 导出完成: {out_path}  ({size_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description="模型导出工具")
    parser.add_argument("--ckpt",   default="checkpoints/best.pth")
    parser.add_argument("--format", choices=["onnx", "ts", "mobile", "all"],
                        default="all")
    parser.add_argument("--out_dir", default="exported")
    parser.add_argument("--opset",  type=int, default=12)
    args = parser.parse_args()

    Path(args.out_dir).mkdir(exist_ok=True)

    if args.format in ("onnx", "all"):
        export_onnx(args.ckpt,
                    str(Path(args.out_dir) / "fall_detector.onnx"),
                    args.opset)

    if args.format in ("ts", "all"):
        export_torchscript(args.ckpt,
                           str(Path(args.out_dir) / "fall_detector.pt"))

    if args.format in ("mobile", "all"):
        export_torchscript_mobile(
            args.ckpt,
            str(Path(args.out_dir) / "fall_detector_mobile.ptl"))


if __name__ == "__main__":
    main()
