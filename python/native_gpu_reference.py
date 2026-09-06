#!/usr/bin/env python3
"""Run existing native reference inputs on a deliberate CUDA environment.

This imports neither checkpoints nor generated outputs from the network and never
exports or writes a model. All mounts supplied to the container can be read-only.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


YOLOX_SOURCE_COMMIT = "e1052df71842031413f6030723c3607b839c80ce"


def binary(path: Path, shape: list[int]):
    import numpy as np
    return np.fromfile(path, dtype="<f4").reshape(shape)


def run_mobilenet(args: argparse.Namespace) -> dict[str, object]:
    import numpy as np
    import safetensors.torch
    import timm
    import torch

    from native_runtime import sha256_file, MOBILENET_SHA256
    if sha256_file(args.weights / "mobilenetv3_small_100.lamb_in1k.safetensors") != MOBILENET_SHA256:
        raise RuntimeError("MobileNet checkpoint hash mismatch")
    state = safetensors.torch.load_file(str(args.weights / "mobilenetv3_small_100.lamb_in1k.safetensors"), device="cpu")
    model = timm.create_model("mobilenetv3_small_100.lamb_in1k", pretrained=False, num_classes=1000)
    model.load_state_dict(state, strict=True)
    model.eval().to(args.device)
    x = torch.from_numpy(binary(args.inputs / "synthetic-raster-v1.mobilenet-224-rgb-nchw-f32.bin", [1, 3, 224, 224])).to(args.device)
    with torch.inference_mode():
        y = model(x).float().cpu().numpy()
    reference = binary(args.references / "mobilenetv3_small_100.lamb_in1k-224.output-f32.bin", [1, 1000])
    return {"model": "mobilenet", "shape": list(y.shape), "finite": bool(np.isfinite(y).all()), "max_abs_from_cpu_reference": float(np.max(np.abs(y - reference)))}


def run_yolox(args: argparse.Namespace) -> dict[str, object]:
    import numpy as np
    import torch

    commit = subprocess.run(["git", "-c", f"safe.directory={args.yolox_source}", "-C", str(args.yolox_source), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    if commit != YOLOX_SOURCE_COMMIT:
        raise RuntimeError(f"YOLOX source must be {YOLOX_SOURCE_COMMIT}, got {commit}")
    sys.path.insert(0, str(args.yolox_source))
    from yolox.exp import get_exp  # type: ignore[import-not-found]

    from native_runtime import sha256_file, YOLOX_SHA256
    if sha256_file(args.weights / "yolox_nano.pth") != YOLOX_SHA256:
        raise RuntimeError("YOLOX checkpoint hash mismatch")
    checkpoint = torch.load(args.weights / "yolox_nano.pth", map_location="cpu", weights_only=True)
    model = get_exp(None, "yolox-nano").get_model()
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    model.eval().to(args.device)
    model.head.decode_in_inference = False
    x = torch.from_numpy(binary(args.inputs / "synthetic-raster-v1.yolox-416-bgr-nchw-f32.bin", [1, 3, 416, 416])).to(args.device)
    with torch.inference_mode():
        y = model(x).float().cpu().numpy()
    reference = binary(args.references / "yolox_nano-coco-416-raw.output-f32.bin", [1, 3549, 85])
    return {"model": "yolox", "shape": list(y.shape), "finite": bool(np.isfinite(y).all()), "max_abs_from_cpu_reference": float(np.max(np.abs(y - reference)))}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--yolox-source", type=Path, required=True)
    p.add_argument("--model", choices=("mobilenet", "yolox", "both"), default="both")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--enable-cudnn", action="store_true", help="Reproduce the known failing default cuDNN probe; not the supported configuration")
    args = p.parse_args()
    import torch
    # The pinned container's default cuDNN path fails this model parity gate on
    # GB10. The same tensors agree when convolution uses the non-cuDNN path.
    torch.backends.cudnn.enabled = args.enable_cudnn
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this selected environment")
    runners = {"mobilenet": run_mobilenet, "yolox": run_yolox}
    selected = runners if args.model == "both" else {args.model: runners[args.model]}
    device = torch.cuda.get_device_name(0) if args.device == "cuda" else "CPU"
    capability = list(torch.cuda.get_device_capability(0)) if args.device == "cuda" else None
    report = {"torch": torch.__version__, "cuda": torch.version.cuda, "device": device, "capability": capability, "cudnn_enabled": torch.backends.cudnn.enabled, "results": [runner(args) for runner in selected.values()]}
    print(json.dumps(report, sort_keys=True))
    if any(not result["finite"] or result["max_abs_from_cpu_reference"] > 1e-3 for result in report["results"]):
        raise RuntimeError("Native device parity failed (maximum absolute tolerance 1e-3)")


if __name__ == "__main__":
    main()
