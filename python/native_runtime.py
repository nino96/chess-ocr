#!/usr/bin/env python3
"""Bounded, local-only checkpoint admission and ONNX reference export.

Artifacts belong below ignored cache/, work/, and artifacts/.  This module never
loads an arbitrary pickle: YOLOX state is deserialized only by
torch.load(..., weights_only=True), while the timm state is safetensors-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
import urllib.parse
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache" / "native"
WORK = ROOT / "work" / "native"
ARTIFACTS = ROOT / "artifacts" / "native"
MAX_DOWNLOAD_BYTES = 4 * 1024**3
SYNTHETIC_VERSION = "synthetic-raster-v1"
YOLOX_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth"
YOLOX_SHA256 = "cd28f55fbbc1829f99d9ac9b38a16d259a22889739c8728ea877610201feff7b"
YOLOX_SOURCE_COMMIT = "e1052df71842031413f6030723c3607b839c80ce"
HF_REPO = "timm/mobilenetv3_small_100.lamb_in1k"
HF_REVISION = "1824797e7887cbec1990e4adbd6675960a36c589"
MOBILENET_SHA256 = "46d2c063b18125884c48937afa4c49e18128869e52e8db96df48bf0a4d7ff697"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def record_downloaded_bytes(count: int) -> int:
    """Persist all transferred bytes, including failed attempts, against one cap."""
    budget_path = WORK / "download-budget.json"
    prior = json.loads(budget_path.read_text(encoding="utf-8")) if budget_path.exists() else {"downloaded_bytes": 0}
    total = int(prior["downloaded_bytes"]) + count
    write_json(budget_path, {"downloaded_bytes": total, "limit_bytes": MAX_DOWNLOAD_BYTES})
    return total


def checked_download(url: str, destination: Path, expected_host: str, expected_sha256: str) -> str:
    """Fetch a single HTTPS artifact within the issue's cumulative size ceiling."""
    partial = destination.with_suffix(destination.suffix + ".partial")
    if any(path.is_symlink() for path in (CACHE, destination, partial, *destination.parents)):
        raise RuntimeError("refusing symlinked native download path")
    if destination.exists():
        digest = sha256_file(destination)
        if digest != expected_sha256:
            raise RuntimeError(f"cached artifact hash mismatch for {destination.name}")
        return digest
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "chess-ocr-native-runtime/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        final = response.url
        host = urllib.parse.urlparse(final).hostname
        if not final.startswith("https://") or host not in {expected_host, "github.com", "release-assets.githubusercontent.com", "cas-bridge.xethub.hf.co", "cdn-lfs.hf.co", "us.aws.cdn.hf.co"}:
            raise RuntimeError(f"unexpected final download host: {host}")
        length = response.headers.get("Content-Length")
        prior_budget = json.loads((WORK / "download-budget.json").read_text(encoding="utf-8"))["downloaded_bytes"] if (WORK / "download-budget.json").exists() else 0
        if length is not None and prior_budget + int(length) > MAX_DOWNLOAD_BYTES:
            raise RuntimeError(f"refusing {length} byte download above {MAX_DOWNLOAD_BYTES} byte ceiling")
        tmp = destination.with_suffix(destination.suffix + ".partial")
        total = 0
        h = hashlib.sha256()
        with tmp.open("wb") as out:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                total += len(block)
                if record_downloaded_bytes(len(block)) > MAX_DOWNLOAD_BYTES:
                    out.close()
                    tmp.unlink(missing_ok=True)
                    raise RuntimeError("native download total exceeded 4 GiB ceiling")
                h.update(block)
                out.write(block)
        digest = h.hexdigest()
        if digest != expected_sha256:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"download hash mismatch for {destination.name}")
        tmp.replace(destination)
    return digest


def synthetic_rgb(width: int = 416, height: int = 416) -> bytes:
    """A documented raster: opaque RGB, no fonts or third-party image content."""
    buf = bytearray(width * height * 3)
    for y in range(height):
        for x in range(width):
            tile = ((x // 52) + (y // 52)) & 1
            r = 238 if tile else 92
            g = 210 if tile else 119
            b = 154 if tile else 78
            # Fixed coloured diagonals make BGR/RGB errors observable.
            if abs(x - y) <= 2:
                r, g, b = 255, 24, 24
            if abs(x - (width - 1 - y)) <= 2:
                r, g, b = 24, 80, 255
            i = (y * width + x) * 3
            buf[i : i + 3] = bytes((r, g, b))
    return bytes(buf)


def prepare_inputs() -> None:
    import numpy as np
    from PIL import Image

    raw = synthetic_rgb()
    raw_path = WORK / f"{SYNTHETIC_VERSION}.rgb"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(raw)
    image = Image.frombytes("RGB", (416, 416), raw)
    # YOLOX ONNX demo preproc: source RGB is converted to BGR, padded to 416
    # using value 114, and emitted NCHW float32 in 0..255 without normalization.
    yolo = np.full((416, 416, 3), 114, dtype=np.float32)
    yolo[:, :, :] = np.asarray(image, dtype=np.float32)[:, :, ::-1]
    yolo_nchw = np.ascontiguousarray(yolo.transpose(2, 0, 1))[None, ...]
    # timm pretrained config: bicubic resize shortest edge to 256 (224/.875),
    # centre crop 224, RGB then ImageNet mean/std.
    resized = image.resize((256, 256), Image.Resampling.BICUBIC)
    crop = resized.crop((16, 16, 240, 240))
    mobile_u8 = np.ascontiguousarray(np.asarray(crop, dtype=np.uint8))
    mobile = mobile_u8.astype(np.float32) / 255.0
    mobile = (mobile - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    mobile_nchw = np.ascontiguousarray(mobile.transpose(2, 0, 1))[None, ...]
    yolo_path = WORK / f"{SYNTHETIC_VERSION}.yolox-416-bgr-nchw-f32.bin"
    mobile_path = WORK / f"{SYNTHETIC_VERSION}.mobilenet-224-rgb-nchw-f32.bin"
    mobile_rgb_path = WORK / f"{SYNTHETIC_VERSION}.mobilenet-224-rgb-u8.bin"
    yolo_nchw.tofile(yolo_path)
    mobile_nchw.tofile(mobile_path)
    mobile_u8.tofile(mobile_rgb_path)
    manifest = {
        "schema_version": 1,
        "input": {"path": str(raw_path.relative_to(ROOT)), "sha256": sha256_file(raw_path), "encoding": "raw-rgb", "width": 416, "height": 416, "algorithm": SYNTHETIC_VERSION},
        "yolox": {"tensor_path": str(yolo_path.relative_to(ROOT)), "sha256": sha256_file(yolo_path), "shape": [1, 3, 416, 416], "dtype": "float32-le", "source_channels": "RGB", "model_channels": "BGR", "resize": "aspect-fit 416x416; synthetic input already 416x416", "pad_value": 114, "normalization": "none; 0..255"},
        "mobilenet": {"cropped_rgb_path": str(mobile_rgb_path.relative_to(ROOT)), "cropped_rgb_sha256": sha256_file(mobile_rgb_path), "cropped_rgb_shape": [224, 224, 3], "cropped_rgb_dtype": "uint8", "tensor_path": str(mobile_path.relative_to(ROOT)), "sha256": sha256_file(mobile_path), "shape": [1, 3, 224, 224], "dtype": "float32-le", "source_channels": "RGB", "resize": "Pillow bicubic 416x416 -> 256x256", "crop": "center (16,16)-(240,240)", "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}},
    }
    write_json(WORK / "parity-input-manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


def save_output(name: str, array: Any, model_path: Path, input_manifest: dict[str, Any]) -> None:
    import numpy as np
    output = np.asarray(array, dtype=np.float32)
    out_path = ARTIFACTS / f"{name}.output-f32.bin"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.tofile(out_path)
    write_json(ARTIFACTS / f"{name}.manifest.json", {
        "schema_version": 1, "model": str(model_path.relative_to(ROOT)), "model_sha256": sha256_file(model_path),
        "input_manifest": str((WORK / "parity-input-manifest.json").relative_to(ROOT)),
        "input_manifest_sha256": sha256_file(WORK / "parity-input-manifest.json"),
        "output": str(out_path.relative_to(ROOT)), "output_sha256": sha256_file(out_path),
        "dtype": "float32-le", "shape": list(output.shape), "preprocessing": input_manifest,
    })


def refresh_manifest(name: str, preprocessing_key: str) -> None:
    """Rebind an existing output to changed input-manifest metadata without inference."""
    manifest_path = ARTIFACTS / f"{name}.manifest.json"
    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    inputs = json.loads((WORK / "parity-input-manifest.json").read_text(encoding="utf-8"))
    existing["input_manifest_sha256"] = sha256_file(WORK / "parity-input-manifest.json")
    existing["preprocessing"] = inputs[preprocessing_key]
    write_json(manifest_path, existing)


def export_mobilenet(weights: Path) -> None:
    import numpy as np
    import safetensors.torch
    import timm
    import torch

    inputs = json.loads((WORK / "parity-input-manifest.json").read_text())
    state = safetensors.torch.load_file(str(weights), device="cpu")
    model = timm.create_model("mobilenetv3_small_100.lamb_in1k", pretrained=False, num_classes=1000)
    model.load_state_dict(state, strict=True)
    model.eval()
    tensor = np.fromfile(ROOT / inputs["mobilenet"]["tensor_path"], dtype="<f4").reshape(inputs["mobilenet"]["shape"])
    example = torch.from_numpy(tensor)
    onnx_path = ARTIFACTS / "mobilenetv3_small_100.lamb_in1k-224.onnx"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, example, onnx_path, input_names=["input"], output_names=["logits"], opset_version=17, dynamo=False)
    with torch.inference_mode():
        native = model(example).detach().cpu().numpy()
    save_output("mobilenetv3_small_100.lamb_in1k-224", native, onnx_path, inputs["mobilenet"])


def export_yolox(weights: Path, source: Path) -> None:
    import numpy as np
    import torch

    if not source.is_dir() or not (source / "yolox").is_dir():
        raise RuntimeError("YOLOX source checkout is required; set --yolox-source to an official pinned checkout")
    commit = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    if commit != YOLOX_SOURCE_COMMIT:
        raise RuntimeError(f"YOLOX source must be {YOLOX_SOURCE_COMMIT}, got {commit}")
    sys.path.insert(0, str(source))
    from yolox.exp import get_exp  # type: ignore[import-not-found]

    inputs = json.loads((WORK / "parity-input-manifest.json").read_text())
    # The checkpoint is a PyTorch zip/state-dict only; weights_only blocks code
    # execution if a release asset is replaced with a malicious pickle.
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    state = checkpoint.get("model", checkpoint)
    exp = get_exp(None, "yolox-nano")
    model = exp.get_model()
    model.load_state_dict(state, strict=True)
    model.eval()
    model.head.decode_in_inference = False  # Browser owns documented decode/NMS.
    tensor = np.fromfile(ROOT / inputs["yolox"]["tensor_path"], dtype="<f4").reshape(inputs["yolox"]["shape"])
    example = torch.from_numpy(tensor)
    onnx_path = ARTIFACTS / "yolox_nano-coco-416-raw.onnx"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, example, onnx_path, input_names=["images"], output_names=["predictions"], opset_version=13, do_constant_folding=True, dynamo=False)
    with torch.inference_mode():
        native = model(example).detach().cpu().numpy()
    save_output("yolox_nano-coco-416-raw", native, onnx_path, inputs["yolox"])


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare-inputs")
    dl = sub.add_parser("download")
    dl.add_argument("--yolox", action="store_true")
    dl.add_argument("--mobilenet", action="store_true")
    ex = sub.add_parser("export")
    ex.add_argument("--model", required=True, choices=("yolox", "mobilenet"))
    ex.add_argument("--weights", type=Path, required=True)
    ex.add_argument("--yolox-source", type=Path)
    refresh = sub.add_parser("refresh-manifest")
    refresh.add_argument("--model", required=True, choices=("yolox", "mobilenet"))
    args = p.parse_args()
    if args.command == "prepare-inputs":
        prepare_inputs()
    elif args.command == "download":
        if not args.yolox and not args.mobilenet:
            p.error("choose --yolox and/or --mobilenet")
        manifest_path = WORK / "download-manifest.json"
        admitted: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"downloads": {}}
        admitted["reviewed_on"] = str(date.today())
        if args.yolox:
            path = CACHE / "yolox_nano.pth"
            admitted["downloads"]["yolox"] = {"url": YOLOX_URL, "source_commit": YOLOX_SOURCE_COMMIT, "path": str(path.relative_to(ROOT)), "sha256": checked_download(YOLOX_URL, path, "github.com", YOLOX_SHA256)}
        if args.mobilenet:
            path = CACHE / "mobilenetv3_small_100.lamb_in1k.safetensors"
            url = f"https://huggingface.co/{HF_REPO}/resolve/{HF_REVISION}/model.safetensors"
            admitted["downloads"]["mobilenet"] = {"url": url, "revision": HF_REVISION, "path": str(path.relative_to(ROOT)), "sha256": checked_download(url, path, "huggingface.co", MOBILENET_SHA256)}
        write_json(WORK / "download-manifest.json", admitted)
        print(json.dumps(admitted, indent=2))
    elif args.command == "export":
        if not (WORK / "parity-input-manifest.json").exists():
            raise RuntimeError("run prepare-inputs first")
        if args.model == "mobilenet":
            export_mobilenet(args.weights)
        else:
            if args.yolox_source is None:
                p.error("--yolox-source is required")
            export_yolox(args.weights, args.yolox_source)
    else:
        if args.model == "mobilenet":
            refresh_manifest("mobilenetv3_small_100.lamb_in1k-224", "mobilenet")
        else:
            refresh_manifest("yolox_nano-coco-416-raw", "yolox")


if __name__ == "__main__":
    main()
