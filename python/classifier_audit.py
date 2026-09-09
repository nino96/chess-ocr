#!/usr/bin/env python3
"""Private exact-tile audit helpers; all payload input/output stays ignored."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

try:
    from python.classifier_preprocessing import classifier_tensor, label_indices, rectify_rgba
except ImportError:  # Direct script and pinned-container execution expose python/.
    from classifier_preprocessing import classifier_tensor, label_indices, rectify_rgba


MAX_CONTROL = 64 * 1024
MAX_RGBA = 64_000_000
MAX_TENSOR = 64 * 3 * 96 * 96 * 4
MAX_CHECKPOINT = 256 * 1024 * 1024


class Invalid(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Invalid(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_control(path: Path) -> tuple[Path, dict]:
    path = path.resolve()
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_CONTROL,
            "unsafe audit control")
    root = path.parent
    value = json.loads(path.read_text(encoding="utf-8"),
                       parse_constant=lambda _: (_ for _ in ()).throw(Invalid("nonfinite control")))
    require(isinstance(value, dict), "invalid audit control")
    return root, value


def local_file(root: Path, name: object, maximum: int, *, must_exist: bool = True) -> Path:
    require(type(name) is str and name and Path(name).name == name, "invalid audit filename")
    path = root / name
    require(not path.is_symlink(), "symlinked audit file")
    require(path.resolve().parent == root.resolve(), "audit file escaped output directory")
    if must_exist:
        require(path.is_file() and 0 < path.stat().st_size <= maximum, "unsafe audit file")
    return path


def private_write(path: Path, value: bytes) -> None:
    require(not path.exists() and not path.is_symlink(), "audit output already exists")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(value)


def preprocess(control_path: Path) -> None:
    root, control = read_control(control_path)
    require(control.get("schema") == "chess-ocr-exact-tile-preprocess/1", "preprocess schema")
    source = control.get("source")
    require(isinstance(source, dict), "source record")
    rgba_path = local_file(root, source.get("file"), MAX_RGBA)
    rgba = rgba_path.read_bytes()
    require(sha256_bytes(rgba) == source.get("sha256"), "decoded raster hash changed")
    width, height = source.get("width"), source.get("height")
    require(type(width) is int and type(height) is int and len(rgba) == width * height * 4,
            "decoded raster shape changed")
    labels = label_indices(control.get("labels"))
    grid = rectify_rgba(rgba, width, height, control.get("corners"))
    tensor = classifier_tensor(grid).tobytes()
    grid_path = local_file(root, "python-grid.rgb", 768 * 768 * 3, must_exist=False)
    tensor_path = local_file(root, "python-tensor.f32", MAX_TENSOR, must_exist=False)
    private_write(grid_path, grid)
    private_write(tensor_path, tensor)
    report = {
        "schema": "chess-ocr-exact-tile-python-stage/1",
        "grid_sha256": sha256_bytes(grid),
        "tensor_sha256": sha256_bytes(tensor),
        "labels_sha256": sha256_bytes(bytes(labels)),
        "class_order": ".PNBRQKpnbrqk",
        "square_order": "image-relative-row-major",
        "channel_order": "RGB-NCHW",
        "normalization": "imagenet-mean-std",
        "interpolation": "deterministic-bilinear-size-minus-one",
    }
    private_write(local_file(root, "python-stage.json", MAX_CONTROL, must_exist=False),
                  (json.dumps(report, sort_keys=True) + "\n").encode())


def native(control_path: Path) -> None:
    root, control = read_control(control_path)
    require(control.get("schema") == "chess-ocr-exact-tile-native/1", "native schema")
    tensor_path = local_file(root, control.get("tensor_file"), MAX_TENSOR)
    require(sha256(tensor_path) == control.get("tensor_sha256") and
            tensor_path.stat().st_size == MAX_TENSOR, "audit tensor identity changed")
    checkpoint = Path(control.get("checkpoint_path", ""))
    native_root = Path(control.get("native_root", ""))
    require(checkpoint.is_absolute() and checkpoint.is_file() and not checkpoint.is_symlink() and
            checkpoint.stat().st_size <= MAX_CHECKPOINT, "unsafe classifier checkpoint")
    require(sha256(checkpoint) == control.get("checkpoint_sha256"), "classifier checkpoint changed")
    require(native_root.is_absolute() and native_root.is_dir() and not native_root.is_symlink(),
            "unsafe native root")

    import numpy as np
    import torch
    import training

    model = training.classifier_model(native_root, torch.device("cpu"))
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    require(saved.get("schema") == "chess-ocr-training-checkpoint/1" and
            isinstance(saved.get("model"), dict), "invalid trainable checkpoint")
    model.load_state_dict(saved["model"])
    model.eval()
    values = np.fromfile(tensor_path, dtype="<f4")
    require(values.size == 64 * 3 * 96 * 96 and np.isfinite(values).all(), "invalid audit tensor")
    with torch.inference_mode():
        logits = model(torch.from_numpy(values.reshape(64, 3, 96, 96))).detach().cpu().numpy()
    require(logits.shape == (64, 13) and np.isfinite(logits).all(), "invalid native logits")
    output = local_file(root, "native-logits.f32", 64 * 13 * 4, must_exist=False)
    private_write(output, logits.astype("<f4", copy=False).tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preprocess", "native"))
    parser.add_argument("control", type=Path)
    args = parser.parse_args()
    try:
        (preprocess if args.stage == "preprocess" else native)(args.control)
    except (Invalid, ValueError, OSError, KeyError) as error:
        raise SystemExit(f"classifier audit failed: {error}") from None


if __name__ == "__main__":
    main()
