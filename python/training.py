#!/usr/bin/env python3
"""Native training and evaluation for the frozen issue #3 synthetic bootstrap.

This file runs only inside the pinned, network-disabled training container. It
reads hash-frozen inputs and writes checkpoints/evidence below the ignored run
directory supplied by the controller.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Iterable

import numpy as np
import PIL
from PIL import Image
import cv2
import safetensors.torch
import torch
from torch import nn
import torch.nn.functional as F


LABELS = ".PNBRQKpnbrqk"
LEGACY_DETECTOR_PREPROCESSING = "legacy-bgr-div255-v1"
V2_DETECTOR_PREPROCESSING = "yolox-rgb-imagenet-v2"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
LOADERS = ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1))


class Stopped(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> Any:
    require(path.is_file() and not path.is_symlink(), f"missing or unsafe file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def directory_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file() and not path.is_symlink())


def resource_guard(run: Path, config: dict[str, Any], current_cpu_seconds: float) -> None:
    prior_cpu = sum(read_json(path).get("cpu_seconds", 0) for path in run.glob("resource-*.json"))
    require(prior_cpu + current_cpu_seconds <= config["resources"]["cpu_seconds"], "CPU budget exhausted")
    require(directory_size(run) <= config["resources"]["storage_bytes"], "training output storage ceiling exceeded")
    usage = os.statvfs(run)
    require(usage.f_bavail / max(1, usage.f_blocks) >= config["resources"]["minimum_free_space_ratio"], "filesystem free-space floor crossed")


def configure(seed: int) -> torch.device:
    require(torch.cuda.is_available(), "CUDA is unavailable")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    return torch.device("cuda")


def freeze_batch_norm(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)


def load_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], dict[int, str]]:
    config = read_json(args.recipe)
    require(config.get("schema") in {"chess-ocr-training-recipe/1", "chess-ocr-training-recipe/2"},
            "recipe schema")
    dependencies = config["environment"]["dependencies"]
    require(np.__version__ == dependencies["numpy"] and PIL.__version__ == dependencies["pillow"], "training NumPy/Pillow versions")
    require(importlib.metadata.version("onnx") == dependencies["onnx"], "training ONNX version")
    require(importlib.metadata.version("onnxruntime") == dependencies["onnxruntime"],
            "training ONNX Runtime version")
    require(importlib.metadata.version("opencv-python-headless") == dependencies["opencv-python-headless"],
            "training OpenCV version")
    require(importlib.metadata.version("safetensors") == dependencies["safetensors"],
            "training safetensors version")
    require(importlib.metadata.version("timm") == dependencies["timm"], "training timm version")
    require(config["dataset"]["label_order"] == LABELS, "label order")
    require(sha256(args.dataset / "frozen.json") == config["dataset"]["frozen_sha256"], "dataset frozen hash")
    require(sha256(args.dataset / "recipes.json") == config["dataset"]["recipes_sha256"], "dataset recipes hash")
    require(sha256(args.dataset / "coverage.json") == config["dataset"]["coverage_sha256"], "dataset coverage hash")
    recipes = read_json(args.dataset / "recipes.json")
    split = read_json(args.split)
    require(split.get("dataset_run_id") == config["dataset"]["run_id"], "split dataset identity")
    page_split = {int(key): value for key, value in split["page_split"].items()}
    require(set(page_split) == set(range(len(recipes))), "split membership")
    return config, recipes, page_split


def solve(matrix: list[list[float]], values: list[float]) -> list[float]:
    rows = [list(map(float, row)) + [float(value)] for row, value in zip(matrix, values)]
    for index in range(len(rows)):
        pivot = max(range(index, len(rows)), key=lambda row: abs(rows[row][index]))
        rows[index], rows[pivot] = rows[pivot], rows[index]
        require(abs(rows[index][index]) > 1e-10, "singular grid")
        scale = rows[index][index]
        rows[index] = [value / scale for value in rows[index]]
        for row in range(len(rows)):
            if row != index:
                scale = rows[row][index]
                rows[row] = [value - scale * current for value, current in zip(rows[row], rows[index])]
    return [row[-1] for row in rows]


def rectified_grid(image: Image.Image, corners: list[list[float]]) -> Image.Image:
    require(len(corners) == 4 and all(len(point) == 2 for point in corners), "board corners")
    matrix, values = [], []
    for (u, v), (x, y) in zip(((0, 0), (768, 0), (768, 768), (0, 768)), corners):
        matrix.extend(((u, v, 1, 0, 0, 0, -x * u, -x * v),
                       (0, 0, 0, u, v, 1, -y * u, -y * v)))
        values.extend((x, y))
    return image.transform((768, 768), Image.Transform.PERSPECTIVE, solve(matrix, values), Image.Resampling.BICUBIC)


class ClassifierBoards:
    def __init__(self, root: Path, recipes: list[dict[str, Any]], page_split: dict[int, str], split: str):
        self.root = root
        self.recipes = recipes
        self.boards: list[tuple[int, int]] = []
        for page_index, page in enumerate(recipes):
            if page_split[page_index] == split and page.get("kind") == "boards":
                self.boards.extend((page_index, board_index) for board_index in range(len(page["boards"])))
        require(self.boards, f"no classifier boards in {split}")

    def __len__(self) -> int:
        return len(self.boards)

    def get(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        page_index, board_index = self.boards[index]
        page = self.recipes[page_index]
        image_path = self.root / "images" / f"page-{page_index:06d}.png"
        with Image.open(image_path) as source:
            source.load()
            image = source.convert("RGB")
        board = page["boards"][board_index]
        require(len(board.get("labels", [])) == 64 and all(label in LABELS for label in board["labels"]), "classifier labels")
        grid = rectified_grid(image, board["corners"])
        array = np.asarray(grid, dtype=np.float32) / 255.0
        tiles = array.reshape(8, 96, 8, 96, 3).transpose(0, 2, 4, 1, 3).reshape(64, 3, 96, 96)
        tiles = (tiles - np.array([.485, .456, .406], dtype=np.float32)[None, :, None, None]) / np.array([.229, .224, .225], dtype=np.float32)[None, :, None, None]
        labels = torch.tensor([LABELS.index(label) for label in board["labels"]], dtype=torch.long)
        metadata = {"page": page_index, "board": board_index, "set": board["set"],
                    "effect": page["condition"]["degradation"]["variant"], "layout": page["layout"],
                    "orientation": board["orientation"]}
        return torch.from_numpy(np.ascontiguousarray(tiles)), labels, metadata

class DetectorPages:
    def __init__(self, root: Path, recipes: list[dict[str, Any]], page_split: dict[int, str], split: str,
                 preprocessing: str = LEGACY_DETECTOR_PREPROCESSING):
        self.root = root
        self.recipes = recipes
        require(preprocessing in {LEGACY_DETECTOR_PREPROCESSING, V2_DETECTOR_PREPROCESSING},
                "detector preprocessing identifier")
        self.preprocessing = preprocessing
        # Partial/unsupported diagrams contain visible board-like structure but
        # no board the complete-path detector is allowed to return.  They are
        # excluded from optimization and included as no-valid-board cases in
        # development/calibration so their false detections cannot disappear
        # from reported metrics or threshold selection.
        admitted_kinds = {"boards", "negative"}
        if split != "train":
            admitted_kinds.update(("partial", "unsupported"))
        self.pages = [index for index, page in enumerate(recipes)
                      if page_split[index] == split and page.get("kind") in admitted_kinds]
        require(self.pages, f"no detector pages in {split}")

    def __len__(self) -> int:
        return len(self.pages)

    def get(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        page_index = self.pages[index]
        page = self.recipes[page_index]
        path = self.root / "images" / f"page-{page_index:06d}.png"
        with Image.open(path) as source:
            source.load()
            image = source.convert("RGB")
        width, height = image.size
        rgb = np.asarray(image, dtype=np.uint8)
        canvas, scale = detector_letterbox_rgb(rgb)
        if self.preprocessing == LEGACY_DETECTOR_PREPROCESSING:
            canvas = canvas[[2, 1, 0]]
        targets = torch.zeros((8, 5), dtype=torch.float32)
        boxes = []
        require(len(page.get("boards", [])) <= 8, "detector target count")
        for target_index, board in enumerate(page.get("boards", [])):
            xs = [point[0] * scale for point in board["corners"]]
            ys = [point[1] * scale for point in board["corners"]]
            left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
            targets[target_index] = torch.tensor([0, (left + right) / 2, (top + bottom) / 2,
                                                   right - left, bottom - top])
            boxes.append([left, top, right, bottom])
        kind = str(page.get("kind"))
        metadata = {"page": page_index, "scale": scale, "boxes": boxes, "negative": kind == "negative",
                    "no_valid_board": not boxes, "evaluation_case": kind,
                    "effect": page["condition"]["degradation"]["variant"], "layout": page["layout"]}
        return torch.from_numpy(np.ascontiguousarray(canvas)), targets, metadata


def detector_letterbox_rgb(rgb: np.ndarray, size: int = 416) -> tuple[np.ndarray, float]:
    require(rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.dtype == np.uint8,
            "detector RGB source")
    height, width = rgb.shape[:2]
    require(width > 0 and height > 0 and size > 0, "detector source dimensions")
    scale = min(size / width, size / height)
    resized = cv2.resize(rgb, (int(width * scale), int(height * scale)),
                         interpolation=cv2.INTER_LINEAR).astype(np.float32)
    canvas = np.full((size, size, 3), 114, dtype=np.float32)
    canvas[:resized.shape[0], :resized.shape[1]] = resized
    return np.ascontiguousarray(canvas.transpose(2, 0, 1)), scale


def letterbox_targets(normalized: list[list[float]], width: int, height: int,
                      size: int = 416) -> torch.Tensor:
    """Convert source-relative xywh targets to YOLOX's uncentered letterbox frame."""
    require(width > 0 and height > 0 and size > 0, "detector target dimensions")
    require(len(normalized) <= 8, "detector target count")
    scale = min(size / width, size / height)
    result = torch.zeros((8, 5), dtype=torch.float32)
    for index, target in enumerate(normalized):
        require(len(target) == 4 and all(math.isfinite(float(value)) for value in target),
                "detector normalized target")
        cx, cy, box_width, box_height = map(float, target)
        result[index] = torch.tensor((0, cx * width * scale, cy * height * scale,
                                      box_width * width * scale, box_height * height * scale))
    return result


class CyclingOrder:
    def __init__(self, length: int, seed: int, state: dict[str, Any] | None = None):
        self.length = length
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.empty(0, dtype=torch.long)
        self.cursor = 0
        if state:
            self.generator.set_state(state["generator"])
            self.order = state["order"]
            self.cursor = int(state["cursor"])

    def take(self, count: int) -> list[int]:
        result = []
        while len(result) < count:
            if self.cursor >= len(self.order):
                self.order = torch.randperm(self.length, generator=self.generator)
                self.cursor = 0
            amount = min(count - len(result), len(self.order) - self.cursor)
            result.extend(self.order[self.cursor:self.cursor + amount].tolist())
            self.cursor += amount
        return result

    def state(self) -> dict[str, Any]:
        return {"generator": self.generator.get_state(), "order": self.order, "cursor": self.cursor}


def cosine_lr(base: float, step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup - 1)
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def classifier_model(native: Path, device: torch.device) -> nn.Module:
    import timm
    path = native / "cache/native/mobilenetv3_small_100.lamb_in1k.safetensors"
    state = safetensors.torch.load_file(str(path), device="cpu")
    source = timm.create_model("mobilenetv3_small_100.lamb_in1k", pretrained=False, num_classes=1000)
    source.load_state_dict(state, strict=True)
    model = timm.create_model("mobilenetv3_small_100.lamb_in1k", pretrained=False, num_classes=13)
    target = model.state_dict()
    compatible = {key: value for key, value in source.state_dict().items()
                  if key in target and target[key].shape == value.shape}
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    require(not unexpected and set(missing) == {"classifier.weight", "classifier.bias"}, "classifier transfer keys")
    return model.to(device)


def detector_model(native: Path, device: torch.device) -> nn.Module:
    import sys
    source = native / "cache/native/yolox"
    require(source.is_dir(), "YOLOX source")
    sys.path.insert(0, str(source))
    from yolox.exp import get_exp
    exp = get_exp(None, "yolox-nano")
    exp.num_classes = 1
    model = exp.get_model()
    checkpoint = torch.load(native / "cache/native/yolox_nano.pth", map_location="cpu", weights_only=True)
    source_state = checkpoint.get("model", checkpoint)
    target = model.state_dict()
    compatible = {key: value for key, value in source_state.items()
                  if key in target and target[key].shape == value.shape}
    result = model.load_state_dict(compatible, strict=False)
    require(not result.unexpected_keys, "detector transfer keys")
    classifier_missing = {key for key in result.missing_keys if ".cls_preds." in f".{key}."}
    require(set(result.missing_keys) == classifier_missing and classifier_missing, "detector incompatible keys")
    return model.to(device)


def batch_classifier(dataset: ClassifierBoards, indices: Iterable[int], device: torch.device):
    items = list(LOADERS.map(dataset.get, indices))
    return torch.cat([item[0] for item in items]).to(device), torch.cat([item[1] for item in items]).to(device)


def batch_detector(dataset: DetectorPages, indices: Iterable[int], device: torch.device):
    items = list(LOADERS.map(dataset.get, indices))
    inputs = detector_training_tensor(torch.stack([item[0] for item in items]).to(device),
                                      getattr(dataset, "preprocessing", LEGACY_DETECTOR_PREPROCESSING))
    return inputs, torch.stack([item[1] for item in items]).to(device), [item[2] for item in items]


def detector_preprocessing(config: dict[str, Any]) -> str:
    return config["detector"].get("preprocessing", LEGACY_DETECTOR_PREPROCESSING)


def detector_training_tensor(raw: torch.Tensor,
                             preprocessing: str = LEGACY_DETECTOR_PREPROCESSING) -> torch.Tensor:
    require(torch.is_floating_point(raw), "detector raw tensor dtype")
    require(torch.isfinite(raw).all() and float(raw.min()) >= 0 and float(raw.max()) <= 255,
            "detector raw tensor range")
    if preprocessing == LEGACY_DETECTOR_PREPROCESSING:
        return raw / 255.0
    require(preprocessing == V2_DETECTOR_PREPROCESSING, "detector preprocessing identifier")
    mean = raw.new_tensor(IMAGENET_MEAN)[None, :, None, None]
    std = raw.new_tensor(IMAGENET_STD)[None, :, None, None]
    return (raw / 255.0 - mean) / std


def rng_state() -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], numpy_state[1].numpy(), numpy_state[2], numpy_state[3], numpy_state[4]))
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
               sampler: CyclingOrder, stage: int, stage_step: int, global_step: int,
               extra: dict[str, Any] | None = None) -> None:
    numpy_state = np.random.get_state()
    state = {"schema": "chess-ocr-training-checkpoint/1", "model": model.state_dict(),
             "optimizer": optimizer.state_dict(), "sampler": sampler.state(), "stage": stage,
             "stage_step": stage_step, "global_step": global_step,
             "rng": {"python": random.getstate(), "numpy": (numpy_state[0], torch.from_numpy(numpy_state[1].copy()), numpy_state[2], numpy_state[3], numpy_state[4]),
                     "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()},
             "extra": extra or {}}
    atomic_torch_save(path, state)


def latest_checkpoint(directory: Path) -> Path | None:
    paths = sorted(directory.glob("checkpoint-*.pt"))
    return paths[-1] if paths else None


@torch.inference_mode()
def evaluate_classifier(model: nn.Module, dataset: ClassifierBoards, device: torch.device) -> dict[str, Any]:
    model.eval()
    confusion = torch.zeros((13, 13), dtype=torch.int64)
    total_loss = 0.0
    exact = 0
    confident_wrong = 0
    confidences = []
    correctness = []
    orientation = {name: {"boards": 0, "exact_boards": 0, "squares": 0, "correct_squares": 0,
                          "nll_sum": 0.0, "confidence_sum": 0.0, "confident_wrong_squares_at_0_99": 0,
                          "_confidences": [], "_correctness": []}
                   for name in ("white-bottom", "black-bottom")}
    for start in range(0, len(dataset), 8):
        indices = list(range(start, min(len(dataset), start + 8)))
        inputs, labels = batch_classifier(dataset, indices, device)
        logits = model(inputs)
        total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
        probabilities = logits.softmax(1)
        confidence, predictions = probabilities.max(1)
        for truth, prediction in zip(labels.cpu(), predictions.cpu()):
            confusion[truth, prediction] += 1
        boards = labels.numel() // 64
        board_correct = predictions.reshape(boards, 64) == labels.reshape(boards, 64)
        exact += int(board_correct.all(1).sum())
        for offset, index in enumerate(indices):
            page, board = dataset.boards[index]
            name = dataset.recipes[page]["boards"][board]["orientation"]
            orientation[name]["boards"] += 1
            orientation[name]["exact_boards"] += int(board_correct[offset].all())
            orientation[name]["squares"] += 64
            orientation[name]["correct_squares"] += int(board_correct[offset].sum())
            square_slice = slice(offset * 64, (offset + 1) * 64)
            board_logits, board_labels = logits[square_slice], labels[square_slice]
            board_confidence = confidence[square_slice]
            board_predictions = predictions[square_slice]
            orientation[name]["nll_sum"] += F.cross_entropy(board_logits, board_labels, reduction="sum").item()
            orientation[name]["confidence_sum"] += float(board_confidence.sum())
            orientation[name]["confident_wrong_squares_at_0_99"] += int(
                ((board_predictions != board_labels) & (board_confidence >= 0.99)).sum())
            orientation[name]["_confidences"].extend(board_confidence.cpu().tolist())
            orientation[name]["_correctness"].extend((board_predictions == board_labels).cpu().tolist())
        confident_wrong += int(((predictions != labels) & (confidence >= 0.99)).sum())
        confidences.extend(confidence.cpu().tolist())
        correctness.extend((predictions == labels).cpu().tolist())
    occupied_f1 = []
    for cls in range(1, 13):
        tp = confusion[cls, cls].item()
        fp = confusion[:, cls].sum().item() - tp
        fn = confusion[cls, :].sum().item() - tp
        occupied_f1.append(2 * tp / max(1, 2 * tp + fp + fn))
    total = int(confusion.sum())
    empty_errors = int(confusion[0, 1:].sum() + confusion[1:, 0].sum())
    color_errors = int(confusion[1:7, 7:13].sum() + confusion[7:13, 1:7].sum())
    piece_class_errors = 0
    for truth in range(1, 13):
        for prediction in range(1, 13):
            if (truth - 1) % 6 != (prediction - 1) % 6:
                piece_class_errors += int(confusion[truth, prediction])
    coverage = {}
    for threshold in (0.5, 0.7, 0.9, 0.95, 0.99):
        accepted = [index for index, confidence in enumerate(confidences) if confidence >= threshold]
        coverage[str(threshold)] = {"coverage": len(accepted) / total,
                                    "wrong": sum(not correctness[index] for index in accepted)}
    orientation_report = {}
    for name, values in orientation.items():
        orientation_coverage = {}
        for threshold in (0.5, 0.7, 0.9, 0.95, 0.99):
            accepted = [index for index, value in enumerate(values["_confidences"]) if value >= threshold]
            orientation_coverage[str(threshold)] = {
                "coverage": len(accepted) / values["squares"],
                "wrong": sum(not values["_correctness"][index] for index in accepted)}
        orientation_report[name] = {key: value for key, value in values.items()
                                    if not key.startswith("_") and key not in {"nll_sum", "confidence_sum"}}
        orientation_report[name].update({"exact_board_accuracy": values["exact_boards"] / values["boards"],
                                         "square_accuracy": values["correct_squares"] / values["squares"],
                                         "nll": values["nll_sum"] / values["squares"],
                                         "mean_confidence": values["confidence_sum"] / values["squares"],
                                         "confidence_coverage": orientation_coverage})
    return {"boards": len(dataset), "squares": total, "exact_boards": exact,
            "exact_board_accuracy": exact / len(dataset), "square_accuracy": int(confusion.diag().sum()) / total,
            "occupied_macro_f1": sum(occupied_f1) / len(occupied_f1), "nll": total_loss / total,
            "confident_wrong_squares_at_0_99": confident_wrong, "mean_confidence": sum(confidences) / len(confidences),
            "empty_occupied_errors": empty_errors, "color_errors": color_errors,
            "piece_class_errors": piece_class_errors, "confidence_coverage": coverage,
            "orientation": orientation_report, "confusion": confusion.tolist()}


@torch.inference_mode()
def calibrate_classifier(model: nn.Module, dataset: ClassifierBoards, device: torch.device) -> dict[str, Any]:
    model.eval()
    logits, labels = [], []
    for start in range(0, len(dataset), 8):
        inputs, truth = batch_classifier(dataset, range(start, min(len(dataset), start + 8)), device)
        logits.append(model(inputs).cpu())
        labels.append(truth.cpu())
    scores, truth = torch.cat(logits), torch.cat(labels)
    candidates = [0.25 + index * 0.025 for index in range(151)]
    temperature = min(candidates, key=lambda value: float(F.cross_entropy(scores / value, truth)))
    probabilities = (scores / temperature).softmax(1)
    confidence, prediction = probabilities.max(1)
    wrong = confidence[prediction != truth]
    threshold = min(1.0, float(wrong.max()) + 1e-6) if len(wrong) else 0.0
    accepted = confidence >= threshold
    return {"temperature": temperature, "uncertainty_threshold": threshold,
            "accepted_square_coverage": float(accepted.float().mean()),
            "accepted_wrong_squares": int(((prediction != truth) & accepted).sum()),
            "truth": "synthetic TRAIN-purpose calibration; not production calibration"}


def set_trainable_classifier(model: nn.Module, names: list[str]) -> list[dict[str, Any]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    groups = []
    for name in names:
        module = model
        for part in name.split("."):
            module = getattr(module, part)
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    freeze_batch_norm(model)
    backbone, head = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (head if name.startswith(("conv_head", "classifier")) else backbone).append(parameter)
    if backbone:
        groups.append({"params": backbone, "role": "backbone"})
    groups.append({"params": head, "role": "head"})
    return groups


def classifier_optimizer(model: nn.Module, stage: dict[str, Any]) -> torch.optim.AdamW:
    parameter_groups = []
    for group in set_trainable_classifier(model, stage["trainable"]):
        base = (stage.get("backbone_learning_rate", stage.get("learning_rate"))
                if group["role"] == "backbone" else stage.get("head_learning_rate", stage.get("learning_rate")))
        parameter_groups.append({"params": group["params"], "base_lr": base, "lr": base})
    return torch.optim.AdamW(parameter_groups, weight_decay=stage["weight_decay"])


def add_count(counts: dict[str, int], key: Any, amount: int = 1) -> None:
    name = str(key)
    counts[name] = counts.get(name, 0) + amount


def validate_checkpoint_state(saved: dict[str, Any], stages: list[dict[str, Any]], role: str) -> None:
    require(saved.get("schema") == "chess-ocr-training-checkpoint/1", f"{role} checkpoint schema")
    stage_index = saved.get("stage")
    require(type(stage_index) is int and 0 <= stage_index < len(stages), f"{role} checkpoint stage")
    stage_step = saved.get("stage_step")
    require(type(stage_step) is int and 0 <= stage_step <= stages[stage_index]["updates"],
            f"{role} checkpoint stage step")
    expected_global = sum(stage["updates"] for stage in stages[:stage_index]) + stage_step
    require(saved.get("global_step") == expected_global, f"{role} checkpoint global step")
    require(all(key in saved for key in ("model", "optimizer", "sampler", "rng", "extra")),
            f"{role} checkpoint state")


def train_classifier(args: argparse.Namespace, config: dict[str, Any], recipes: list[dict[str, Any]],
                     page_split: dict[int, str], device: torch.device) -> None:
    directory = args.run / "classifier"
    directory.mkdir(parents=True, exist_ok=True)
    train = ClassifierBoards(args.dataset, recipes, page_split, "train")
    development = ClassifierBoards(args.dataset, recipes, page_split, "development")
    model = classifier_model(args.native, device)
    stages = config["classifier"]["stages"]
    resume = latest_checkpoint(directory)
    resume_state = torch.load(resume, map_location="cpu", weights_only=True) if resume else None
    if resume_state:
        validate_checkpoint_state(resume_state, stages, "classifier")
        model.load_state_dict(resume_state["model"], strict=True)
        restore_rng(resume_state["rng"])
    global_step = int(resume_state["global_step"]) if resume_state else 0
    curves = read_json(directory / "curves.json") if (directory / "curves.json").exists() else []
    interval_loss = 0.0
    interval_updates = 0
    exposure = {"classes": {}, "sets": {}, "effects": {}, "orientations": {}}
    for stage_index, stage in enumerate(stages):
        if resume_state and stage_index < resume_state["stage"]:
            continue
        optimizer = classifier_optimizer(model, stage)
        stage_step = 0
        sampler_state = None
        if resume_state and stage_index == resume_state["stage"]:
            optimizer.load_state_dict(resume_state["optimizer"])
            stage_step = int(resume_state["stage_step"])
            sampler_state = resume_state["sampler"]
        sampler = CyclingOrder(len(train), config["seed"] + stage_index, sampler_state)
        model.train()
        freeze_batch_norm(model)
        while stage_step < stage["updates"]:
            board_count = config["classifier"]["batch_size"] // 64
            indices = sampler.take(board_count)
            inputs, labels = batch_classifier(train, indices, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = F.cross_entropy(logits, labels)
            require(torch.isfinite(loss).item(), "non-finite classifier loss")
            loss.backward()
            require(all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None), "non-finite classifier gradients")
            for group in optimizer.param_groups:
                group["lr"] = cosine_lr(group["base_lr"], stage_step, stage["updates"], stage["warmup_updates"])
            optimizer.step()
            interval_loss += float(loss.detach())
            interval_updates += 1
            for class_index, count in enumerate(torch.bincount(labels.detach().cpu(), minlength=13).tolist()):
                add_count(exposure["classes"], LABELS[class_index], count)
            for index in indices:
                page_index, board_index = train.boards[index]
                board = recipes[page_index]["boards"][board_index]
                add_count(exposure["sets"], board["set"])
                add_count(exposure["effects"], recipes[page_index]["condition"]["degradation"]["variant"])
                add_count(exposure["orientations"], board["orientation"])
            stage_step += 1
            global_step += 1
            if global_step % config["classifier"]["checkpoint_interval_updates"] == 0 or stage_step == stage["updates"]:
                metrics = evaluate_classifier(model, development, device)
                curves.append({"global_step": global_step, "stage": stage["name"], "stage_step": stage_step,
                               "training_loss": interval_loss / interval_updates,
                               "interval_updates": interval_updates, "exposure": exposure,
                               "learning_rates": [g["lr"] for g in optimizer.param_groups],
                               "development": metrics})
                write_json(directory / "curves.json", curves)
                checkpoint(directory / f"checkpoint-{global_step:06d}.pt", model, optimizer, sampler,
                           stage_index, stage_step, global_step)
                resource_guard(args.run, config, time.process_time())
                write_json(directory / "progress.json", {"state": "running", "global_step": global_step,
                           "scheduled_updates": 10000, "stage": stage["name"], "development": metrics})
                if (args.run / "stop").exists():
                    raise Stopped("operator stop requested")
                interval_loss = 0.0
                interval_updates = 0
                exposure = {"classes": {}, "sets": {}, "effects": {}, "orientations": {}}
            model.train()
            freeze_batch_norm(model)
        resume_state = None
    best = max(curves, key=lambda row: (row["development"]["exact_board_accuracy"],
                                       row["development"]["occupied_macro_f1"],
                                       -row["development"]["nll"]))
    selected = torch.load(directory / f"checkpoint-{best['global_step']:06d}.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(selected["model"], strict=True)
    final = evaluate_classifier(model, development, device)
    calibration = calibrate_classifier(model, ClassifierBoards(args.dataset, recipes, page_split, "calibration"), device)
    export_classifier(model, args.run, config, {"development": final, "calibration": calibration,
                                               "selected_global_step": best["global_step"]})
    write_json(directory / "progress.json", {"state": "complete", "global_step": global_step,
               "scheduled_updates": 10000, "selected_global_step": best["global_step"],
               "development": final, "calibration": calibration})
    resource_guard(args.run, config, time.process_time())


def box_iou(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lt = torch.maximum(left[:, None, :2], right[None, :, :2])
    rb = torch.minimum(left[:, None, 2:], right[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    intersection = wh[:, :, 0] * wh[:, :, 1]
    area_left = (left[:, 2] - left[:, 0]) * (left[:, 3] - left[:, 1])
    area_right = (right[:, 2] - right[:, 0]) * (right[:, 3] - right[:, 1])
    return intersection / (area_left[:, None] + area_right[None, :] - intersection).clamp(min=1e-9)


def decode_detector(output: torch.Tensor, confidence: float, nms_iou: float) -> list[torch.Tensor]:
    results = []
    for page in output:
        scores = page[:, 4] * page[:, 5]
        keep = scores >= confidence
        xywh = page[keep, :4]
        scores = scores[keep]
        boxes = torch.stack((xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2,
                             xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2), 1) if len(xywh) else xywh.new_zeros((0, 4))
        chosen = []
        order = scores.argsort(descending=True)
        while len(order):
            current = int(order[0])
            chosen.append(current)
            if len(order) == 1:
                break
            ious = box_iou(boxes[current:current + 1], boxes[order[1:]])[0]
            order = order[1:][ious <= nms_iou]
        if chosen:
            chosen_indices = torch.tensor(chosen, device=boxes.device, dtype=torch.long)
            results.append(torch.cat((boxes[chosen_indices], scores[chosen_indices, None]), 1))
        else:
            results.append(boxes.new_zeros((0, 5)))
    return results


def score_ordered_matches(predicted: torch.Tensor, truth: torch.Tensor,
                          iou_threshold: float) -> list[tuple[int, int, float]]:
    """Greedily match scored detections to one unused eligible target each."""
    require(predicted.ndim == 2 and predicted.shape[1] == 5, "detector predictions")
    require(truth.ndim == 2 and truth.shape[1] == 4, "detector truth boxes")
    require(0 <= iou_threshold <= 1, "detector IoU threshold")
    if not len(predicted) or not len(truth):
        return []
    ious = box_iou(predicted[:, :4], truth)
    order = torch.argsort(predicted[:, 4], descending=True, stable=True)
    used: set[int] = set()
    matches = []
    for prediction_index in order.tolist():
        eligible = [(float(iou), truth_index) for truth_index, iou in enumerate(ious[prediction_index])
                    if truth_index not in used and float(iou) >= iou_threshold]
        if not eligible:
            continue
        iou, truth_index = max(eligible, key=lambda item: (item[0], -item[1]))
        used.add(truth_index)
        matches.append((prediction_index, truth_index, iou))
    return matches


def unit_interval(value: float) -> float:
    """Clamp floating-point metric accumulation to its mathematical range."""
    require(math.isfinite(value), "non-finite bounded metric")
    return min(1.0, max(0.0, float(value)))


def metric_threshold_key(value: float) -> str:
    """Serialize the fixed IoU grid without binary floating-point artifacts."""
    return f"{value:.2f}".rstrip("0").rstrip(".")


@torch.inference_mode()
def evaluate_detector(model: nn.Module, dataset: DetectorPages, device: torch.device,
                      nms_iou: float) -> dict[str, Any]:
    model.eval()
    predictions = []
    total_targets = 0
    negative_pages = 0
    false_on_negative = 0
    no_valid_pages: dict[str, int] = {"negative": 0, "partial": 0, "unsupported": 0}
    false_on_no_valid: dict[str, int] = {"negative": 0, "partial": 0, "unsupported": 0}
    matched_iou_sum = 0.0
    normalized_box_errors = []
    strata: dict[str, dict[str, dict[str, float]]] = {key: {} for key in ("effect", "layout", "board_count", "size")}
    def observe(kind: str, name: str, targets: int, matched: int, false_positives: int,
                iou_sum: float, error_sum: float = 0.0, error_matches: int = 0) -> None:
        row = strata[kind].setdefault(name, {"targets": 0, "matched_at_iou_0_5": 0,
                                            "false_positives": 0, "iou_sum": 0.0,
                                            "normalized_box_error_sum": 0.0,
                                            "normalized_box_error_matches": 0})
        row["targets"] += targets; row["matched_at_iou_0_5"] += matched
        row["false_positives"] += false_positives; row["iou_sum"] += iou_sum
        row["normalized_box_error_sum"] += error_sum
        row["normalized_box_error_matches"] += error_matches
    for start in range(0, len(dataset), 16):
        indices = range(start, min(len(dataset), start + 16))
        inputs, _, metadata = batch_detector(dataset, indices, device)
        output = model(inputs)
        decoded = decode_detector(output, 0.01, nms_iou)
        for result, meta in zip(decoded, metadata):
            truth = torch.tensor(meta["boxes"], device=device, dtype=torch.float32).reshape(-1, 4)
            total_targets += len(truth)
            case = str(meta.get("evaluation_case", "negative" if meta.get("negative") else "boards"))
            if meta.get("no_valid_board", meta.get("negative", False)):
                no_valid_pages.setdefault(case, 0)
                false_on_no_valid.setdefault(case, 0)
                no_valid_pages[case] += 1
                false_on_no_valid[case] += len(result)
            negative_pages += int(case == "negative")
            false_on_negative += len(result) if case == "negative" else 0
            predictions.append((result.detach().cpu(), truth.detach().cpu()))
            matches = score_ordered_matches(result, truth, 0.5)
            matched = len(matches)
            iou_sum = sum(iou for _, _, iou in matches)
            matched_iou_sum += iou_sum
            errors_by_truth = {}
            for prediction_index, truth_index, _ in matches:
                error = float((result[prediction_index, :4] - truth[truth_index]).abs().mean() / 416)
                normalized_box_errors.append(error)
                errors_by_truth[truth_index] = error
            matched_by_truth = {truth_index: iou for _, truth_index, iou in matches}
            for truth_index, target in enumerate(truth):
                relative = float(torch.sqrt((target[2] - target[0]) * (target[3] - target[1])) / 416)
                size = "small" if relative < 0.4 else "medium" if relative < 0.7 else "large"
                observe("size", size, 1, int(truth_index in matched_by_truth), 0,
                        matched_by_truth.get(truth_index, 0.0), errors_by_truth.get(truth_index, 0.0),
                        int(truth_index in errors_by_truth))
            false_positives = max(0, len(result) - matched)
            error_sum = sum(errors_by_truth.values())
            for kind in ("effect", "layout"):
                observe(kind, str(meta[kind]), len(truth), matched, false_positives, iou_sum,
                        error_sum, len(errors_by_truth))
            observe("board_count", str(len(truth)), len(truth), matched, false_positives, iou_sum,
                    error_sum, len(errors_by_truth))
    recalls = {}
    precisions = {}
    aps = []
    ap_by_iou = {}
    for threshold in [0.5 + i * 0.05 for i in range(10)]:
        threshold_key = metric_threshold_key(threshold)
        scored = []
        positives = 0
        for predicted, truth in predictions:
            positives += len(truth)
            matches = score_ordered_matches(predicted, truth, threshold)
            true_predictions = {prediction_index for prediction_index, _, _ in matches}
            order = torch.argsort(predicted[:, 4], descending=True, stable=True)
            for prediction_index in order.tolist():
                scored.append((float(predicted[prediction_index, 4]),
                               int(prediction_index in true_predictions)))
        scored.sort(key=lambda item: -item[0])
        tp = 0
        precision_curve, recall_curve = [], []
        for rank, (_, good) in enumerate(scored, 1):
            tp += good
            precision_curve.append(tp / rank)
            recall_curve.append(tp / max(1, positives))
        ap = 0.0
        for recall_level in [i / 100 for i in range(101)]:
            ap += max((p for p, r in zip(precision_curve, recall_curve) if r >= recall_level), default=0) / 101
        bounded_ap = unit_interval(ap)
        aps.append(bounded_ap)
        ap_by_iou[threshold_key] = bounded_ap
        recalls[threshold_key] = unit_interval(recall_curve[-1] if recall_curve else 0)
        precisions[threshold_key] = unit_interval(precision_curve[-1] if precision_curve else 0)
    normalized_strata = {kind: {name: {**row,
        "recall_at_iou_0_5": row["matched_at_iou_0_5"] / max(1, row["targets"]),
        "mean_iou_at_0_5_including_misses": row["iou_sum"] / max(1, row["targets"]),
        "mean_normalized_box_error_on_matches": row["normalized_box_error_sum"] /
        max(1, row["normalized_box_error_matches"])} for name, row in values.items()}
        for kind, values in strata.items()}
    return {"pages": len(dataset), "targets": total_targets, "negative_pages": negative_pages,
            "ap50_95": unit_interval(sum(aps) / len(aps)), "ap_by_iou": ap_by_iou,
            "recall": recalls, "precision": precisions,
            "false_detections_on_negative_pages_at_0_01": false_on_negative,
            "no_valid_board_pages": no_valid_pages,
            "false_detections_on_no_valid_board_pages_at_0_01": false_on_no_valid,
            "mean_iou_at_0_5_including_misses": matched_iou_sum / max(1, total_targets),
            "mean_normalized_box_error": sum(normalized_box_errors) / max(1, len(normalized_box_errors)),
            "normalized_box_error_definition": "mean coordinate MAE / 416 on one-to-one IoU>=0.5 matches only",
            "strata": normalized_strata}


@torch.inference_mode()
def calibrate_detector(model: nn.Module, dataset: DetectorPages, device: torch.device,
                       nms_iou: float) -> dict[str, Any]:
    model.eval()
    pages = []
    for start in range(0, len(dataset), 16):
        inputs, _, metadata = batch_detector(dataset, range(start, min(len(dataset), start + 16)), device)
        raw = model(inputs)
        for result, meta in zip(decode_detector(raw, 0.001, nms_iou), metadata):
            pages.append((result.cpu(), torch.tensor(meta["boxes"], dtype=torch.float32).reshape(-1, 4),
                          str(meta.get("evaluation_case", "negative" if meta.get("negative") else "boards"))))
    candidates = sorted({float(row[4]) for result, _, _ in pages for row in result}, reverse=True)
    candidates = candidates[::max(1, len(candidates) // 1000)] + [1.0]
    no_valid_counts = {kind: sum(not len(truth) and case == kind for _, truth, case in pages)
                       for kind in ("negative", "partial", "unsupported")}
    no_valid_total = sum(no_valid_counts.values())
    total_targets = sum(len(truth) for _, truth, _ in pages)
    selected = {"threshold": 1.0, "recall": 0.0,
                "false_positives_per_no_valid_board_page": 0.0,
                "false_positives_per_negative_page": 0.0,
                "false_detections_on_no_valid_board_pages": {kind: 0 for kind in no_valid_counts},
                "no_valid_board_pages": no_valid_counts}
    for threshold in candidates:
        matched = 0
        false_by_kind = {kind: 0 for kind in no_valid_counts}
        for result, truth, case in pages:
            result = result[result[:, 4] >= threshold]
            if not len(truth):
                false_by_kind.setdefault(case, 0)
                false_by_kind[case] += len(result)
            else:
                matched += len(score_ordered_matches(result, truth, 0.5))
        fp_rate = sum(false_by_kind.values()) / max(1, no_valid_total)
        recall = matched / max(1, total_targets)
        if fp_rate <= 0.05 and recall > selected["recall"]:
            selected = {"threshold": threshold, "recall": recall,
                        "false_positives_per_no_valid_board_page": fp_rate,
                        "false_positives_per_negative_page": false_by_kind.get("negative", 0) /
                        max(1, no_valid_counts.get("negative", 0)),
                        "false_detections_on_no_valid_board_pages": false_by_kind,
                        "no_valid_board_pages": no_valid_counts}
    selected["truth"] = "synthetic-development-only calibration; not production calibration or qualification"
    return selected


class EMA:
    def __init__(self, model: nn.Module, decay: float, warmup_updates: int | None = None):
        require(0 <= decay < 1, "EMA decay")
        require(warmup_updates is None or warmup_updates > 0, "EMA warmup updates")
        self.base_decay = decay
        self.warmup_updates = warmup_updates
        self.updates = 0
        self.state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    @property
    def decay(self) -> float:
        if self.warmup_updates is None:
            return self.base_decay
        return self.base_decay * (1 - math.exp(-self.updates / self.warmup_updates))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        decay = self.decay
        for key, value in model.state_dict().items():
            if value.is_floating_point():
                self.state[key].mul_(decay).add_(value.detach(), alpha=1 - decay)
            else:
                self.state[key].copy_(value)

    def checkpoint_state(self) -> dict[str, Any]:
        return {"weights": self.state, "updates": self.updates,
                "base_decay": self.base_decay, "warmup_updates": self.warmup_updates}

    def load_checkpoint_state(self, value: dict[str, Any], device: torch.device) -> None:
        # Retained v1 checkpoints stored the weight mapping directly.
        if "weights" not in value:
            self.state = {key: tensor.to(device) for key, tensor in value.items()}
            self.updates = 0
            return
        require(value.get("base_decay") == self.base_decay and
                value.get("warmup_updates") == self.warmup_updates and
                type(value.get("updates")) is int and value["updates"] >= 0,
                "EMA checkpoint configuration")
        self.state = {key: tensor.to(device) for key, tensor in value["weights"].items()}
        self.updates = value["updates"]


def set_trainable_detector(model: nn.Module, stage: str) -> list[nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(stage == "full-model")
    if stage == "head":
        for parameter in model.head.parameters():
            parameter.requires_grad_(True)
    freeze_batch_norm(model)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def detector_optimizer(model: nn.Module, config: dict[str, Any], stage: dict[str, Any]) -> torch.optim.SGD:
    parameters = set_trainable_detector(model, stage["name"])
    return torch.optim.SGD(parameters, lr=stage["learning_rate"], momentum=config["detector"]["momentum"],
                           nesterov=True, weight_decay=config["detector"]["weight_decay"])


def detector_transition_gate_failed(curves: list[dict[str, Any]], global_step: int,
                                    live_metrics: dict[str, Any] | None,
                                    gate: dict[str, Any] | None = None) -> bool:
    gate = gate or {"step": 2000, "prerequisite_step": 1500,
                    "prerequisite_live_recall_at_0_5_minimum": .10,
                    "stop_live_recall_at_0_5_below": .05}
    if global_step != gate["step"] or live_metrics is None:
        return False
    head = next((item.get("live_development") for item in curves
                 if item.get("global_step") == gate["prerequisite_step"]), None)
    return bool(head is not None and
                head["recall"]["0.5"] >= gate["prerequisite_live_recall_at_0_5_minimum"] and
                live_metrics["recall"]["0.5"] < gate["stop_live_recall_at_0_5_below"])


def finish_detector_lifecycle(model: nn.Module, args: argparse.Namespace, config: dict[str, Any],
                              calibration_data: DetectorPages, device: torch.device,
                              lifecycle: dict[str, Any]) -> None:
    directory = args.run / "detector"
    lifecycle["state"] = "calibrating"
    write_json(directory / "progress.json", lifecycle)
    calibration = calibrate_detector(model, calibration_data, device, config["detector"]["nms_iou"])
    lifecycle.update(state="exporting", calibration=calibration)
    write_json(directory / "progress.json", lifecycle)
    try:
        export_detector(model, args.run, config, {
            "development": lifecycle["final"], "calibration": calibration,
            "selected_global_step": lifecycle["selected_global_step"]})
    except Exception as error:
        lifecycle.update(state="export_failed", export_error=str(error))
        write_json(directory / "progress.json", lifecycle)
        raise
    lifecycle["state"] = "complete"
    lifecycle.pop("export_error", None)
    write_json(directory / "progress.json", lifecycle)


def train_detector(args: argparse.Namespace, config: dict[str, Any], recipes: list[dict[str, Any]],
                   page_split: dict[int, str], device: torch.device) -> None:
    directory = args.run / "detector"
    directory.mkdir(parents=True, exist_ok=True)
    preprocessing = detector_preprocessing(config)
    train = DetectorPages(args.dataset, recipes, page_split, "train", preprocessing)
    development = DetectorPages(args.dataset, recipes, page_split, "development", preprocessing)
    model = detector_model(args.native, device)
    stages = config["detector"]["stages"]
    dual_evaluation_steps = set(config["detector"].get("evaluation_steps", {}).get(
        "live_and_ema", []))
    resume = latest_checkpoint(directory)
    resume_state = torch.load(resume, map_location="cpu", weights_only=True) if resume else None
    ema = EMA(model, config["detector"]["ema_decay"],
              config["detector"].get("ema_warmup_updates"))
    if resume_state:
        validate_checkpoint_state(resume_state, stages, "detector")
        model.load_state_dict(resume_state["model"], strict=True)
        ema.load_checkpoint_state(resume_state["extra"]["ema"], device)
        restore_rng(resume_state["rng"])
    global_step = int(resume_state["global_step"]) if resume_state else 0
    curves = read_json(directory / "curves.json") if (directory / "curves.json").exists() else []
    interval_loss = 0.0
    interval_updates = 0
    exposure = {"effects": {}, "layouts": {}, "board_counts": {}, "positive_pages": 0,
                "negative_pages": 0}
    for stage_index, stage in enumerate(stages):
        if resume_state and stage_index < resume_state["stage"]:
            continue
        optimizer = detector_optimizer(model, config, stage)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        stage_step = 0
        sampler_state = None
        if resume_state and stage_index == resume_state["stage"]:
            optimizer.load_state_dict(resume_state["optimizer"])
            stage_step = int(resume_state["stage_step"])
            sampler_state = resume_state["sampler"]
        sampler = CyclingOrder(len(train), config["seed"] + 100 + stage_index, sampler_state)
        model.train()
        freeze_batch_norm(model)
        while stage_step < stage["updates"]:
            inputs, targets, metadata = batch_detector(train, sampler.take(config["detector"]["batch_size"]), device)
            optimizer.zero_grad(set_to_none=True)
            losses = model(inputs, targets)
            loss = losses["total_loss"]
            require(torch.isfinite(loss).item(), "non-finite detector loss")
            loss.backward()
            require(all(torch.isfinite(parameter.grad).all() for parameter in parameters if parameter.grad is not None), "non-finite detector gradients")
            torch.nn.utils.clip_grad_norm_(parameters, config["detector"]["gradient_clip_norm"])
            require(all(torch.isfinite(parameter.grad).all() for parameter in parameters if parameter.grad is not None), "non-finite clipped detector gradients")
            optimizer.param_groups[0]["lr"] = cosine_lr(stage["learning_rate"], stage_step, stage["updates"], stage["warmup_updates"])
            optimizer.step()
            ema.update(model)
            interval_loss += float(loss.detach())
            interval_updates += 1
            for item in metadata:
                add_count(exposure["effects"], item["effect"])
                add_count(exposure["layouts"], item["layout"])
                add_count(exposure["board_counts"], len(item["boxes"]))
                key = "negative_pages" if item["negative"] else "positive_pages"
                exposure[key] += 1
            stage_step += 1
            global_step += 1
            if global_step % config["detector"]["checkpoint_interval_updates"] == 0 or stage_step == stage["updates"]:
                live = {key: value.detach().clone() for key, value in model.state_dict().items()}
                live_metrics = (evaluate_detector(model, development, device, config["detector"]["nms_iou"])
                                if global_step in dual_evaluation_steps else None)
                model.load_state_dict(ema.state, strict=True)
                metrics = evaluate_detector(model, development, device, config["detector"]["nms_iou"])
                model.load_state_dict(live, strict=True)
                row = {"global_step": global_step, "stage": stage["name"], "stage_step": stage_step,
                       "training_loss": interval_loss / interval_updates,
                       "interval_updates": interval_updates, "exposure": exposure,
                       "losses": {key: float(value.detach()) if torch.is_tensor(value) else float(value) for key, value in losses.items()},
                       "learning_rate": optimizer.param_groups[0]["lr"], "development": metrics,
                       "ema_development": metrics, "ema_updates": ema.updates,
                       "ema_decay": ema.decay}
                if live_metrics is not None:
                    row["live_development"] = live_metrics
                curves.append(row)
                write_json(directory / "curves.json", curves)
                checkpoint(directory / f"checkpoint-{global_step:06d}.pt", model, optimizer, sampler,
                           stage_index, stage_step, global_step, {"ema": ema.checkpoint_state()})
                resource_guard(args.run, config, time.process_time())
                write_json(directory / "progress.json", {"state": "running", "global_step": global_step,
                           "scheduled_updates": sum(item["updates"] for item in stages),
                           "stage": stage["name"], "development": metrics,
                           "live_development": live_metrics, "ema_updates": ema.updates})
                gate = config["detector"].get("evaluation_steps", {}).get("transition_gate")
                if detector_transition_gate_failed(curves, global_step, live_metrics, gate):
                    raise Stopped("detector stage-transition gate: live recall collapsed below 5% at step 2000")
                if (args.run / "stop").exists():
                    raise Stopped("operator stop requested")
                interval_loss = 0.0
                interval_updates = 0
                exposure = {"effects": {}, "layouts": {}, "board_counts": {}, "positive_pages": 0,
                            "negative_pages": 0}
            model.train()
            freeze_batch_norm(model)
        resume_state = None
    best = max(curves, key=lambda row: (row["development"]["ap50_95"],
                                       row["development"]["recall"]["0.5"],
                                       -row["development"]["mean_normalized_box_error"]))
    scheduled = sum(item["updates"] for item in stages)
    lifecycle = {"state": "trained", "global_step": global_step,
                 "scheduled_updates": scheduled, "selected_global_step": best["global_step"],
                 "selection_development": best["development"]}
    write_json(directory / "progress.json", lifecycle)
    selected = torch.load(directory / f"checkpoint-{best['global_step']:06d}.pt", map_location="cpu", weights_only=True)
    selected_ema = selected["extra"]["ema"]
    selected_weights = selected_ema.get("weights", selected_ema)
    model.load_state_dict({key: value.to(device) for key, value in selected_weights.items()}, strict=True)
    final = evaluate_detector(model, development, device, config["detector"]["nms_iou"])
    lifecycle["final"] = final
    write_json(directory / "progress.json", lifecycle)
    finish_detector_lifecycle(
        model, args, config,
        DetectorPages(args.dataset, recipes, page_split, "calibration", preprocessing),
        device, lifecycle)
    resource_guard(args.run, config, time.process_time())


def export_detector_checkpoint(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Retry only ONNX export from durable selection/evaluation/calibration evidence."""
    directory = args.run / "detector"
    progress = read_json(directory / "progress.json")
    require(progress.get("state") in {"exporting", "export_failed"},
            "detector export retry requires exporting/export_failed state")
    require(type(progress.get("selected_global_step")) is int,
            "detector export retry selected checkpoint")
    require(isinstance(progress.get("final"), dict), "detector export retry final evaluation")
    require(isinstance(progress.get("calibration"), dict), "detector export retry calibration")
    selected_path = directory / f"checkpoint-{progress['selected_global_step']:06d}.pt"
    require(selected_path.is_file() and not selected_path.is_symlink(),
            "detector export retry checkpoint")
    selected = torch.load(selected_path, map_location="cpu", weights_only=True)
    selected_ema = selected.get("extra", {}).get("ema")
    require(isinstance(selected_ema, dict), "detector export retry EMA checkpoint")
    selected_weights = selected_ema.get("weights", selected_ema)
    model = detector_model(args.native, torch.device("cpu"))
    model.load_state_dict(selected_weights, strict=True)
    progress["state"] = "exporting"
    progress.pop("export_error", None)
    write_json(directory / "progress.json", progress)
    try:
        export_detector(model, args.run, config, {
            "development": progress["final"], "calibration": progress["calibration"],
            "selected_global_step": progress["selected_global_step"]})
    except Exception as error:
        progress.update(state="export_failed", export_error=str(error))
        write_json(directory / "progress.json", progress)
        raise
    progress["state"] = "complete"
    write_json(directory / "progress.json", progress)


@torch.inference_mode()
def audit_detector(args: argparse.Namespace, config: dict[str, Any], recipes: list[dict[str, Any]],
                   page_split: dict[int, str]) -> None:
    """Re-score a frozen detector without mutating its training lifecycle."""
    require(args.audit_output is not None, "detector audit output")
    output = args.audit_output.resolve()
    require(args.run.resolve() in output.parents and output.name.endswith(".json"),
            "detector audit output must be a JSON file below the run root")
    progress = read_json(args.run / "detector" / "progress.json")
    require(progress.get("state") == "complete" and
            progress.get("global_step") == progress.get("scheduled_updates") and
            type(progress.get("selected_global_step")) is int,
            "detector audit requires a completed selected run")
    checkpoint_path = args.run / "detector" / f"checkpoint-{progress['selected_global_step']:06d}.pt"
    onnx_path = args.run / "detector" / "selected.onnx"
    manifest_path = onnx_path.with_suffix(".manifest.json")
    manifest = read_json(manifest_path)
    require(manifest.get("sha256") == sha256(onnx_path), "detector audit ONNX identity")
    selected = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    validate_checkpoint_state(selected, config["detector"]["stages"], "detector")
    selected_ema = selected.get("extra", {}).get("ema")
    require(isinstance(selected_ema, dict), "detector audit EMA checkpoint")
    weights = selected_ema.get("weights", selected_ema)
    device = torch.device("cpu")
    model = detector_model(args.native, device)
    model.load_state_dict(weights, strict=True)
    model.eval()
    model.head.decode_in_inference = False
    preprocessing = detector_preprocessing(config)
    development = DetectorPages(args.dataset, recipes, page_split, "development", preprocessing)
    calibration_pages = DetectorPages(args.dataset, recipes, page_split, "calibration", preprocessing)

    # The complete evaluation uses the frozen native selected weights.  Execute
    # the already-exported ONNX on the same raw page tensor as a bounded proof
    # that this audit loaded both frozen artifacts and that their graph boundary
    # still agrees.
    import onnxruntime as ort
    raw, _, _ = development.get(0)
    raw_batch = raw.numpy()[None]
    native = DetectorExportWrapper(model, preprocessing).eval()(torch.from_numpy(raw_batch)).numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx = session.run(None, {session.get_inputs()[0].name: raw_batch})[0]
    parity_max_abs = float(np.max(np.abs(native - onnx)))
    require(parity_max_abs <= 1e-3, "detector audit native/ONNX parity")
    model.head.decode_in_inference = True

    started = time.time()
    development_metrics = evaluate_detector(model, development, device, config["detector"]["nms_iou"])
    calibration_metrics = evaluate_detector(model, calibration_pages, device, config["detector"]["nms_iou"])
    calibration = calibrate_detector(model, calibration_pages, device, config["detector"]["nms_iou"])
    write_json(output, {
        "schema": "chess-ocr-synthetic-detector-audit/1",
        "truth": "synthetic-development-only; not real development, calibration, or qualification",
        "source_run_id": read_json(args.run / "frozen.json")["run_id"],
        "selected_global_step": progress["selected_global_step"],
        "artifacts": {"checkpoint_sha256": sha256(checkpoint_path),
                      "onnx_sha256": sha256(onnx_path),
                      "onnx_manifest_sha256": sha256(manifest_path),
                      "split_sha256": sha256(args.split)},
        "preprocessing": preprocessing,
        "nms_iou": config["detector"]["nms_iou"],
        "native_onnx_max_abs": parity_max_abs,
        "development": development_metrics,
        "calibration_evaluation": calibration_metrics,
        "calibration": calibration,
        "started_at": started,
        "finished_at": time.time(),
    })


def verify_onnx(path: Path, inputs: dict[str, np.ndarray], expected: np.ndarray,
                maximum_absolute_difference: float = 1e-4) -> float:
    import onnx
    import onnxruntime as ort
    require(0 < maximum_absolute_difference <= 1e-3, "ONNX parity tolerance")
    onnx.checker.check_model(onnx.load(path))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(None, inputs)[0]
    require(actual.shape == expected.shape and np.isfinite(actual).all() and np.isfinite(expected).all(),
            "ONNX output shape/finite parity")
    maximum = float(np.max(np.abs(actual - expected)))
    require(maximum <= maximum_absolute_difference, f"ONNX output parity mismatch: {maximum}")
    return maximum


def export_classifier(model: nn.Module, run: Path, config: dict[str, Any], metrics: dict[str, Any]) -> None:
    model.eval().cpu()
    (run / "classifier").mkdir(parents=True, exist_ok=True)
    destination = run / "classifier" / "selected.onnx"
    example = torch.linspace(-2.0, 2.0, steps=2 * 3 * 96 * 96).reshape(2, 3, 96, 96)
    with torch.inference_mode():
        expected = model(example).numpy()
    torch.onnx.export(model, example, destination, input_names=["tiles"],
                      output_names=["logits"], dynamic_axes={"tiles": {0: "squares"}, "logits": {0: "squares"}},
                      opset_version=17, dynamo=False)
    metrics = {**metrics, "onnx_native_max_abs": verify_onnx(destination, {"tiles": example.numpy()}, expected)}
    write_json(destination.with_suffix(".manifest.json"), {"schema": "chess-ocr-model/1", "role": "square-classifier",
               "sha256": sha256(destination), "labels": LABELS, "preprocessing": config["classifier"]["input"],
               "metrics": metrics, "publication": "not-authorized"})


def export_classifier_checkpoint(args: argparse.Namespace, config: dict[str, Any],
                                 recipes: list[dict[str, Any]], page_split: dict[int, str]) -> None:
    """Materialize a completed classifier checkpoint for detector-only runs."""
    checkpoint_path = args.classifier_checkpoint
    require(checkpoint_path is not None and checkpoint_path.is_file() and not checkpoint_path.is_symlink(),
            "classifier checkpoint is missing or unsafe")
    model = classifier_model(args.native, torch.device("cpu"))
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    stages = config["classifier"]["stages"]
    scheduled = sum(stage["updates"] for stage in stages)
    validate_checkpoint_state(saved, stages, "classifier")
    model.load_state_dict(saved["model"], strict=True)
    record = read_json(args.run / "frozen.json")["classifier_checkpoint"]
    require(record["sha256"] == sha256(checkpoint_path) and
            record["scheduled_updates"] == scheduled and
            record["selected_global_step"] == saved["global_step"],
            "classifier checkpoint frozen identity")
    development = record["development"]
    calibration = {"state": "not-recomputed", "reason": "source run failed before persisting calibration"}
    selected_step = int(record["selected_global_step"])
    export_classifier(model, args.run, config, {"development": development, "calibration": calibration,
                                               "selected_global_step": selected_step,
                                               "source_checkpoint_sha256": sha256(checkpoint_path)})
    write_json(args.run / "classifier" / "progress.json", {"state": "complete", "global_step": scheduled,
               "scheduled_updates": 10000, "selected_global_step": selected_step,
               "development": development, "calibration": calibration,
               "source_checkpoint": str(checkpoint_path), "source_checkpoint_sha256": sha256(checkpoint_path)})
    resource_guard(args.run, config, time.process_time())


class DetectorExportWrapper(nn.Module):
    def __init__(self, detector: nn.Module, contract: str):
        super().__init__()
        require(contract in {LEGACY_DETECTOR_PREPROCESSING, V2_DETECTOR_PREPROCESSING},
                "detector export preprocessing")
        self.detector = detector
        self.contract = contract
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN)[None, :, None, None])
        self.register_buffer("std", torch.tensor(IMAGENET_STD)[None, :, None, None])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        values = images / 255.0
        if self.contract == V2_DETECTOR_PREPROCESSING:
            values = (values - self.mean) / self.std
        return self.detector(values)


def export_detector(model: nn.Module, run: Path, config: dict[str, Any], metrics: dict[str, Any]) -> None:
    model.eval().cpu()
    model.head.decode_in_inference = False
    destination = run / "detector" / "selected.onnx"
    raw = torch.linspace(0.0, 255.0, steps=3 * 416 * 416).reshape(1, 3, 416, 416)
    preprocessing = detector_preprocessing(config)
    exported = DetectorExportWrapper(model, preprocessing).eval()
    with torch.inference_mode():
        expected = exported(raw).numpy()
    torch.onnx.export(exported, raw, destination, input_names=["images"],
                      output_names=["predictions"], opset_version=13, do_constant_folding=True, dynamo=False)
    # YOLOX native/browser probes use a 1e-3 raw-output ceiling. The trained
    # graph's bounded diagnosis found <=4.19e-4 raw drift across 16 development
    # pages without changing decoded counts or materially changing boxes/scores.
    metrics = {**metrics, "onnx_native_max_abs": verify_onnx(
        destination, {"images": raw.numpy()}, expected, maximum_absolute_difference=1e-3)}
    write_json(destination.with_suffix(".manifest.json"), {"schema": "chess-ocr-model/1", "role": "inner-grid-detector",
               "sha256": sha256(destination), "labels": ["inner-grid"], "preprocessing": preprocessing,
               "nms_iou": config["detector"]["nms_iou"], "metrics": metrics, "publication": "not-authorized"})


def dataset_record(dataset: Path, page_index: int) -> dict[str, Any]:
    batch_start = page_index // 64 * 64
    return read_json(dataset / f"batch-{batch_start:06d}.json")[page_index - batch_start]


def pinned_yolox_preproc(native: Path):
    """Load the reviewed preprocessing file without YOLOX's optional COCO imports."""
    path = native / "cache/native/yolox/yolox/data/data_augment.py"
    require(path.is_file() and not path.is_symlink(), "pinned YOLOX preprocessing source")
    import sys
    source = str(native / "cache/native/yolox")
    if source not in sys.path:
        sys.path.insert(0, source)
    spec = importlib.util.spec_from_file_location("_chess_ocr_yolox_data_augment", path)
    require(spec is not None and spec.loader is not None, "load pinned YOLOX preprocessing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.preproc


def corrected_upstream_preprocessing_parity(yolox_preproc, source_bgr: np.ndarray) -> tuple[float, float]:
    """Compare the v2 RGB tensor with the pinned helper from an OpenCV BGR source."""
    require(source_bgr.ndim == 3 and source_bgr.shape[2] == 3 and source_bgr.dtype == np.uint8,
            "OpenCV BGR source")
    official, official_scale = yolox_preproc(
        source_bgr, (416, 416), np.asarray(IMAGENET_MEAN, dtype=np.float32),
        np.asarray(IMAGENET_STD, dtype=np.float32))
    raw_rgb, candidate_scale = detector_letterbox_rgb(np.ascontiguousarray(source_bgr[:, :, ::-1]))
    candidate = detector_training_tensor(torch.from_numpy(raw_rgb)[None],
                                         V2_DETECTOR_PREPROCESSING)[0].numpy()
    return float(np.max(np.abs(candidate - official))), abs(candidate_scale - float(official_scale))


def validate_preprocessing(args: argparse.Namespace, config: dict[str, Any], recipes: list[dict[str, Any]],
                           page_split: dict[int, str]) -> dict[str, Any]:
    """Run input-contract checks that must pass without allocating a GPU."""
    classifier_data = ClassifierBoards(args.dataset, recipes, page_split, "train")
    detector_data = DetectorPages(args.dataset, recipes, page_split, "train", detector_preprocessing(config))
    import synthetic_training as reference

    # Exercise both image-relative orientations. The same tile order is required
    # regardless of which player's pieces appear at the bottom of the source.
    classifier_indices = []
    for orientation in ("white-bottom", "black-bottom"):
        classifier_indices.append(next(index for index, (page_index, board_index) in enumerate(classifier_data.boards)
                                       if recipes[page_index]["boards"][board_index]["orientation"] == orientation))
    classifier_max_abs = 0.0
    for index in classifier_indices:
        page_index, board_index = classifier_data.boards[index]
        record = dataset_record(args.dataset, page_index)
        candidate_tiles, candidate_labels, _ = classifier_data.get(index)
        expected = reference.load_board(args.dataset / "images" / f"page-{page_index:06d}.png", record, board_index)
        expected_tensor = torch.from_numpy(np.frombuffer(expected["tensor"], dtype="<f4").reshape(expected["shape"]))
        classifier_max_abs = max(classifier_max_abs, float((candidate_tiles - expected_tensor).abs().max()))
        require(candidate_labels.tolist() == expected["labels"], "classifier label/order parity")
    require(classifier_max_abs <= 1e-6, "classifier preprocessing parity")

    # Select one page for every frozen source-size/kind combination, bounded to
    # keep this an operator-friendly CPU gate.
    detector_indices: list[int] = []
    seen: set[tuple[int, int, str]] = set()
    for positive in (True, False):
        index = next(index for index, page_index in enumerate(detector_data.pages)
                     if bool(recipes[page_index].get("boards")) is positive)
        page = recipes[detector_data.pages[index]]
        seen.add((int(page["width"]), int(page["height"]), str(page["kind"])))
        detector_indices.append(index)
    for index, page_index in enumerate(detector_data.pages):
        page = recipes[page_index]
        key = (int(page["width"]), int(page["height"]), str(page["kind"]))
        if key not in seen:
            seen.add(key)
            detector_indices.append(index)
        if len(detector_indices) >= 32:
            break
    detector_max_abs = 0.0
    detector_image_max_abs = 0.0
    yolox_preproc = pinned_yolox_preproc(args.native)
    preprocessing = detector_preprocessing(config)
    for index in detector_indices:
        candidate_image, candidate_targets, metadata = detector_data.get(index)
        page_index = metadata["page"]
        record = dataset_record(args.dataset, page_index)
        with Image.open(args.dataset / "images" / f"page-{page_index:06d}.png") as source:
            source.load()
            source_rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
        if preprocessing == V2_DETECTOR_PREPROCESSING:
            image_difference, scale_difference = corrected_upstream_preprocessing_parity(
                yolox_preproc, np.ascontiguousarray(source_rgb[:, :, ::-1]))
            expected_image = detector_training_tensor(candidate_image[None], preprocessing)[0]
        else:
            official_image, official_scale = yolox_preproc(source_rgb, (416, 416), None, None)
            expected_image = torch.from_numpy(np.ascontiguousarray(official_image * 255.0))
            image_difference = float((candidate_image - expected_image).abs().max())
            scale_difference = abs(float(metadata["scale"]) - float(official_scale))
        normalized = reference.detector_targets(record)
        expected_targets = letterbox_targets(normalized, record["recipe"]["width"], record["recipe"]["height"])
        detector_max_abs = max(detector_max_abs, float((candidate_targets - expected_targets).abs().max()))
        detector_image_max_abs = max(detector_image_max_abs, image_difference)
        require(len(metadata["boxes"]) == len(normalized), "detector target count parity")
        require(scale_difference < 1e-12,
                "detector image scale parity")
        require(candidate_image.shape == (3, 416, 416) and torch.isfinite(candidate_image).all(),
                "detector image preprocessing")
        require(float(candidate_image.min()) >= 0 and float(candidate_image.max()) <= 255,
                "detector image range")
    require(detector_max_abs / 416 < 1e-6, "detector target preprocessing parity")
    tolerance = 1e-6 if preprocessing == V2_DETECTOR_PREPROCESSING else 2e-5
    require(detector_image_max_abs <= tolerance, "detector image preprocessing parity")

    # A fixed non-square OpenCV BGR raster catches accidental RGB/BGR symmetry
    # even if the selected dataset pages happen to be nearly grayscale.
    sentinel_bgr = np.zeros((173, 311, 3), dtype=np.uint8)
    sentinel_bgr[:, :103] = (251, 17, 43)
    sentinel_bgr[:, 103:207] = (7, 229, 83)
    sentinel_bgr[:, 207:] = (61, 101, 241)
    sentinel_max_abs, sentinel_scale_abs = corrected_upstream_preprocessing_parity(
        yolox_preproc, sentinel_bgr)
    require(sentinel_max_abs <= 1e-6 and sentinel_scale_abs < 1e-12,
            "detector colored-sentinel upstream preprocessing parity")

    # The first optimizer update must not make the YOLOX loss non-finite. This
    # CPU-only smoke catches raw-pixel/normalization mistakes before GPU spend.
    smoke_indices = detector_indices[:min(4, len(detector_indices))]
    smoke_model = detector_model(args.native, torch.device("cpu"))
    smoke_stage = config["detector"]["stages"][0]
    smoke_inputs, smoke_targets, _ = batch_detector(detector_data, smoke_indices, torch.device("cpu"))
    smoke_optimizer = detector_optimizer(smoke_model, config, smoke_stage)
    smoke_losses = []
    for _ in range(3):
        smoke_optimizer.zero_grad(set_to_none=True)
        smoke_loss = smoke_model(smoke_inputs, smoke_targets)["total_loss"]
        require(torch.isfinite(smoke_loss), "detector CPU smoke loss")
        smoke_loss.backward()
        require(all(torch.isfinite(parameter.grad).all() for parameter in smoke_model.parameters()
                    if parameter.grad is not None), "detector CPU smoke gradients")
        torch.nn.utils.clip_grad_norm_([parameter for parameter in smoke_model.parameters() if parameter.requires_grad],
                                       config["detector"]["gradient_clip_norm"])
        smoke_optimizer.step()
        smoke_losses.append(float(smoke_loss.detach()))
    report = {"state": "passed", "finished_at": time.time(),
              "numpy": np.__version__, "pillow": PIL.__version__,
              "classifier_examples": len(classifier_indices),
              "classifier_orientations": ["white-bottom", "black-bottom"],
              "classifier_preprocessing_max_abs": classifier_max_abs,
              "detector_examples": len(detector_indices),
              "detector_source_size_kinds": [list(value) for value in sorted(seen)[:len(detector_indices)]],
              "detector_target_max_abs": detector_max_abs,
              "detector_image_max_abs": detector_image_max_abs,
              "detector_colored_sentinel_max_abs": sentinel_max_abs,
              "detector_cpu_smoke_losses": smoke_losses,
              "detector_training_input_range": [float(smoke_inputs.min()), float(smoke_inputs.max())]}
    write_json(args.run / "validation" / "report.json", report)
    resource_guard(args.run, config, time.process_time())
    return report


def preflight(args: argparse.Namespace, config: dict[str, Any], recipes: list[dict[str, Any]],
              page_split: dict[int, str], device: torch.device) -> None:
    report: dict[str, Any] = {"started_at": time.time(), "device": torch.cuda.get_device_name(0),
                              "capability": list(torch.cuda.get_device_capability(0)), "torch": torch.__version__,
                              "cuda": torch.version.cuda, "cudnn_enabled": torch.backends.cudnn.enabled}
    preprocessing = validate_preprocessing(args, config, recipes, page_split)
    classifier_data = ClassifierBoards(args.dataset, recipes, page_split, "train")
    detector_data = DetectorPages(args.dataset, recipes, page_split, "train", detector_preprocessing(config))
    from types import SimpleNamespace
    from native_gpu_reference import run_mobilenet, run_yolox
    native_args = SimpleNamespace(weights=args.native / "cache/native", inputs=args.native / "work/native",
                                  references=args.native / "artifacts/native", yolox_source=args.native / "cache/native/yolox",
                                  device="cuda", enable_cudnn=False)
    parity = [run_mobilenet(native_args), run_yolox(native_args)]
    require(all(result["finite"] and result["max_abs_from_cpu_reference"] <= 1e-3 for result in parity), "native GPU parity")

    classifier = classifier_model(args.native, device)
    classifier_stage = config["classifier"]["stages"][0]
    c_inputs, c_labels = batch_classifier(classifier_data, range(8), device)
    c_before = classifier.classifier.weight.detach().clone()
    c_optimizer = classifier_optimizer(classifier, classifier_stage)
    c_optimizer.zero_grad(set_to_none=True)
    c_loss = F.cross_entropy(classifier(c_inputs), c_labels)
    c_loss.backward()
    require(torch.isfinite(c_loss) and all(torch.isfinite(p.grad).all() for p in classifier.parameters() if p.grad is not None), "classifier preflight gradients")
    c_optimizer.step()
    require(not torch.equal(c_before, classifier.classifier.weight), "classifier parameters did not update")
    # A fixed known-label board must be memorized by the new head; this is a mechanics gate.
    tiny_inputs, tiny_labels = c_inputs[:64], c_labels[:64]
    tiny_initial = float(F.cross_entropy(classifier(tiny_inputs), tiny_labels).detach())
    for _ in range(100):
        c_optimizer.zero_grad(set_to_none=True)
        tiny_loss = F.cross_entropy(classifier(tiny_inputs), tiny_labels)
        tiny_loss.backward(); c_optimizer.step()
    tiny_accuracy = float((classifier(tiny_inputs).argmax(1) == tiny_labels).float().mean())
    require(tiny_accuracy >= 0.98 and float(tiny_loss.detach()) <= tiny_initial * 0.2, "classifier tiny known-label fit")

    detector = detector_model(args.native, device)
    detector_stage = config["detector"]["stages"][0]
    positive = [index for index, page_number in enumerate(detector_data.pages) if recipes[page_number].get("boards")][:2]
    negative = [index for index, page_number in enumerate(detector_data.pages) if not recipes[page_number].get("boards")][:2]
    require(len(positive) == 2 and len(negative) == 2, "detector positive/negative preflight coverage")
    d_inputs, d_targets, metadata = batch_detector(detector_data, positive + negative, device)
    require(any(meta["negative"] for meta in metadata) and any(not meta["negative"] for meta in metadata), "detector mixed preflight batch")
    d_optimizer = detector_optimizer(detector, config, detector_stage)
    d_optimizer.zero_grad(set_to_none=True)
    d_loss = detector(d_inputs, d_targets)["total_loss"]
    d_loss.backward()
    require(torch.isfinite(d_loss) and all(torch.isfinite(p.grad).all() for p in detector.parameters() if p.grad is not None), "detector preflight gradients")
    torch.nn.utils.clip_grad_norm_([p for p in detector.parameters() if p.requires_grad], config["detector"]["gradient_clip_norm"])
    d_optimizer.step()
    negative_inputs, negative_targets, _ = batch_detector(detector_data, negative, device)
    negative_loss = detector(negative_inputs, negative_targets)["total_loss"]
    require(torch.isfinite(negative_loss), "detector negative-only loss")
    tiny_detector_initial = float(detector(d_inputs, d_targets)["total_loss"].detach())
    for _ in range(30):
        d_optimizer.zero_grad(set_to_none=True)
        tiny_detector_loss = detector(d_inputs, d_targets)["total_loss"]
        tiny_detector_loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in detector.parameters() if p.requires_grad], config["detector"]["gradient_clip_norm"])
        d_optimizer.step()
    require(float(tiny_detector_loss.detach()) <= tiny_detector_initial * 0.8, "detector tiny-set fit")

    # Time every scheduled stage with its exact trainable set, optimizer and batch.
    classifier_timings = []
    c_inputs, c_labels = batch_classifier(classifier_data, range(8, 16), device)
    for stage in config["classifier"]["stages"]:
        timed_optimizer = classifier_optimizer(classifier, stage)
        for group in timed_optimizer.param_groups:
            group["lr"] = cosine_lr(group["base_lr"], 0, stage["updates"], stage["warmup_updates"])
        torch.cuda.synchronize(); started = time.monotonic()
        timed_optimizer.zero_grad(set_to_none=True)
        timed_c_loss = F.cross_entropy(classifier(c_inputs), c_labels)
        timed_c_loss.backward(); timed_optimizer.step(); torch.cuda.synchronize()
        classifier_timings.append({"stage": stage["name"], "updates": stage["updates"],
                                   "step_seconds": time.monotonic() - started})
    classifier_projection = sum(row["step_seconds"] * row["updates"] for row in classifier_timings) * 1.5

    detector_timings = []
    detector_indices = CyclingOrder(len(detector_data), config["seed"] + 700).take(config["detector"]["batch_size"])
    d_inputs, d_targets, _ = batch_detector(detector_data, detector_indices, device)
    for stage in config["detector"]["stages"]:
        timed_optimizer = detector_optimizer(detector, config, stage)
        timed_optimizer.param_groups[0]["lr"] = cosine_lr(
            stage["learning_rate"], 0, stage["updates"], stage["warmup_updates"])
        timed_parameters = [parameter for parameter in detector.parameters() if parameter.requires_grad]
        torch.cuda.synchronize(); started = time.monotonic()
        timed_optimizer.zero_grad(set_to_none=True)
        timed_d_loss = detector(d_inputs, d_targets)["total_loss"]
        timed_d_loss.backward()
        torch.nn.utils.clip_grad_norm_(timed_parameters, config["detector"]["gradient_clip_norm"])
        timed_optimizer.step(); torch.cuda.synchronize()
        detector_timings.append({"stage": stage["name"], "updates": stage["updates"],
                                 "step_seconds": time.monotonic() - started})
    detector_projection = sum(row["step_seconds"] * row["updates"] for row in detector_timings) * 1.5
    if config["resources"]["classifier_gpu_seconds"]:
        require(classifier_projection <= config["resources"]["classifier_gpu_seconds"],
                "classifier schedule does not fit allocation")
    require(detector_projection <= config["resources"]["detector_gpu_seconds"], "detector schedule does not fit allocation")
    # Actual stochastic-path recovery: saved sampler/RNG/model/optimizer state
    # must reproduce the next update in every frozen stage on this GPU.
    recovery_dir = args.run / "preflight"
    classifier_recovery = []
    for stage_index, stage in enumerate(config["classifier"]["stages"]):
        seed = config["seed"] + 999 + stage_index
        sampler = CyclingOrder(len(classifier_data), seed)
        optimizer = classifier_optimizer(classifier, stage)
        path = recovery_dir / f"classifier-recovery-stage-{stage_index}.pt"
        global_step = sum(item["updates"] for item in config["classifier"]["stages"][:stage_index])
        checkpoint(path, classifier, optimizer, sampler, stage_index, 0, global_step)
        indices = sampler.take(2)
        inputs, labels = batch_classifier(classifier_data, indices, device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(classifier(inputs), labels); loss.backward()
        for group in optimizer.param_groups:
            group["lr"] = cosine_lr(group["base_lr"], 0, stage["updates"], stage["warmup_updates"])
        optimizer.step()
        expected = {key: value.detach().clone() for key, value in classifier.state_dict().items()}
        saved = torch.load(path, map_location="cpu", weights_only=True)
        validate_checkpoint_state(saved, config["classifier"]["stages"], "classifier")
        classifier.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
        sampler = CyclingOrder(len(classifier_data), seed, saved["sampler"]); restore_rng(saved["rng"])
        require(indices == sampler.take(2), "classifier sampler recovery mismatch")
        inputs, labels = batch_classifier(classifier_data, indices, device)
        optimizer.zero_grad(set_to_none=True)
        replay_loss = F.cross_entropy(classifier(inputs), labels); replay_loss.backward()
        for group in optimizer.param_groups:
            group["lr"] = cosine_lr(group["base_lr"], 0, stage["updates"], stage["warmup_updates"])
        optimizer.step()
        maximum = max(float((expected[key] - value).abs().max())
                      for key, value in classifier.state_dict().items())
        require(maximum == 0, f"classifier stage {stage_index} recovery mismatch: {maximum}")
        classifier_recovery.append({"stage": stage["name"], "max_abs": maximum})

    detector_recovery = []
    for stage_index, stage in enumerate(config["detector"]["stages"]):
        seed = config["seed"] + 1999 + stage_index
        detector_sampler = CyclingOrder(len(detector_data), seed)
        detector_optimizer_state = detector_optimizer(detector, config, stage)
        detector_ema = EMA(detector, config["detector"]["ema_decay"],
                           config["detector"].get("ema_warmup_updates"))
        path = recovery_dir / f"detector-recovery-stage-{stage_index}.pt"
        global_step = sum(item["updates"] for item in config["detector"]["stages"][:stage_index])
        checkpoint(path, detector, detector_optimizer_state, detector_sampler,
                   stage_index, 0, global_step, {"ema": detector_ema.checkpoint_state()})
        recovery_indices = detector_sampler.take(config["detector"]["batch_size"])
        inputs, targets, _ = batch_detector(detector_data, recovery_indices, device)
        detector_optimizer_state.zero_grad(set_to_none=True)
        recovery_loss = detector(inputs, targets)["total_loss"]
        recovery_loss.backward()
        recovery_parameters = [parameter for parameter in detector.parameters() if parameter.requires_grad]
        torch.nn.utils.clip_grad_norm_(recovery_parameters, config["detector"]["gradient_clip_norm"])
        detector_optimizer_state.param_groups[0]["lr"] = cosine_lr(
            stage["learning_rate"], 0, stage["updates"], stage["warmup_updates"])
        detector_optimizer_state.step(); detector_ema.update(detector)
        expected_detector = {key: value.detach().clone() for key, value in detector.state_dict().items()}
        expected_ema = {key: value.detach().clone() for key, value in detector_ema.state.items()}
        saved = torch.load(path, map_location="cpu", weights_only=True)
        validate_checkpoint_state(saved, config["detector"]["stages"], "detector")
        detector.load_state_dict(saved["model"]); detector_optimizer_state.load_state_dict(saved["optimizer"])
        detector_sampler = CyclingOrder(len(detector_data), seed, saved["sampler"]); restore_rng(saved["rng"])
        detector_ema.load_checkpoint_state(saved["extra"]["ema"], device)
        require(recovery_indices == detector_sampler.take(config["detector"]["batch_size"]),
                "detector sampler recovery mismatch")
        inputs, targets, _ = batch_detector(detector_data, recovery_indices, device)
        detector_optimizer_state.zero_grad(set_to_none=True)
        replay_detector_loss = detector(inputs, targets)["total_loss"]
        replay_detector_loss.backward()
        recovery_parameters = [parameter for parameter in detector.parameters() if parameter.requires_grad]
        torch.nn.utils.clip_grad_norm_(recovery_parameters, config["detector"]["gradient_clip_norm"])
        detector_optimizer_state.param_groups[0]["lr"] = cosine_lr(
            stage["learning_rate"], 0, stage["updates"], stage["warmup_updates"])
        detector_optimizer_state.step(); detector_ema.update(detector)
        detector_maximum = max(float((expected_detector[key] - value).abs().max())
                               for key, value in detector.state_dict().items())
        ema_maximum = max(float((expected_ema[key] - value).abs().max())
                          for key, value in detector_ema.state.items())
        require(detector_maximum == 0 and ema_maximum == 0,
                f"detector stage {stage_index} recovery mismatch: {detector_maximum}/{ema_maximum}")
        detector_recovery.append({"stage": stage["name"], "model_max_abs": detector_maximum,
                                  "ema_max_abs": ema_maximum})
    report.update(native_parity=parity, preprocessing=preprocessing,
                  classifier_loss=float(c_loss.detach()), classifier_tiny_accuracy=tiny_accuracy,
                  classifier_tiny_loss_ratio=float(tiny_loss.detach()) / tiny_initial,
                  detector_loss=float(d_loss.detach()), detector_negative_loss=float(negative_loss.detach()),
                  detector_tiny_loss_ratio=float(tiny_detector_loss.detach()) / tiny_detector_initial,
                  classifier_stage_timings=classifier_timings, classifier_projected_seconds=classifier_projection,
                  detector_stage_timings=detector_timings, detector_projected_seconds=detector_projection,
                  classifier_recovery=classifier_recovery, detector_recovery=detector_recovery,
                  finished_at=time.time(), state="passed")
    write_json(recovery_dir / "report.json", report)
    resource_guard(args.run, config, time.process_time())


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("segment", choices=("validate", "preflight", "classifier-export", "detector-export",
                                           "detector-audit", "classifier", "detector"))
    value.add_argument("--recipe", type=Path, required=True)
    value.add_argument("--dataset", type=Path, required=True)
    value.add_argument("--native", type=Path, required=True)
    value.add_argument("--run", type=Path, required=True)
    value.add_argument("--split", type=Path, required=True)
    value.add_argument("--classifier-checkpoint", type=Path)
    value.add_argument("--audit-output", type=Path)
    return value


def main() -> None:
    args = parser().parse_args()
    started_wall, started_cpu = time.monotonic(), time.process_time()
    try:
        config, recipes, page_split = load_inputs(args)
        if args.segment == "validate":
            validate_preprocessing(args, config, recipes, page_split)
            return
        if args.segment == "classifier-export":
            export_classifier_checkpoint(args, config, recipes, page_split)
            return
        if args.segment == "detector-export":
            export_detector_checkpoint(args, config)
            return
        if args.segment == "detector-audit":
            audit_detector(args, config, recipes, page_split)
            return
        device = configure(config["seed"])
        if args.segment == "preflight":
            preflight(args, config, recipes, page_split, device)
        elif args.segment == "classifier":
            train_classifier(args, config, recipes, page_split, device)
        else:
            train_detector(args, config, recipes, page_split, device)
    except Stopped as error:
        print(json.dumps({"state": "stopped", "reason": str(error)}))
        raise SystemExit(75)
    finally:
        if args.run.is_dir():
            resource_root = (args.audit_output.parent if args.segment == "detector-audit" and
                             args.audit_output is not None else args.run)
            write_json(resource_root / f"resource-{args.segment}-{os.getpid()}.json", {
                "segment": args.segment, "pid": os.getpid(), "wall_seconds": time.monotonic() - started_wall,
                "cpu_seconds": time.process_time() - started_cpu, "finished_at": time.time()})


if __name__ == "__main__":
    main()
