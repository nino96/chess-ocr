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
    require(config.get("schema") == "chess-ocr-training-recipe/1", "recipe schema")
    dependencies = config["environment"]["dependencies"]
    require(np.__version__ == dependencies["numpy"] and PIL.__version__ == dependencies["pillow"], "training NumPy/Pillow versions")
    require(importlib.metadata.version("onnx") == dependencies["onnx"], "training ONNX version")
    require(importlib.metadata.version("opencv-python-headless") == dependencies["opencv-python-headless"],
            "training OpenCV version")
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
    def __init__(self, root: Path, recipes: list[dict[str, Any]], page_split: dict[int, str], split: str):
        self.root = root
        self.recipes = recipes
        self.pages = [index for index, page in enumerate(recipes)
                      if page_split[index] == split and page.get("kind") in {"boards", "negative"}]
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
        scale = min(416 / width, 416 / height)
        rgb = np.asarray(image, dtype=np.uint8)
        resized = cv2.resize(rgb, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        canvas = np.full((416, 416, 3), 114, dtype=np.float32)
        canvas[:resized.shape[0], :resized.shape[1]] = resized[:, :, ::-1]
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
        metadata = {"page": page_index, "scale": scale, "boxes": boxes, "negative": not boxes,
                    "effect": page["condition"]["degradation"]["variant"], "layout": page["layout"]}
        return torch.from_numpy(np.ascontiguousarray(canvas.transpose(2, 0, 1))), targets, metadata


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
    # DetectorPages retains raw BGR 0..255 for independent raster parity; the
    # pinned YOLOX training path consumes the same tensor divided by 255.
    inputs = detector_training_tensor(torch.stack([item[0] for item in items]).to(device))
    return inputs, torch.stack([item[1] for item in items]).to(device), [item[2] for item in items]


def detector_training_tensor(raw: torch.Tensor) -> torch.Tensor:
    require(torch.is_floating_point(raw), "detector raw tensor dtype")
    require(torch.isfinite(raw).all() and float(raw.min()) >= 0 and float(raw.max()) <= 255,
            "detector raw tensor range")
    return raw / 255.0


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
    orientation = {name: {"boards": 0, "exact_boards": 0, "squares": 0, "correct_squares": 0}
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
    return {"boards": len(dataset), "squares": total, "exact_boards": exact,
            "exact_board_accuracy": exact / len(dataset), "square_accuracy": int(confusion.diag().sum()) / total,
            "occupied_macro_f1": sum(occupied_f1) / len(occupied_f1), "nll": total_loss / total,
            "confident_wrong_squares_at_0_99": confident_wrong, "mean_confidence": sum(confidences) / len(confidences),
            "empty_occupied_errors": empty_errors, "color_errors": color_errors,
            "piece_class_errors": piece_class_errors, "confidence_coverage": coverage,
            "orientation": {name: {**values,
                "exact_board_accuracy": values["exact_boards"] / values["boards"],
                "square_accuracy": values["correct_squares"] / values["squares"]}
                for name, values in orientation.items()}, "confusion": confusion.tolist()}


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
        model.load_state_dict(resume_state["model"], strict=True)
        restore_rng(resume_state["rng"])
    global_step = int(resume_state["global_step"]) if resume_state else 0
    curves = read_json(directory / "curves.json") if (directory / "curves.json").exists() else []
    for stage_index, stage in enumerate(stages):
        if resume_state and stage_index < resume_state["stage"]:
            continue
        groups = set_trainable_classifier(model, stage["trainable"])
        parameter_groups = []
        for group in groups:
            base = stage.get("backbone_learning_rate", stage.get("learning_rate")) if group["role"] == "backbone" else stage.get("head_learning_rate", stage.get("learning_rate"))
            parameter_groups.append({"params": group["params"], "base_lr": base, "lr": base})
        optimizer = torch.optim.AdamW(parameter_groups, weight_decay=stage["weight_decay"])
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
            inputs, labels = batch_classifier(train, sampler.take(board_count), device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = F.cross_entropy(logits, labels)
            require(torch.isfinite(loss).item(), "non-finite classifier loss")
            loss.backward()
            require(all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None), "non-finite classifier gradients")
            for group in optimizer.param_groups:
                group["lr"] = cosine_lr(group["base_lr"], stage_step, stage["updates"], stage["warmup_updates"])
            optimizer.step()
            stage_step += 1
            global_step += 1
            if global_step % config["classifier"]["checkpoint_interval_updates"] == 0 or stage_step == stage["updates"]:
                metrics = evaluate_classifier(model, development, device)
                curves.append({"global_step": global_step, "stage": stage["name"], "stage_step": stage_step,
                               "training_loss": float(loss.detach()), "learning_rates": [g["lr"] for g in optimizer.param_groups],
                               "development": metrics})
                write_json(directory / "curves.json", curves)
                checkpoint(directory / f"checkpoint-{global_step:06d}.pt", model, optimizer, sampler,
                           stage_index, stage_step, global_step)
                resource_guard(args.run, config, time.process_time())
                write_json(directory / "progress.json", {"state": "running", "global_step": global_step,
                           "scheduled_updates": 10000, "stage": stage["name"], "development": metrics})
                if (args.run / "stop").exists():
                    raise Stopped("operator stop requested")
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
            current = order[0]
            chosen.append(current)
            if len(order) == 1:
                break
            ious = box_iou(boxes[current:current + 1], boxes[order[1:]])[0]
            order = order[1:][ious <= nms_iou]
        results.append(torch.cat((boxes[chosen], scores[chosen, None]), 1) if chosen else boxes.new_zeros((0, 5)))
    return results


@torch.inference_mode()
def evaluate_detector(model: nn.Module, dataset: DetectorPages, device: torch.device,
                      nms_iou: float) -> dict[str, Any]:
    model.eval()
    predictions = []
    total_targets = 0
    negative_pages = 0
    false_on_negative = 0
    matched_ious = []
    normalized_box_errors = []
    strata: dict[str, dict[str, dict[str, float]]] = {key: {} for key in ("effect", "layout", "board_count", "size")}
    def observe(kind: str, name: str, targets: int, matched: int, false_positives: int, iou_sum: float) -> None:
        row = strata[kind].setdefault(name, {"targets": 0, "matched_at_iou_0_5": 0, "false_positives": 0, "iou_sum": 0.0})
        row["targets"] += targets; row["matched_at_iou_0_5"] += matched
        row["false_positives"] += false_positives; row["iou_sum"] += iou_sum
    for start in range(0, len(dataset), 16):
        indices = range(start, min(len(dataset), start + 16))
        inputs, _, metadata = batch_detector(dataset, indices, device)
        output = model(inputs)
        decoded = decode_detector(output, 0.01, nms_iou)
        for result, meta in zip(decoded, metadata):
            truth = torch.tensor(meta["boxes"], device=device)
            total_targets += len(truth)
            negative_pages += int(meta["negative"])
            false_on_negative += len(result) if meta["negative"] else 0
            predictions.append((result.detach().cpu(), truth.detach().cpu()))
            matched = 0
            iou_sum = 0.0
            if len(result) and len(truth):
                ious = box_iou(result[:, :4], truth)
                best_prediction = ious.max(0).indices
                best_ious = ious.max(0).values
                matched = int((best_ious >= 0.5).sum())
                iou_sum = float(best_ious.sum())
                matched_ious.extend(best_ious.cpu().tolist())
                normalized_box_errors.extend(((result[best_prediction, :4] - truth).abs().mean(1) / 416).cpu().tolist())
                for target in truth:
                    relative = float(torch.sqrt((target[2] - target[0]) * (target[3] - target[1])) / 416)
                    size = "small" if relative < 0.4 else "medium" if relative < 0.7 else "large"
                    target_iou = float(box_iou(result[:, :4], target[None])[..., 0].max())
                    observe("size", size, 1, int(target_iou >= 0.5), 0, target_iou)
            false_positives = max(0, len(result) - matched)
            for kind in ("effect", "layout"):
                observe(kind, str(meta[kind]), len(truth), matched, false_positives, iou_sum)
            observe("board_count", str(len(truth)), len(truth), matched, false_positives, iou_sum)
    recalls = {}
    precisions = {}
    aps = []
    for threshold in [0.5 + i * 0.05 for i in range(10)]:
        scored = []
        positives = 0
        for predicted, truth in predictions:
            positives += len(truth)
            used = set()
            for row in predicted[predicted[:, 4].argsort(descending=True)]:
                if not len(truth):
                    scored.append((float(row[4]), 0))
                    continue
                ious = box_iou(row[None, :4], truth)[0]
                best = int(ious.argmax())
                good = float(ious[best]) >= threshold and best not in used
                if good:
                    used.add(best)
                scored.append((float(row[4]), int(good)))
        scored.sort(reverse=True)
        tp = 0
        precision_curve, recall_curve = [], []
        for rank, (_, good) in enumerate(scored, 1):
            tp += good
            precision_curve.append(tp / rank)
            recall_curve.append(tp / max(1, positives))
        ap = 0.0
        for recall_level in [i / 100 for i in range(101)]:
            ap += max((p for p, r in zip(precision_curve, recall_curve) if r >= recall_level), default=0) / 101
        aps.append(ap)
        recalls[str(threshold)] = recall_curve[-1] if recall_curve else 0
        precisions[str(threshold)] = precision_curve[-1] if precision_curve else 0
    normalized_strata = {kind: {name: {**row,
        "recall_at_iou_0_5": row["matched_at_iou_0_5"] / max(1, row["targets"]),
        "mean_best_iou": row["iou_sum"] / max(1, row["targets"])} for name, row in values.items()}
        for kind, values in strata.items()}
    return {"pages": len(dataset), "targets": total_targets, "negative_pages": negative_pages,
            "ap50_95": sum(aps) / len(aps), "recall": recalls, "precision": precisions,
            "false_detections_on_negative_pages_at_0_01": false_on_negative,
            "mean_best_target_iou": sum(matched_ious) / max(1, len(matched_ious)),
            "mean_normalized_box_error": sum(normalized_box_errors) / max(1, len(normalized_box_errors)),
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
            pages.append((result.cpu(), torch.tensor(meta["boxes"])))
    candidates = sorted({float(row[4]) for result, _ in pages for row in result}, reverse=True)
    candidates = candidates[::max(1, len(candidates) // 1000)] + [1.0]
    negative_pages = sum(not len(truth) for _, truth in pages)
    total_targets = sum(len(truth) for _, truth in pages)
    selected = {"threshold": 1.0, "recall": 0.0, "false_positives_per_negative_page": 0.0}
    for threshold in candidates:
        matched, negative_false = 0, 0
        for result, truth in pages:
            result = result[result[:, 4] >= threshold]
            if not len(truth):
                negative_false += len(result)
            elif len(result):
                matched += int((box_iou(result[:, :4], truth).max(0).values >= 0.5).sum())
        fp_rate = negative_false / max(1, negative_pages)
        recall = matched / max(1, total_targets)
        if fp_rate <= 0.05 and recall > selected["recall"]:
            selected = {"threshold": threshold, "recall": recall,
                        "false_positives_per_negative_page": fp_rate}
    selected["truth"] = "synthetic TRAIN-purpose calibration; not production calibration"
    return selected


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for key, value in model.state_dict().items():
            if value.is_floating_point():
                self.state[key].mul_(self.decay).add_(value.detach(), alpha=1 - self.decay)
            else:
                self.state[key].copy_(value)


def set_trainable_detector(model: nn.Module, stage: str) -> list[nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(stage == "full-model")
    if stage == "head":
        for parameter in model.head.parameters():
            parameter.requires_grad_(True)
    freeze_batch_norm(model)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def train_detector(args: argparse.Namespace, config: dict[str, Any], recipes: list[dict[str, Any]],
                   page_split: dict[int, str], device: torch.device) -> None:
    directory = args.run / "detector"
    directory.mkdir(parents=True, exist_ok=True)
    train = DetectorPages(args.dataset, recipes, page_split, "train")
    development = DetectorPages(args.dataset, recipes, page_split, "development")
    model = detector_model(args.native, device)
    stages = config["detector"]["stages"]
    resume = latest_checkpoint(directory)
    resume_state = torch.load(resume, map_location="cpu", weights_only=True) if resume else None
    ema = EMA(model, config["detector"]["ema_decay"])
    if resume_state:
        model.load_state_dict(resume_state["model"], strict=True)
        ema.state = {key: value.to(device) for key, value in resume_state["extra"]["ema"].items()}
        restore_rng(resume_state["rng"])
    global_step = int(resume_state["global_step"]) if resume_state else 0
    curves = read_json(directory / "curves.json") if (directory / "curves.json").exists() else []
    for stage_index, stage in enumerate(stages):
        if resume_state and stage_index < resume_state["stage"]:
            continue
        parameters = set_trainable_detector(model, stage["name"])
        optimizer = torch.optim.SGD(parameters, lr=stage["learning_rate"], momentum=config["detector"]["momentum"],
                                    nesterov=True, weight_decay=config["detector"]["weight_decay"])
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
            inputs, targets, _ = batch_detector(train, sampler.take(config["detector"]["batch_size"]), device)
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
            stage_step += 1
            global_step += 1
            if global_step % config["detector"]["checkpoint_interval_updates"] == 0 or stage_step == stage["updates"]:
                live = {key: value.detach().clone() for key, value in model.state_dict().items()}
                model.load_state_dict(ema.state, strict=True)
                metrics = evaluate_detector(model, development, device, config["detector"]["nms_iou"])
                model.load_state_dict(live, strict=True)
                curves.append({"global_step": global_step, "stage": stage["name"], "stage_step": stage_step,
                               "training_loss": float(loss.detach()), "losses": {key: float(value.detach()) if torch.is_tensor(value) else float(value) for key, value in losses.items()},
                               "learning_rate": optimizer.param_groups[0]["lr"], "development": metrics})
                write_json(directory / "curves.json", curves)
                checkpoint(directory / f"checkpoint-{global_step:06d}.pt", model, optimizer, sampler,
                           stage_index, stage_step, global_step, {"ema": ema.state})
                resource_guard(args.run, config, time.process_time())
                write_json(directory / "progress.json", {"state": "running", "global_step": global_step,
                           "scheduled_updates": 9000, "stage": stage["name"], "development": metrics})
                if (args.run / "stop").exists():
                    raise Stopped("operator stop requested")
            model.train()
            freeze_batch_norm(model)
        resume_state = None
    best = max(curves, key=lambda row: (row["development"]["ap50_95"],
                                       row["development"]["recall"]["0.5"],
                                       -row["development"]["mean_normalized_box_error"]))
    selected = torch.load(directory / f"checkpoint-{best['global_step']:06d}.pt", map_location="cpu", weights_only=True)
    model.load_state_dict({key: value.to(device) for key, value in selected["extra"]["ema"].items()}, strict=True)
    final = evaluate_detector(model, development, device, config["detector"]["nms_iou"])
    calibration = calibrate_detector(model, DetectorPages(args.dataset, recipes, page_split, "calibration"),
                                     device, config["detector"]["nms_iou"])
    export_detector(model, args.run, config, {"development": final, "calibration": calibration,
                                             "selected_global_step": best["global_step"]})
    write_json(directory / "progress.json", {"state": "complete", "global_step": global_step,
               "scheduled_updates": 9000, "selected_global_step": best["global_step"],
               "development": final, "calibration": calibration})
    resource_guard(args.run, config, time.process_time())


def export_classifier(model: nn.Module, run: Path, config: dict[str, Any], metrics: dict[str, Any]) -> None:
    model.eval().cpu()
    (run / "classifier").mkdir(parents=True, exist_ok=True)
    destination = run / "classifier" / "selected.onnx"
    torch.onnx.export(model, torch.zeros((1, 3, 96, 96)), destination, input_names=["tiles"],
                      output_names=["logits"], dynamic_axes={"tiles": {0: "squares"}, "logits": {0: "squares"}},
                      opset_version=17, dynamo=False)
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
    model.load_state_dict(saved["model"], strict=True)
    development = evaluate_classifier(model, ClassifierBoards(args.dataset, recipes, page_split, "development"),
                                      torch.device("cpu"))
    calibration = calibrate_classifier(model, ClassifierBoards(args.dataset, recipes, page_split, "calibration"),
                                       torch.device("cpu"))
    selected_step = int(saved.get("global_step", 0))
    export_classifier(model, args.run, config, {"development": development, "calibration": calibration,
                                                "selected_global_step": selected_step,
                                                "source_checkpoint_sha256": sha256(checkpoint_path)})
    write_json(args.run / "classifier" / "progress.json", {"state": "complete", "global_step": selected_step,
               "scheduled_updates": 10000, "selected_global_step": selected_step,
               "development": development, "calibration": calibration,
               "source_checkpoint": str(checkpoint_path), "source_checkpoint_sha256": sha256(checkpoint_path)})


def export_detector(model: nn.Module, run: Path, config: dict[str, Any], metrics: dict[str, Any]) -> None:
    model.eval().cpu()
    model.head.decode_in_inference = False
    destination = run / "detector" / "selected.onnx"
    torch.onnx.export(model, torch.zeros((1, 3, 416, 416)), destination, input_names=["images"],
                      output_names=["predictions"], opset_version=13, do_constant_folding=True, dynamo=False)
    write_json(destination.with_suffix(".manifest.json"), {"schema": "chess-ocr-model/1", "role": "inner-grid-detector",
               "sha256": sha256(destination), "labels": ["inner-grid"], "preprocessing": config["detector"]["input"],
               "nms_iou": config["detector"]["nms_iou"], "metrics": metrics, "publication": "not-authorized"})


def dataset_record(dataset: Path, page_index: int) -> dict[str, Any]:
    batch_start = page_index // 64 * 64
    return read_json(dataset / f"batch-{batch_start:06d}.json")[page_index - batch_start]


def pinned_yolox_preproc(native: Path):
    """Load the reviewed preprocessing file without YOLOX's optional COCO imports."""
    path = native / "cache/native/yolox/yolox/data/data_augment.py"
    require(path.is_file() and not path.is_symlink(), "pinned YOLOX preprocessing source")
    spec = importlib.util.spec_from_file_location("_chess_ocr_yolox_data_augment", path)
    require(spec is not None and spec.loader is not None, "load pinned YOLOX preprocessing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.preproc


def validate_preprocessing(args: argparse.Namespace, config: dict[str, Any], recipes: list[dict[str, Any]],
                           page_split: dict[int, str]) -> dict[str, Any]:
    """Run input-contract checks that must pass without allocating a GPU."""
    classifier_data = ClassifierBoards(args.dataset, recipes, page_split, "train")
    detector_data = DetectorPages(args.dataset, recipes, page_split, "train")
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
    for index in detector_indices:
        candidate_image, candidate_targets, metadata = detector_data.get(index)
        page_index = metadata["page"]
        record = dataset_record(args.dataset, page_index)
        with Image.open(args.dataset / "images" / f"page-{page_index:06d}.png") as source:
            source.load()
            source_rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
        official_image, official_scale = yolox_preproc(source_rgb, (416, 416), None, None)
        expected_image = torch.from_numpy(np.ascontiguousarray(official_image * 255.0))
        normalized = reference.detector_targets(record)
        expected_targets = letterbox_targets(normalized, record["recipe"]["width"], record["recipe"]["height"])
        detector_max_abs = max(detector_max_abs, float((candidate_targets - expected_targets).abs().max()))
        detector_image_max_abs = max(detector_image_max_abs, float((candidate_image - expected_image).abs().max()))
        require(len(metadata["boxes"]) == len(normalized), "detector target count parity")
        require(abs(float(metadata["scale"]) - float(official_scale)) < 1e-12,
                "detector image scale parity")
        require(candidate_image.shape == (3, 416, 416) and torch.isfinite(candidate_image).all(),
                "detector image preprocessing")
        require(float(candidate_image.min()) >= 0 and float(candidate_image.max()) <= 255,
                "detector image range")
    require(detector_max_abs / 416 < 1e-6, "detector target preprocessing parity")
    require(detector_image_max_abs <= 2e-5, "detector image preprocessing parity")

    # The first optimizer update must not make the YOLOX loss non-finite. This
    # CPU-only smoke catches raw-pixel/normalization mistakes before GPU spend.
    smoke_indices = detector_indices[:min(4, len(detector_indices))]
    smoke_model = detector_model(args.native, torch.device("cpu"))
    set_trainable_detector(smoke_model, "head")
    smoke_inputs, smoke_targets, _ = batch_detector(detector_data, smoke_indices, torch.device("cpu"))
    smoke_optimizer = torch.optim.SGD([parameter for parameter in smoke_model.parameters() if parameter.requires_grad],
                                      lr=5e-3, momentum=.9)
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
    detector_data = DetectorPages(args.dataset, recipes, page_split, "train")
    from types import SimpleNamespace
    from native_gpu_reference import run_mobilenet, run_yolox
    native_args = SimpleNamespace(weights=args.native / "cache/native", inputs=args.native / "work/native",
                                  references=args.native / "artifacts/native", yolox_source=args.native / "cache/native/yolox",
                                  device="cuda", enable_cudnn=False)
    parity = [run_mobilenet(native_args), run_yolox(native_args)]
    require(all(result["finite"] and result["max_abs_from_cpu_reference"] <= 1e-3 for result in parity), "native GPU parity")

    classifier = classifier_model(args.native, device)
    set_trainable_classifier(classifier, ["conv_head", "classifier"])
    c_inputs, c_labels = batch_classifier(classifier_data, range(8), device)
    c_before = classifier.classifier.weight.detach().clone()
    c_optimizer = torch.optim.AdamW([p for p in classifier.parameters() if p.requires_grad], lr=1e-3)
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
    set_trainable_detector(detector, "head")
    positive = [index for index, page_number in enumerate(detector_data.pages) if recipes[page_number].get("boards")][:2]
    negative = [index for index, page_number in enumerate(detector_data.pages) if not recipes[page_number].get("boards")][:2]
    require(len(positive) == 2 and len(negative) == 2, "detector positive/negative preflight coverage")
    d_inputs, d_targets, metadata = batch_detector(detector_data, positive + negative, device)
    require(any(meta["negative"] for meta in metadata) and any(not meta["negative"] for meta in metadata), "detector mixed preflight batch")
    d_optimizer = torch.optim.SGD([p for p in detector.parameters() if p.requires_grad], lr=5e-3, momentum=.9)
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

    # Warm once, then project the exact frozen update counts from planned batch sizes.
    c_inputs, c_labels = batch_classifier(classifier_data, range(8, 16), device)
    torch.cuda.synchronize(); started = time.monotonic()
    c_optimizer.zero_grad(set_to_none=True); timed_c_loss = F.cross_entropy(classifier(c_inputs), c_labels)
    timed_c_loss.backward(); c_optimizer.step(); torch.cuda.synchronize()
    classifier_step_seconds = time.monotonic() - started
    detector_indices = CyclingOrder(len(detector_data), config["seed"] + 700).take(config["detector"]["batch_size"])
    d_inputs, d_targets, _ = batch_detector(detector_data, detector_indices, device)
    torch.cuda.synchronize(); started = time.monotonic()
    d_optimizer.zero_grad(set_to_none=True); timed_d_loss = detector(d_inputs, d_targets)["total_loss"]
    timed_d_loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in detector.parameters() if p.requires_grad], config["detector"]["gradient_clip_norm"])
    d_optimizer.step(); torch.cuda.synchronize()
    detector_step_seconds = time.monotonic() - started
    classifier_projection = classifier_step_seconds * 10000 * 1.5
    detector_projection = detector_step_seconds * 9000 * 1.5
    require(classifier_projection <= config["resources"]["classifier_gpu_seconds"], "classifier schedule does not fit allocation")
    require(detector_projection <= config["resources"]["detector_gpu_seconds"], "detector schedule does not fit allocation")
    # Actual stochastic-path recovery: the saved sampler/RNG/model/optimizer state
    # must reproduce the next classifier update exactly on this GPU configuration.
    recovery_dir = args.run / "preflight"
    sampler = CyclingOrder(len(classifier_data), config["seed"] + 999)
    optimizer = torch.optim.AdamW([p for p in classifier.parameters() if p.requires_grad], lr=1e-3)
    checkpoint(recovery_dir / "recovery.pt", classifier, optimizer, sampler, 0, 0, 0)
    indices = sampler.take(2)
    inputs, labels = batch_classifier(classifier_data, indices, device)
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(classifier(inputs), labels); loss.backward(); optimizer.step()
    expected = {key: value.detach().clone() for key, value in classifier.state_dict().items()}
    saved = torch.load(recovery_dir / "recovery.pt", map_location="cpu", weights_only=True)
    classifier.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
    sampler = CyclingOrder(len(classifier_data), config["seed"] + 999, saved["sampler"]); restore_rng(saved["rng"])
    require(indices == sampler.take(2), "sampler recovery mismatch")
    inputs, labels = batch_classifier(classifier_data, indices, device)
    optimizer.zero_grad(set_to_none=True)
    replay_loss = F.cross_entropy(classifier(inputs), labels); replay_loss.backward(); optimizer.step()
    maximum = max(float((expected[key] - value).abs().max()) for key, value in classifier.state_dict().items())
    require(maximum == 0, f"stochastic recovery mismatch: {maximum}")
    report.update(native_parity=parity, preprocessing=preprocessing,
                  classifier_loss=float(c_loss.detach()), classifier_tiny_accuracy=tiny_accuracy,
                  classifier_tiny_loss_ratio=float(tiny_loss.detach()) / tiny_initial,
                  detector_loss=float(d_loss.detach()), detector_negative_loss=float(negative_loss.detach()),
                  detector_tiny_loss_ratio=float(tiny_detector_loss.detach()) / tiny_detector_initial,
                  classifier_step_seconds=classifier_step_seconds, classifier_projected_seconds=classifier_projection,
                  detector_step_seconds=detector_step_seconds, detector_projected_seconds=detector_projection,
                  recovery_max_abs=maximum, finished_at=time.time(), state="passed")
    write_json(recovery_dir / "report.json", report)
    resource_guard(args.run, config, time.process_time())


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("segment", choices=("validate", "preflight", "classifier-export", "classifier", "detector"))
    value.add_argument("--recipe", type=Path, required=True)
    value.add_argument("--dataset", type=Path, required=True)
    value.add_argument("--native", type=Path, required=True)
    value.add_argument("--run", type=Path, required=True)
    value.add_argument("--split", type=Path, required=True)
    value.add_argument("--classifier-checkpoint", type=Path)
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
            write_json(args.run / f"resource-{args.segment}-{os.getpid()}.json", {
                "segment": args.segment, "pid": os.getpid(), "wall_seconds": time.monotonic() - started_wall,
                "cpu_seconds": time.process_time() - started_cpu, "finished_at": time.time()})


if __name__ == "__main__":
    main()
