"""Hash-bound local ONNX candidate used only for interactive, human-reviewed proposals."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

from PIL import Image


SCHEMA = "chess-ocr-candidate-bundle/1"
LABELS = ["empty", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]


class Invalid(ValueError):
    pass


def require(value, message):
    if not value:
        raise Invalid(message)


def safe_file(path: Path, root: Path, maximum: int) -> Path:
    path = path.absolute()
    require(path.is_file() and not path.is_symlink(), "candidate file is missing or unsafe")
    require(path.resolve().is_relative_to(root.resolve()), "candidate file must remain under work/")
    for parent in (path, *path.parents):
        if parent == root.parent:
            break
        require(not parent.is_symlink(), "symlink in candidate path")
    require(0 < path.stat().st_size <= maximum, "candidate file is outside its size bound")
    return path


class LocalCandidate:
    def __init__(self, manifest_path: Path, classifier_path: Path, detector_path: Path,
                 overlay_root: Path, workspace: Path):
        work = workspace / "work"
        manifest_path = safe_file(manifest_path, work, 64 * 1024)
        classifier_path = safe_file(classifier_path, work, 32 * 1024 * 1024)
        detector_path = safe_file(detector_path, work, 64 * 1024 * 1024)
        overlay_root = overlay_root.absolute()
        require(overlay_root.is_dir() and not overlay_root.is_symlink()
                and overlay_root.resolve().is_relative_to(work.resolve()),
                "candidate dependency overlay must remain under work/")
        try:
            manifest = json.loads(manifest_path.read_text())
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise Invalid("candidate manifest is not valid UTF-8 JSON") from None
        self._validate_manifest(manifest)
        import hashlib
        def digest(path):
            value = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    value.update(chunk)
            return value.hexdigest()
        require(classifier_path.stat().st_size == manifest["classifier"]["bytes"]
                and digest(classifier_path) == manifest["classifier"]["sha256"],
                "classifier does not match candidate manifest")
        require(detector_path.stat().st_size == manifest["detector"]["bytes"]
                and digest(detector_path) == manifest["detector"]["sha256"],
                "detector does not match candidate manifest")
        sys.path.insert(0, str(overlay_root))
        try:
            import numpy as np
            import onnxruntime as ort
        except ImportError:
            raise Invalid("candidate dependencies are unavailable in the selected overlay") from None
        self.np = np
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        self.classifier = ort.InferenceSession(str(classifier_path), sess_options=options,
                                               providers=["CPUExecutionProvider"])
        self.detector = ort.InferenceSession(str(detector_path), sess_options=options,
                                             providers=["CPUExecutionProvider"])
        self.manifest = manifest
        self._validate_session(self.classifier, manifest["classifier"], [None, 3, 96, 96], [None, 13])
        self._validate_session(self.detector, manifest["detector"], [1, 3, 416, 416], [1, 3549, 6])

    @staticmethod
    def _validate_manifest(value):
        require(isinstance(value, dict) and set(value) == {
            "schema", "name", "version", "qualification", "classifier", "detector"
        }, "invalid candidate manifest")
        require(value["schema"] == SCHEMA and value["qualification"] == "synthetic-development-only",
                "unsupported candidate manifest")
        require(all(isinstance(value[key], str) and 0 < len(value[key]) <= 120
                    for key in ("name", "version")), "invalid candidate identity")
        for role, ceiling in (("classifier", 32 * 1024 * 1024), ("detector", 64 * 1024 * 1024)):
            model = value[role]
            expected = {"sha256", "bytes", "input", "output"}
            if role == "classifier":
                expected.add("labels")
            else:
                expected.update(("scoreThreshold", "nmsIou"))
            require(isinstance(model, dict) and set(model) == expected, "invalid candidate model")
            require(isinstance(model["sha256"], str) and len(model["sha256"]) == 64
                    and all(char in "0123456789abcdef" for char in model["sha256"]),
                    "invalid candidate hash")
            require(type(model["bytes"]) is int and 0 < model["bytes"] <= ceiling,
                    "invalid candidate size")
            require(all(isinstance(model[key], str) and 0 < len(model[key]) <= 80
                        for key in ("input", "output")), "invalid candidate tensor name")
        require(value["classifier"]["labels"] == LABELS, "invalid classifier labels")
        require(type(value["detector"]["scoreThreshold"]) in (int, float)
                and .001 <= value["detector"]["scoreThreshold"] <= 1,
                "invalid detector score threshold")
        require(type(value["detector"]["nmsIou"]) in (int, float)
                and 0 <= value["detector"]["nmsIou"] <= 1,
                "invalid detector NMS threshold")

    @staticmethod
    def _validate_session(session, model, input_shape, output_shape):
        inputs, outputs = session.get_inputs(), session.get_outputs()
        require(len(inputs) == 1 and inputs[0].name == model["input"], "candidate input tensor mismatch")
        require(len(outputs) == 1 and outputs[0].name == model["output"], "candidate output tensor mismatch")
        for actual, expected in ((inputs[0].shape, input_shape), (outputs[0].shape, output_shape)):
            require(len(actual) == len(expected), "candidate tensor rank mismatch")
            require(all(want is None or got == want for got, want in zip(actual, expected)),
                    "candidate tensor shape mismatch")

    @property
    def public_identity(self):
        return {"name": self.manifest["name"], "version": self.manifest["version"],
                "qualification": self.manifest["qualification"]}

    def _detector_input(self, image):
        np = self.np
        import cv2
        width, height = image.size
        scale = min(416 / width, 416 / height)
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        resized = cv2.resize(rgb, (max(1, int(width * scale)), max(1, int(height * scale))),
                             interpolation=cv2.INTER_LINEAR)
        canvas = np.full((416, 416, 3), 114, dtype=np.float32)
        canvas[:resized.shape[0], :resized.shape[1]] = resized[:, :, ::-1]
        return np.ascontiguousarray(canvas.transpose(2, 0, 1)[None]), scale, resized.shape[1], resized.shape[0]

    def _decode_detector(self, raw, scale, resized_width, resized_height):
        np = self.np
        require(raw.shape == (1, 3549, 6) and np.isfinite(raw).all(), "invalid detector output")
        decoded = raw[0].copy()
        offset = 0
        for stride in (8, 16, 32):
            side = 416 // stride
            count = side * side
            yy, xx = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
            decoded[offset:offset + count, 0] = (decoded[offset:offset + count, 0] + xx.reshape(-1)) * stride
            decoded[offset:offset + count, 1] = (decoded[offset:offset + count, 1] + yy.reshape(-1)) * stride
            decoded[offset:offset + count, 2:4] = np.exp(decoded[offset:offset + count, 2:4]) * stride
            offset += count
        scores = decoded[:, 4] * decoded[:, 5]
        selected = np.flatnonzero(scores >= self.manifest["detector"]["scoreThreshold"])
        boxes = []
        for index in selected:
            cx, cy, width, height = map(float, decoded[index, :4])
            left, top = max(0., cx - width / 2), max(0., cy - height / 2)
            right, bottom = min(float(resized_width), cx + width / 2), min(float(resized_height), cy + height / 2)
            if right > left + 1 and bottom > top + 1:
                boxes.append(([left, top, right, bottom], float(scores[index])))
        boxes.sort(key=lambda item: item[1], reverse=True)
        kept = []
        for box, score in boxes:
            if all(self._iou(box, prior[0]) <= self.manifest["detector"]["nmsIou"] for prior in kept):
                kept.append((box, score))
            if len(kept) == 16:
                break
        return [[value / scale for value in box] for box, _ in kept]

    @staticmethod
    def _iou(left, right):
        overlap = max(0., min(left[2], right[2]) - max(left[0], right[0])) * max(
            0., min(left[3], right[3]) - max(left[1], right[1]))
        area_left = (left[2] - left[0]) * (left[3] - left[1])
        area_right = (right[2] - right[0]) * (right[3] - right[1])
        return overlap / max(1e-9, area_left + area_right - overlap)

    def _classify(self, image, boxes):
        np = self.np
        grids = []
        for left, top, right, bottom in boxes:
            grid = image.convert("RGB").transform(
                (768, 768), Image.Transform.EXTENT, (left, top, right, bottom),
                resample=Image.Resampling.BICUBIC)
            array = np.asarray(grid, dtype=np.float32) / 255.
            grids.append(array.reshape(8, 96, 8, 96, 3).transpose(0, 2, 4, 1, 3).reshape(64, 3, 96, 96))
        tiles = np.concatenate(grids)
        tiles = (tiles - np.array([.485, .456, .406], dtype=np.float32)[None, :, None, None]) / np.array(
            [.229, .224, .225], dtype=np.float32)[None, :, None, None]
        logits = self.classifier.run([self.manifest["classifier"]["output"]], {
            self.manifest["classifier"]["input"]: np.ascontiguousarray(tiles)
        })[0]
        require(logits.shape == (len(boxes) * 64, 13) and np.isfinite(logits).all(),
                "invalid classifier output")
        return logits.argmax(1).reshape(len(boxes), 64)

    def recognize(self, image, selection=None):
        np = self.np
        if selection is None:
            input_tensor, scale, resized_width, resized_height = self._detector_input(image)
            raw = self.detector.run([self.manifest["detector"]["output"]], {
                self.manifest["detector"]["input"]: input_tensor
            })[0]
            boxes = self._decode_detector(raw, scale, resized_width, resized_height)
        else:
            boxes = [selection]
        if not boxes:
            return {"boards": [], "model": self.public_identity,
                    "warning": "Synthetic candidate found no board; inspect the page manually."}
        predictions = self._classify(image, boxes)
        boards = []
        for box, labels in zip(boxes, predictions):
            left, top, right, bottom = box
            require(all(math.isfinite(value) for value in box), "nonfinite candidate geometry")
            boards.append({"corners": [[left, top], [right, top], [right, bottom], [left, bottom]],
                           "labels": ["." if LABELS[int(value)] == "empty" else LABELS[int(value)] for value in labels],
                           "orientation": "unknown"})
        return {"boards": boards, "model": self.public_identity,
                "warning": "Synthetic-only unqualified proposal; inspect every grid and square before human acceptance."}
