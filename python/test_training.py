import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import torch


PATH = Path(__file__).with_name("training.py")
SPEC = importlib.util.spec_from_file_location("training", PATH)
training = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(training)


class TrainingTest(unittest.TestCase):
    def test_rectification_matches_identity_grid_and_tile_order(self):
        pixels = np.zeros((768, 768, 3), dtype=np.uint8)
        for row in range(8):
            for column in range(8):
                pixels[row * 96:(row + 1) * 96, column * 96:(column + 1) * 96] = (row, column, row * 8 + column)
        image = Image.fromarray(pixels, "RGB")
        result = np.asarray(training.rectified_grid(image, [[0, 0], [768, 0], [768, 768], [0, 768]]))
        self.assertEqual(result[48, 48].tolist(), [0, 0, 0])
        self.assertEqual(result[7 * 96 + 48, 7 * 96 + 48].tolist(), [7, 7, 63])

    def test_cycling_order_resume_is_exact(self):
        original = training.CyclingOrder(11, 123)
        original.take(7)
        resumed = training.CyclingOrder(11, 123, original.state())
        self.assertEqual(original.take(30), resumed.take(30))

    def test_cosine_schedule_and_iou(self):
        self.assertAlmostEqual(training.cosine_lr(1.0, 0, 10, 2), 0.5)
        self.assertAlmostEqual(training.cosine_lr(1.0, 1, 10, 2), 1.0)
        self.assertAlmostEqual(training.cosine_lr(1.0, 9, 10, 2), 0.0)
        left = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        right = torch.tensor([[0.0, 0.0, 10.0, 10.0], [10.0, 10.0, 20.0, 20.0]])
        self.assertEqual(training.box_iou(left, right).tolist(), [[1.0, 0.0]])

    def test_letterbox_targets_convert_source_frame_for_non_square_page(self):
        result = training.letterbox_targets([[.5, .5, 1.0, .5]], 800, 400)
        self.assertEqual(result[0].tolist(), [0.0, 208.0, 104.0, 416.0, 104.0])
        self.assertTrue(torch.equal(result[1:], torch.zeros((7, 5))))

    def test_letterbox_targets_use_continuous_yolox_scale_when_resize_rounds(self):
        result = training.letterbox_targets([[.25, .75, .5, .25]], 1001, 333)
        scale = 416 / 1001
        expected = torch.tensor([0, .25 * 1001 * scale, .75 * 333 * scale,
                                 .5 * 1001 * scale, .25 * 333 * scale])
        self.assertTrue(torch.allclose(result[0], expected, atol=1e-5, rtol=0))

    def test_detector_training_tensor_normalizes_raw_pixels(self):
        raw = torch.tensor([[[[0.0, 255.0]]]])
        self.assertTrue(torch.equal(training.detector_training_tensor(raw), torch.tensor([[[[0.0, 1.0]]]])))
        with self.assertRaisesRegex(RuntimeError, "raw tensor range"):
            training.detector_training_tensor(torch.tensor([[[[256.0]]]]))

    def test_classifier_export_creates_directory_before_onnx_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 96 * 96, 13))
            def fake_export(_model, _example, destination, **_kwargs):
                self.assertTrue(Path(destination).parent.is_dir())
                Path(destination).write_bytes(b"onnx")
            with mock.patch.object(training.torch.onnx, "export", side_effect=fake_export), \
                 mock.patch.object(training, "verify_onnx", return_value=0.0):
                training.export_classifier(model, root, {"classifier": {"input": "test"}}, {})
            self.assertTrue((root / "classifier" / "selected.onnx").is_file())

    def test_checkpoint_export_reuses_frozen_metrics_without_dataset_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint")
            stages = [{"updates": 2}, {"updates": 8}]
            model = torch.nn.Linear(1, 1)
            saved = {"schema": "chess-ocr-training-checkpoint/1", "global_step": 8,
                     "stage": 1, "stage_step": 6, "model": model.state_dict(),
                     "optimizer": {}, "sampler": {}, "rng": {}, "extra": {}}
            development = {"exact_board_accuracy": 1.0}
            (root / "frozen.json").write_text(json.dumps({"classifier_checkpoint": {
                "sha256": training.sha256(checkpoint), "selected_global_step": 8,
                "scheduled_updates": 10,
                "development": development}}))
            args = type("Args", (), {"classifier_checkpoint": checkpoint, "native": root,
                                      "run": root, "dataset": root})()
            config = {"classifier": {"stages": stages}}
            with mock.patch.object(training, "classifier_model", return_value=model), \
                 mock.patch.object(training.torch, "load", return_value=saved), \
                 mock.patch.object(training, "export_classifier") as export, \
                 mock.patch.object(training, "evaluate_classifier", side_effect=AssertionError("full evaluation")), \
                 mock.patch.object(training, "calibrate_classifier", side_effect=AssertionError("full calibration")), \
                 mock.patch.object(training, "resource_guard"):
                training.export_classifier_checkpoint(args, config, [], {})
            self.assertEqual(export.call_args.args[3]["development"], development)

    def test_detector_export_embeds_raw_pixel_normalization(self):
        class Detector(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = type("Head", (), {"decode_in_inference": True})()

            def forward(self, images):
                return images.mean(dim=(2, 3))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "detector").mkdir()
            captured = {}
            def fake_export(model, example, destination, **_kwargs):
                captured["example"] = example
                captured["output"] = model(example)
                Path(destination).write_bytes(b"onnx")
            with mock.patch.object(training.torch.onnx, "export", side_effect=fake_export), \
                 mock.patch.object(training, "verify_onnx", return_value=0.0) as verify:
                training.export_detector(Detector(), root, {"detector": {"input": "raw", "nms_iou": .5}}, {})
            expected = captured["example"].mean(dim=(2, 3)) / 255
            self.assertGreater(float(captured["example"].max() - captured["example"].min()), 0)
            self.assertTrue(torch.allclose(captured["output"], expected))
            self.assertEqual(verify.call_args.kwargs["maximum_absolute_difference"], 1e-3)


if __name__ == "__main__":
    unittest.main()
