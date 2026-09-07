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

    def test_corrected_detector_preprocessing_matches_pinned_upstream_helper(self):
        bgr = np.zeros((173, 311, 3), dtype=np.uint8)
        bgr[:, :103] = (251, 17, 43)
        bgr[:, 103:207] = (7, 229, 83)
        bgr[:, 207:] = (61, 101, 241)
        helper = training.pinned_yolox_preproc(Path(__file__).parents[1])
        maximum, scale = training.corrected_upstream_preprocessing_parity(helper, bgr)
        self.assertLessEqual(maximum, 1e-6)
        self.assertEqual(scale, 0)

    def test_ramped_ema_formula_and_resume_are_exact(self):
        model = torch.nn.Linear(1, 1, bias=False)
        model.weight.data.fill_(0)
        ema = training.EMA(model, 0.9998, 2000)
        model.weight.data.fill_(1)
        ema.update(model)
        expected_decay = 0.9998 * (1 - np.exp(-1 / 2000))
        self.assertEqual(ema.updates, 1)
        self.assertAlmostEqual(ema.decay, expected_decay, places=15)
        self.assertLess(ema.decay, .001)
        expected_weight = torch.zeros((1, 1)).mul_(expected_decay).add_(
            torch.ones((1, 1)), alpha=1 - expected_decay)
        self.assertTrue(torch.equal(ema.state["weight"], expected_weight))
        saved = ema.checkpoint_state()
        resumed = training.EMA(model, 0.9998, 2000)
        resumed.load_checkpoint_state(saved, torch.device("cpu"))
        model.weight.data.fill_(2)
        ema.update(model)
        resumed.update(model)
        self.assertEqual(resumed.updates, 2)
        self.assertTrue(torch.equal(ema.state["weight"], resumed.state["weight"]))

    def test_detector_stage_transition_gate_uses_live_weights(self):
        curves = [{"global_step": 1500, "live_development": {"recall": {"0.5": .10}}}]
        self.assertTrue(training.detector_transition_gate_failed(
            curves, 2000, {"recall": {"0.5": .049}}))
        self.assertFalse(training.detector_transition_gate_failed(
            curves, 2000, {"recall": {"0.5": .05}}))
        self.assertFalse(training.detector_transition_gate_failed(
            [{"global_step": 1500, "live_development": {"recall": {"0.5": .099}}}],
            2000, {"recall": {"0.5": 0}}))

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

    def test_v2_detector_export_embeds_rgb_imagenet_normalization(self):
        class Detector(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = type("Head", (), {"decode_in_inference": True})()

            def forward(self, images):
                return images.mean(dim=(2, 3))

        raw = torch.tensor([[[[255.]], [[127.5]], [[0.]]]])
        wrapper = training.DetectorExportWrapper(
            Detector(), training.V2_DETECTOR_PREPROCESSING)
        expected = training.detector_training_tensor(
            raw, training.V2_DETECTOR_PREPROCESSING).mean(dim=(2, 3))
        self.assertTrue(torch.equal(wrapper(raw), expected))

    def test_detector_export_retry_reuses_durable_evidence_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            detector = root / "detector"
            detector.mkdir()
            (detector / "checkpoint-009000.pt").write_bytes(b"checkpoint")
            progress = {"state": "export_failed", "selected_global_step": 9000,
                        "final": {"ap50_95": .5}, "calibration": {"threshold": .2}}
            (detector / "progress.json").write_text(json.dumps(progress))
            model = torch.nn.Linear(1, 1, bias=False)
            model.head = type("Head", (), {"decode_in_inference": True})()
            saved = {"extra": {"ema": {"weights": model.state_dict(), "updates": 9000,
                                          "base_decay": .9998, "warmup_updates": 2000}}}
            args = type("Args", (), {"run": root, "native": root})()
            config = {"detector": {"preprocessing": training.V2_DETECTOR_PREPROCESSING}}
            with mock.patch.object(training.torch, "load", return_value=saved), \
                 mock.patch.object(training, "detector_model", return_value=model), \
                 mock.patch.object(training, "export_detector") as export, \
                 mock.patch.object(training, "evaluate_detector", side_effect=AssertionError("evaluation repeated")), \
                 mock.patch.object(training, "calibrate_detector", side_effect=AssertionError("calibration repeated")):
                training.export_detector_checkpoint(args, config)
            self.assertEqual(export.call_args.args[3]["development"], progress["final"])
            self.assertEqual(json.loads((detector / "progress.json").read_text())["state"], "complete")

    def test_detector_export_failure_preserves_retryable_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            detector = root / "detector"
            detector.mkdir()
            (detector / "checkpoint-009000.pt").write_bytes(b"checkpoint")
            (detector / "progress.json").write_text(json.dumps({
                "state": "exporting", "selected_global_step": 9000,
                "final": {"ap50_95": .5}, "calibration": {"threshold": .2}}))
            model = torch.nn.Linear(1, 1, bias=False)
            model.head = type("Head", (), {"decode_in_inference": True})()
            saved = {"extra": {"ema": model.state_dict()}}
            args = type("Args", (), {"run": root, "native": root})()
            with mock.patch.object(training.torch, "load", return_value=saved), \
                 mock.patch.object(training, "detector_model", return_value=model), \
                 mock.patch.object(training, "export_detector", side_effect=RuntimeError("injected export")):
                with self.assertRaisesRegex(RuntimeError, "injected export"):
                    training.export_detector_checkpoint(args, {"detector": {}})
            retained = json.loads((detector / "progress.json").read_text())
            self.assertEqual(retained["state"], "export_failed")
            self.assertEqual(retained["final"], {"ap50_95": .5})
            self.assertEqual(retained["calibration"], {"threshold": .2})

    def test_calibration_failure_retains_selected_and_final_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "detector").mkdir()
            lifecycle = {"state": "trained", "global_step": 9000,
                         "selected_global_step": 8500, "final": {"ap50_95": .4}}
            args = type("Args", (), {"run": root})()
            with mock.patch.object(training, "calibrate_detector",
                                   side_effect=RuntimeError("injected calibration")), \
                 mock.patch.object(training, "export_detector") as export:
                with self.assertRaisesRegex(RuntimeError, "injected calibration"):
                    training.finish_detector_lifecycle(
                        torch.nn.Identity(), args, {"detector": {"nms_iou": .65}},
                        object(), torch.device("cpu"), lifecycle)
            export.assert_not_called()
            retained = json.loads((root / "detector" / "progress.json").read_text())
            self.assertEqual(retained["state"], "calibrating")
            self.assertEqual(retained["selected_global_step"], 8500)
            self.assertEqual(retained["final"], {"ap50_95": .4})


if __name__ == "__main__":
    unittest.main()
