import importlib.util
from pathlib import Path
import unittest

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


if __name__ == "__main__":
    unittest.main()
