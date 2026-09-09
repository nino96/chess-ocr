"""Procedural, original-fixture checks for the exact classifier input contract."""
import array
import unittest

from python import classifier_preprocessing as p


class ClassifierPreprocessingTest(unittest.TestCase):
    def test_identity_rectification_preserves_rgb_and_discards_alpha(self):
        width = height = 4
        rgba = bytes(value for y in range(height) for x in range(width)
                     for value in (x * 31 + y, y * 47 + x, x * 13 + y * 17, 19))
        result = p.rectify_rgba(
            rgba, width, height, [[0, 0], [3, 0], [3, 3], [0, 3]], size=4)
        expected = bytes(value for y in range(height) for x in range(width)
                         for value in (x * 31 + y, y * 47 + x, x * 13 + y * 17))
        self.assertEqual(result, expected)

    def test_bilinear_interpolation_uses_half_up_byte_rounding(self):
        rgba = bytes((0, 10, 20, 255, 100, 110, 120, 255,
                      200, 210, 220, 255, 255, 250, 240, 255))
        result = p.rectify_rgba(
            rgba, 2, 2, [[.5, .5], [1, .5], [1, 1], [.5, 1]], size=2)
        self.assertEqual(result[:3], bytes((139, 145, 150)))
        self.assertEqual(result[-3:], bytes((255, 250, 240)))

    def test_vectorized_and_dependency_free_rectification_are_identical(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("NumPy is not part of the dataset-only environment")
        width, height = 17, 13
        rgba = bytes((index * 37 + 11) % 256 for index in range(width * height * 4))
        corners = [[1.25, .75], [15.5, 1.1], [16, 11.75], [.2, 12]]
        scalar = p.rectify_rgba_scalar(rgba, width, height, corners, size=31)
        self.assertEqual(p.rectify_rgba_numpy(rgba, width, height, corners, size=31), scalar)

    def test_tile_channel_and_row_major_order_with_normalization(self):
        rgb = bytearray(768 * 768 * 3)
        for row in range(8):
            for column in range(8):
                square = row * 8 + column
                for y in range(row * 96, (row + 1) * 96):
                    for x in range(column * 96, (column + 1) * 96):
                        offset = (y * 768 + x) * 3
                        rgb[offset:offset + 3] = bytes((square, 128, 255 - square))
        tensor = p.classifier_tensor(rgb)
        if __import__("sys").byteorder != "little":
            tensor.byteswap()
        values = array.array("f")
        values.frombytes(tensor.tobytes())
        plane = 96 * 96
        self.assertAlmostEqual(values[0], (0 / 255 - .485) / .229, places=6)
        self.assertAlmostEqual(values[63 * 3 * plane], (63 / 255 - .485) / .229, places=6)
        self.assertAlmostEqual(values[plane], (128 / 255 - .456) / .224, places=6)
        try:
            accelerated = p.classifier_tensor_numpy(rgb)
        except ImportError:
            pass
        else:
            self.assertEqual(accelerated.tobytes(), tensor.tobytes())

    def test_class_order_and_corrupt_inputs_are_rejected(self):
        self.assertEqual(p.label_indices(list(p.LABELS) + ["."] * 51)[:13], list(range(13)))
        with self.assertRaisesRegex(p.Invalid, "RGBA byte length"):
            p.rectify_rgba(b"bad", 2, 2, [[0, 0], [1, 0], [1, 1], [0, 1]])
        with self.assertRaisesRegex(p.Invalid, "rectified RGB"):
            p.classifier_tensor(b"bad")
        with self.assertRaisesRegex(p.Invalid, "classifier label"):
            p.label_indices(["x"] * 64)


if __name__ == "__main__":
    unittest.main()
