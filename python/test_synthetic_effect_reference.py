from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path

from PIL import Image

from python.synthetic_effect_reference import apply_effect, validate_exact_config

ROOT = Path(__file__).resolve().parents[1]
PIXELS = bytes([0, 20, 40, 80, 100, 120, 200, 220, 240, 10, 30, 50])
SOFT = {"variant": "soft", "seed": 123456, "paper_noise": 0.7, "ink_fade": 0.035, "illumination": 0.02, "blur_radius": 1}


class SyntheticEffectReferenceTests(unittest.TestCase):
    def test_blank_is_new_identity_image_and_config_is_strict(self) -> None:
        image = Image.frombytes("RGB", (2, 2), PIXELS)
        blank = {"variant": "blank", "seed": 0, "paper_noise": 0, "ink_fade": 0, "illumination": 0, "blur_radius": 0}
        result = apply_effect(image, blank)
        self.assertEqual(result.tobytes(), PIXELS)
        self.assertIsNot(result, image)
        with self.assertRaises(ValueError):
            validate_exact_config({**blank, "extra": 1})
        with self.assertRaises(ValueError):
            validate_exact_config({**blank, "seed": True})

    def test_known_soft_fixture_is_deterministic_and_byte_bounded(self) -> None:
        image = Image.frombytes("RGB", (2, 2), PIXELS)
        first = apply_effect(image, SOFT).tobytes()
        self.assertEqual(first, apply_effect(image, SOFT).tobytes())
        self.assertEqual(first, bytes([24, 44, 64, 82, 102, 121, 179, 199, 218, 33, 52, 72]))
        self.assertEqual(len(first), len(PIXELS))
        self.assertTrue(all(0 <= value <= 255 for value in first))

    def test_small_fixture_matches_javascript_reference(self) -> None:
        script = """
import { degradePixels } from './scripts/synthetic-degradation.mjs';
const input = JSON.parse(process.argv[1]);
const rgba = new Uint8ClampedArray(input.rgb.flatMap((value) => [value[0], value[1], value[2], 255]));
const out = degradePixels(rgba, input.width, input.height, input.config);
console.log(JSON.stringify(Array.from(out).filter((_, index) => index % 4 !== 3)));
"""
        payload = {"width": 2, "height": 2, "rgb": [list(PIXELS[index:index + 3]) for index in range(0, len(PIXELS), 3)], "config": SOFT}
        result = subprocess.run(["node", "--input-type=module", "-e", script, json.dumps(payload)], cwd=ROOT, check=True, text=True, capture_output=True)
        expected = bytes(json.loads(result.stdout))
        image = Image.frombytes("RGB", (2, 2), PIXELS)
        self.assertEqual(apply_effect(image, SOFT).tobytes(), expected)

    def test_rejects_non_rgb_and_excessive_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            apply_effect(Image.new("RGBA", (1, 1)), SOFT)
        with self.assertRaises(ValueError):
            apply_effect(Image.new("RGB", (2_000_001, 2)), SOFT)


if __name__ == "__main__":
    unittest.main()
