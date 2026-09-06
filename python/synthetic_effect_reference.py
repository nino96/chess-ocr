"""Independent Pillow reference for conservative synthetic pixel effects.

This mirrors the JavaScript byte math for fidelity checks only; it is not an
empirically calibrated model of printed-page degradation.
"""
from __future__ import annotations

import math
from typing import Any

from PIL import Image

MAX_PIXELS = 4_000_000
_FIELDS = {"variant", "seed", "paper_noise", "ink_fade", "illumination", "blur_radius"}
_VARIANTS = {"blank", "paper", "faded", "soft"}


def validate_exact_config(config: Any) -> dict[str, Any]:
    """Return a validated exact JavaScript degradation configuration."""
    if not isinstance(config, dict) or set(config) != _FIELDS:
        raise ValueError("config has unknown or missing fields")
    if config["variant"] not in _VARIANTS:
        raise ValueError("config variant invalid")
    seed = config["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("config seed invalid")
    for name, maximum in (("paper_noise", 3), ("ink_fade", 0.08), ("illumination", 0.06)):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= maximum:
            raise ValueError(f"config {name} invalid")
    if config["blur_radius"] not in (0, 1) or isinstance(config["blur_radius"], bool):
        raise ValueError("blur_radius must be 0 or 1")
    if config["variant"] == "blank" and any(config[name] != 0 for name in ("paper_noise", "ink_fade", "illumination", "blur_radius")):
        raise ValueError("blank must be identity")
    return config


def _mix(value: int) -> float:
    value = (value + 0x6D2B79F5) & 0xFFFFFFFF
    value = ((value ^ (value >> 15)) * (value | 1)) & 0xFFFFFFFF
    value ^= (value + (((value ^ (value >> 7)) * (value | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF
    return ((value ^ (value >> 14)) & 0xFFFFFFFF) / 2**32


def _byte(value: float) -> int:
    """Uint8ClampedArray conversion: clamp then ties-to-even rounding."""
    return int(round(max(0.0, min(255.0, value))))


def _blur_rgb(source: bytearray, width: int, height: int) -> bytearray:
    horizontal = bytearray(len(source))
    output = bytearray(len(source))
    for y in range(height):
        for x in range(width):
            target = (y * width + x) * 3
            for channel in range(3):
                total = 0
                for dx in (-1, 0, 1):
                    source_x = min(width - 1, max(0, x + dx))
                    total += source[(y * width + source_x) * 3 + channel]
                horizontal[target + channel] = _byte(total / 3)
    for y in range(height):
        for x in range(width):
            target = (y * width + x) * 3
            for channel in range(3):
                total = 0
                for dy in (-1, 0, 1):
                    source_y = min(height - 1, max(0, y + dy))
                    total += horizontal[(source_y * width + x) * 3 + channel]
                output[target + channel] = _byte(total / 3)
    return output


def apply_effect(image: Image.Image, config: dict[str, Any]) -> Image.Image:
    """Return a new RGB image with deterministic, geometry-preserving effects."""
    validate_exact_config(config)
    if image.mode != "RGB":
        raise ValueError("image must be RGB")
    width, height = image.size
    if width <= 0 or height <= 0 or width * height > MAX_PIXELS:
        raise ValueError("dimensions must be positive and at most four million pixels")
    source = image.tobytes()
    if config["variant"] == "blank":
        return Image.frombytes("RGB", image.size, source)
    output = bytearray(source)
    for pixel in range(width * height):
        offset = pixel * 3
        x, y = pixel % width, pixel // width
        wave = (
            math.sin((x / max(1, width - 1)) * math.pi * 1.7 + config["seed"] * 0.00001) * 0.5
            + math.cos((y / max(1, height - 1)) * math.pi * 1.3) * 0.5
        )
        illumination = 1 + config["illumination"] * wave
        noise = (_mix(config["seed"] ^ pixel) - 0.5) * 2 * config["paper_noise"]
        for channel in range(3):
            value = source[offset + channel]
            output[offset + channel] = _byte((value + (255 - value) * config["ink_fade"]) * illumination + noise)
    if config["blur_radius"] == 1:
        blurred = _blur_rgb(output, width, height)
        output = bytearray(_byte(a*0.75+b*0.25) for a,b in zip(output,blurred))
    return Image.frombytes("RGB", image.size, bytes(output))
