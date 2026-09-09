"""Deterministic classifier grid rectification and RGB96 tensor extraction.

This mirrors src/grid.ts and src/candidate-runtime.ts without depending on a
platform image resampler. Callers must decode the bounded source raster first.
"""
from __future__ import annotations

import array
import math
import sys
from collections.abc import Sequence


GRID_SIZE = 768
SQUARE_SIZE = 96
LABELS = ".PNBRQKpnbrqk"
MEAN = (.485, .456, .406)
STD = (.229, .224, .225)
MAX_DIMENSION = 8192
MAX_PIXELS = 16_000_000


class Invalid(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Invalid(message)


def _point(value: object) -> tuple[float, float]:
    if isinstance(value, dict):
        x, y = value.get("x"), value.get("y")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
        x, y = value
    else:
        raise Invalid("invalid grid point")
    require(type(x) in (int, float) and type(y) in (int, float), "invalid grid point")
    x, y = float(x), float(y)
    require(math.isfinite(x) and math.isfinite(y), "invalid grid point")
    return x, y


def corners(value: object, width: int, height: int) -> tuple[tuple[float, float], ...]:
    require(isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 4,
            "four grid corners required")
    parsed = tuple(_point(point) for point in value)
    require(all(0 <= x <= width and 0 <= y <= height for x, y in parsed), "grid corner outside raster")
    # Dataset geometry uses continuous image edges and permits width/height;
    # Canvas pixels use centers through width-1/height-1. Only the outer edge is
    # converted, matching the browser's manual-selection clamp.
    result = tuple((min(x, width - 1), min(y, height - 1)) for x, y in parsed)
    area = sum(result[index][0] * result[(index + 1) % 4][1] -
               result[index][1] * result[(index + 1) % 4][0] for index in range(4)) / 2
    require(area > 1e-6, "grid corners must be clockwise from top-left")
    return result


def inverse_homography(value: object, width: int, height: int,
                       size: int = GRID_SIZE) -> tuple[float, ...]:
    """Return the browser's destination-to-source homography."""
    require(type(size) is int and 1 <= size <= GRID_SIZE, "invalid rectified size")
    points = corners(value, width, height)
    edge = size - 1
    destination = ((0, 0), (edge, 0), (edge, edge), (0, edge))
    rows: list[list[float]] = []
    for (x, y), (sx, sy) in zip(destination, points):
        rows.extend((
            [x, y, 1, 0, 0, 0, -x * sx, -y * sx, sx],
            [0, 0, 0, x, y, 1, -x * sy, -y * sy, sy],
        ))
    for column in range(8):
        pivot = max(range(column, 8), key=lambda row: abs(rows[row][column]))
        require(abs(rows[pivot][column]) >= 1e-10, "degenerate grid homography")
        rows[column], rows[pivot] = rows[pivot], rows[column]
        scale = rows[column][column]
        for index in range(column, 9):
            rows[column][index] /= scale
        for row in range(8):
            if row == column:
                continue
            factor = rows[row][column]
            for index in range(column, 9):
                rows[row][index] -= factor * rows[column][index]
    return tuple(rows[index][8] for index in range(8)) + (1.0,)


def _rectification_input(rgba: bytes | bytearray | memoryview, width: int, height: int,
                         value: object, size: int) -> tuple[memoryview, tuple[float, ...]]:
    require(type(width) is int and type(height) is int and 0 < width <= MAX_DIMENSION and
            0 < height <= MAX_DIMENSION and width * height <= MAX_PIXELS, "invalid bounded raster")
    source = memoryview(rgba).cast("B")
    require(len(source) == width * height * 4, "invalid RGBA byte length")
    return source, inverse_homography(value, width, height, size)


def rectify_rgba_scalar(rgba: bytes | bytearray | memoryview, width: int, height: int,
                        value: object, size: int = GRID_SIZE) -> bytes:
    """Dependency-free reference for browser-identical bilinear sampling."""
    source, transform = _rectification_input(rgba, width, height, value, size)
    output = bytearray(size * size * 3)
    h0, h1, h2, h3, h4, h5, h6, h7, h8 = transform
    source_width = width
    source_height = height
    for y in range(size):
        for x in range(size):
            divisor = h6 * x + h7 * y + h8
            sx = (h0 * x + h1 * y + h2) / divisor
            sy = (h3 * x + h4 * y + h5) / divisor
            x0 = max(0, min(source_width - 1, math.floor(sx)))
            y0 = max(0, min(source_height - 1, math.floor(sy)))
            x1 = min(source_width - 1, x0 + 1)
            y1 = min(source_height - 1, y0 + 1)
            fx = max(0.0, min(1.0, sx - x0))
            fy = max(0.0, min(1.0, sy - y0))
            a = (y0 * source_width + x0) * 4
            b = (y0 * source_width + x1) * 4
            c = (y1 * source_width + x0) * 4
            d = (y1 * source_width + x1) * 4
            target = (y * size + x) * 3
            for channel in range(3):
                sample = (source[a + channel] * (1 - fx) * (1 - fy) +
                          source[b + channel] * fx * (1 - fy) +
                          source[c + channel] * (1 - fx) * fy +
                          source[d + channel] * fx * fy)
                # All samples are non-negative, so this is JavaScript Math.round.
                output[target + channel] = math.floor(sample + .5)
    return bytes(output)


def rectify_rgba_numpy(rgba: bytes | bytearray | memoryview, width: int, height: int,
                       value: object, size: int = GRID_SIZE) -> bytes:
    """Vectorized form of the scalar contract for the native training environment."""
    import numpy as np

    source, transform = _rectification_input(rgba, width, height, value, size)
    h0, h1, h2, h3, h4, h5, h6, h7, h8 = transform
    y, x = np.indices((size, size), dtype=np.float64)
    divisor = h6 * x + h7 * y + h8
    sx = (h0 * x + h1 * y + h2) / divisor
    sy = (h3 * x + h4 * y + h5) / divisor
    x0 = np.clip(np.floor(sx), 0, width - 1).astype(np.int64)
    y0 = np.clip(np.floor(sy), 0, height - 1).astype(np.int64)
    x1 = np.minimum(width - 1, x0 + 1)
    y1 = np.minimum(height - 1, y0 + 1)
    fx = np.clip(sx - x0, 0, 1)
    fy = np.clip(sy - y0, 0, 1)
    pixels = np.frombuffer(source, dtype=np.uint8).reshape(height, width, 4)[..., :3]
    samples = (pixels[y0, x0] * (1 - fx[..., None]) * (1 - fy[..., None]) +
               pixels[y0, x1] * fx[..., None] * (1 - fy[..., None]) +
               pixels[y1, x0] * (1 - fx[..., None]) * fy[..., None] +
               pixels[y1, x1] * fx[..., None] * fy[..., None])
    return np.floor(samples + .5).astype(np.uint8).tobytes()


def rectify_rgba(rgba: bytes | bytearray | memoryview, width: int, height: int,
                 value: object, size: int = GRID_SIZE) -> bytes:
    """Rectify through the fast native path when NumPy is already available."""
    try:
        import numpy  # noqa: F401
    except ImportError:
        return rectify_rgba_scalar(rgba, width, height, value, size)
    return rectify_rgba_numpy(rgba, width, height, value, size)


def _tensor_input(rectified_rgb: bytes | bytearray | memoryview, size: int) -> memoryview:
    require(size == GRID_SIZE, "classifier grid must be RGB768")
    source = memoryview(rectified_rgb).cast("B")
    require(len(source) == size * size * 3, "invalid rectified RGB byte length")
    return source


def classifier_tensor_scalar(rectified_rgb: bytes | bytearray | memoryview,
                             size: int = GRID_SIZE) -> array.array:
    """Dependency-free image-row-major RGB96 normalized NCHW extraction."""
    source = _tensor_input(rectified_rgb, size)
    result = array.array("f")
    append = result.append
    for row in range(8):
        for column in range(8):
            for channel, (mean, std) in enumerate(zip(MEAN, STD)):
                for y in range(SQUARE_SIZE):
                    start = (((row * SQUARE_SIZE + y) * size + column * SQUARE_SIZE) * 3 + channel)
                    stop = start + SQUARE_SIZE * 3
                    for index in range(start, stop, 3):
                        append((source[index] / 255 - mean) / std)
    if sys.byteorder != "little":
        result.byteswap()
    return result


def classifier_tensor_numpy(rectified_rgb: bytes | bytearray | memoryview,
                            size: int = GRID_SIZE) -> array.array:
    """Vectorized form of the scalar tensor contract for native training."""
    import numpy as np

    source = _tensor_input(rectified_rgb, size)
    pixels = np.frombuffer(source, dtype=np.uint8).reshape(8, SQUARE_SIZE, 8, SQUARE_SIZE, 3)
    tiles = pixels.transpose(0, 2, 4, 1, 3).reshape(64, 3, SQUARE_SIZE, SQUARE_SIZE)
    normalized = ((tiles.astype(np.float64) / 255 - np.asarray(MEAN)[None, :, None, None]) /
                  np.asarray(STD)[None, :, None, None]).astype("<f4")
    result = array.array("f")
    result.frombytes(normalized.tobytes())
    return result


def classifier_tensor(rectified_rgb: bytes | bytearray | memoryview,
                      size: int = GRID_SIZE) -> array.array:
    """Extract image-row-major RGB squares into ImageNet-normalized NCHW float32."""
    try:
        import numpy  # noqa: F401
    except ImportError:
        return classifier_tensor_scalar(rectified_rgb, size)
    return classifier_tensor_numpy(rectified_rgb, size)


def label_indices(labels: object) -> list[int]:
    require(isinstance(labels, Sequence) and not isinstance(labels, (str, bytes)) and len(labels) == 64,
            "64 image-relative labels required")
    canonical = ["." if label == "empty" else label for label in labels]
    require(all(type(label) is str and len(label) == 1 and label in LABELS for label in canonical),
            "invalid classifier label")
    return [LABELS.index(label) for label in canonical]
