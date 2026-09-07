import {
  LEGACY_DETECTOR_PREPROCESSING,
  V2_DETECTOR_PREPROCESSING,
  type detectorPreprocessingSchema,
} from "./candidate.ts";
import { decodeYolox, type Detection } from "./geometry.ts";
import type { z } from "zod";

export const DETECTOR_SIZE = 416;
export const GRID_SIZE = 768;
export const SQUARE_SIZE = 96;
const MEAN = [0.485, 0.456, 0.406] as const;
const STD = [0.229, 0.224, 0.225] as const;

export type DetectorPreprocessing = z.infer<typeof detectorPreprocessingSchema>;
export type DetectorRaster = {
  input: Float32Array;
  scale: number;
  resizedWidth: number;
  resizedHeight: number;
};

function assertRgba(
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
): void {
  if (
    !Number.isInteger(width) ||
    !Number.isInteger(height) ||
    width < 1 ||
    height < 1 ||
    width > 8192 ||
    height > 8192 ||
    width * height > 16_000_000 ||
    rgba.length !== width * height * 4
  )
    throw new Error("Invalid bounded RGBA raster");
}

/** Deterministic half-pixel bilinear resize into YOLOX's top-left letterbox. */
export function detectorRasterFromRgba(
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
  preprocessing: DetectorPreprocessing,
): DetectorRaster {
  assertRgba(rgba, width, height);
  const scale = Math.min(DETECTOR_SIZE / width, DETECTOR_SIZE / height);
  const resizedWidth = Math.max(1, Math.trunc(width * scale));
  const resizedHeight = Math.max(1, Math.trunc(height * scale));
  const plane = DETECTOR_SIZE * DETECTOR_SIZE;
  const input = new Float32Array(plane * 3);
  input.fill(114);
  const legacy = preprocessing === LEGACY_DETECTOR_PREPROCESSING;
  if (!legacy && preprocessing !== V2_DETECTOR_PREPROCESSING)
    throw new Error("Unsupported detector preprocessing");
  const channelSource = legacy ? [2, 1, 0] : [0, 1, 2];
  for (let y = 0; y < resizedHeight; y++) {
    const sourceY = (y + 0.5) / scale - 0.5;
    const y0 = Math.max(0, Math.min(height - 1, Math.floor(sourceY)));
    const y1 = Math.min(height - 1, y0 + 1);
    const wy = Math.max(0, Math.min(1, sourceY - y0));
    for (let x = 0; x < resizedWidth; x++) {
      const sourceX = (x + 0.5) / scale - 0.5;
      const x0 = Math.max(0, Math.min(width - 1, Math.floor(sourceX)));
      const x1 = Math.min(width - 1, x0 + 1);
      const wx = Math.max(0, Math.min(1, sourceX - x0));
      for (let channel = 0; channel < 3; channel++) {
        const sourceChannel = channelSource[channel]!;
        const top =
          rgba[(y0 * width + x0) * 4 + sourceChannel]! * (1 - wx) +
          rgba[(y0 * width + x1) * 4 + sourceChannel]! * wx;
        const bottom =
          rgba[(y1 * width + x0) * 4 + sourceChannel]! * (1 - wx) +
          rgba[(y1 * width + x1) * 4 + sourceChannel]! * wx;
        input[channel * plane + y * DETECTOR_SIZE + x] =
          top * (1 - wy) + bottom * wy;
      }
    }
  }
  return { input, scale, resizedWidth, resizedHeight };
}

export type SourceDetection = Detection & {
  detectorBox: Detection["box"];
};

export function sourceDetections(
  raw: Float32Array,
  raster: DetectorRaster,
  scoreThreshold: number,
  nmsIou: number,
  maxResults = 16,
): SourceDetection[] {
  return decodeYolox(raw, DETECTOR_SIZE, scoreThreshold, 1, nmsIou, maxResults)
    .map((detection) => {
      const left = Math.max(0, Math.min(raster.resizedWidth, detection.box.x));
      const top = Math.max(0, Math.min(raster.resizedHeight, detection.box.y));
      const right = Math.max(
        0,
        Math.min(raster.resizedWidth, detection.box.x + detection.box.width),
      );
      const bottom = Math.max(
        0,
        Math.min(raster.resizedHeight, detection.box.y + detection.box.height),
      );
      return {
        ...detection,
        detectorBox: detection.box,
        box: {
          x: left / raster.scale,
          y: top / raster.scale,
          width: (right - left) / raster.scale,
          height: (bottom - top) / raster.scale,
        },
      };
    })
    .filter((detection) => detection.box.width > 1 && detection.box.height > 1);
}

/** Form 64 RGB96 ImageNet-normalized NCHW inputs from a rectified RGB768 grid. */
export function classifierTiles(rectifiedRgb: Uint8Array): Float32Array {
  if (rectifiedRgb.length !== GRID_SIZE * GRID_SIZE * 3)
    throw new Error("Invalid rectified RGB grid");
  const result = new Float32Array(64 * 3 * SQUARE_SIZE * SQUARE_SIZE);
  for (let row = 0; row < 8; row++)
    for (let column = 0; column < 8; column++) {
      const square = row * 8 + column;
      for (let channel = 0; channel < 3; channel++)
        for (let y = 0; y < SQUARE_SIZE; y++)
          for (let x = 0; x < SQUARE_SIZE; x++) {
            const source =
              ((row * SQUARE_SIZE + y) * GRID_SIZE + column * SQUARE_SIZE + x) *
                3 +
              channel;
            const target =
              (square * 3 + channel) * SQUARE_SIZE * SQUARE_SIZE +
              y * SQUARE_SIZE +
              x;
            result[target] =
              (rectifiedRgb[source]! / 255 - MEAN[channel]!) / STD[channel]!;
          }
    }
  return result;
}

export function classProbabilities(
  logits: Float32Array,
  squares: number,
): number[][] {
  if (!Number.isInteger(squares) || squares < 1 || squares > 1024)
    throw new Error("Invalid classifier square count");
  if (logits.length !== squares * 13)
    throw new Error("Invalid classifier output");
  return Array.from({ length: squares }, (_, square) => {
    const start = square * 13;
    let maximum = -Infinity;
    for (let index = 0; index < 13; index++) {
      const value = logits[start + index]!;
      if (!Number.isFinite(value))
        throw new Error("Nonfinite classifier output");
      maximum = Math.max(maximum, value);
    }
    const values = Array.from({ length: 13 }, (_, index) =>
      Math.exp(logits[start + index]! - maximum),
    );
    const total = values.reduce((sum, value) => sum + value, 0);
    if (!Number.isFinite(total) || total <= 0)
      throw new Error("Invalid classifier probability total");
    return values.map((value) => value / total);
  });
}
