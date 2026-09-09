import assert from "node:assert/strict";
import test from "node:test";
import {
  classifierTiles,
  classProbabilities,
  detectorRasterFromRgba,
} from "../src/candidate-runtime.ts";
import { LABELS } from "../src/contract.ts";

test("detector preprocessing preserves colored channels and top-left non-square letterbox", () => {
  const rgba = new Uint8ClampedArray([
    240, 20, 3, 255, 240, 20, 3, 255, 240, 20, 3, 255, 240, 20, 3, 255,
  ]);
  const v2 = detectorRasterFromRgba(rgba, 2, 2, "yolox-rgb-imagenet-v2");
  const legacy = detectorRasterFromRgba(rgba, 2, 2, "legacy-bgr-div255-v1");
  assert.equal(v2.input[0], 240);
  assert.equal(v2.input[416 * 416], 20);
  assert.equal(v2.input[2 * 416 * 416], 3);
  assert.equal(legacy.input[0], 3);
  assert.equal(legacy.input[2 * 416 * 416], 240);

  const wide = new Uint8ClampedArray(4 * 2 * 4).fill(255);
  const boxed = detectorRasterFromRgba(wide, 4, 2, "yolox-rgb-imagenet-v2");
  assert.equal(boxed.resizedWidth, 416);
  assert.equal(boxed.resizedHeight, 208);
  assert.equal(boxed.input[207 * 416], 255);
  assert.equal(boxed.input[208 * 416], 114);
});

test("classifier tiling is image-row-major NCHW with ImageNet normalization", () => {
  assert.deepEqual(LABELS, [
    "empty",
    "P",
    "N",
    "B",
    "R",
    "Q",
    "K",
    "p",
    "n",
    "b",
    "r",
    "q",
    "k",
  ]);
  const rgb = new Uint8Array(768 * 768 * 3);
  for (let row = 0; row < 8; row++)
    for (let column = 0; column < 8; column++)
      for (let y = 0; y < 96; y++)
        for (let x = 0; x < 96; x++) {
          const offset = ((row * 96 + y) * 768 + column * 96 + x) * 3;
          rgb[offset] = row * 8 + column;
          rgb[offset + 1] = 128;
          rgb[offset + 2] = 255 - (row * 8 + column);
        }
  const tiles = classifierTiles(rgb);
  const plane = 96 * 96;
  assert.ok(Math.abs(tiles[0]! - (0 / 255 - 0.485) / 0.229) < 1e-6);
  assert.ok(
    Math.abs(tiles[63 * 3 * plane]! - (63 / 255 - 0.485) / 0.229) < 1e-6,
  );
  assert.ok(Math.abs(tiles[plane]! - (128 / 255 - 0.456) / 0.224) < 1e-6);
  assert.throws(() => classifierTiles(rgb.subarray(1)), /Invalid rectified/);
});

test("classifier probability conversion is stable and rejects corrupt outputs", () => {
  const logits = new Float32Array(13);
  logits[4] = 100;
  const probabilities = classProbabilities(logits, 1)[0]!;
  assert.equal(probabilities.indexOf(Math.max(...probabilities)), 4);
  assert.ok(Math.abs(probabilities.reduce((a, b) => a + b, 0) - 1) < 1e-12);
  logits[2] = Number.NaN;
  assert.throws(() => classProbabilities(logits, 1), /Nonfinite/);
});
