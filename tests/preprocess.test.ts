import { test } from "node:test";
import assert from "node:assert/strict";
import { resizeBicubic, mobileNormalize, yolox416 } from "../src/preprocess.ts";
test("bicubic constant fields and channel normalization preserve channel ordering", () => {
  const rgb = new Uint8Array(12 * 12 * 3);
  for (let i = 0; i < 144; i++) {
    rgb[i * 3] = 255;
    rgb[i * 3 + 1] = 128;
  }
  const resized = resizeBicubic(rgb, 12, 12, 6, 6);
  for (let i = 0; i < 36; i++)
    assert.deepEqual([...resized.slice(i * 3, i * 3 + 3)], [255, 128, 0]);
  const black = mobileNormalize(new Uint8Array(224 * 224 * 3));
  assert.ok(Math.abs(black[0]! + 0.485 / 0.229) < 1e-5);
  assert.ok(Math.abs(black[224 * 224]! + 0.456 / 0.224) < 1e-5);
  const source = new Uint8Array(416 * 416 * 3);
  source.set([5, 10, 20]);
  const input = yolox416(source);
  assert.equal(input[0], 20);
  assert.equal(input[416 * 416 * 2], 5);
  assert.throws(() => resizeBicubic(new Uint8Array(1), 1, 1, 1, 1), /RGB/);
});
