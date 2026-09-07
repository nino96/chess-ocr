import { test } from "node:test";
import assert from "node:assert/strict";
import { decodeYolox, nms, pageTiles } from "../src/geometry.ts";
test("YOLOX grid decode and class-aware NMS preserve class and score semantics", () => {
  const raw = new Float32Array(3549 * 85);
  raw.set([2, 3, Math.log(2), Math.log(4), 0.9, 0.8]);
  const d = decodeYolox(raw);
  assert.equal(d.length, 1);
  assert.equal(d[0]!.classId, 0);
  assert.ok(Math.abs(d[0]!.box.x - 8) < 1e-5);
  assert.ok(Math.abs(d[0]!.box.height - 32) < 1e-5);
  const a = {
    box: { x: 0, y: 0, width: 20, height: 20 },
    score: 0.9,
    classId: 0,
  };
  assert.equal(nms([a, { ...a, score: 0.8 }, { ...a, classId: 1 }]).length, 2);
  raw[0] = NaN;
  assert.throws(() => decodeYolox(raw), /Nonfinite/);
});
test("one-class trained detector uses the same bounded YOLOX grid decoder", () => {
  const raw = new Float32Array(3549 * 6);
  raw.set([2, 3, Math.log(2), Math.log(4), 0.9, 0.8]);
  const detection = decodeYolox(raw, 416, 0.3, 1, 0.65, 16);
  assert.equal(detection.length, 1);
  assert.equal(detection[0]!.classId, 0);
  assert.ok(Math.abs(detection[0]!.box.x - 8) < 1e-5);
  assert.throws(
    () => decodeYolox(new Float32Array(3549 * 7), 416, 0.3, 1),
    /tensor/,
  );
});
test("page windows cover source boundaries with bounded overlap and work", () => {
  const tiles = pageTiles(1600, 1200);
  assert.equal(tiles.length, 4);
  assert.ok(
    tiles.some((t) => t.x + t.width === 1600 && t.y + t.height === 1200),
  );
  assert.deepEqual(pageTiles(200, 100), [
    { x: 0, y: 0, width: 200, height: 100 },
  ]);
  assert.throws(() => pageTiles(4000, 4000, 128, 127), /tiles|windows/);
});
