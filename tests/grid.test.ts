import { test } from "node:test";
import assert from "node:assert/strict";
import {
  deduplicateGrids,
  findInnerGrid,
  rectifyGrid,
  type GridCorners,
  type Point,
} from "../src/grid.ts";

function raster(width = 260, height = 240) {
  return { width, height, data: new Uint8Array(width * height * 4).fill(255) };
}
function put(r: ReturnType<typeof raster>, x: number, y: number, value = 0) {
  if (x >= 0 && y >= 0 && x < r.width && y < r.height) {
    const i = (y * r.width + x) * 4;
    r.data.fill(value, i, i + 3);
    r.data[i + 3] = 255;
  }
}
function line(r: ReturnType<typeof raster>, a: Point, b: Point) {
  const steps = Math.ceil(Math.max(Math.abs(a.x - b.x), Math.abs(a.y - b.y)));
  for (let i = 0; i <= steps; i++) {
    const x = Math.round(a.x + ((b.x - a.x) * i) / steps),
      y = Math.round(a.y + ((b.y - a.y) * i) / steps);
    for (let dx = -1; dx <= 1; dx++)
      for (let dy = -1; dy <= 1; dy++) put(r, x + dx, y + dy);
  }
}
function grid(r: ReturnType<typeof raster>, origin: Point, u: Point, v: Point) {
  for (let i = 0; i < 9; i++) {
    line(
      r,
      { x: origin.x + v.x * i, y: origin.y + v.y * i },
      { x: origin.x + u.x * 8 + v.x * i, y: origin.y + u.y * 8 + v.y * i },
    );
    line(
      r,
      { x: origin.x + u.x * i, y: origin.y + u.y * i },
      { x: origin.x + v.x * 8 + u.x * i, y: origin.y + v.y * 8 + u.y * i },
    );
  }
}

test("findInnerGrid detects a skewed nine-line grid inside an expanded detector region", () => {
  const r = raster();
  grid(r, { x: 55, y: 42 }, { x: 18, y: 4 }, { x: -3, y: 19 });
  const found = findInnerGrid(r, { x: 25, y: 15, width: 210, height: 220 });
  assert.equal(found.ok, true);
  if (!found.ok) return;
  assert.ok(found.score > 0.2);
  assert.ok(Math.abs(found.corners[0].x - 55) < 8);
  assert.ok(Math.abs(found.corners[0].y - 42) < 8);
  assert.equal(found.evidence.verticalLines.length, 9);
  assert.equal(found.evidence.horizontalLines.length, 9);
});
test("expanded detector region excludes its border rather than returning its rectangle", () => {
  const r = raster();
  grid(r, { x: 65, y: 55 }, { x: 16, y: 2 }, { x: -2, y: 17 });
  const found = findInnerGrid(r, { x: 20, y: 20, width: 220, height: 200 });
  assert.equal(found.ok, true);
  if (found.ok) assert.ok(found.corners[0].x > 45 && found.corners[0].y > 35);
});
test("low evidence and partial grids are explicit rejections", () => {
  const r = raster();
  assert.equal(
    findInnerGrid(r, { x: 10, y: 10, width: 220, height: 200 }).ok,
    false,
  );
  grid(r, { x: 10, y: 30 }, { x: 16, y: 0 }, { x: 0, y: 17 });
  const partial = findInnerGrid(r, { x: 10, y: 10, width: 220, height: 200 });
  assert.deepEqual(
    partial.ok ? undefined : partial.reason,
    "partial-or-border-grid",
  );
});
test("inverse homography sampling is deterministic and maps output corners to source corners", () => {
  const r = raster(4, 4);
  for (let y = 0; y < 4; y++)
    for (let x = 0; x < 4; x++) {
      const i = (y * 4 + x) * 4;
      r.data[i] = x * 40;
      r.data[i + 1] = y * 50;
      r.data[i + 2] = 7;
      r.data[i + 3] = 255;
    }
  const corners = [
    { x: 0, y: 0 },
    { x: 3, y: 0 },
    { x: 3, y: 3 },
    { x: 0, y: 3 },
  ] as const;
  const a = rectifyGrid(r, corners, 4, 4),
    b = rectifyGrid(r, corners, 4, 4);
  assert.deepEqual(a.data, b.data);
  assert.deepEqual([...a.data.slice(0, 4)], [0, 0, 7, 255]);
  assert.deepEqual([...a.data.slice(15 * 4, 64)], [120, 150, 7, 255]);
});

test("refined-grid deduplication removes overlap but preserves separate boards", () => {
  const corners = (x: number): GridCorners => [
    { x, y: 10 },
    { x: x + 80, y: 10 },
    { x: x + 80, y: 90 },
    { x, y: 90 },
  ];
  const values = [
    { id: "best", corners: corners(10) },
    { id: "duplicate", corners: corners(12) },
    { id: "separate", corners: corners(120) },
  ];
  assert.deepEqual(
    deduplicateGrids(values).map((value) => value.id),
    ["best", "separate"],
  );
});
