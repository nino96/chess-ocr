import { test } from "node:test";
import assert from "node:assert/strict";
import { inspectRaster } from "../src/image.ts";
function png(width: number, height: number) {
  const b = new Uint8Array(24);
  b.set([137, 80, 78, 71, 13, 10, 26, 10]);
  const v = new DataView(b.buffer);
  v.setUint32(8, 13);
  v.setUint32(12, 0x49484452);
  v.setUint32(16, width);
  v.setUint32(20, height);
  return b;
}
test("raster admission bounds decoded pixels before browser decode", () => {
  assert.deepEqual(inspectRaster(png(256, 256)), {
    width: 256,
    height: 256,
    mime: "image/png",
  });
  for (const b of [
    png(8192, 8192),
    png(0, 20),
    new Uint8Array(30),
    new Uint8Array(21 * 1024 * 1024),
  ])
    assert.throws(() => inspectRaster(b));
});
