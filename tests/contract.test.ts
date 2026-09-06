import { test } from "node:test";
import assert from "node:assert/strict";
import {
  VERSION,
  requestSchema,
  resultSchema,
  manualBoard,
} from "../src/contract.ts";
import { Editor } from "../src/editor.ts";
const request = {
  schema: VERSION,
  requestId: "a",
  image: { width: 256, height: 256 },
  selection: null,
};
const result = () => ({
  ...request,
  selection: undefined,
  status: "ok",
  boards: [manualBoard({ x: 0, y: 0, width: 256, height: 256 })],
  warnings: [],
  model: null,
  preprocessing: "test",
  timings: { totalMs: 1 },
});
function response() {
  const { selection: _, ...r } = result();
  return r;
}
test("contract rejects bounds, nonfinite numbers, unknown fields and excessive pixels", () => {
  for (const bad of [
    { ...request, image: { width: 8192, height: 8192 } },
    { ...request, selection: { x: 250, y: 0, width: 10, height: 10 } },
    { ...request, image: { width: NaN, height: 256 } },
    { ...request, filename: "secret" },
  ])
    assert.throws(() => requestSchema.parse(bad));
  assert.equal(requestSchema.parse(request).schema, VERSION);
});
test("result validates all 64 squares, probabilities and clockwise source geometry", () => {
  const r = response();
  assert.equal(resultSchema.parse(r).boards[0]!.squares.length, 64);
  r.boards[0]!.squares.pop();
  assert.throws(() => resultSchema.parse(r));
  const r2 = response();
  r2.boards[0]!.corners.reverse();
  assert.throws(() => resultSchema.parse(r2));
  const r3 = response();
  r3.boards[0]!.squares[0]!.probabilities = Array(13).fill(1);
  assert.throws(() => resultSchema.parse(r3));
  const r4 = response();
  r4.boards[0]!.corners[0].x = 300;
  assert.throws(() => resultSchema.parse(r4));
});
test("edits and orientation survive retries and out-of-order backend results", () => {
  const e = new Editor();
  e.reset(response().boards[0]);
  e.begin("a");
  e.edit(0, "K");
  e.orient("black-bottom");
  e.begin("b");
  assert.equal(e.apply(response()), false);
  assert.equal(e.apply({ ...response(), requestId: "b" }), true);
  assert.equal(e.board!.squares[0]!.label, "K");
  assert.equal(e.board!.orientation, "black-bottom");
  e.invalidate();
  assert.equal(e.apply({ ...response(), requestId: "b" }), false);
  e.reset();
  assert.equal(e.board, null);
  assert.equal(e.isEdited(0), false);
});
test("placement is absent until fully labelled and oriented, and never invents FEN state", () => {
  const e = new Editor();
  e.reset(response().boards[0]);
  assert.equal(e.placement(), null);
  for (let i = 0; i < 64; i++) e.edit(i, "empty");
  e.edit(0, "K");
  assert.equal(e.placement(), null);
  e.orient("white-bottom");
  assert.equal(e.placement(), "K7/8/8/8/8/8/8/8");
  e.orient("black-bottom");
  assert.equal(e.placement(), "8/8/8/8/8/8/8/7K");
  assert.equal("castling" in e.export(), false);
});
test("a changed detected grid cannot silently relocate corrections", () => {
  const e = new Editor();
  e.reset(response().boards[0]);
  e.begin("a");
  e.edit(0, "K");
  const r = response();
  r.boards[0] = manualBoard({ x: 10, y: 10, width: 200, height: 200 });
  assert.equal(e.apply(r), false);
  assert.equal(e.board!.corners[0].x, 0);
  assert.equal(e.board!.squares[0]!.label, "K");
});
