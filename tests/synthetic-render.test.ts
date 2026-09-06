import assert from "node:assert/strict";
import { test } from "node:test";
// @ts-expect-error MJS tooling API is exercised by Node's ESM test runner.
import { makeRecipe, validateRecipe } from "../scripts/synthetic-render.mjs";

test("synthetic recipes are deterministic and label every square in image row-major order", () => {
  const first = makeRecipe("bootstrap-v1", 7);
  assert.deepEqual(makeRecipe("bootstrap-v1", 7), first);
  assert.notDeepEqual(makeRecipe("bootstrap-v1", 8), first);
  for (const board of first.boards) {
    assert.equal(board.labels.length, 64);
    assert.match(board.labels.join(""), /^[.PNBRQKpnbrqk]{64}$/);
    assert.equal(board.corners.length, 4);
    assert.deepEqual(
      board.corners[0].map((v: number) => Number.isFinite(v)),
      [true, true],
    );
  }
});

test("invalid off-axis geometry is rejected instead of mislabeled", () => {
  const recipe = makeRecipe(1, 2);
  for (let i = 0; i < 12000; i++) validateRecipe(makeRecipe(20260907, i));
  recipe.boards[0].corners[2][0] += 20;
  assert.throws(() => validateRecipe(recipe), /parallelograms/);
  recipe.boards[0].corners[3][0] = -1;
  assert.throws(() => validateRecipe(recipe), /outside page/);
});

test("black-bottom recipes reverse visual labels rather than only changing metadata", () => {
  const recipe = makeRecipe("bootstrap-v1", 0); // Legal initial position, deliberately black bottom.
  const board = recipe.boards[0];
  assert.equal(board.orientation, "black-bottom");
  assert.equal(board.labels.length, 64);
  assert.equal(board.labels[0], "R"); // a1 is image top-left after 180-degree rotation.
  assert.equal(board.labels[63], "r");
});

test("recipe layouts include small, large, and multiple boards with bounded geometry", () => {
  const kinds = new Set(
    Array.from({ length: 4 }, (_, index) => makeRecipe("layout", index).layout),
  );
  assert.deepEqual(
    kinds,
    new Set(["small-board", "large-board", "multiple-boards"]),
  );
  const close = (left: number, right: number) =>
    assert.ok(Math.abs(left - right) < 1e-9);
  for (let index = 0; index < 12; index++)
    for (const board of makeRecipe("layout", index).boards) {
      const [tl, tr, br, bl] = board.corners;
      close(tr[0]! - tl[0]!, br[0]! - bl[0]!);
      close(tr[1]! - tl[1]!, br[1]! - bl[1]!);
      close(br[0]! - tr[0]!, bl[0]! - tl[0]!);
      close(br[1]! - tr[1]!, bl[1]! - tl[1]!);
    }
  const rotated = makeRecipe("layout", 5).boards[0]!;
  assert.notEqual(rotated.corners[0]![1], rotated.corners[1]![1]);
  const negative = makeRecipe("layout", 10);
  assert.equal(negative.kind, "negative");
  assert.deepEqual(negative.boards, []);
});
