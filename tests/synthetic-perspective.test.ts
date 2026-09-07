import assert from "node:assert/strict";
import { test } from "node:test";
import {
  perspectiveRecipe,
  projectPoint,
  perspectiveCss,
  validatePerspective,
  // @ts-expect-error MJS offline tooling API.
} from "../scripts/synthetic-perspective.mjs";
// @ts-expect-error MJS offline tooling API.
import { makeRecipe, validateRecipe } from "../scripts/synthetic-render.mjs";

test("perspective is bounded non-affine whole-page geometry with correct CSS ordering", () => {
  const m = perspectiveRecipe(612, 792, 4);
  const tl = projectPoint(m, [0, 0]),
    tr = projectPoint(m, [612, 0]),
    bl = projectPoint(m, [0, 792]),
    br = projectPoint(m, [612, 792]);
  assert.ok(Math.abs(br[0] - tr[0] - bl[0] + tl[0]) > 1);
  for (const [x, y] of [tl, tr, bl, br]) {
    assert.ok(x >= 0 && x <= 612);
    assert.ok(y >= 0 && y <= 792);
  }
  const css = perspectiveCss(m).slice(9, -1).split(",").map(Number);
  for (const [x, y] of [
    [100, 200],
    [600, 700],
  ] as const) {
    const z = css[3] * x + css[7] * y + css[15];
    assert.deepEqual(
      [
        (css[0] * x + css[4] * y + css[12]) / z,
        (css[1] * x + css[5] * y + css[13]) / z,
      ],
      projectPoint(m, [x, y]),
    );
  }
  assert.throws(() => validatePerspective([...m.slice(0, 8), 0], 612, 792, 4));
  const recipe = makeRecipe(20260907, 4);
  validateRecipe(recipe);
  recipe.boards[0].corners[0][0] += 1;
  assert.throws(() => validateRecipe(recipe), /projected corners/);
  const clipped = makeRecipe(20260907, 4);
  clipped.boards[0].render_corners = [
    [-1, 50],
    [200, 50],
    [200, 250],
    [-1, 250],
  ];
  clipped.boards[0].corners = clipped.boards[0].render_corners.map(
    (p: number[]) => projectPoint(clipped.condition.perspective, p),
  );
  assert.throws(() => validateRecipe(clipped), /source board outside page/);
  const changedEffect = makeRecipe(20260907, 4);
  const canonical = makeRecipe(20260907, 4);
  canonical.condition.degradation = Object.fromEntries(
    Object.entries(canonical.condition.degradation).sort(),
  );
  validateRecipe(canonical);
  changedEffect.condition.degradation.paper_noise = 2;
  assert.throws(
    () => validateRecipe(changedEffect),
    /changed degradation recipe/,
  );
});
