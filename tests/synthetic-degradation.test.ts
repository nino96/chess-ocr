import assert from "node:assert/strict";
import { test } from "node:test";
const { degradationRecipe, degradePixels, validateDegradation } = await import(
  // @ts-ignore MJS tooling API is exercised by Node's ESM test runner.
  "../scripts/synthetic-degradation.mjs"
);

const pixels = new Uint8ClampedArray([
  0, 20, 40, 11, 80, 100, 120, 99, 200, 220, 240, 255, 10, 30, 50, 0,
]);

test("degradation recipes are deterministic and retain an explicit identity variant", () => {
  assert.deepEqual(degradationRecipe("seed", 2), degradationRecipe("seed", 2));
  assert.notDeepEqual(
    degradationRecipe("seed", 2),
    degradationRecipe("other", 2),
  );
  assert.deepEqual(degradationRecipe("seed", 0), {
    variant: "blank",
    seed: degradationRecipe("seed", 0).seed,
    paper_noise: 0,
    ink_fade: 0,
    illumination: 0,
    blur_radius: 0,
  });
});

test("pixel degradation is deterministic, bounded, alpha preserving, and non-mutating", () => {
  const config = degradationRecipe("seed", 3);
  const source = new Uint8ClampedArray(pixels);
  const one = degradePixels(pixels, 2, 2, config);
  assert.deepEqual(one, degradePixels(pixels, 2, 2, config));
  assert.deepEqual(pixels, source);
  assert.notStrictEqual(one, pixels);
  for (let offset = 0; offset < one.length; offset += 4) {
    assert.equal(one[offset + 3], pixels[offset + 3]);
    for (let channel = 0; channel < 3; channel++)
      assert.ok(one[offset + channel] >= 0 && one[offset + channel] <= 255);
  }
  assert.deepEqual(
    degradePixels(pixels, 2, 2, degradationRecipe("seed", 0)),
    pixels,
  );
});

test("degradation rejects malformed configs and unsafe dimensions", () => {
  assert.throws(() =>
    validateDegradation({ ...degradationRecipe("seed", 1), paper_noise: 4 }),
  );
  assert.throws(() =>
    validateDegradation({ ...degradationRecipe("seed", 1), extra: true }),
  );
  assert.throws(() =>
    degradePixels(pixels, 0, 2, degradationRecipe("seed", 1)),
  );
  assert.throws(() =>
    degradePixels(pixels, 2_000_001, 2, degradationRecipe("seed", 1)),
  );
  assert.throws(() =>
    degradePixels(new Uint8ClampedArray(3), 1, 1, degradationRecipe("seed", 1)),
  );
});
