import { test } from "node:test";
import assert from "node:assert/strict";
import { imageSquares } from "../src/squares.ts";
import { LABELS } from "../src/contract.ts";
test("FENShot bottom-up ranks and class order map to original-image row-major probabilities", () => {
  const input = new Float32Array(832);
  for (let i = 0; i < 64; i++) input[i * 13] = 1;
  input[0] = 0;
  input[1] = 1; // bottom-left white king
  input[56 * 13] = 0;
  input[56 * 13 + 12] = 1; // top-left black pawn
  const s = imageSquares(input);
  assert.equal(s[0]!.label, "p");
  assert.equal(s[56]!.label, "K");
  assert.equal(s[0]!.probabilities[LABELS.indexOf("p")], 1);
  assert.equal(s[56]!.uncertain, false);
  assert.throws(() => imageSquares(new Float32Array(13)), /shape/);
});
