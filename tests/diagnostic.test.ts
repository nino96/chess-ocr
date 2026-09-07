import assert from "node:assert/strict";
import test from "node:test";
import { matchingImportFiles, sha256 } from "../src/diagnostic.ts";
import {
  prepareManualGridInput,
  restoreManualGridResult,
} from "../src/browser.ts";
import type { Result } from "../src/contract.ts";
import type { PrivateEvaluation } from "../src/evaluation.ts";

const identity = { name: "model", version: "1", sha256: "a".repeat(64) };

test("private import requires every reselected file to match its recorded hash", async () => {
  const first = new File(["first"], "first.png", { type: "image/png" });
  const second = new File(["second"], "second.png", { type: "image/png" });
  const emptyResult: Result = {
    schema: "chess-ocr/1",
    requestId: "result",
    image: { width: 1, height: 1 },
    status: "unsupported",
    boards: [],
    warnings: [],
    model: identity,
    preprocessing: "test",
    timings: { totalMs: 1 },
  };
  const session: PrivateEvaluation = {
    schema: "chess-ocr-private-evaluation/1",
    sensitive: true,
    warning:
      "SENSITIVE LOCAL EVIDENCE: contains image hashes, geometry, positions, and raw model results; do not publish",
    createdAt: "2026-09-07T00:00:00.000Z",
    device: {
      label: "x",
      os: "x",
      browser: "x",
      browserVersion: "x",
      cpu: "x",
      memoryGiB: null,
      mainThreadHeapPeakBytes: null,
      peakMemoryMethod: "x",
    },
    models: { v2: identity, fenshot: identity },
    entries: [0, 1].map((index, item) => ({
      index,
      image: {
        width: 1,
        height: 1,
        bytes: [first, second][item]!.size,
        sha256: "",
      },
      mode: "automatic",
      reference: { kind: "no-board" },
      results: {
        v2: {
          result: emptyResult,
          cold: true,
          totalMs: 1,
          workerHeapBytes: null,
        },
        fenshot: {
          result: emptyResult,
          cold: true,
          totalMs: 1,
          workerHeapBytes: null,
        },
      },
    })),
  };
  session.entries[0]!.image.sha256 = await sha256(first);
  session.entries[1]!.image.sha256 = await sha256(second);
  assert.equal(await matchingImportFiles(session, [first, second]), true);
  assert.equal(
    await matchingImportFiles(
      { ...session, entries: [...session.entries].reverse() },
      [first, second],
    ),
    true,
  );
  assert.equal(await matchingImportFiles(session, [second, first]), false);
  assert.equal(await matchingImportFiles(session, [first]), false);
});

test("manual-grid pairing rectifies identical RGBA and restores source geometry", () => {
  const rgba = new Uint8ClampedArray(12 * 12 * 4);
  rgba.fill(255);
  const corners = [
    { x: 1, y: 1 },
    { x: 10, y: 2 },
    { x: 9, y: 10 },
    { x: 2, y: 9 },
  ] as const;
  const prepared = prepareManualGridInput(rgba, 12, 12, corners);
  assert.equal(prepared.length, 768 * 768 * 4);
  const base = {
    schema: "chess-ocr/1",
    requestId: "manual",
    image: { width: 768, height: 768 },
    status: "ok",
    boards: [
      {
        id: "board",
        corners: [
          { x: 0, y: 0 },
          { x: 767, y: 0 },
          { x: 767, y: 767 },
          { x: 0, y: 767 },
        ],
        geometrySource: "detected",
        squares: Array.from({ length: 64 }, () => ({
          label: "empty",
          probabilities: Array.from({ length: 13 }, (_, index) =>
            index === 0 ? 1 : 0,
          ),
          uncertain: true,
        })),
        orientation: "unknown",
        orientationEvidence: "unknown",
        warnings: [],
      },
    ],
    warnings: [],
    model: identity,
    preprocessing: "unchanged-model",
    timings: { totalMs: 1 },
  } as Result;
  const restored = restoreManualGridResult(
    base,
    { width: 12, height: 12 },
    corners,
  );
  assert.deepEqual(restored.boards[0]!.corners, corners);
  assert.equal(restored.boards[0]!.geometrySource, "manual");
  assert.match(restored.preprocessing, /manual-grid/);
});
