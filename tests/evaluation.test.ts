import assert from "node:assert/strict";
import test from "node:test";
import {
  compareResult,
  evaluationSummarySchema,
  privateEvaluationSchema,
  summarizeEvaluation,
} from "../src/evaluation.ts";
import { LABELS, VERSION, type Result } from "../src/contract.ts";

const hash = "a".repeat(64);
const identity = { name: "model", version: "1", sha256: hash };
const corners: [
  { x: number; y: number },
  { x: number; y: number },
  { x: number; y: number },
  { x: number; y: number },
] = [
  { x: 8, y: 8 },
  { x: 72, y: 8 },
  { x: 72, y: 72 },
  { x: 8, y: 72 },
];
const result = (
  labels = Array(64).fill("empty") as (typeof LABELS)[number][],
): Result => ({
  schema: VERSION,
  requestId: "request",
  image: { width: 80, height: 80 },
  status: "ok",
  boards: [
    {
      id: "board",
      corners,
      geometrySource: "detected",
      squares: labels.map((label) => ({
        label,
        probabilities: LABELS.map((item) => (item === label ? 1 : 0)),
        uncertain: true,
      })),
      orientation: "unknown",
      orientationEvidence: "unknown",
      warnings: [],
    },
  ],
  warnings: [],
  model: identity,
  preprocessing: "test",
  timings: { totalMs: 10 },
});

test("paired comparison counts geometry, class, color and confident errors", () => {
  const labels = Array(64).fill("empty") as (typeof LABELS)[number][];
  labels[0] = "P";
  labels[1] = "n";
  const predicted = [...labels];
  predicted[0] = "p";
  predicted[1] = "B";
  const comparison = compareResult(
    { kind: "board", corners, orientation: "unknown", labels },
    result(predicted),
  );
  assert.equal(comparison.wrongSquares, 2);
  assert.equal(comparison.pieceClassErrors, 1);
  assert.equal(comparison.colorErrors, 2);
  assert.equal(comparison.confidentWrongSquares, 2);
});

test("a missing prediction counts against occupied reference evidence", () => {
  const labels = Array(64).fill("empty") as (typeof LABELS)[number][];
  labels[0] = "P";
  const missing = result();
  missing.boards[0]!.squares[0] = {
    label: null,
    probabilities: null,
    uncertain: true,
  };
  const comparison = compareResult(
    { kind: "board", corners, orientation: "unknown", labels },
    missing,
  );
  assert.equal(comparison.occupiedEmptyErrors, 1);
  assert.equal(comparison.wrongSquares, 1);
});

test("private sessions are bounded/sensitive and public summaries omit private fields", () => {
  const session = privateEvaluationSchema.parse({
    schema: "chess-ocr-private-evaluation/1",
    sensitive: true,
    warning:
      "SENSITIVE LOCAL EVIDENCE: contains image hashes, geometry, positions, and raw model results; do not publish",
    createdAt: "2026-09-07T00:00:00.000Z",
    device: {
      label: "laptop",
      os: "test",
      browser: "test",
      browserVersion: "1",
      cpu: "test",
      memoryGiB: null,
      mainThreadHeapPeakBytes: null,
      peakMemoryMethod: "task manager",
    },
    models: { v2: identity, fenshot: identity },
    entries: [
      {
        index: 0,
        image: { width: 80, height: 80, bytes: 100, sha256: hash },
        mode: "manual-grid",
        reference: {
          kind: "board",
          corners,
          orientation: "unknown",
          labels: Array(64).fill("empty"),
        },
        results: {
          v2: {
            result: result(),
            cold: true,
            totalMs: 10,
            workerHeapBytes: null,
          },
          fenshot: {
            result: result(),
            cold: true,
            totalMs: 20,
            workerHeapBytes: null,
          },
        },
      },
    ],
  });
  const summary = summarizeEvaluation(session);
  evaluationSummarySchema.parse(summary);
  const serialized = JSON.stringify(summary);
  assert.doesNotMatch(
    serialized,
    /sha256|corners|positions|filename|path|requestId/,
  );
  assert.equal(summary.counts.v2.exactBoards, 1);
  assert.equal(summary.counts.v2.cornerDisplacementCount, 1);
});

test("private import rejects duplicate image/mode and more than twenty inputs", () => {
  const entry = {
    index: 0,
    image: { width: 80, height: 80, bytes: 100, sha256: hash },
    mode: "automatic",
    reference: { kind: "no-board" },
    results: {
      v2: { result: result(), cold: false, totalMs: 1, workerHeapBytes: null },
      fenshot: {
        result: result(),
        cold: false,
        totalMs: 1,
        workerHeapBytes: null,
      },
    },
  };
  const base = {
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
  };
  assert.throws(() =>
    privateEvaluationSchema.parse({ ...base, entries: [entry, entry] }),
  );
  assert.throws(() =>
    privateEvaluationSchema.parse({
      ...base,
      entries: Array.from({ length: 21 }, (_, index) => ({ ...entry, index })),
    }),
  );
});
