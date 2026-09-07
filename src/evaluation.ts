import { z } from "zod";
import {
  boardSchema,
  identitySchema,
  imageSchema,
  labelSchema,
  orientationSchema,
  pointSchema,
  resultSchema,
  type Board,
  type Label,
  type Result,
} from "./contract.ts";

export const PRIVATE_EVALUATION_VERSION =
  "chess-ocr-private-evaluation/1" as const;
export const EVALUATION_SUMMARY_VERSION =
  "chess-ocr-evaluation-summary/1" as const;
const finite = z.number().finite();
const hash = z.string().regex(/^[a-f0-9]{64}$/);
const corners = z.tuple([pointSchema, pointSchema, pointSchema, pointSchema]);

export const deviceSchema = z
  .object({
    label: z.string().min(1).max(120),
    os: z.string().min(1).max(120),
    browser: z.string().min(1).max(120),
    browserVersion: z.string().min(1).max(80),
    cpu: z.string().min(1).max(160),
    memoryGiB: finite.positive().max(4096).nullable(),
    mainThreadHeapPeakBytes: finite.int().nonnegative().nullable(),
    peakMemoryMethod: z.string().min(1).max(300),
  })
  .strict();

export const referenceSchema = z
  .discriminatedUnion("kind", [
    z.object({ kind: z.enum(["no-board", "partial"]) }).strict(),
    z
      .object({
        kind: z.literal("board"),
        corners,
        orientation: orientationSchema,
        labels: z.array(labelSchema).length(64),
      })
      .strict(),
  ])
  .superRefine((reference, context) => {
    if (reference.kind !== "board") return;
    const candidate = {
      id: "reference",
      corners: reference.corners,
      geometrySource: "manual",
      squares: reference.labels.map((label) => ({
        label,
        probabilities: null,
        uncertain: true,
      })),
      orientation: reference.orientation,
      orientationEvidence: "user",
      warnings: [],
    };
    const parsed = boardSchema.safeParse(candidate);
    if (!parsed.success)
      context.addIssue({
        code: "custom",
        message: "Invalid reference corners",
      });
  });

export type Reference = z.infer<typeof referenceSchema>;

const modelRunSchema = z
  .object({
    result: resultSchema,
    cold: z.boolean(),
    totalMs: finite.nonnegative(),
    workerHeapBytes: finite.int().nonnegative().nullable(),
  })
  .strict();

export const privateEntrySchema = z
  .object({
    index: z.number().int().nonnegative().max(19),
    image: imageSchema.extend({
      bytes: z.number().int().positive(),
      sha256: hash,
    }),
    mode: z.enum(["automatic", "manual-grid"]),
    reference: referenceSchema,
    results: z.object({ v2: modelRunSchema, fenshot: modelRunSchema }).strict(),
  })
  .strict()
  .superRefine((entry, context) => {
    for (const run of Object.values(entry.results))
      if (
        run.result.image.width !== entry.image.width ||
        run.result.image.height !== entry.image.height
      )
        context.addIssue({ code: "custom", message: "Result image changed" });
  });

export const privateEvaluationSchema = z
  .object({
    schema: z.literal(PRIVATE_EVALUATION_VERSION),
    sensitive: z.literal(true),
    warning: z.literal(
      "SENSITIVE LOCAL EVIDENCE: contains image hashes, geometry, positions, and raw model results; do not publish",
    ),
    createdAt: z.string().datetime(),
    device: deviceSchema,
    models: z.object({ v2: identitySchema, fenshot: identitySchema }).strict(),
    entries: z.array(privateEntrySchema).max(40),
  })
  .strict()
  .superRefine((session, context) => {
    const keys = new Set<string>();
    for (const entry of session.entries) {
      const key = `${entry.index}:${entry.mode}`;
      if (keys.has(key))
        context.addIssue({
          code: "custom",
          message: "Duplicate image/mode evidence",
        });
      keys.add(key);
    }
    if (new Set(session.entries.map((entry) => entry.index)).size > 20)
      context.addIssue({
        code: "custom",
        message: "Evaluation exceeds 20 inputs",
      });
  });

export type PrivateEvaluation = z.infer<typeof privateEvaluationSchema>;

export type Comparison = {
  expectedBoards: number;
  returnedBoards: number;
  missedBoards: number;
  falseBoards: number;
  refinementFailures: number;
  cornerDisplacementSquareWidths: number | null;
  exactBoard: boolean;
  occupiedEmptyErrors: number;
  pieceClassErrors: number;
  colorErrors: number;
  wrongSquares: number;
  confidentWrongSquares: number;
};

function averageSquareWidth(
  boardCorners: Reference & { kind: "board" },
): number {
  const lengths = boardCorners.corners.map((point, index) => {
    const next = boardCorners.corners[(index + 1) % 4]!;
    return Math.hypot(next.x - point.x, next.y - point.y);
  });
  return lengths.reduce((sum, value) => sum + value, 0) / lengths.length / 8;
}

function cornerError(
  reference: Reference & { kind: "board" },
  board: Board,
): number {
  const pixels =
    reference.corners.reduce((sum, point, index) => {
      const candidate = board.corners[index]!;
      return sum + Math.hypot(candidate.x - point.x, candidate.y - point.y);
    }, 0) / 4;
  return pixels / Math.max(1e-9, averageSquareWidth(reference));
}

function isWhite(label: Label): boolean | null {
  if (label === "empty") return null;
  return label === label.toUpperCase();
}

export function compareResult(
  reference: Reference,
  result: Result,
  confidentThreshold = 0.99,
): Comparison {
  referenceSchema.parse(reference);
  resultSchema.parse(result);
  if (
    !Number.isFinite(confidentThreshold) ||
    confidentThreshold < 0 ||
    confidentThreshold > 1
  )
    throw new Error("Invalid confident threshold");
  if (reference.kind !== "board")
    return {
      expectedBoards: 0,
      returnedBoards: result.boards.length,
      missedBoards: 0,
      falseBoards: result.boards.length,
      refinementFailures: result.warnings.some((warning) =>
        warning.toLowerCase().includes("refinement"),
      )
        ? 1
        : 0,
      cornerDisplacementSquareWidths: null,
      exactBoard: result.boards.length === 0,
      occupiedEmptyErrors: 0,
      pieceClassErrors: 0,
      colorErrors: 0,
      wrongSquares: 0,
      confidentWrongSquares: 0,
    };
  const board = result.boards.length
    ? result.boards.reduce((best, candidate) =>
        cornerError(reference, candidate) < cornerError(reference, best)
          ? candidate
          : best,
      )
    : null;
  if (!board)
    return {
      expectedBoards: 1,
      returnedBoards: 0,
      missedBoards: 1,
      falseBoards: 0,
      refinementFailures: result.warnings.some((warning) =>
        warning.toLowerCase().includes("refinement"),
      )
        ? 1
        : 0,
      cornerDisplacementSquareWidths: null,
      exactBoard: false,
      occupiedEmptyErrors: 0,
      pieceClassErrors: 0,
      colorErrors: 0,
      wrongSquares: 64,
      confidentWrongSquares: 0,
    };
  let occupiedEmptyErrors = 0,
    pieceClassErrors = 0,
    colorErrors = 0,
    wrongSquares = 0,
    confidentWrongSquares = 0;
  for (let index = 0; index < 64; index++) {
    const expected = reference.labels[index]!;
    const square = board.squares[index]!;
    const actual = square.label;
    if (actual === expected) continue;
    wrongSquares++;
    const expectedOccupied = expected !== "empty";
    const actualOccupied = actual !== null && actual !== "empty";
    if (expectedOccupied !== actualOccupied) occupiedEmptyErrors++;
    if (expected !== "empty" && actual !== null && actual !== "empty") {
      if (expected.toLowerCase() !== actual.toLowerCase()) pieceClassErrors++;
      if (isWhite(expected) !== isWhite(actual)) colorErrors++;
    }
    if (
      square.probabilities &&
      Math.max(...square.probabilities) >= confidentThreshold
    )
      confidentWrongSquares++;
  }
  const displacement = cornerError(reference, board);
  return {
    expectedBoards: 1,
    returnedBoards: result.boards.length,
    missedBoards: 0,
    falseBoards: Math.max(0, result.boards.length - 1),
    refinementFailures: 0,
    cornerDisplacementSquareWidths: displacement,
    exactBoard: wrongSquares === 0 && displacement <= 0.25,
    occupiedEmptyErrors,
    pieceClassErrors,
    colorErrors,
    wrongSquares,
    confidentWrongSquares,
  };
}

const countsSchema = z
  .object({
    inputs: z.number().int().nonnegative(),
    automaticInputs: z.number().int().nonnegative(),
    manualGridInputs: z.number().int().nonnegative(),
    expectedBoards: z.number().int().nonnegative(),
    returnedBoards: z.number().int().nonnegative(),
    missedBoards: z.number().int().nonnegative(),
    falseBoards: z.number().int().nonnegative(),
    refinementFailures: z.number().int().nonnegative(),
    exactBoards: z.number().int().nonnegative(),
    occupiedEmptyErrors: z.number().int().nonnegative(),
    pieceClassErrors: z.number().int().nonnegative(),
    colorErrors: z.number().int().nonnegative(),
    wrongSquares: z.number().int().nonnegative(),
    confidentWrongSquares: z.number().int().nonnegative(),
    cornerDisplacementCount: z.number().int().nonnegative(),
    cornerDisplacementTotalSquareWidths: finite.nonnegative(),
    coldMs: z.array(finite.nonnegative()).max(40),
    warmMs: z.array(finite.nonnegative()).max(40),
  })
  .strict();

export const evaluationSummarySchema = z
  .object({
    schema: z.literal(EVALUATION_SUMMARY_VERSION),
    createdAt: z.string().datetime(),
    device: deviceSchema,
    models: z
      .object({
        v2: identitySchema.pick({ name: true, version: true }),
        fenshot: identitySchema.pick({ name: true, version: true }),
      })
      .strict(),
    counts: z.object({ v2: countsSchema, fenshot: countsSchema }).strict(),
  })
  .strict();

export type EvaluationSummary = z.infer<typeof evaluationSummarySchema>;

export function summarizeEvaluation(
  session: PrivateEvaluation,
): EvaluationSummary {
  const parsed = privateEvaluationSchema.parse(session);
  const counts = (model: "v2" | "fenshot") => {
    const comparisons = parsed.entries.map((entry) =>
      compareResult(entry.reference, entry.results[model].result),
    );
    const sum = (key: keyof Comparison) =>
      comparisons.reduce((total, value) => {
        const item = value[key];
        return total + (typeof item === "boolean" ? Number(item) : (item ?? 0));
      }, 0);
    return {
      inputs: parsed.entries.length,
      automaticInputs: parsed.entries.filter(
        (entry) => entry.mode === "automatic",
      ).length,
      manualGridInputs: parsed.entries.filter(
        (entry) => entry.mode === "manual-grid",
      ).length,
      expectedBoards: sum("expectedBoards"),
      returnedBoards: sum("returnedBoards"),
      missedBoards: sum("missedBoards"),
      falseBoards: sum("falseBoards"),
      refinementFailures: sum("refinementFailures"),
      exactBoards: sum("exactBoard"),
      occupiedEmptyErrors: sum("occupiedEmptyErrors"),
      pieceClassErrors: sum("pieceClassErrors"),
      colorErrors: sum("colorErrors"),
      wrongSquares: sum("wrongSquares"),
      confidentWrongSquares: sum("confidentWrongSquares"),
      cornerDisplacementCount: comparisons.filter(
        (comparison) => comparison.cornerDisplacementSquareWidths !== null,
      ).length,
      cornerDisplacementTotalSquareWidths: comparisons.reduce(
        (total, comparison) =>
          total + (comparison.cornerDisplacementSquareWidths ?? 0),
        0,
      ),
      coldMs: parsed.entries
        .filter((entry) => entry.results[model].cold)
        .map((entry) => entry.results[model].totalMs),
      warmMs: parsed.entries
        .filter((entry) => !entry.results[model].cold)
        .map((entry) => entry.results[model].totalMs),
    };
  };
  return evaluationSummarySchema.parse({
    schema: EVALUATION_SUMMARY_VERSION,
    createdAt: parsed.createdAt,
    device: parsed.device,
    models: {
      v2: { name: parsed.models.v2.name, version: parsed.models.v2.version },
      fenshot: {
        name: parsed.models.fenshot.name,
        version: parsed.models.fenshot.version,
      },
    },
    counts: { v2: counts("v2"), fenshot: counts("fenshot") },
  });
}
