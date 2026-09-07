import assert from "node:assert/strict";
import test from "node:test";
import type { InferenceSession } from "onnxruntime-web/wasm";
import {
  PROPOSAL_LABELS,
  PROPOSAL_VERSION,
  PROVIDER_MANIFEST_VERSION,
  composeProviders,
  createClassicalGridLocalizationProvider,
  createFENShotLocalizationProvider,
  createOnnxLabelProvider,
  createOnnxLocalizationProvider,
  fenshotSquares,
  labelResultSchema,
  localizationResultSchema,
  providerManifestSchema,
  type LabelProvider,
  type LocalizationProvider,
} from "../src/proposals/index.ts";

const hash = "a".repeat(64);
const manifest = (
  id: string,
  capability: "localization" | "labels",
  runtime: "classical-grid-v1" | "fenshot-labeler-v1",
) => ({
  schema: PROVIDER_MANIFEST_VERSION,
  id,
  capability,
  runtime,
  model: { name: "m", version: "1", sha256: hash },
  preprocessing: "x",
  artifact: null,
  limits: {
    max_pixels: 16_000_000,
    max_dimension: 8192,
    timeout_ms: 30_000,
    max_candidates: 16,
  },
});
function raster(width: number, height: number): Uint8ClampedArray {
  const rgba = new Uint8ClampedArray(width * height * 4);
  rgba.fill(255);
  return rgba;
}
function checker(
  rgba: Uint8ClampedArray,
  width: number,
  x0: number,
  y0: number,
  tile = 12,
): void {
  for (let y = y0; y < y0 + tile * 8; y++)
    for (let x = x0; x < x0 + tile * 8; x++) {
      const value =
        (Math.floor((x - x0) / tile) + Math.floor((y - y0) / tile)) % 2
          ? 210
          : 25;
      const index = (y * width + x) * 4;
      rgba[index] = rgba[index + 1] = rgba[index + 2] = value;
      rgba[index + 3] = 255;
    }
}
const input = (rgba: Uint8ClampedArray, width: number, height: number) => ({
  schema: PROPOSAL_VERSION,
  requestId: "request",
  image: { width, height },
  rgba,
});

test("strict provider contracts reject unknown runtimes and invalid geometry", () => {
  assert.throws(() =>
    providerManifestSchema.parse({
      ...manifest("bad/id", "labels", "fenshot-labeler-v1"),
    }),
  );
  assert.throws(() =>
    localizationResultSchema.parse({
      schema: PROPOSAL_VERSION,
      requestId: "r",
      image: { width: 10, height: 10 },
      provider: manifest("classical", "localization", "classical-grid-v1"),
      candidates: [
        {
          id: "x",
          providerRuntimeId: "classical",
          score: 1,
          corners: [
            { x: 0, y: 0 },
            { x: 10, y: 0 },
            { x: 0, y: 10 },
            { x: 10, y: 10 },
          ],
        },
      ],
      warnings: [],
    }),
  );
  assert.throws(() =>
    labelResultSchema.parse({
      schema: PROPOSAL_VERSION,
      requestId: "r",
      image: { width: 10, height: 10 },
      provider: manifest("labels", "labels", "fenshot-labeler-v1"),
      candidateId: "x",
      squares: [],
      warnings: [],
    }),
  );
});

test("classical localizer is deterministic, bounded, and finds separate synthetic grids", async () => {
  const width = 260,
    height = 130,
    rgba = raster(width, height);
  checker(rgba, width, 15, 10);
  checker(rgba, width, 145, 22);
  const provider = createClassicalGridLocalizationProvider();
  const first = await provider.localize(input(rgba, width, height));
  const second = await provider.localize(input(rgba, width, height));
  assert.deepEqual(first.candidates, second.candidates);
  assert.equal(first.candidates.length, 2);
  assert.ok(first.candidates.length <= 16);
  assert.ok(
    first.candidates.every(
      (candidate) =>
        candidate.corners[2].x <= width && candidate.corners[2].y <= height,
    ),
  );
  const blank = await provider.localize(
    input(raster(width, height), width, height),
  );
  assert.deepEqual(blank.candidates, []);
});

test("FENShot class ordering maps A1-origin output to image-relative row order", () => {
  const raw = new Float32Array(64 * 13);
  for (let tile = 0; tile < 64; tile++) raw[tile * 13 + 1] = 1; // FENShot K class
  const squares = fenshotSquares(raw);
  assert.equal(squares.length, 64);
  assert.equal(squares[56]!.label, "K"); // A1 becomes first file of bottom image row.
  assert.equal(squares[0]!.label, "K");
  assert.deepEqual(
    squares[0]!.probabilities,
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
  );
});

test("FENShot localization downsamples large pages and restores source geometry", async () => {
  const width = 2400;
  const height = 1800;
  const rgba = raster(width, height);
  checker(rgba, width, 600, 300, 100);
  const result = await createFENShotLocalizationProvider().localize(
    input(rgba, width, height),
  );
  assert.equal(result.candidates.length, 1);
  const corners = result.candidates[0]!.corners;
  assert.ok(corners[0].x > 500 && corners[0].x < 800);
  assert.ok(corners[0].y > 200 && corners[0].y < 400);
  assert.ok(corners[2].x > 1300 && corners[2].x < 1600);
  assert.ok(corners[2].y > 1000 && corners[2].y < 1300);
});

test("composition preserves localizer candidates and labels each candidate", async () => {
  const localizationManifest = providerManifestSchema.parse(
    manifest("classical", "localization", "classical-grid-v1"),
  );
  const labelManifest = providerManifestSchema.parse(
    manifest("labels", "labels", "fenshot-labeler-v1"),
  );
  const candidate = {
    id: "board",
    providerRuntimeId: localizationManifest.id,
    score: 1,
    corners: [
      { x: 0, y: 0 },
      { x: 8, y: 0 },
      { x: 8, y: 8 },
      { x: 0, y: 8 },
    ],
  } as const;
  const localizer: LocalizationProvider = {
    manifest: localizationManifest,
    async localize(value) {
      return localizationResultSchema.parse({
        schema: PROPOSAL_VERSION,
        requestId: value.requestId,
        image: value.image,
        provider: localizationManifest,
        candidates: [candidate],
        warnings: [],
      });
    },
  };
  const labeler: LabelProvider = {
    manifest: labelManifest,
    async label(value, board) {
      return labelResultSchema.parse({
        schema: PROPOSAL_VERSION,
        requestId: value.requestId,
        image: value.image,
        provider: labelManifest,
        candidateId: board.id,
        squares: Array.from({ length: 64 }, () => ({
          label: null,
          probabilities: null,
          uncertain: true,
        })),
        warnings: [],
      });
    },
  };
  const result = await composeProviders(
    localizer,
    labeler,
  )(input(raster(8, 8), 8, 8));
  assert.equal(result.localization.candidates[0]!.id, "board");
  assert.equal(result.labels[0]!.candidateId, "board");
});

const refinement = {
  id: "nine-line-grid-refiner-v1",
  implementationSha256: "b".repeat(64),
  regionExpansion: 0.12,
  outputSize: 768,
  classifierTileSize: 96,
  maxCandidates: 4,
} as const;
const onnxManifest = (capability: "localization" | "labels") =>
  providerManifestSchema.parse({
    schema: PROVIDER_MANIFEST_VERSION,
    id: `onnx-${capability}`,
    capability,
    runtime:
      capability === "localization"
        ? "chess-ocr-onnx-localizer-v1"
        : "chess-ocr-onnx-labeler-v1",
    model: { name: "v2", version: "frozen", sha256: hash },
    preprocessing:
      capability === "localization"
        ? "yolox-rgb-imagenet-v2"
        : "nine-line-grid-refiner-v1/rgb768-bilinear/rgb96-imagenet-v1",
    artifact: { path: "work/model.onnx", sha256: hash },
    configuration:
      capability === "localization"
        ? {
            kind: "chess-ocr-onnx-localizer/1",
            input: "images",
            output: "predictions",
            inputShape: [1, 3, 416, 416],
            proposalScoreThreshold: 0.01,
            calibratedAcceptanceThreshold: 1,
            nmsIou: 0.65,
            refinement,
          }
        : {
            kind: "chess-ocr-onnx-labeler/1",
            input: "tiles",
            output: "logits",
            inputShape: ["squares", 3, 96, 96],
            outputShape: ["squares", 13],
            labels: [...PROPOSAL_LABELS],
            refinement,
          },
    limits: {
      max_pixels: 16_000_000,
      max_dimension: 8192,
      timeout_ms: 30_000,
      max_candidates: 4,
    },
  });

test("ONNX provider contracts reject missing configuration and preprocessing mismatch", () => {
  const valid = onnxManifest("localization");
  assert.throws(
    () => providerManifestSchema.parse({ ...valid, configuration: undefined }),
    /configuration is required/,
  );
  assert.throws(
    () => providerManifestSchema.parse({ ...valid, preprocessing: "wrong" }),
    /detector preprocessing/,
  );
});

test("ONNX localizer runs shared preprocessing and returns no board for empty predictions", async () => {
  const fake = {
    async run() {
      return {
        predictions: { data: new Float32Array(3549 * 6), dispose() {} },
      };
    },
  } as unknown as InferenceSession;
  const provider = createOnnxLocalizationProvider(
    fake,
    onnxManifest("localization"),
  );
  const localized = await provider.localize(input(raster(80, 60), 80, 60));
  assert.deepEqual(localized.candidates, []);
});

test("ONNX labeler rectifies a manual quadrilateral and keeps every class uncertain", async () => {
  const fake = {
    async run() {
      const logits = new Float32Array(64 * 13);
      for (let square = 0; square < 64; square++) logits[square * 13 + 1] = 9;
      return { logits: { data: logits, dispose() {} } };
    },
  } as unknown as InferenceSession;
  const provider = createOnnxLabelProvider(fake, onnxManifest("labels"));
  const labeled = await provider.label(input(raster(100, 100), 100, 100), {
    id: "manual",
    providerRuntimeId: "manual-grid",
    score: 1,
    corners: [
      { x: 5, y: 5 },
      { x: 94, y: 7 },
      { x: 92, y: 94 },
      { x: 7, y: 92 },
    ],
  });
  assert.equal(labeled.squares.length, 64);
  assert.ok(
    labeled.squares.every((square) => square.label === "P" && square.uncertain),
  );
});
