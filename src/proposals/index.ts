import { Tensor, type InferenceSession } from "onnxruntime-web/wasm";
import { z } from "zod";
import {
  classProbabilities,
  classifierTiles,
  detectorRasterFromRgba,
  sourceDetections,
} from "../candidate-runtime.ts";
import {
  deduplicateGrids,
  findInnerGrid,
  rectifyGrid,
  type GridCorners,
} from "../grid.ts";

/** Stable, deliberately small contract for locally registered proposal adapters. */
export const PROPOSAL_VERSION = "chess-ocr-dataset-proposal/1" as const;
export const PROVIDER_MANIFEST_VERSION =
  "chess-ocr-provider-manifest/1" as const;
export const PROPOSAL_LABELS = [
  "empty",
  "P",
  "N",
  "B",
  "R",
  "Q",
  "K",
  "p",
  "n",
  "b",
  "r",
  "q",
  "k",
] as const;
const finite = z.number().finite();
const runtimeId = z.string().regex(/^[a-z][a-z0-9-]{0,63}$/);
const pointSchema = z
  .object({ x: finite.nonnegative(), y: finite.nonnegative() })
  .strict();
const imageSchema = z
  .object({
    width: finite.int().positive().max(8192),
    height: finite.int().positive().max(8192),
  })
  .strict()
  .refine(
    (image) => image.width * image.height <= 16_000_000,
    "Image exceeds pixel limit",
  );
const cornersSchema = z.tuple([
  pointSchema,
  pointSchema,
  pointSchema,
  pointSchema,
]);
const identitySchema = z
  .object({
    name: z.string().min(1).max(120),
    version: z.string().min(1).max(120),
    sha256: z.string().regex(/^[a-f0-9]{64}$/),
  })
  .strict();

const onnxRefinementSchema = z
  .object({
    id: z.literal("nine-line-grid-refiner-v1"),
    implementationSha256: z.string().regex(/^[a-f0-9]{64}$/),
    regionExpansion: finite.min(0).max(0.5),
    outputSize: z.literal(768),
    classifierTileSize: z.literal(96),
    maxCandidates: finite.int().positive().max(16),
  })
  .strict();
const providerConfigurationSchema = z
  .discriminatedUnion("kind", [
    z
      .object({
        kind: z.literal("chess-ocr-onnx-localizer/1"),
        input: z.string().min(1).max(80),
        output: z.string().min(1).max(80),
        inputShape: z.tuple([
          z.literal(1),
          z.literal(3),
          z.literal(416),
          z.literal(416),
        ]),
        proposalScoreThreshold: finite.min(0.001).max(1),
        calibratedAcceptanceThreshold: finite.min(0.001).max(1),
        nmsIou: finite.min(0).max(1),
        refinement: onnxRefinementSchema,
      })
      .strict(),
    z
      .object({
        kind: z.literal("chess-ocr-onnx-labeler/1"),
        input: z.string().min(1).max(80),
        output: z.string().min(1).max(80),
        inputShape: z.tuple([
          z.literal("squares"),
          z.literal(3),
          z.literal(96),
          z.literal(96),
        ]),
        outputShape: z.tuple([z.literal("squares"), z.literal(13)]),
        labels: z.array(z.enum(PROPOSAL_LABELS)).length(13),
        refinement: onnxRefinementSchema,
      })
      .strict(),
  ])
  .nullable();

export const providerManifestSchema = z
  .object({
    schema: z.literal(PROVIDER_MANIFEST_VERSION),
    id: runtimeId,
    capability: z.enum(["localization", "labels"]),
    runtime: z.enum([
      "fenshot-localizer-v1",
      "fenshot-labeler-v1",
      "fenshot-rectified-labeler-v1",
      "classical-grid-v1",
      "chess-ocr-onnx-localizer-v1",
      "chess-ocr-onnx-labeler-v1",
    ]),
    model: identitySchema,
    preprocessing: z.string().min(1).max(120),
    artifact: z
      .object({
        path: z.string().min(1).max(300),
        sha256: z.string().regex(/^[a-f0-9]{64}$/),
      })
      .strict()
      .nullable(),
    configuration: providerConfigurationSchema.optional(),
    limits: z
      .object({
        max_pixels: finite.int().positive().max(16_000_000),
        max_dimension: finite.int().positive().max(8192),
        timeout_ms: finite.int().positive().max(120_000),
        max_candidates: finite.int().positive().max(16),
      })
      .strict(),
  })
  .strict()
  .superRefine((manifest, context) => {
    if (
      manifest.runtime === "chess-ocr-onnx-localizer-v1" &&
      manifest.configuration?.kind !== "chess-ocr-onnx-localizer/1"
    )
      context.addIssue({
        code: "custom",
        message: "ONNX localizer configuration is required",
      });
    if (
      manifest.runtime === "chess-ocr-onnx-localizer-v1" &&
      manifest.preprocessing !== "yolox-rgb-imagenet-v2" &&
      manifest.preprocessing !== "legacy-bgr-div255-v1"
    )
      context.addIssue({
        code: "custom",
        message: "Unsupported ONNX detector preprocessing",
      });
    if (
      manifest.runtime === "chess-ocr-onnx-labeler-v1" &&
      manifest.configuration?.kind !== "chess-ocr-onnx-labeler/1"
    )
      context.addIssue({
        code: "custom",
        message: "ONNX labeler configuration is required",
      });
    if (
      manifest.runtime === "chess-ocr-onnx-labeler-v1" &&
      manifest.preprocessing !==
        "nine-line-grid-refiner-v1/rgb768-bilinear/rgb96-imagenet-v1"
    )
      context.addIssue({
        code: "custom",
        message: "Unsupported ONNX classifier preprocessing",
      });
    if (
      manifest.configuration?.kind === "chess-ocr-onnx-localizer/1" &&
      manifest.configuration.proposalScoreThreshold >
        manifest.configuration.calibratedAcceptanceThreshold
    )
      context.addIssue({
        code: "custom",
        message: "ONNX proposal threshold exceeds calibrated acceptance",
      });
    if (
      manifest.configuration?.kind === "chess-ocr-onnx-labeler/1" &&
      !manifest.configuration.labels.every(
        (label, index) => label === PROPOSAL_LABELS[index],
      )
    )
      context.addIssue({
        code: "custom",
        message: "ONNX label order mismatch",
      });
  });
export type ProviderManifest = z.infer<typeof providerManifestSchema>;

export const proposalInputSchema = z
  .object({
    schema: z.literal(PROPOSAL_VERSION),
    requestId: z.string().min(1).max(100),
    image: imageSchema,
  })
  .strict();
export type ProposalInput = z.infer<typeof proposalInputSchema>;
export type DecodedInput = ProposalInput & { rgba: Uint8ClampedArray };
export const candidateSchema = z
  .object({
    id: z.string().min(1).max(100),
    corners: cornersSchema,
    score: finite.min(0).max(1),
    providerRuntimeId: runtimeId,
  })
  .strict();
export type BoardCandidate = z.infer<typeof candidateSchema>;
export const localizationResultSchema = z
  .object({
    schema: z.literal(PROPOSAL_VERSION),
    requestId: z.string().min(1).max(100),
    image: imageSchema,
    provider: providerManifestSchema,
    candidates: z.array(candidateSchema).max(16),
    warnings: z.array(z.string().max(300)).max(30),
  })
  .strict()
  .superRefine((result, ctx) => {
    for (const candidate of result.candidates)
      validateCorners(candidate.corners, result.image, ctx);
  });
export type LocalizationResult = z.infer<typeof localizationResultSchema>;
const squareSchema = z
  .object({
    label: z.enum(PROPOSAL_LABELS).nullable(),
    probabilities: z.array(finite.min(0).max(1)).length(13).nullable(),
    uncertain: z.boolean(),
  })
  .strict()
  .superRefine((square, ctx) => {
    if (
      square.probabilities &&
      Math.abs(
        square.probabilities.reduce((sum, value) => sum + value, 0) - 1,
      ) > 1e-4
    )
      ctx.addIssue({
        code: "custom",
        message: "Probabilities must sum to one",
      });
    if (!square.uncertain && (!square.label || !square.probabilities))
      ctx.addIssue({
        code: "custom",
        message: "Missing evidence must remain uncertain",
      });
  });
export const labelResultSchema = z
  .object({
    schema: z.literal(PROPOSAL_VERSION),
    requestId: z.string().min(1).max(100),
    image: imageSchema,
    provider: providerManifestSchema,
    candidateId: z.string().min(1).max(100),
    squares: z.array(squareSchema).length(64),
    warnings: z.array(z.string().max(300)).max(30),
  })
  .strict();
export type LabelResult = z.infer<typeof labelResultSchema>;

function validateCorners(
  corners: z.infer<typeof cornersSchema>,
  image: z.infer<typeof imageSchema>,
  ctx: z.RefinementCtx,
): void {
  for (const point of corners)
    if (point.x > image.width || point.y > image.height)
      ctx.addIssue({
        code: "custom",
        message: "Geometry is outside the image",
      });
  for (let index = 0; index < 4; index++) {
    const a = corners[index]!;
    const b = corners[(index + 1) % 4]!;
    const c = corners[(index + 2) % 4]!;
    if ((b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x) <= 0)
      ctx.addIssue({
        code: "custom",
        message: "Corners must be convex and clockwise",
      });
  }
}
function assertRaster(input: DecodedInput): void {
  proposalInputSchema.parse({
    schema: input.schema,
    requestId: input.requestId,
    image: input.image,
  });
  if (input.rgba.length !== input.image.width * input.image.height * 4)
    throw new Error("Invalid decoded RGBA length");
}

export interface LocalizationProvider {
  readonly manifest: ProviderManifest;
  localize(input: DecodedInput): Promise<LocalizationResult>;
}
export interface LabelProvider {
  readonly manifest: ProviderManifest;
  label(input: DecodedInput, candidate: BoardCandidate): Promise<LabelResult>;
}

const codeHash = "0".repeat(64);
const defaultLimits = {
  max_pixels: 16_000_000,
  max_dimension: 8192,
  timeout_ms: 30_000,
  max_candidates: 16,
};
const fenshotLocalizationManifest = providerManifestSchema.parse({
  schema: PROVIDER_MANIFEST_VERSION,
  id: "fenshot-localizer",
  capability: "localization",
  runtime: "fenshot-localizer-v1",
  model: { name: "@scoriiu/fenshot", version: "0.1.4", sha256: codeHash },
  preprocessing: "fenshot-0.1.4/rgba-gray/1",
  artifact: null,
  limits: defaultLimits,
});
export function createFENShotLocalizationProvider(
  manifest: ProviderManifest = fenshotLocalizationManifest,
): LocalizationProvider {
  if (
    manifest.runtime !== "fenshot-localizer-v1" ||
    manifest.capability !== "localization"
  )
    throw new Error("Invalid FENShot localization manifest");
  return {
    manifest,
    async localize(input) {
      assertRaster(input);
      const { findChessboardCorners, rgbaToGray } = await loadFenshot();
      const found = detectFenshotBox(
        rgbaToGray(input.rgba, input.image.width, input.image.height),
        findChessboardCorners,
      );
      const candidates = found
        ? [
            {
              id: "fenshot-0",
              score: 0.5,
              providerRuntimeId: manifest.id,
              corners: cornersFromBox(found),
            },
          ]
        : [];
      const distinct = deduplicateGrids(candidates);
      return localizationResultSchema.parse({
        schema: PROPOSAL_VERSION,
        requestId: input.requestId,
        image: input.image,
        provider: manifest,
        candidates: distinct,
        warnings: found
          ? ["FENShot returns at most one axis-aligned candidate."]
          : ["No FENShot grid candidate."],
      });
    },
  };
}

export function createFENShotLabelProvider(
  session: InferenceSession,
  model: z.infer<typeof identitySchema>,
  supplied?: ProviderManifest,
): LabelProvider {
  const manifest =
    supplied ??
    providerManifestSchema.parse({
      schema: PROVIDER_MANIFEST_VERSION,
      id: "fenshot-labeler",
      capability: "labels",
      runtime: "fenshot-labeler-v1",
      model,
      preprocessing: "fenshot-0.1.4/rgba-gray-bilinear-256/1",
      artifact: null,
      limits: defaultLimits,
    });
  if (
    manifest.runtime !== "fenshot-labeler-v1" ||
    manifest.capability !== "labels" ||
    manifest.model.sha256 !== model.sha256
  )
    throw new Error("Invalid FENShot label manifest");
  return {
    manifest,
    async label(input, candidate) {
      assertRaster(input);
      candidateSchema.parse(candidate);
      const { extractTiles, rgbaToGray } = await loadFenshot();
      const gray = rgbaToGray(
        input.rgba,
        input.image.width,
        input.image.height,
      );
      const box = {
        x0: candidate.corners[0].x,
        y0: candidate.corners[0].y,
        x1: candidate.corners[2].x,
        y1: candidate.corners[2].y,
      };
      const tensor = new Tensor("float32", extractTiles(gray, box), [64, 1024]);
      try {
        const output = await session.run({ tiles: tensor });
        try {
          const probabilities = output.probs?.data;
          if (!probabilities || probabilities.length !== 64 * 13)
            throw new Error("Invalid FENShot output shape");
          const values = probabilities as ArrayLike<unknown>;
          const numeric = Float32Array.from(
            Array.from({ length: values.length }, (_, index) => {
              const value = values[index];
              if (typeof value !== "number")
                throw new Error("Invalid FENShot probability type");
              return value;
            }),
          );
          return labelResultSchema.parse({
            schema: PROPOSAL_VERSION,
            requestId: input.requestId,
            image: input.image,
            provider: manifest,
            candidateId: candidate.id,
            squares: fenshotSquares(numeric),
            warnings: [
              "Uncalibrated FENShot confidence; inspect every square.",
            ],
          });
        } finally {
          for (const value of Object.values(output)) value.dispose();
        }
      } finally {
        tensor.dispose();
      }
    },
  };
}

/** FENShot labels over the exact shared four-corner RGB768 rectification contract. */
export function createFENShotRectifiedLabelProvider(
  session: InferenceSession,
  model: z.infer<typeof identitySchema>,
  supplied: ProviderManifest,
): LabelProvider {
  const manifest = providerManifestSchema.parse(supplied);
  if (
    manifest.runtime !== "fenshot-rectified-labeler-v1" ||
    manifest.capability !== "labels" ||
    manifest.model.sha256 !== model.sha256
  )
    throw new Error("Invalid rectified FENShot label manifest");
  return {
    manifest,
    async label(input, candidate) {
      assertRaster(input);
      candidateSchema.parse(candidate);
      const source = {
        data: new Uint8Array(
          input.rgba.buffer,
          input.rgba.byteOffset,
          input.rgba.byteLength,
        ),
        width: input.image.width,
        height: input.image.height,
      };
      const rectified = rectifyGrid(
        source,
        candidate.corners as GridCorners,
        4,
        768,
      );
      const { extractTiles, rgbaToGray } = await loadFenshot();
      const gray = rgbaToGray(
        new Uint8ClampedArray(
          rectified.data.buffer,
          rectified.data.byteOffset,
          rectified.data.byteLength,
        ),
        768,
        768,
      );
      const tensor = new Tensor(
        "float32",
        extractTiles(gray, { x0: 0, y0: 0, x1: 768, y1: 768 }),
        [64, 1024],
      );
      try {
        const output = await session.run({ tiles: tensor });
        try {
          const probabilities = output.probs?.data;
          if (!probabilities || probabilities.length !== 64 * 13)
            throw new Error("Invalid FENShot output shape");
          const numeric = Float32Array.from(probabilities as ArrayLike<number>);
          return labelResultSchema.parse({
            schema: PROPOSAL_VERSION,
            requestId: input.requestId,
            image: input.image,
            provider: manifest,
            candidateId: candidate.id,
            squares: fenshotSquares(numeric),
            warnings: [
              "FENShot labels use the reviewer-supplied shared rectification; inspect every square.",
            ],
          });
        } finally {
          for (const value of Object.values(output)) value.dispose();
        }
      } finally {
        tensor.dispose();
      }
    },
  };
}

function onnxConfiguration(
  manifest: ProviderManifest,
  kind: "chess-ocr-onnx-localizer/1",
): Extract<
  NonNullable<ProviderManifest["configuration"]>,
  { kind: typeof kind }
>;
function onnxConfiguration(
  manifest: ProviderManifest,
  kind: "chess-ocr-onnx-labeler/1",
): Extract<
  NonNullable<ProviderManifest["configuration"]>,
  { kind: typeof kind }
>;
function onnxConfiguration(
  manifest: ProviderManifest,
  kind: "chess-ocr-onnx-localizer/1" | "chess-ocr-onnx-labeler/1",
) {
  if (!manifest.artifact || manifest.configuration?.kind !== kind)
    throw new Error("ONNX provider manifest is incomplete");
  return manifest.configuration;
}

function expandedRect(
  box: { x: number; y: number; width: number; height: number },
  width: number,
  height: number,
  fraction: number,
) {
  const padding = Math.max(box.width, box.height) * fraction;
  const x = Math.max(0, box.x - padding);
  const y = Math.max(0, box.y - padding);
  const right = Math.min(width, box.x + box.width + padding);
  const bottom = Math.min(height, box.y + box.height + padding);
  return { x, y, width: right - x, height: bottom - y };
}

/** Shared v2 detector plus deterministic nine-line refinement. */
export function createOnnxLocalizationProvider(
  session: InferenceSession,
  supplied: ProviderManifest,
): LocalizationProvider {
  const manifest = providerManifestSchema.parse(supplied);
  if (
    manifest.runtime !== "chess-ocr-onnx-localizer-v1" ||
    manifest.capability !== "localization"
  )
    throw new Error("Invalid chess-ocr ONNX localization manifest");
  const configuration = onnxConfiguration(
    manifest,
    "chess-ocr-onnx-localizer/1",
  );
  return {
    manifest,
    async localize(input) {
      assertRaster(input);
      const prepared = detectorRasterFromRgba(
        input.rgba,
        input.image.width,
        input.image.height,
        manifest.preprocessing as
          | "legacy-bgr-div255-v1"
          | "yolox-rgb-imagenet-v2",
      );
      const tensor = new Tensor("float32", prepared.input, [1, 3, 416, 416]);
      let detections;
      try {
        const output = await session.run({ [configuration.input]: tensor });
        try {
          const raw = output[configuration.output];
          if (!raw || !(raw.data instanceof Float32Array))
            throw new Error("Invalid ONNX detector output");
          detections = sourceDetections(
            raw.data,
            prepared,
            configuration.proposalScoreThreshold,
            configuration.nmsIou,
            configuration.refinement.maxCandidates,
          );
        } finally {
          for (const value of Object.values(output)) value.dispose();
        }
      } finally {
        tensor.dispose();
      }
      const source = {
        data: new Uint8Array(
          input.rgba.buffer,
          input.rgba.byteOffset,
          input.rgba.byteLength,
        ),
        width: input.image.width,
        height: input.image.height,
      };
      let rejected = 0;
      const candidates: BoardCandidate[] = [];
      for (const detection of detections) {
        const refined = findInnerGrid(
          source,
          expandedRect(
            detection.box,
            input.image.width,
            input.image.height,
            configuration.refinement.regionExpansion,
          ),
        );
        if (!refined.ok) {
          rejected++;
          continue;
        }
        candidates.push({
          id: `onnx-refined-${candidates.length}`,
          corners: refined.corners.map((point) => ({
            ...point,
          })) as BoardCandidate["corners"],
          score: Math.max(0, Math.min(1, detection.score * refined.score)),
          providerRuntimeId: manifest.id,
        });
      }
      return localizationResultSchema.parse({
        schema: PROPOSAL_VERSION,
        requestId: input.requestId,
        image: input.image,
        provider: manifest,
        candidates,
        warnings: [
          "Detector outputs are low-threshold region proposals; only refined nine-line grids are returned.",
          ...(rejected
            ? [
                `Grid refinement rejected ${rejected} detector proposal${rejected === 1 ? "" : "s"}.`,
              ]
            : []),
        ],
      });
    },
  };
}

/** Shared perspective rectification and MobileNetV3 square classifier. */
export function createOnnxLabelProvider(
  session: InferenceSession,
  supplied: ProviderManifest,
): LabelProvider {
  const manifest = providerManifestSchema.parse(supplied);
  if (
    manifest.runtime !== "chess-ocr-onnx-labeler-v1" ||
    manifest.capability !== "labels"
  )
    throw new Error("Invalid chess-ocr ONNX label manifest");
  const configuration = onnxConfiguration(manifest, "chess-ocr-onnx-labeler/1");
  return {
    manifest,
    async label(input, candidate) {
      assertRaster(input);
      candidateSchema.parse(candidate);
      const source = {
        data: new Uint8Array(
          input.rgba.buffer,
          input.rgba.byteOffset,
          input.rgba.byteLength,
        ),
        width: input.image.width,
        height: input.image.height,
      };
      const rectified = rectifyGrid(
        source,
        candidate.corners as GridCorners,
        3,
        configuration.refinement.outputSize,
      );
      const values = classifierTiles(rectified.data);
      const tensor = new Tensor("float32", values, [64, 3, 96, 96]);
      try {
        const output = await session.run({ [configuration.input]: tensor });
        try {
          const raw = output[configuration.output];
          if (!raw || !(raw.data instanceof Float32Array))
            throw new Error("Invalid ONNX classifier output");
          const probabilities = classProbabilities(raw.data, 64);
          return labelResultSchema.parse({
            schema: PROPOSAL_VERSION,
            requestId: input.requestId,
            image: input.image,
            provider: manifest,
            candidateId: candidate.id,
            squares: probabilities.map((values) => {
              const confidence = Math.max(...values);
              return {
                label: PROPOSAL_LABELS[values.indexOf(confidence)]!,
                probabilities: values,
                uncertain: true,
              };
            }),
            warnings: [
              "Synthetic-development-only confidence; inspect all 64 squares.",
            ],
          });
        } finally {
          for (const value of Object.values(output)) value.dispose();
        }
      } finally {
        tensor.dispose();
      }
    },
  };
}

// FENShot emits A1..H8 with 1KQRBNPkqrbnp. Convert to image-row order and this contract's class order.
const FENSHOT_CLASSES = [
  "empty",
  "K",
  "Q",
  "R",
  "B",
  "N",
  "P",
  "k",
  "q",
  "r",
  "b",
  "n",
  "p",
] as const;
export function fenshotSquares(raw: ArrayLike<number>) {
  if (raw.length !== 832) throw new Error("Invalid FENShot output shape");
  return Array.from({ length: 64 }, (_, imageIndex) => {
    const fenshotTile = (7 - Math.floor(imageIndex / 8)) * 8 + (imageIndex % 8);
    const probabilities = PROPOSAL_LABELS.map((label) =>
      Number(raw[fenshotTile * 13 + FENSHOT_CLASSES.indexOf(label)]),
    ).map((value) => {
      if (!Number.isFinite(value) || value < 0 || value > 1)
        throw new Error("Invalid FENShot probability");
      return value;
    });
    const confidence = Math.max(...probabilities);
    const label = PROPOSAL_LABELS[probabilities.indexOf(confidence)]!;
    return { label, probabilities, uncertain: confidence < 0.7 };
  });
}

const classicalManifest = providerManifestSchema.parse({
  schema: PROVIDER_MANIFEST_VERSION,
  id: "classical-grid",
  capability: "localization",
  runtime: "classical-grid-v1",
  model: { name: "chess-ocr-classical-grid", version: "1", sha256: codeHash },
  preprocessing: "gray-edge-grid-v1",
  artifact: null,
  limits: defaultLimits,
});
export function createClassicalGridLocalizationProvider(
  manifest: ProviderManifest = classicalManifest,
): LocalizationProvider {
  if (
    manifest.runtime !== "classical-grid-v1" ||
    manifest.capability !== "localization"
  )
    throw new Error("Invalid classical localization manifest");
  return {
    manifest,
    async localize(input) {
      assertRaster(input);
      const gray = rgbaToGrayLocal(
        input.rgba,
        input.image.width,
        input.image.height,
      );
      const candidates = gridCandidates(
        gray,
        input.image.width,
        input.image.height,
      ).map((candidate, index) => ({
        ...candidate,
        id: `classical-${index}`,
        providerRuntimeId: manifest.id,
      }));
      return localizationResultSchema.parse({
        schema: PROPOSAL_VERSION,
        requestId: input.requestId,
        image: input.image,
        provider: manifest,
        candidates,
        warnings: candidates.length
          ? ["Classical grid evidence is deterministic and requires review."]
          : ["No bounded grid evidence found."],
      });
    },
  };
}

type FenshotApi = {
  rgbaToGray(
    rgba: Uint8ClampedArray,
    width: number,
    height: number,
  ): { data: Float32Array; width: number; height: number };
  findChessboardCorners(gray: {
    data: Float32Array;
    width: number;
    height: number;
  }): { x0: number; y0: number; x1: number; y1: number } | null;
  extractTiles(
    gray: { data: Float32Array; width: number; height: number },
    corners: { x0: number; y0: number; x1: number; y1: number },
  ): Float32Array;
};
async function loadFenshot(): Promise<FenshotApi> {
  // FENShot 0.1.4's compiled package uses extensionless internal ESM imports,
  // which Node does not resolve. Its pinned, reviewed lower-level modules are
  // dependency-free and are loaded directly for the detached proposal worker.
  const entry = new URL(import.meta.resolve("@scoriiu/fenshot"));
  const [detect, tiles] = await Promise.all([
    import(new URL("./detect.js", entry).href),
    import(new URL("./tiles.js", entry).href),
  ]);
  return {
    findChessboardCorners: detect.findChessboardCorners,
    rgbaToGray: tiles.rgbaToGray,
    extractTiles: tiles.extractTiles,
  } as FenshotApi;
}

const FENSHOT_MAX_DETECT_DIM = 1600;

function resizeGray(
  source: { data: Float32Array; width: number; height: number },
  width: number,
  height: number,
): { data: Float32Array; width: number; height: number } {
  const data = new Float32Array(width * height);
  const scaleX = source.width / width;
  const scaleY = source.height / height;
  for (let y = 0; y < height; y++) {
    const sy = (y + 0.5) * scaleY - 0.5;
    const y0 = Math.max(0, Math.floor(sy));
    const y1 = Math.min(source.height - 1, y0 + 1);
    const wy = Math.max(0, Math.min(1, sy - y0));
    for (let x = 0; x < width; x++) {
      const sx = (x + 0.5) * scaleX - 0.5;
      const x0 = Math.max(0, Math.floor(sx));
      const x1 = Math.min(source.width - 1, x0 + 1);
      const wx = Math.max(0, Math.min(1, sx - x0));
      const top =
        source.data[y0 * source.width + x0]! * (1 - wx) +
        source.data[y0 * source.width + x1]! * wx;
      const bottom =
        source.data[y1 * source.width + x0]! * (1 - wx) +
        source.data[y1 * source.width + x1]! * wx;
      data[y * width + x] = top * (1 - wy) + bottom * wy;
    }
  }
  return { data, width, height };
}

function detectFenshotBox(
  gray: { data: Float32Array; width: number; height: number },
  find: FenshotApi["findChessboardCorners"],
): { x0: number; y0: number; x1: number; y1: number } | null {
  const scale = Math.min(
    1,
    FENSHOT_MAX_DETECT_DIM / Math.max(gray.width, gray.height),
  );
  const detection =
    scale < 1
      ? resizeGray(
          gray,
          Math.round(gray.width * scale),
          Math.round(gray.height * scale),
        )
      : gray;
  const found = find(detection);
  if (!found) return null;
  return {
    x0: found.x0 / scale,
    y0: found.y0 / scale,
    x1: found.x1 / scale,
    y1: found.y1 / scale,
  };
}
function rgbaToGrayLocal(
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
): Float32Array {
  const gray = new Float32Array(width * height);
  for (let pixel = 0; pixel < gray.length; pixel++)
    gray[pixel] =
      rgba[pixel * 4]! * 0.299 +
      rgba[pixel * 4 + 1]! * 0.587 +
      rgba[pixel * 4 + 2]! * 0.114;
  return gray;
}

function cornersFromBox(box: {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}): z.infer<typeof cornersSchema> {
  return [
    { x: box.x0, y: box.y0 },
    { x: box.x1, y: box.y0 },
    { x: box.x1, y: box.y1 },
    { x: box.x0, y: box.y1 },
  ];
}
type Span = { start: number; end: number; strength: number };
function gridCandidates(
  data: Float32Array,
  width: number,
  height: number,
): Array<{ corners: z.infer<typeof cornersSchema>; score: number }> {
  const horizontal = new Float64Array(height);
  const vertical = new Float64Array(width);
  for (let y = 1; y < height; y++)
    for (let x = 1; x < width; x++) {
      const current = data[y * width + x]!;
      horizontal[y] =
        horizontal[y]! + Math.abs(current - data[(y - 1) * width + x]!);
      vertical[x] = vertical[x]! + Math.abs(current - data[y * width + x - 1]!);
    }
  const xs = gridSpans(vertical);
  const ys = gridSpans(horizontal);
  const raw: Array<{ corners: z.infer<typeof cornersSchema>; score: number }> =
    [];
  for (const x of xs)
    for (const y of ys) {
      const w = x.end - x.start;
      const h = y.end - y.start;
      const ratio = Math.min(w, h) / Math.max(w, h);
      if (w < 32 || h < 32 || ratio < 0.75) continue;
      raw.push({
        corners: cornersFromBox({
          x0: x.start,
          y0: y.start,
          x1: x.end,
          y1: y.end,
        }),
        score: Math.min(1, (ratio * (x.strength + y.strength)) / 2),
      });
    }
  raw.sort(
    (a, b) =>
      b.score - a.score ||
      a.corners[0].y - b.corners[0].y ||
      a.corners[0].x - b.corners[0].x,
  );
  const kept: typeof raw = [];
  for (const candidate of raw) {
    if (!kept.some((other) => overlap(candidate, other) > 0.5))
      kept.push(candidate);
    if (kept.length === 16) break;
  }
  return kept;
}
function gridSpans(response: Float64Array): Span[] {
  let maximum = 0;
  for (const value of response) maximum = Math.max(maximum, value);
  if (maximum <= 0) return [];
  const peaks: Array<{ position: number; value: number }> = [];
  for (let i = 2; i < response.length - 2; i++)
    if (
      response[i]! >= maximum * 0.25 &&
      response[i]! >= response[i - 1]! &&
      response[i]! > response[i + 1]!
    ) {
      if (peaks.length === 0 || i - peaks[peaks.length - 1]!.position > 2)
        peaks.push({ position: i, value: response[i]! / maximum });
      else if (response[i]! > response[peaks[peaks.length - 1]!.position]!)
        peaks[peaks.length - 1] = {
          position: i,
          value: response[i]! / maximum,
        };
    }
  peaks.sort((a, b) => b.value - a.value || a.position - b.position);
  const bounded = peaks.slice(0, 96).sort((a, b) => a.position - b.position);
  const found: Span[] = [];
  for (let i = 0; i < bounded.length; i++)
    for (let j = i + 8; j < bounded.length; j++) {
      const step = (bounded[j]!.position - bounded[i]!.position) / 8;
      if (step < 4) continue;
      const matches = Array.from({ length: 9 }, (_, k) =>
        bounded.find(
          (peak) =>
            Math.abs(peak.position - (bounded[i]!.position + k * step)) <=
            Math.max(2, step * 0.18),
        ),
      );
      if (matches.every(Boolean))
        found.push({
          start: bounded[i]!.position,
          end: bounded[j]!.position,
          strength: matches.reduce((sum, peak) => sum + peak!.value, 0) / 9,
        });
    }
  found.sort((a, b) => b.strength - a.strength || a.start - b.start);
  return found.slice(0, 24);
}
function overlap(
  a: { corners: z.infer<typeof cornersSchema> },
  b: { corners: z.infer<typeof cornersSchema> },
): number {
  const ax0 = a.corners[0].x;
  const ay0 = a.corners[0].y;
  const ax1 = a.corners[2].x;
  const ay1 = a.corners[2].y;
  const bx0 = b.corners[0].x;
  const by0 = b.corners[0].y;
  const bx1 = b.corners[2].x;
  const by1 = b.corners[2].y;
  const area =
    Math.max(0, Math.min(ax1, bx1) - Math.max(ax0, bx0)) *
    Math.max(0, Math.min(ay1, by1) - Math.max(ay0, by0));
  return area / Math.min((ax1 - ax0) * (ay1 - ay0), (bx1 - bx0) * (by1 - by0));
}

export function composeProviders(
  localizer: LocalizationProvider,
  labeler: LabelProvider,
) {
  return async (input: DecodedInput) => {
    const localization = await localizer.localize(input);
    const labels = await Promise.all(
      localization.candidates.map((candidate) =>
        labeler.label(input, candidate),
      ),
    );
    return { localization, labels };
  };
}
/** Runtime IDs select reviewed code paths; they never name a module, URL, or executable. */
export function supportedManifest(manifest: unknown): ProviderManifest {
  const parsed = providerManifestSchema.parse(manifest);
  const capabilities: Record<
    ProviderManifest["runtime"],
    ProviderManifest["capability"]
  > = {
    "classical-grid-v1": "localization",
    "fenshot-localizer-v1": "localization",
    "fenshot-labeler-v1": "labels",
    "fenshot-rectified-labeler-v1": "labels",
    "chess-ocr-onnx-localizer-v1": "localization",
    "chess-ocr-onnx-labeler-v1": "labels",
  };
  if (capabilities[parsed.runtime] !== parsed.capability)
    throw new Error("Unsupported provider runtime capability");
  return parsed;
}

/** Parse a manifest supplied by a CLI/config boundary before selecting an adapter. */
export function validateManifest(value: unknown): ProviderManifest {
  return supportedManifest(value);
}

const sha256Schema = z.string().regex(/^[a-f0-9]{64}$/);
/** Default built-ins. Hash-bound ONNX manifests are registered explicitly. */
export function builtInManifests(modelSha256: string): ProviderManifest[] {
  const sha256 = sha256Schema.parse(modelSha256);
  return [
    classicalManifest,
    fenshotLocalizationManifest,
    providerManifestSchema.parse({
      schema: PROVIDER_MANIFEST_VERSION,
      id: "fenshot-labeler",
      capability: "labels",
      runtime: "fenshot-labeler-v1",
      model: { name: "@scoriiu/fenshot", version: "0.1.4", sha256 },
      preprocessing: "fenshot-0.1.4/rgba-gray-bilinear-256/1",
      artifact: null,
      limits: defaultLimits,
    }),
    providerManifestSchema.parse({
      schema: PROVIDER_MANIFEST_VERSION,
      id: "fenshot-rectified-labeler",
      capability: "labels",
      runtime: "fenshot-rectified-labeler-v1",
      model: { name: "@scoriiu/fenshot", version: "0.1.4", sha256 },
      preprocessing: "four-corner-rgba768-bilinear/fenshot-gray32-v1",
      artifact: null,
      limits: defaultLimits,
    }),
  ];
}

export interface BuiltInRegistry {
  readonly manifests: readonly ProviderManifest[];
  localization(manifest: unknown): LocalizationProvider;
  labels(manifest: unknown): LabelProvider;
}
export function createBuiltInRegistry(options: {
  /** Backward-compatible FENShot label session. */
  session?: InferenceSession;
  fenshotSession?: InferenceSession;
  onnxLocalizerSession?: InferenceSession;
  onnxLabelerSession?: InferenceSession;
  modelSha256?: string;
  manifests?: ProviderManifest[];
}): BuiltInRegistry {
  const manifests = (
    options.manifests ?? builtInManifests(options.modelSha256 ?? codeHash)
  ).map(validateManifest);
  const localizers: LocalizationProvider[] = manifests.flatMap((manifest) =>
    manifest.runtime === "classical-grid-v1"
      ? [createClassicalGridLocalizationProvider(manifest)]
      : manifest.runtime === "fenshot-localizer-v1"
        ? [createFENShotLocalizationProvider(manifest)]
        : manifest.runtime === "chess-ocr-onnx-localizer-v1" &&
            options.onnxLocalizerSession
          ? [
              createOnnxLocalizationProvider(
                options.onnxLocalizerSession,
                manifest,
              ),
            ]
          : [],
  );
  const fenshotLabels = manifests.find(
    (manifest) => manifest.runtime === "fenshot-labeler-v1",
  );
  const rectifiedFenshotLabels = manifests.find(
    (manifest) => manifest.runtime === "fenshot-rectified-labeler-v1",
  );
  const fenshotSession = options.fenshotSession ?? options.session;
  const onnxLabels = manifests.find(
    (manifest) => manifest.runtime === "chess-ocr-onnx-labeler-v1",
  );
  const labelers: LabelProvider[] = [
    ...(fenshotSession && fenshotLabels
      ? [
          createFENShotLabelProvider(
            fenshotSession,
            fenshotLabels.model,
            fenshotLabels,
          ),
        ]
      : []),
    ...(fenshotSession && rectifiedFenshotLabels
      ? [
          createFENShotRectifiedLabelProvider(
            fenshotSession,
            rectifiedFenshotLabels.model,
            rectifiedFenshotLabels,
          ),
        ]
      : []),
    ...(options.onnxLabelerSession && onnxLabels
      ? [createOnnxLabelProvider(options.onnxLabelerSession, onnxLabels)]
      : []),
  ];
  const exact = <T extends { manifest: ProviderManifest }>(
    providers: readonly T[],
    manifest: unknown,
    capability: ProviderManifest["capability"],
  ): T => {
    const requested = validateManifest(manifest);
    if (requested.capability !== capability)
      throw new Error(`Provider capability must be ${capability}`);
    const provider = providers.find(
      (candidate) => candidate.manifest.runtime === requested.runtime,
    );
    if (
      !provider ||
      JSON.stringify(provider.manifest) !== JSON.stringify(requested)
    )
      throw new Error("Provider is not available in this registry");
    return provider;
  };
  return {
    manifests,
    localization: (manifest) => exact(localizers, manifest, "localization"),
    labels: (manifest) => exact(labelers, manifest, "labels"),
  };
}

const runProposalSchema = z
  .object({
    sampleId: z.string().min(1).max(160),
    revision: finite.int().nonnegative(),
    imageSha256: sha256Schema,
    width: imageSchema.shape.width,
    height: imageSchema.shape.height,
    localizerManifest: providerManifestSchema,
    labelerManifest: providerManifestSchema,
    runId: sha256Schema,
    configSha256: sha256Schema,
  })
  .strict()
  .superRefine((run, ctx) => {
    if (run.width * run.height > 16_000_000)
      ctx.addIssue({ code: "custom", message: "Image exceeds pixel limit" });
    if (run.localizerManifest.capability !== "localization")
      ctx.addIssue({
        code: "custom",
        message: "Localizer manifest has wrong capability",
      });
    if (run.labelerManifest.capability !== "labels")
      ctx.addIssue({
        code: "custom",
        message: "Labeler manifest has wrong capability",
      });
  });
export type ProposalRun = z.infer<typeof runProposalSchema>;

const boardRereadRunSchema = z
  .object({
    sampleId: z.string().min(1).max(160),
    revision: finite.int().nonnegative(),
    imageSha256: sha256Schema,
    draftVersion: finite.int().nonnegative(),
    boardIndex: finite.int().nonnegative().max(63),
    width: imageSchema.shape.width,
    height: imageSchema.shape.height,
    corners: cornersSchema,
    labelerManifest: providerManifestSchema,
    requestId: sha256Schema,
    configSha256: sha256Schema,
  })
  .strict()
  .superRefine((run, ctx) => {
    if (run.width * run.height > 16_000_000)
      ctx.addIssue({ code: "custom", message: "Image exceeds pixel limit" });
    if (run.labelerManifest.capability !== "labels")
      ctx.addIssue({
        code: "custom",
        message: "Labeler manifest has wrong capability",
      });
    validateCorners(run.corners, { width: run.width, height: run.height }, ctx);
  });
export type BoardRereadRun = z.infer<typeof boardRereadRunSchema>;

/** Label one human-supplied quadrilateral without executing localization. */
export async function runBoardReread(
  run: BoardRereadRun & { rgba: Uint8ClampedArray },
  registry: BuiltInRegistry,
) {
  const started = performance.now();
  const parsed = boardRereadRunSchema.parse({
    sampleId: run.sampleId,
    revision: run.revision,
    imageSha256: run.imageSha256,
    draftVersion: run.draftVersion,
    boardIndex: run.boardIndex,
    width: run.width,
    height: run.height,
    corners: run.corners,
    labelerManifest: run.labelerManifest,
    requestId: run.requestId,
    configSha256: run.configSha256,
  });
  if (run.rgba.length !== parsed.width * parsed.height * 4)
    throw new Error("Invalid decoded RGBA length");
  const input: DecodedInput = {
    schema: PROPOSAL_VERSION,
    requestId: parsed.requestId,
    image: { width: parsed.width, height: parsed.height },
    rgba: run.rgba,
  };
  const candidate: BoardCandidate = candidateSchema.parse({
    id: `draft-board-${parsed.boardIndex}`,
    corners: parsed.corners,
    score: 1,
    providerRuntimeId: parsed.labelerManifest.id,
  });
  const labeled = await registry
    .labels(parsed.labelerManifest)
    .label(input, candidate);
  return {
    schema: "chess-ocr-board-reread/1" as const,
    requestId: parsed.requestId,
    sampleId: parsed.sampleId,
    revision: parsed.revision,
    imageSha256: parsed.imageSha256,
    draftVersion: parsed.draftVersion,
    boardIndex: parsed.boardIndex,
    corners: parsed.corners.map((point) => [point.x, point.y]),
    labels: labeled.squares
      .map((square) => square.label ?? "empty")
      .map((label) => (label === "empty" ? "." : label)),
    probabilities: labeled.squares.map((square) => square.probabilities),
    uncertain: labeled.squares.map((square) => square.uncertain),
    provider: labeled.provider,
    warnings: labeled.warnings,
    timings: { totalMs: performance.now() - started },
  };
}

/** Pure Node-CLI-friendly composition: caller owns bytes, storage, and process lifecycle. */
export async function runProposal(
  run: ProposalRun & { rgba: Uint8ClampedArray },
  registry: BuiltInRegistry,
): Promise<{
  schema: typeof PROPOSAL_VERSION;
  runId: string;
  sampleId: string;
  revision: number;
  imageSha256: string;
  status: "ok" | "unsupported";
  boards: Array<{
    id: string;
    corners: [
      [number, number],
      [number, number],
      [number, number],
      [number, number],
    ];
    labels: string[];
    orientation: "unknown";
    probabilities: Array<number[] | null>;
    uncertain: boolean[];
  }>;
  warnings: string[];
  providers: ProviderManifest[];
  timings: { totalMs: number };
}> {
  const started = performance.now();
  const parsed = runProposalSchema.parse({
    sampleId: run.sampleId,
    revision: run.revision,
    imageSha256: run.imageSha256,
    width: run.width,
    height: run.height,
    localizerManifest: run.localizerManifest,
    labelerManifest: run.labelerManifest,
    runId: run.runId,
    configSha256: run.configSha256,
  });
  if (run.rgba.length !== parsed.width * parsed.height * 4)
    throw new Error("Invalid decoded RGBA length");
  const input: DecodedInput = {
    schema: PROPOSAL_VERSION,
    requestId: parsed.runId,
    image: { width: parsed.width, height: parsed.height },
    rgba: run.rgba,
  };
  const localization = await registry
    .localization(parsed.localizerManifest)
    .localize(input);
  const labeler = registry.labels(parsed.labelerManifest);
  const labels = await Promise.all(
    localization.candidates.map((candidate) => labeler.label(input, candidate)),
  );
  const boards = localization.candidates.map((candidate, index) => {
    const labeled = labels[index]!;
    return {
      id: candidate.id,
      corners: candidate.corners.map((point) => [point.x, point.y]) as [
        [number, number],
        [number, number],
        [number, number],
        [number, number],
      ],
      labels: labeled.squares
        .map((square) => square.label ?? "empty")
        .map((label) => (label === "empty" ? "." : label)),
      orientation: "unknown" as const,
      probabilities: labeled.squares.map((square) => square.probabilities),
      uncertain: labeled.squares.map((square) => square.uncertain),
    };
  });
  return {
    schema: PROPOSAL_VERSION,
    runId: parsed.runId,
    sampleId: parsed.sampleId,
    revision: parsed.revision,
    imageSha256: parsed.imageSha256,
    status: boards.length ? "ok" : "unsupported",
    boards,
    warnings: [
      ...localization.warnings,
      ...labels.flatMap((label) => label.warnings),
    ],
    providers: [localization.provider, labeler.manifest],
    timings: { totalMs: performance.now() - started },
  };
}
