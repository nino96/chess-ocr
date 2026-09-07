import { z } from "zod";
import { LABELS, identitySchema } from "./contract.ts";

const hash = z.string().regex(/^[a-f0-9]{64}$/);
export const LEGACY_DETECTOR_PREPROCESSING = "legacy-bgr-div255-v1";
export const V2_DETECTOR_PREPROCESSING = "yolox-rgb-imagenet-v2";
export const detectorPreprocessingSchema = z.enum([
  LEGACY_DETECTOR_PREPROCESSING,
  V2_DETECTOR_PREPROCESSING,
]);

export function detectorInputFromRgba(
  pixels: Uint8ClampedArray,
  preprocessing: z.infer<typeof detectorPreprocessingSchema>,
): Float32Array {
  if (pixels.length !== 416 * 416 * 4)
    throw new Error("Invalid detector raster size");
  const input = new Float32Array(3 * 416 * 416);
  const legacy = preprocessing === LEGACY_DETECTOR_PREPROCESSING;
  for (let i = 0; i < 416 * 416; i++) {
    input[i] = pixels[i * 4 + (legacy ? 2 : 0)]!;
    input[416 * 416 + i] = pixels[i * 4 + 1]!;
    input[2 * 416 * 416 + i] = pixels[i * 4 + (legacy ? 0 : 2)]!;
  }
  return input;
}
const model = z
  .object({
    sha256: hash,
    bytes: z.number().int().positive(),
    input: z.string().min(1).max(80),
    output: z.string().min(1).max(80),
  })
  .strict();

export const candidateManifestSchema = z
  .object({
    schema: z.literal("chess-ocr-candidate-bundle/3"),
    name: z.string().min(1).max(120),
    version: z.string().min(1).max(120),
    qualification: z.literal("synthetic-development-only"),
    preprocessing: detectorPreprocessingSchema,
    classifier: model.extend({
      labels: z.array(z.string()).length(13),
      inputShape: z.tuple([
        z.literal("squares"),
        z.literal(3),
        z.literal(96),
        z.literal(96),
      ]),
      outputShape: z.tuple([z.literal("squares"), z.literal(13)]),
    }),
    detector: model.extend({
      inputShape: z.tuple([
        z.literal(1),
        z.literal(3),
        z.literal(416),
        z.literal(416),
      ]),
      proposalScoreThreshold: z.number().finite().min(0.001).max(1),
      calibratedAcceptanceThreshold: z.number().finite().min(0.001).max(1),
      nmsIou: z.number().finite().min(0).max(1),
    }),
    refinement: z
      .object({
        id: z.literal("nine-line-grid-refiner-v1"),
        implementationSha256: hash,
        regionExpansion: z.number().finite().min(0).max(0.5),
        outputSize: z.literal(768),
        classifierTileSize: z.literal(96),
        maxCandidates: z.number().int().positive().max(16),
      })
      .strict(),
  })
  .strict()
  .superRefine((value, context) => {
    if (!value.classifier.labels.every((label, i) => label === LABELS[i]))
      context.addIssue({
        code: "custom",
        message: "Candidate classifier label order does not match the contract",
      });
    if (value.classifier.bytes > 32 * 1024 * 1024)
      context.addIssue({ code: "custom", message: "Classifier is too large" });
    if (value.detector.bytes > 64 * 1024 * 1024)
      context.addIssue({ code: "custom", message: "Detector is too large" });
    if (
      value.detector.proposalScoreThreshold >
      value.detector.calibratedAcceptanceThreshold
    )
      context.addIssue({
        code: "custom",
        message: "Proposal threshold cannot exceed calibrated acceptance",
      });
  });

export type CandidateManifest = z.infer<typeof candidateManifestSchema>;
export type CandidateConfig = {
  manifest: CandidateManifest;
  identity: z.infer<typeof identitySchema>;
  classifier: Uint8Array;
  detector: Uint8Array;
};

async function digest(bytes: Uint8Array): Promise<string> {
  const copy = Uint8Array.from(bytes);
  return [...new Uint8Array(await crypto.subtle.digest("SHA-256", copy.buffer))]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
}

async function readBounded(file: File, maximum: number): Promise<Uint8Array> {
  if (file.size <= 0 || file.size > maximum)
    throw new Error("Candidate file is outside its size bound");
  return new Uint8Array(await file.arrayBuffer());
}

export async function loadCandidateFiles(
  manifestFile: File,
  classifierFile: File,
  detectorFile: File,
): Promise<CandidateConfig> {
  const manifestBytes = await readBounded(manifestFile, 64 * 1024);
  let value: unknown;
  try {
    value = JSON.parse(
      new TextDecoder("utf-8", { fatal: true }).decode(manifestBytes),
    );
  } catch {
    throw new Error("Candidate manifest is not valid UTF-8 JSON");
  }
  if (
    value &&
    typeof value === "object" &&
    "schema" in value &&
    (value.schema === "chess-ocr-candidate-bundle/1" ||
      value.schema === "chess-ocr-candidate-bundle/2")
  )
    throw new Error(
      `Candidate bundle schema ${value.schema.endsWith("/1") ? "1" : "2"} lacks the shared refinement contract; regenerate it as schema 3`,
    );
  const manifest = candidateManifestSchema.parse(value);
  const [classifier, detector] = await Promise.all([
    readBounded(classifierFile, 32 * 1024 * 1024),
    readBounded(detectorFile, 64 * 1024 * 1024),
  ]);
  if (
    classifier.byteLength !== manifest.classifier.bytes ||
    (await digest(classifier)) !== manifest.classifier.sha256
  )
    throw new Error("Classifier file does not match the candidate manifest");
  if (
    detector.byteLength !== manifest.detector.bytes ||
    (await digest(detector)) !== manifest.detector.sha256
  )
    throw new Error("Detector file does not match the candidate manifest");
  return {
    manifest,
    identity: identitySchema.parse({
      name: manifest.name,
      version: manifest.version,
      sha256: await digest(manifestBytes),
    }),
    classifier,
    detector,
  };
}
