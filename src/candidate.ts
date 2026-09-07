import { z } from "zod";
import { LABELS, identitySchema } from "./contract.ts";

const hash = z.string().regex(/^[a-f0-9]{64}$/);
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
    schema: z.literal("chess-ocr-candidate-bundle/1"),
    name: z.string().min(1).max(120),
    version: z.string().min(1).max(120),
    qualification: z.literal("synthetic-development-only"),
    classifier: model.extend({
      labels: z.array(z.string()).length(13),
    }),
    detector: model.extend({
      scoreThreshold: z.number().finite().min(0.001).max(1),
      nmsIou: z.number().finite().min(0).max(1),
    }),
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
