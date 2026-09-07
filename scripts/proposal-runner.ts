import { createHash, randomUUID } from "node:crypto";
import { readFile, realpath, rename, stat, writeFile } from "node:fs/promises";
import { dirname, relative, resolve } from "node:path";
import * as ort from "onnxruntime-web/wasm";
import {
  createBuiltInRegistry,
  runProposal,
  validateManifest,
} from "../src/proposals/index.ts";

const repository = resolve(import.meta.dirname, "..");
const requestPath = process.argv[2];
const outputPath = process.argv[3];
if (!requestPath || !outputPath)
  throw new Error("request and output paths required");

function inside(path: string, root: string): boolean {
  const value = relative(root, path);
  return value !== "" && !value.startsWith("..") && !value.startsWith("/");
}

async function safeFile(path: string, allowedRoot: string, maximum: number) {
  const absolute = resolve(path);
  const actual = await realpath(absolute);
  if (!inside(actual, allowedRoot))
    throw new Error("unsafe proposal input path");
  const info = await stat(actual);
  if (!info.isFile() || info.size <= 0 || info.size > maximum)
    throw new Error("proposal input size rejected");
  return readFile(actual);
}

function sha256(bytes: Uint8Array): string {
  return createHash("sha256").update(bytes).digest("hex");
}

function asciiJson(value: unknown): string {
  const encoded = JSON.stringify(value);
  if (encoded === undefined) throw new Error("invalid canonical value");
  return encoded.replace(
    /[\u007f-\uffff]/g,
    (character) =>
      `\\u${character.charCodeAt(0).toString(16).padStart(4, "0")}`,
  );
}

function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.entries(value as Record<string, unknown>)
      .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
      .map(([key, item]) => `${asciiJson(key)}:${canonical(item)}`)
      .join(",")}}`;
  }
  return asciiJson(value);
}

const stagingRoot =
  process.env.CHESS_OCR_TESTING === "1"
    ? resolve(dirname(resolve(requestPath)), "../..")
    : resolve(repository, "work/dataset/proposals/staging");
const requestBytes = await safeFile(requestPath, stagingRoot, 2 * 1024 * 1024);
const request: unknown = JSON.parse(requestBytes.toString("utf8"));
if (!request || typeof request !== "object" || Array.isArray(request))
  throw new Error("invalid proposal request");
const value = request as Record<string, unknown>;
const localizer = validateManifest(value.localizer);
const labeler = validateManifest(value.labeler);
if (localizer.capability !== "localization" || labeler.capability !== "labels")
  throw new Error("provider capability mismatch");
const gridSource = await readFile(resolve(repository, "src/grid.ts"));
for (const manifest of [localizer, labeler]) {
  const configuration = manifest.configuration;
  if (
    configuration &&
    sha256(gridSource) !== configuration.refinement.implementationSha256
  )
    throw new Error("provider refinement implementation hash mismatch");
}
if (
  typeof value.config_sha256 !== "string" ||
  sha256(Buffer.from(canonical({ localizer, labeler }))) !== value.config_sha256
)
  throw new Error("provider configuration identity mismatch");
if (
  typeof value.width !== "number" ||
  typeof value.height !== "number" ||
  !Number.isInteger(value.width) ||
  !Number.isInteger(value.height) ||
  value.width <= 0 ||
  value.height <= 0 ||
  value.width * value.height > 16_000_000 ||
  typeof value.rgba_path !== "string"
)
  throw new Error("invalid proposal raster declaration");
const rgbaBytes = await safeFile(
  value.rgba_path,
  stagingRoot,
  value.width * value.height * 4,
);
if (rgbaBytes.length !== value.width * value.height * 4)
  throw new Error("decoded raster length mismatch");

async function providerArtifact(manifest: typeof labeler) {
  if (!manifest.artifact) throw new Error("model provider artifact required");
  const modelPath = resolve(repository, manifest.artifact.path);
  const modelRoot = manifest.artifact.path.startsWith("node_modules/")
    ? resolve(repository, "node_modules")
    : manifest.artifact.path.startsWith("artifacts/")
      ? resolve(repository, "artifacts")
      : resolve(repository, "work");
  const model = await safeFile(modelPath, modelRoot, 256 * 1024 * 1024);
  if (
    sha256(model) !== manifest.artifact.sha256 ||
    manifest.artifact.sha256 !== manifest.model.sha256
  )
    throw new Error(`${manifest.capability} provider artifact hash mismatch`);
  return model;
}

ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths =
  resolve(repository, "node_modules/onnxruntime-web/dist") + "/";
const fenshotModel =
  labeler.runtime === "fenshot-labeler-v1"
    ? await providerArtifact(labeler)
    : null;
const onnxLocalizerModel =
  localizer.runtime === "chess-ocr-onnx-localizer-v1"
    ? await providerArtifact(localizer)
    : null;
const onnxLabelerModel =
  labeler.runtime === "chess-ocr-onnx-labeler-v1"
    ? await providerArtifact(labeler)
    : null;
let fenshotSession: ort.InferenceSession | null = null;
let onnxLocalizerSession: ort.InferenceSession | null = null;
let onnxLabelerSession: ort.InferenceSession | null = null;
try {
  if (fenshotModel)
    fenshotSession = await ort.InferenceSession.create(fenshotModel, {
      executionProviders: ["wasm"],
    });
  if (onnxLocalizerModel)
    onnxLocalizerSession = await ort.InferenceSession.create(
      onnxLocalizerModel,
      { executionProviders: ["wasm"] },
    );
  if (onnxLabelerModel)
    onnxLabelerSession = await ort.InferenceSession.create(onnxLabelerModel, {
      executionProviders: ["wasm"],
    });
  const result = await runProposal(
    {
      sampleId: String(value.sample_id),
      revision: Number(value.revision),
      imageSha256: String(value.image_sha256),
      width: value.width,
      height: value.height,
      localizerManifest: localizer,
      labelerManifest: labeler,
      runId: String(value.run_id),
      configSha256: String(value.config_sha256),
      rgba: new Uint8ClampedArray(
        rgbaBytes.buffer,
        rgbaBytes.byteOffset,
        rgbaBytes.byteLength,
      ),
    },
    createBuiltInRegistry({
      fenshotSession: fenshotSession ?? undefined,
      onnxLocalizerSession: onnxLocalizerSession ?? undefined,
      onnxLabelerSession: onnxLabelerSession ?? undefined,
      modelSha256: labeler.model.sha256,
      manifests: [localizer, labeler],
    }),
  );
  const output = resolve(outputPath);
  if (
    !inside(output, stagingRoot) ||
    dirname(output) !== dirname(resolve(requestPath))
  )
    throw new Error("unsafe proposal output path");
  const temporary = `${output}.tmp-${randomUUID()}`;
  await writeFile(temporary, `${JSON.stringify(result)}\n`, { flag: "wx" });
  await rename(temporary, output);
} finally {
  await Promise.allSettled(
    [fenshotSession, onnxLocalizerSession, onnxLabelerSession]
      .filter((session): session is ort.InferenceSession => session !== null)
      .map((session) => session.release()),
  );
}
