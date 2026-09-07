import { createHash } from "node:crypto";
import { readFile, realpath, stat } from "node:fs/promises";
import { relative, resolve } from "node:path";
import * as ort from "onnxruntime-web/wasm";
import {
  PROPOSAL_VERSION,
  createOnnxLabelProvider,
  createOnnxLocalizationProvider,
  validateManifest,
} from "../src/proposals/index.ts";

const repository = resolve(import.meta.dirname, "..");
const [localizerPath, labelerPath] = process.argv.slice(2);
if (!localizerPath || !labelerPath)
  throw new Error(
    "Usage: candidate-verify LOCALIZER_MANIFEST LABELER_MANIFEST",
  );

const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");
async function bounded(path, maximum) {
  const resolved = await realpath(resolve(repository, path));
  const location = relative(resolve(repository, "work"), resolved);
  if (location.startsWith("..") || resolve(location) === resolve(".."))
    throw new Error("Verification inputs must remain under ignored work/");
  const info = await stat(resolved);
  if (!info.isFile() || info.size <= 0 || info.size > maximum)
    throw new Error("Verification input is outside its size bound");
  return readFile(resolved);
}
async function manifest(path, capability) {
  const parsed = validateManifest(
    JSON.parse((await bounded(path, 64 * 1024)).toString("utf8")),
  );
  if (parsed.capability !== capability || !parsed.artifact)
    throw new Error(`Expected ${capability} ONNX provider`);
  const model = await bounded(parsed.artifact.path, 64 * 1024 * 1024);
  if (
    sha256(model) !== parsed.artifact.sha256 ||
    parsed.model.sha256 !== parsed.artifact.sha256
  )
    throw new Error(`${capability} artifact hash mismatch`);
  return { parsed, model };
}

const localizer = await manifest(localizerPath, "localization");
const labeler = await manifest(labelerPath, "labels");
const gridHash = sha256(await readFile(resolve(repository, "src/grid.ts")));
for (const provider of [localizer.parsed, labeler.parsed])
  if (provider.configuration?.refinement.implementationSha256 !== gridHash)
    throw new Error("Grid implementation hash mismatch");

ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths =
  resolve(repository, "node_modules/onnxruntime-web/dist") + "/";
let detectorSession = null;
let classifierSession = null;
try {
  detectorSession = await ort.InferenceSession.create(localizer.model, {
    executionProviders: ["wasm"],
  });
  classifierSession = await ort.InferenceSession.create(labeler.model, {
    executionProviders: ["wasm"],
  });
  const width = 256;
  const height = 256;
  const rgba = new Uint8ClampedArray(width * height * 4);
  rgba.fill(255);
  for (let y = 32; y <= 224; y++)
    for (let x = 32; x <= 224; x++) {
      const index = (y * width + x) * 4;
      const line = (x - 32) % 24 === 0 || (y - 32) % 24 === 0;
      const square =
        (Math.floor((x - 32) / 24) + Math.floor((y - 32) / 24)) % 2;
      const value = line ? 8 : square ? 200 : 70;
      rgba[index] = rgba[index + 1] = rgba[index + 2] = value;
      rgba[index + 3] = 255;
    }
  const input = {
    schema: PROPOSAL_VERSION,
    requestId: "candidate-node-wasm-verification",
    image: { width, height },
    rgba,
  };
  const started = performance.now();
  const localized = await createOnnxLocalizationProvider(
    detectorSession,
    localizer.parsed,
  ).localize(input);
  const labeled = await createOnnxLabelProvider(
    classifierSession,
    labeler.parsed,
  ).label(input, {
    id: "manual-grid",
    providerRuntimeId: localizer.parsed.id,
    score: 1,
    corners: [
      { x: 32, y: 32 },
      { x: 224, y: 32 },
      { x: 224, y: 224 },
      { x: 32, y: 224 },
    ],
  });
  if (
    labeled.squares.length !== 64 ||
    labeled.squares.some(
      (square) => !square.probabilities || square.probabilities.length !== 13,
    )
  )
    throw new Error("Classifier verification result is incomplete");
  process.stdout.write(
    `${JSON.stringify({
      schema: "chess-ocr-candidate-node-wasm-verification/1",
      detector: localizer.parsed.model,
      classifier: labeler.parsed.model,
      detectorCandidates: localized.candidates.length,
      classifierSquares: labeled.squares.length,
      totalMs: performance.now() - started,
    })}\n`,
  );
} finally {
  await Promise.allSettled(
    [detectorSession, classifierSession]
      .filter(Boolean)
      .map((session) => session.release()),
  );
}
