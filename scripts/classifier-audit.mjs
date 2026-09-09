import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import {
  chmod,
  mkdir,
  readFile,
  realpath,
  stat,
  writeFile,
} from "node:fs/promises";
import { arch, platform } from "node:os";
import { basename, dirname, relative, resolve } from "node:path";
import { createServer } from "vite";
import { chromium } from "@playwright/test";
import * as ort from "onnxruntime-web/wasm";
import { privateEvaluationSchema } from "../src/evaluation.ts";

const repository = resolve(import.meta.dirname, "..");
const args = new Map();
const values = process.argv.slice(2);
if (values[0] === "--") values.shift();
for (let index = 0; index < values.length; index += 2) {
  const key = values[index];
  const value = values[index + 1];
  if (!key?.startsWith("--") || !value)
    throw new Error("Invalid classifier audit arguments");
  args.set(key.slice(2), value);
}
for (const required of ["session", "image", "entry", "run", "output"])
  if (!args.has(required))
    throw new Error(
      "Usage: classifier-audit --session PRIVATE_JSON --image IMAGE --entry INDEX --run TRAINING_RUN --output NEW_WORK_DIRECTORY",
    );

const sha256 = (value) => createHash("sha256").update(value).digest("hex");
const json = (value) => JSON.stringify(value, null, 2) + "\n";
const maximums = {
  session: 8 * 1024 * 1024,
  image: 20 * 1024 * 1024,
  onnx: 64 * 1024 * 1024,
  checkpoint: 256 * 1024 * 1024,
};

function ignored(location) {
  const name = relative(repository, location).split(/[\\/]/)[0];
  return ["work", "artifacts", "cache", "models", "data"].includes(name);
}

async function bounded(path, maximum, purpose) {
  const location = await realpath(resolve(repository, path));
  if (!ignored(location))
    throw new Error(`${purpose} must stay below an ignored payload root`);
  const info = await stat(location);
  if (!info.isFile() || info.size <= 0 || info.size > maximum)
    throw new Error(`${purpose} is outside its byte bound`);
  return { location, info, bytes: await readFile(location) };
}

async function command(program, commandArgs, options = {}) {
  return new Promise((resolveCommand, reject) => {
    const child = spawn(program, commandArgs, {
      cwd: repository,
      stdio: ["ignore", "pipe", "pipe"],
      ...options,
    });
    const stdout = [];
    const stderr = [];
    child.stdout.on("data", (value) => stdout.push(value));
    child.stderr.on("data", (value) => stderr.push(value));
    child.on("error", reject);
    child.on("exit", (code) => {
      if (code === 0) resolveCommand(Buffer.concat(stdout).toString("utf8"));
      else
        reject(
          new Error(
            `Bounded ${basename(program)} subprocess failed (${code}); private subprocess output was suppressed`,
          ),
        );
    });
  });
}

function maximumByteDifference(left, right) {
  if (left.length !== right.length)
    throw new Error("Audit stage byte shapes disagree");
  let maximum = 0;
  for (let index = 0; index < left.length; index++)
    maximum = Math.max(maximum, Math.abs(left[index] - right[index]));
  return maximum;
}

function floats(value) {
  if (value.byteOffset % 4 || value.byteLength % 4)
    throw new Error("Misaligned float stage");
  return new Float32Array(value.buffer, value.byteOffset, value.byteLength / 4);
}

function maximumFloatDifference(leftBytes, rightBytes) {
  const left = floats(leftBytes);
  const right = floats(rightBytes);
  if (left.length !== right.length)
    throw new Error("Audit float shapes disagree");
  let maximum = 0;
  for (let index = 0; index < left.length; index++) {
    if (!Number.isFinite(left[index]) || !Number.isFinite(right[index]))
      throw new Error("Nonfinite audit stage");
    maximum = Math.max(maximum, Math.abs(left[index] - right[index]));
  }
  return maximum;
}

function labels(logitBytes) {
  const values = floats(logitBytes);
  if (values.length !== 64 * 13)
    throw new Error("Invalid classifier logit shape");
  return Array.from({ length: 64 }, (_, square) => {
    let best = 0;
    for (let index = 1; index < 13; index++)
      if (values[square * 13 + index] > values[square * 13 + best])
        best = index;
    return best;
  });
}

const sessionFile = await bounded(
  args.get("session"),
  maximums.session,
  "Private session",
);
const imageFile = await bounded(
  args.get("image"),
  maximums.image,
  "Private image",
);
const session = privateEvaluationSchema.parse(
  JSON.parse(sessionFile.bytes.toString("utf8")),
);
const entryIndex = Number(args.get("entry"));
if (!Number.isInteger(entryIndex)) throw new Error("Entry must be an integer");
const entry = session.entries.find(
  (candidate) =>
    candidate.index === entryIndex && candidate.mode === "manual-grid",
);
if (!entry || entry.reference.kind !== "board")
  throw new Error(
    "Entry must be a human-confirmed manual-grid board reference",
  );
if (
  sha256(imageFile.bytes) !== entry.image.sha256 ||
  imageFile.info.size !== entry.image.bytes
)
  throw new Error(
    "Private image no longer matches its local evaluation record",
  );

const runLocation = await realpath(resolve(repository, args.get("run")));
if (!ignored(runLocation) || !(await stat(runLocation)).isDirectory())
  throw new Error("Training run must stay below an ignored payload root");
const frozenPath = resolve(runLocation, "frozen.json");
const frozen = JSON.parse(await readFile(frozenPath, "utf8"));
const onnx = await bounded(
  relative(repository, resolve(runLocation, "classifier/selected.onnx")),
  maximums.onnx,
  "Classifier ONNX",
);
const modelManifest = JSON.parse(
  await readFile(
    resolve(runLocation, "classifier/selected.manifest.json"),
    "utf8",
  ),
);
if (
  modelManifest.schema !== "chess-ocr-model/1" ||
  modelManifest.role !== "square-classifier" ||
  modelManifest.labels !== ".PNBRQKpnbrqk" ||
  modelManifest.sha256 !== sha256(onnx.bytes)
)
  throw new Error("Classifier ONNX manifest identity changed");
const checkpointRecord = frozen.classifier_checkpoint;
if (!checkpointRecord?.path || !checkpointRecord.sha256)
  throw new Error(
    "Frozen run does not identify the exact retained classifier checkpoint",
  );
const checkpoint = await bounded(
  relative(repository, resolve(checkpointRecord.path)),
  maximums.checkpoint,
  "Classifier checkpoint",
);
if (sha256(checkpoint.bytes) !== checkpointRecord.sha256)
  throw new Error("Classifier checkpoint identity changed");

const requestedOutput = resolve(repository, args.get("output"));
const outputParent = await realpath(dirname(requestedOutput));
const output = resolve(outputParent, basename(requestedOutput));
if (
  requestedOutput !== output ||
  !ignored(outputParent) ||
  relative(repository, output).startsWith("..")
)
  throw new Error("Audit output must stay below an ignored payload root");
await mkdir(output, { recursive: false, mode: 0o700 });
await chmod(output, 0o700);
const writePrivate = async (name, value) => {
  const path = resolve(output, name);
  if (dirname(path) !== output) throw new Error("Invalid audit output name");
  await writeFile(path, value, { flag: "wx", mode: 0o600 });
};

let server;
let browser;
try {
  server = await createServer({
    server: { host: "127.0.0.1", port: 4176, strictPort: true, hmr: false },
    logLevel: "silent",
  });
  await server.listen();
  browser = await chromium.launch();
  const page = await browser.newPage();
  const external = [];
  page.on("request", (request) => {
    if (
      !request.url().startsWith("http://127.0.0.1:4176") &&
      !request.url().startsWith("blob:")
    )
      external.push("blocked");
  });
  await page.goto("http://127.0.0.1:4176/tests/classifier-audit.html");
  await page.waitForFunction(() => Boolean(window.classifierAuditHarness));
  const browserResult = await page.evaluate(
    ({ image, corners, model }) =>
      window.classifierAuditHarness.run(image, corners, model),
    {
      image: imageFile.bytes.toString("base64"),
      corners: entry.reference.corners,
      model: onnx.bytes.toString("base64"),
    },
  );
  if (external.length)
    throw new Error("Classifier audit attempted an external request");
  if (
    browserResult.width !== entry.image.width ||
    browserResult.height !== entry.image.height
  )
    throw new Error("Browser-decoded dimensions changed");
  const stages = Object.fromEntries(
    ["rgba", "grid", "tensor", "logits"].map((name) => [
      name,
      Buffer.from(browserResult[name], "base64"),
    ]),
  );
  if (
    stages.rgba.length !== entry.image.width * entry.image.height * 4 ||
    stages.grid.length !== 768 * 768 * 3 ||
    stages.tensor.length !== 64 * 3 * 96 * 96 * 4 ||
    stages.logits.length !== 64 * 13 * 4
  )
    throw new Error("Browser audit stage shape changed");
  await Promise.all([
    writePrivate("browser-source.rgba", stages.rgba),
    writePrivate("browser-grid.rgb", stages.grid),
    writePrivate("browser-tensor.f32", stages.tensor),
    writePrivate("browser-logits.f32", stages.logits),
  ]);

  await writePrivate(
    "preprocess-control.json",
    json({
      schema: "chess-ocr-exact-tile-preprocess/1",
      source: {
        file: "browser-source.rgba",
        sha256: sha256(stages.rgba),
        width: entry.image.width,
        height: entry.image.height,
      },
      corners: entry.reference.corners,
      labels: entry.reference.labels,
    }),
  );
  let python = process.env.DATASET_PYTHON;
  if (!python) {
    try {
      python = await realpath(
        resolve(repository, "work/dataset-venv/bin/python"),
      );
    } catch {
      python = "python3";
    }
  }
  await command(python, [
    "python/classifier_audit.py",
    "preprocess",
    resolve(output, "preprocess-control.json"),
  ]);
  const pythonGrid = await readFile(resolve(output, "python-grid.rgb"));
  const pythonTensor = await readFile(resolve(output, "python-tensor.f32"));

  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths =
    resolve(repository, "node_modules/onnxruntime-web/dist") + "/";
  const nodeSession = await ort.InferenceSession.create(onnx.bytes, {
    executionProviders: ["wasm"],
  });
  let nodeLogits;
  const nodeTensor = new ort.Tensor(
    "float32",
    Float32Array.from(floats(stages.tensor)),
    [64, 3, 96, 96],
  );
  try {
    const result = await nodeSession.run({
      [nodeSession.inputNames[0]]: nodeTensor,
    });
    nodeLogits = Buffer.from(
      Float32Array.from(result[nodeSession.outputNames[0]].data).buffer,
    );
    for (const value of Object.values(result)) value.dispose();
  } finally {
    nodeTensor.dispose();
    await nodeSession.release();
  }
  await writePrivate("node-logits.f32", nodeLogits);

  const nativeRoot = resolve(frozen.native_root);
  const overlayRoot = resolve(frozen.dependency_overlay_root);
  const image = frozen.recipe?.environment?.image;
  if (!image || !frozen.container_id)
    throw new Error("Frozen container identity is incomplete");
  const actualContainer = (
    await command("docker", ["image", "inspect", image, "--format", "{{.Id}}"])
  ).trim();
  if (actualContainer !== frozen.container_id)
    throw new Error("Frozen training container identity changed");
  await writePrivate(
    "native-control.json",
    json({
      schema: "chess-ocr-exact-tile-native/1",
      tensor_file: "browser-tensor.f32",
      tensor_sha256: sha256(stages.tensor),
      checkpoint_path: `/checkpoint/${basename(checkpoint.location)}`,
      checkpoint_sha256: checkpointRecord.sha256,
      native_root: "/native",
    }),
  );
  await command("docker", [
    "run",
    "--rm",
    "--network",
    "none",
    "--cpus",
    "2",
    "--memory",
    "4g",
    "--pids-limit",
    "256",
    "--user",
    `${process.getuid()}:${process.getgid()}`,
    "--entrypoint",
    "python",
    "-v",
    `${repository}:/repo:ro`,
    "-v",
    `${output}:/audit`,
    "-v",
    `${dirname(checkpoint.location)}:/checkpoint:ro`,
    "-v",
    `${nativeRoot}:/native:ro`,
    "-v",
    `${overlayRoot}:/overlay:ro`,
    "-e",
    "PYTHONPATH=/overlay:/repo/python",
    image,
    "/repo/python/classifier_audit.py",
    "native",
    "/audit/native-control.json",
  ]);
  const nativeLogits = await readFile(resolve(output, "native-logits.f32"));

  const differences = {
    rectifiedRgb: maximumByteDifference(stages.grid, pythonGrid),
    normalizedTensor: maximumFloatDifference(stages.tensor, pythonTensor),
    nodeVsBrowserLogits: maximumFloatDifference(nodeLogits, stages.logits),
    nativeVsExportedLogits: maximumFloatDifference(nativeLogits, stages.logits),
  };
  const nativeLabels = labels(nativeLogits);
  const exportedLabels = labels(stages.logits);
  const labelAgreement = nativeLabels.every(
    (value, index) => value === exportedLabels[index],
  );
  const expectedLabels = entry.reference.labels.map((label) =>
    ".PNBRQKpnbrqk".indexOf(label === "empty" ? "." : label),
  );
  const referenceAgreement = expectedLabels.every(
    (value, index) => value === exportedLabels[index],
  );
  const stagesInOrder = [
    ["rectified-rgb", differences.rectifiedRgb === 0],
    ["normalized-tensor", differences.normalizedTensor <= 1e-6],
    ["node-wasm-logits", differences.nodeVsBrowserLogits <= 1e-6],
    ["checkpoint-export-logits", differences.nativeVsExportedLogits <= 1e-3],
    ["exported-labels", labelAgreement],
  ];
  const firstDivergentStage =
    stagesInOrder.find(([, passed]) => !passed)?.[0] ?? null;
  const code = {};
  for (const name of [
    "src/grid.ts",
    "src/candidate-runtime.ts",
    "tests/classifier-audit.ts",
    "tests/parity-worker.ts",
    "python/classifier_preprocessing.py",
    "python/classifier_audit.py",
    "python/training.py",
    "scripts/classifier-audit.mjs",
    "pnpm-lock.yaml",
  ])
    code[name] = sha256(await readFile(resolve(repository, name)));
  const report = {
    schema: "chess-ocr-exact-tile-audit/1",
    createdAt: new Date().toISOString(),
    private: true,
    publishable: false,
    result: firstDivergentStage ? "failed" : "passed",
    conclusion: firstDivergentStage
      ? "preprocessing-or-runtime-divergence"
      : referenceAgreement
        ? "parity-passed-reference-reproduced"
        : "classifier-generalization-failure",
    firstDivergentStage,
    maximumDifferences: differences,
    tolerances: {
      rectifiedRgb: 0,
      normalizedTensor: 1e-6,
      nodeVsBrowserLogits: 1e-6,
      nativeVsExportedLogits: 1e-3,
    },
    exportedRuntimeLabelsReproduceNatively: labelAgreement,
    exportedRuntimeLabelsMatchHumanReference: referenceAgreement,
    manualGrid: {
      source: "human-confirmed",
      refinementApplied: false,
      identityBound: true,
    },
    contract: {
      classOrder: ".PNBRQKpnbrqk",
      squareOrder: "image-relative-row-major",
      channelOrder: "RGB-NCHW",
      normalization: "imagenet-mean-std",
      interpolation: "deterministic-bilinear-size-minus-one",
    },
    identities: {
      encodedInputSha256: entry.image.sha256,
      decodedRgbaSha256: sha256(stages.rgba),
      referenceSha256: sha256(Buffer.from(JSON.stringify(entry.reference))),
      modelSha256: modelManifest.sha256,
      checkpointSha256: checkpointRecord.sha256,
      code,
    },
    environment: {
      platform: platform(),
      arch: arch(),
      node: process.version,
      browser: "chromium",
      browserVersion: browser.version(),
      onnxRuntimeWeb: "1.29.0",
      trainingContainerId: frozen.container_id,
    },
  };
  await writePrivate("report.json", json(report));
  process.stdout.write(
    `${JSON.stringify({ state: report.result, firstDivergentStage, report: "ignored local audit report" })}\n`,
  );
  if (firstDivergentStage) process.exitCode = 1;
} finally {
  await browser?.close();
  await server?.close();
}
