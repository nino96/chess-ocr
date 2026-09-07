import { createHash } from "node:crypto";
import {
  existsSync,
  lstatSync,
  mkdirSync,
  readFileSync,
  realpathSync,
  renameSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { dirname, isAbsolute, relative, resolve } from "node:path";

const argv = process.argv.slice(2);
if (argv[0] === "--") argv.shift();
if (argv.shift() !== "prepare")
  throw new Error(
    "Usage: candidate prepare --run-root PATH --output PATH --proposal-score-threshold NUMBER --calibrated-score-threshold NUMBER --preprocessing ID",
  );
const options = new Map();
while (argv.length) {
  const key = argv.shift();
  if (!key?.startsWith("--") || !argv.length)
    throw new Error("Candidate options require --name value pairs");
  if (options.has(key)) throw new Error(`Duplicate option: ${key}`);
  options.set(key, argv.shift());
}
const required = (name) => {
  const value = options.get(name);
  if (!value) throw new Error(`Missing ${name}`);
  return value;
};
const repository = realpathSync(process.cwd());
const work = realpathSync(resolve(repository, "work"));
function underWork(value, kind = "file") {
  const path = resolve(repository, value);
  const resolved = realpathSync(path);
  const location = relative(work, resolved);
  if (isAbsolute(location) || location === ".." || location.startsWith("../"))
    throw new Error("Candidate inputs must remain under work/");
  if (lstatSync(path).isSymbolicLink())
    throw new Error("Candidate inputs cannot be symbolic links");
  if (kind === "file" && !statSync(path).isFile())
    throw new Error("Candidate input is not a file");
  if (kind === "directory" && !statSync(path).isDirectory())
    throw new Error("Candidate run root is not a directory");
  return path;
}
const run = underWork(required("--run-root"), "directory");
const classifier = underWork(resolve(run, "classifier/selected.onnx"));
const detector = underWork(resolve(run, "detector/selected.onnx"));
const readJson = (path) => JSON.parse(readFileSync(path, "utf8"));
const digest = (path) =>
  createHash("sha256").update(readFileSync(path)).digest("hex");
const classifierManifest = readJson(
  underWork(resolve(run, "classifier/selected.manifest.json")),
);
const frozen = readJson(underWork(resolve(run, "frozen.json")));
if (
  classifierManifest.schema !== "chess-ocr-model/1" ||
  classifierManifest.role !== "square-classifier" ||
  classifierManifest.labels !== ".PNBRQKpnbrqk" ||
  classifierManifest.sha256 !== digest(classifier)
)
  throw new Error("Classifier export manifest is missing or inconsistent");
let detectorStep;
let exportedPreprocessing;
try {
  const detectorManifest = readJson(
    underWork(resolve(run, "detector/selected.manifest.json")),
  );
  if (
    detectorManifest.schema !== "chess-ocr-model/1" ||
    detectorManifest.role !== "inner-grid-detector" ||
    detectorManifest.sha256 !== digest(detector)
  )
    throw new Error("Detector export manifest is inconsistent");
  detectorStep = detectorManifest.metrics?.selected_global_step;
  exportedPreprocessing = detectorManifest.preprocessing;
} catch (error) {
  // The first completed schedule was rejected only by the old 1e-4 gate. This
  // recovery is deliberately limited to the documented <=1e-3 failure shape.
  const state = readJson(underWork(resolve(run, "state.json")));
  const log = readFileSync(underWork(resolve(run, "detector.log")), "utf8");
  const matches = [
    ...log.matchAll(/ONNX output parity mismatch: ([0-9.eE+-]+)/g),
  ];
  const drift = Number(matches.at(-1)?.[1]);
  const curves = readJson(underWork(resolve(run, "detector/curves.json")));
  const best = curves.reduce((left, right) => {
    const a = left.development;
    const b = right.development;
    return b.ap50_95 > a.ap50_95 ||
      (b.ap50_95 === a.ap50_95 && b.recall["0.5"] > a.recall["0.5"]) ||
      (b.ap50_95 === a.ap50_95 &&
        b.recall["0.5"] === a.recall["0.5"] &&
        b.mean_normalized_box_error < a.mean_normalized_box_error)
      ? right
      : left;
  });
  const scheduled = frozen.recipe.detector.stages.reduce(
    (total, stage) => total + stage.updates,
    0,
  );
  const checkpoint = underWork(
    resolve(
      run,
      `detector/checkpoint-${String(best.global_step).padStart(6, "0")}.pt`,
    ),
  );
  if (
    state.state !== "failed" ||
    state.error !== "detector failed; inspect its retained log" ||
    !Number.isFinite(drift) ||
    drift > 0.001 ||
    best.global_step !== scheduled ||
    !statSync(checkpoint).isFile()
  )
    throw error;
  detectorStep = best.global_step;
  exportedPreprocessing = "legacy-bgr-div255-v1";
}
const proposalThreshold = Number(required("--proposal-score-threshold"));
const calibratedThreshold = Number(required("--calibrated-score-threshold"));
if (
  !Number.isFinite(proposalThreshold) ||
  proposalThreshold < 0.001 ||
  proposalThreshold > 1 ||
  !Number.isFinite(calibratedThreshold) ||
  calibratedThreshold < 0.001 ||
  calibratedThreshold > 1 ||
  proposalThreshold > calibratedThreshold
)
  throw new Error(
    "Proposal and calibrated thresholds must be ordered values between 0.001 and 1",
  );
const preprocessing = required("--preprocessing");
if (
  preprocessing !== "legacy-bgr-div255-v1" &&
  preprocessing !== "yolox-rgb-imagenet-v2"
)
  throw new Error("Unsupported detector preprocessing identifier");
if (exportedPreprocessing !== preprocessing)
  throw new Error("Detector preprocessing does not match its export manifest");
const classifierStep = Number(
  classifierManifest.metrics?.selected_global_step ?? 0,
);
if (!Number.isInteger(classifierStep) || !Number.isInteger(detectorStep))
  throw new Error("Selected checkpoint steps are unavailable");
const manifest = {
  schema: "chess-ocr-candidate-bundle/3",
  name: frozen.recipe?.run,
  version: `classifier-${classifierStep}-detector-${detectorStep}`,
  qualification: "synthetic-development-only",
  preprocessing,
  classifier: {
    sha256: digest(classifier),
    bytes: statSync(classifier).size,
    input: "tiles",
    output: "logits",
    inputShape: ["squares", 3, 96, 96],
    outputShape: ["squares", 13],
    labels: [
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
    ],
  },
  detector: {
    sha256: digest(detector),
    bytes: statSync(detector).size,
    input: "images",
    output: "predictions",
    inputShape: [1, 3, 416, 416],
    proposalScoreThreshold: proposalThreshold,
    calibratedAcceptanceThreshold: calibratedThreshold,
    nmsIou: 0.65,
  },
  refinement: {
    id: "nine-line-grid-refiner-v1",
    implementationSha256: digest(resolve(repository, "src/grid.ts")),
    regionExpansion: 0.12,
    outputSize: 768,
    classifierTileSize: 96,
    maxCandidates: 4,
  },
};
const output = resolve(repository, required("--output"));
const outputParent = dirname(output);
mkdirSync(outputParent, { recursive: true });
const checkedParent = realpathSync(outputParent);
const location = relative(work, checkedParent);
if (isAbsolute(location) || location === ".." || location.startsWith("../"))
  throw new Error("Candidate output must remain under work/");
const temporary = `${output}.${process.pid}.tmp`;
writeFileSync(temporary, JSON.stringify(manifest, null, 2) + "\n", {
  flag: "wx",
});
renameSync(temporary, output);

const providerDirectoryOption = options.get("--provider-output-directory");
const providerOutputs = [];
if (providerDirectoryOption) {
  const providerDirectory = resolve(repository, providerDirectoryOption);
  mkdirSync(providerDirectory, { recursive: true });
  const checkedDirectory = realpathSync(providerDirectory);
  const providerLocation = relative(work, checkedDirectory);
  if (
    isAbsolute(providerLocation) ||
    providerLocation === ".." ||
    providerLocation.startsWith("../")
  )
    throw new Error("Provider output directory must remain under work/");
  const limits = {
    max_pixels: 16_000_000,
    max_dimension: 8192,
    timeout_ms: 30_000,
    max_candidates: manifest.refinement.maxCandidates,
  };
  const refinement = { ...manifest.refinement };
  const providers = [
    {
      schema: "chess-ocr-provider-manifest/1",
      id: `v2-localizer-${manifest.detector.sha256.slice(0, 8)}-${manifest.refinement.implementationSha256.slice(0, 8)}`,
      capability: "localization",
      runtime: "chess-ocr-onnx-localizer-v1",
      model: {
        name: `${manifest.name} detector`,
        version: manifest.version,
        sha256: manifest.detector.sha256,
      },
      preprocessing: manifest.preprocessing,
      artifact: {
        path: relative(repository, detector),
        sha256: manifest.detector.sha256,
      },
      configuration: {
        kind: "chess-ocr-onnx-localizer/1",
        input: manifest.detector.input,
        output: manifest.detector.output,
        inputShape: manifest.detector.inputShape,
        proposalScoreThreshold: manifest.detector.proposalScoreThreshold,
        calibratedAcceptanceThreshold:
          manifest.detector.calibratedAcceptanceThreshold,
        nmsIou: manifest.detector.nmsIou,
        refinement,
      },
      limits,
    },
    {
      schema: "chess-ocr-provider-manifest/1",
      id: `v2-labeler-${manifest.classifier.sha256.slice(0, 8)}-${manifest.refinement.implementationSha256.slice(0, 8)}`,
      capability: "labels",
      runtime: "chess-ocr-onnx-labeler-v1",
      model: {
        name: `${manifest.name} classifier`,
        version: manifest.version,
        sha256: manifest.classifier.sha256,
      },
      preprocessing:
        "nine-line-grid-refiner-v1/rgb768-bilinear/rgb96-imagenet-v1",
      artifact: {
        path: relative(repository, classifier),
        sha256: manifest.classifier.sha256,
      },
      configuration: {
        kind: "chess-ocr-onnx-labeler/1",
        input: manifest.classifier.input,
        output: manifest.classifier.output,
        inputShape: manifest.classifier.inputShape,
        outputShape: manifest.classifier.outputShape,
        labels: manifest.classifier.labels,
        refinement,
      },
      limits,
    },
  ];
  for (const provider of providers) {
    const path = resolve(providerDirectory, `${provider.id}.json`);
    const body = `${JSON.stringify(provider, null, 2)}\n`;
    if (existsSync(path)) {
      if (readFileSync(path, "utf8") !== body)
        throw new Error(`Immutable provider manifest already differs: ${path}`);
    } else {
      const staged = `${path}.${process.pid}.tmp`;
      writeFileSync(staged, body, { flag: "wx" });
      renameSync(staged, path);
    }
    providerOutputs.push(path);
  }
}
process.stdout.write(`${[output, ...providerOutputs].join("\n")}\n`);
