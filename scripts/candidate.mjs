import { createHash } from "node:crypto";
import {
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
    "Usage: candidate prepare --run-root PATH --output PATH --score-threshold NUMBER --preprocessing ID",
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
const threshold = Number(required("--score-threshold"));
if (!Number.isFinite(threshold) || threshold < 0.001 || threshold > 1)
  throw new Error("Score threshold must be between 0.001 and 1");
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
  schema: "chess-ocr-candidate-bundle/2",
  name: frozen.recipe?.run,
  version: `classifier-${classifierStep}-detector-${detectorStep}`,
  qualification: "synthetic-development-only",
  preprocessing,
  classifier: {
    sha256: digest(classifier),
    bytes: statSync(classifier).size,
    input: "tiles",
    output: "logits",
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
    scoreThreshold: threshold,
    nmsIou: 0.65,
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
process.stdout.write(`${output}\n`);
