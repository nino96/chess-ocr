import { test } from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { File as NodeFile } from "node:buffer";
import {
  candidateManifestSchema,
  detectorInputFromRgba,
  loadCandidateFiles,
} from "../src/candidate.ts";

const manifest = {
  schema: "chess-ocr-candidate-bundle/3",
  name: "test candidate",
  version: "classifier-1-detector-1",
  qualification: "synthetic-development-only",
  preprocessing: "yolox-rgb-imagenet-v2",
  classifier: {
    sha256: "a".repeat(64),
    bytes: 1024,
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
    sha256: "b".repeat(64),
    bytes: 1024,
    input: "images",
    output: "predictions",
    inputShape: [1, 3, 416, 416],
    proposalScoreThreshold: 0.01,
    calibratedAcceptanceThreshold: 1,
    nmsIou: 0.65,
  },
  refinement: {
    id: "nine-line-grid-refiner-v1",
    implementationSha256: "c".repeat(64),
    regionExpansion: 0.12,
    outputSize: 768,
    classifierTileSize: 96,
    maxCandidates: 4,
  },
};

test("candidate manifest fixes roles, label order, provenance state and bounds", () => {
  assert.equal(
    candidateManifestSchema.parse(manifest).classifier.labels[0],
    "empty",
  );
  assert.throws(
    () =>
      candidateManifestSchema.parse({
        ...manifest,
        qualification: "production",
      }),
    /synthetic-development-only/,
  );
  assert.throws(
    () =>
      candidateManifestSchema.parse({
        ...manifest,
        classifier: {
          ...manifest.classifier,
          labels: [...manifest.classifier.labels].reverse(),
        },
      }),
    /label order/,
  );
});

test("candidate detector packing follows the declared preprocessing", () => {
  const rgba = new Uint8ClampedArray(416 * 416 * 4);
  rgba.set([17, 31, 47, 255]);
  const rgb = detectorInputFromRgba(rgba, "yolox-rgb-imagenet-v2");
  const bgr = detectorInputFromRgba(rgba, "legacy-bgr-div255-v1");
  assert.deepEqual([rgb[0], rgb[416 * 416], rgb[2 * 416 * 416]], [17, 31, 47]);
  assert.deepEqual([bgr[0], bgr[416 * 416], bgr[2 * 416 * 416]], [47, 31, 17]);
});

test("older bundles require regeneration", async () => {
  const old = { ...manifest, schema: "chess-ocr-candidate-bundle/1" };
  const file = new NodeFile(
    [JSON.stringify(old)],
    "candidate.json",
  ) as unknown as File;
  await assert.rejects(
    loadCandidateFiles(file, file, file),
    /lacks the shared refinement contract; regenerate it as schema 3/,
  );
  await assert.rejects(
    loadCandidateFiles(
      new NodeFile(
        [
          JSON.stringify({
            ...manifest,
            schema: "chess-ocr-candidate-bundle/2",
          }),
        ],
        "candidate.json",
      ) as unknown as File,
      file,
      file,
    ),
    /schema 2 lacks the shared refinement contract/,
  );
});

test("proposal and calibrated acceptance thresholds remain distinct and ordered", () => {
  assert.equal(
    candidateManifestSchema.parse(manifest).detector
      .calibratedAcceptanceThreshold,
    1,
  );
  assert.throws(
    () =>
      candidateManifestSchema.parse({
        ...manifest,
        detector: {
          ...manifest.detector,
          proposalScoreThreshold: 0.5,
          calibratedAcceptanceThreshold: 0.4,
        },
      }),
    /Proposal threshold cannot exceed/,
  );
});

test("candidate files must match their manifest hashes and byte lengths", async () => {
  const classifier = Buffer.from("classifier fixture");
  const detector = Buffer.from("detector fixture");
  const value = {
    ...manifest,
    classifier: {
      ...manifest.classifier,
      bytes: classifier.length,
      sha256: createHash("sha256").update(classifier).digest("hex"),
    },
    detector: {
      ...manifest.detector,
      bytes: detector.length,
      sha256: createHash("sha256").update(detector).digest("hex"),
    },
  };
  const manifestFile = () =>
    new NodeFile([JSON.stringify(value)], "candidate.json") as unknown as File;
  const classifierFile = (bytes: Buffer) =>
    new NodeFile([bytes], "classifier.onnx") as unknown as File;
  const detectorFile = () =>
    new NodeFile([detector], "detector.onnx") as unknown as File;
  const loaded = await loadCandidateFiles(
    manifestFile(),
    classifierFile(classifier),
    detectorFile(),
  );
  assert.equal(loaded.classifier.byteLength, classifier.length);
  await assert.rejects(
    loadCandidateFiles(
      manifestFile(),
      classifierFile(Buffer.from("wrong bytes")),
      detectorFile(),
    ),
    /Classifier file does not match/,
  );
});
