import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { test } from "node:test";
import { LABELS } from "../src/contract.ts";
import {
  REMOTE_CANDIDATE_ENDPOINTS,
  REMOTE_CANDIDATE_REQUEST_HEADER,
  REMOTE_CANDIDATE_REQUEST_VALUE,
} from "../src/remote-candidate-api.ts";
import { loadConfiguredRemoteCandidate } from "../src/remote-candidate.ts";

const sha256 = (bytes: Uint8Array): string =>
  createHash("sha256").update(bytes).digest("hex");

function fixture(classifier: Uint8Array, detector: Uint8Array): Uint8Array {
  return Buffer.from(
    JSON.stringify({
      schema: "chess-ocr-candidate-bundle/3",
      name: "remote fixture",
      version: "1",
      qualification: "synthetic-development-only",
      preprocessing: "yolox-rgb-imagenet-v2",
      classifier: {
        sha256: sha256(classifier),
        bytes: classifier.byteLength,
        input: "tiles",
        output: "logits",
        inputShape: ["squares", 3, 96, 96],
        outputShape: ["squares", 13],
        labels: LABELS,
      },
      detector: {
        sha256: sha256(detector),
        bytes: detector.byteLength,
        input: "images",
        output: "predictions",
        inputShape: [1, 3, 416, 416],
        proposalScoreThreshold: 0.01,
        calibratedAcceptanceThreshold: 1,
        nmsIou: 0.65,
      },
      refinement: {
        id: "nine-line-grid-refiner-v1",
        implementationSha256: "a".repeat(64),
        regionExpansion: 0.12,
        outputSize: 768,
        classifierTileSize: 96,
        maxCandidates: 4,
      },
    }),
  );
}

test("configured remote candidate uses only fixed same-origin roles and verifies again", async () => {
  const classifier = Buffer.from("classifier");
  const detector = Buffer.from("detector");
  const manifest = fixture(classifier, detector);
  const requested: string[] = [];
  const original = globalThis.fetch;
  globalThis.fetch = async (input, init) => {
    const path = String(input);
    requested.push(path);
    assert.equal(init?.credentials, "same-origin");
    assert.equal(init?.redirect, "error");
    assert.equal(init?.cache, "no-store");
    assert.equal(
      new Headers(init?.headers).get(REMOTE_CANDIDATE_REQUEST_HEADER),
      REMOTE_CANDIDATE_REQUEST_VALUE,
    );
    const bytes =
      path === REMOTE_CANDIDATE_ENDPOINTS.manifest
        ? manifest
        : path === REMOTE_CANDIDATE_ENDPOINTS.classifier
          ? classifier
          : detector;
    return new Response(Buffer.from(bytes), {
      headers: { "Content-Length": String(bytes.byteLength) },
    });
  };
  try {
    const loaded = await loadConfiguredRemoteCandidate(
      new AbortController().signal,
    );
    assert.equal(loaded.manifest.name, "remote fixture");
    assert.deepEqual(requested, [
      REMOTE_CANDIDATE_ENDPOINTS.manifest,
      REMOTE_CANDIDATE_ENDPOINTS.classifier,
      REMOTE_CANDIDATE_ENDPOINTS.detector,
    ]);
  } finally {
    globalThis.fetch = original;
  }
});

test("configured remote candidate rejects changed lengths and cancels peer downloads", async () => {
  const classifier = Buffer.from("classifier");
  const detector = Buffer.from("detector");
  const manifest = fixture(classifier, detector);
  const original = globalThis.fetch;
  let modelRequests = 0;
  let aborted = 0;
  globalThis.fetch = async (input, init) => {
    if (String(input) === REMOTE_CANDIDATE_ENDPOINTS.manifest)
      return new Response(Buffer.from(manifest), {
        headers: { "Content-Length": String(manifest.byteLength) },
      });
    modelRequests++;
    return await new Promise<Response>((_resolve, reject) => {
      init?.signal?.addEventListener(
        "abort",
        () => {
          aborted++;
          reject(
            init.signal?.reason ?? new DOMException("Aborted", "AbortError"),
          );
        },
        { once: true },
      );
    });
  };
  const controller = new AbortController();
  try {
    const loading = loadConfiguredRemoteCandidate(controller.signal);
    while (modelRequests < 2)
      await new Promise((resolve) => setTimeout(resolve));
    controller.abort(new DOMException("Stopped", "AbortError"));
    await assert.rejects(loading, /Stopped|Aborted/);
    assert.equal(aborted, 2);
  } finally {
    globalThis.fetch = original;
  }
});

test("configured remote candidate cancels a response stream that exceeds its bound", async () => {
  const original = globalThis.fetch;
  let cancelled = false;
  globalThis.fetch = async () =>
    new Response(
      new ReadableStream({
        start(controller) {
          controller.enqueue(new Uint8Array([1, 2]));
        },
        cancel() {
          cancelled = true;
        },
      }),
      { headers: { "Content-Length": "1" } },
    );
  try {
    await assert.rejects(
      loadConfiguredRemoteCandidate(new AbortController().signal),
      /response size changed/,
    );
    assert.equal(cancelled, true);
  } finally {
    globalThis.fetch = original;
  }
});

test("configured remote candidate cancels an unauthorized response body", async () => {
  const original = globalThis.fetch;
  let cancelled = false;
  globalThis.fetch = async () =>
    new Response(
      new ReadableStream({
        start(controller) {
          controller.enqueue(new Uint8Array([1]));
        },
        cancel() {
          cancelled = true;
        },
      }),
      { status: 403 },
    );
  try {
    await assert.rejects(
      loadConfiguredRemoteCandidate(new AbortController().signal),
      /session is unavailable/,
    );
    assert.equal(cancelled, true);
  } finally {
    globalThis.fetch = original;
  }
});
