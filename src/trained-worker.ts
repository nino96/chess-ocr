import * as ort from "onnxruntime-web/wasm";
import { verifiedAsset } from "./assets.ts";
import {
  candidateManifestSchema,
  detectorInputFromRgba,
  type CandidateManifest,
} from "./candidate.ts";
import {
  identitySchema,
  LABELS,
  manualBoard,
  requestSchema,
  resultSchema,
  VERSION,
  type Rect,
  type Request,
} from "./contract.ts";
import { decodeYolox } from "./geometry.ts";

type Sessions = {
  classifier: ort.InferenceSession;
  detector: ort.InferenceSession;
};
type Identity = ReturnType<typeof identitySchema.parse>;
let manifest: CandidateManifest | null = null;
let identity: Identity | null = null;
let sessions: Promise<Sessions> | null = null;
let busy = false;

async function configureRuntime(): Promise<() => void> {
  const [mjs, wasm] = await Promise.all(
    ["ort-mjs", "ort-wasm"].map(verifiedAsset),
  );
  const mjsUrl = URL.createObjectURL(
    new Blob([mjs!], { type: "text/javascript" }),
  );
  const wasmUrl = URL.createObjectURL(
    new Blob([wasm!], { type: "application/wasm" }),
  );
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = { mjs: mjsUrl, wasm: wasmUrl };
  return () => {
    URL.revokeObjectURL(mjsUrl);
    URL.revokeObjectURL(wasmUrl);
  };
}

function configure(message: unknown): void {
  if (
    !message ||
    typeof message !== "object" ||
    !("classifier" in message) ||
    !(message.classifier instanceof Uint8Array) ||
    !("detector" in message) ||
    !(message.detector instanceof Uint8Array)
  )
    throw new Error("Invalid candidate configuration");
  const candidate = candidateManifestSchema.parse(
    "manifest" in message ? message.manifest : undefined,
  );
  const modelIdentity = identitySchema.parse(
    "identity" in message ? message.identity : undefined,
  );
  if (
    message.classifier.byteLength !== candidate.classifier.bytes ||
    message.detector.byteLength !== candidate.detector.bytes
  )
    throw new Error("Candidate model size changed");
  manifest = candidate;
  identity = modelIdentity;
  const classifierBytes = message.classifier;
  const detectorBytes = message.detector;
  sessions = (async () => {
    const cleanup = await configureRuntime();
    try {
      const [classifier, detector] = await Promise.all([
        ort.InferenceSession.create(classifierBytes, {
          executionProviders: ["wasm"],
        }),
        ort.InferenceSession.create(detectorBytes, {
          executionProviders: ["wasm"],
        }),
      ]);
      return { classifier, detector };
    } finally {
      cleanup();
    }
  })();
}

function sourceCanvas(
  request: Request,
  rgba: Uint8ClampedArray,
): OffscreenCanvas {
  const canvas = new OffscreenCanvas(request.image.width, request.image.height);
  canvas
    .getContext("2d")!
    .putImageData(
      new ImageData(
        Uint8ClampedArray.from(rgba),
        request.image.width,
        request.image.height,
      ),
      0,
      0,
    );
  return canvas;
}

async function detect(
  request: Request,
  source: OffscreenCanvas,
  session: ort.InferenceSession,
  candidate: CandidateManifest,
): Promise<Rect[]> {
  const scale = Math.min(416 / request.image.width, 416 / request.image.height);
  const resizedWidth = Math.max(1, Math.trunc(request.image.width * scale));
  const resizedHeight = Math.max(1, Math.trunc(request.image.height * scale));
  const canvas = new OffscreenCanvas(416, 416);
  const context = canvas.getContext("2d")!;
  context.fillStyle = "rgb(114 114 114)";
  context.fillRect(0, 0, 416, 416);
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "low";
  context.drawImage(source, 0, 0, resizedWidth, resizedHeight);
  const pixels = context.getImageData(0, 0, 416, 416).data;
  const input = detectorInputFromRgba(pixels, candidate.preprocessing);
  const tensor = new ort.Tensor("float32", input, [1, 3, 416, 416]);
  try {
    const output = await session.run({ [candidate.detector.input]: tensor });
    try {
      const raw = output[candidate.detector.output];
      if (!raw || !(raw.data instanceof Float32Array))
        throw new Error("Invalid detector output");
      return decodeYolox(
        raw.data,
        416,
        candidate.detector.scoreThreshold,
        1,
        candidate.detector.nmsIou,
        16,
      )
        .map(({ box }) => {
          const left = Math.max(0, Math.min(resizedWidth, box.x));
          const top = Math.max(0, Math.min(resizedHeight, box.y));
          const right = Math.max(0, Math.min(resizedWidth, box.x + box.width));
          const bottom = Math.max(
            0,
            Math.min(resizedHeight, box.y + box.height),
          );
          return {
            x: left / scale,
            y: top / scale,
            width: (right - left) / scale,
            height: (bottom - top) / scale,
          };
        })
        .filter((box) => box.width > 1 && box.height > 1);
    } finally {
      for (const value of Object.values(output)) value.dispose();
    }
  } finally {
    tensor.dispose();
  }
}

function classifierInput(source: OffscreenCanvas, boxes: Rect[]): Float32Array {
  const result = new Float32Array(boxes.length * 64 * 3 * 96 * 96);
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];
  boxes.forEach((box, board) => {
    const canvas = new OffscreenCanvas(768, 768);
    const context = canvas.getContext("2d")!;
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = "high";
    context.drawImage(
      source,
      box.x,
      box.y,
      box.width,
      box.height,
      0,
      0,
      768,
      768,
    );
    const pixels = context.getImageData(0, 0, 768, 768).data;
    for (let row = 0; row < 8; row++)
      for (let column = 0; column < 8; column++) {
        const square = board * 64 + row * 8 + column;
        for (let channel = 0; channel < 3; channel++)
          for (let y = 0; y < 96; y++)
            for (let x = 0; x < 96; x++) {
              const sourceIndex =
                ((row * 96 + y) * 768 + column * 96 + x) * 4 + channel;
              const targetIndex = (square * 3 + channel) * 96 * 96 + y * 96 + x;
              result[targetIndex] =
                (pixels[sourceIndex]! / 255 - mean[channel]!) / std[channel]!;
            }
      }
  });
  return result;
}

function probabilities(logits: Float32Array, square: number): number[] {
  const start = square * 13;
  let maximum = -Infinity;
  for (let i = 0; i < 13; i++) maximum = Math.max(maximum, logits[start + i]!);
  const values = Array.from({ length: 13 }, (_, i) =>
    Math.exp(logits[start + i]! - maximum),
  );
  const total = values.reduce((sum, value) => sum + value, 0);
  if (!Number.isFinite(total) || total <= 0)
    throw new Error("Invalid classifier output");
  return values.map((value) => value / total);
}

function preprocessingIdentity(candidate: CandidateManifest): string {
  return `candidate-v2/${candidate.preprocessing}/mobilenetv3-rgb96`;
}

async function recognize(
  request: Request,
  rgba: Uint8ClampedArray,
): Promise<ReturnType<typeof resultSchema.parse>> {
  const start = performance.now();
  if (!manifest || !identity || !sessions)
    throw new Error("Candidate is not configured");
  const candidate = manifest;
  const modelIdentity = identity;
  const loaded = await sessions;
  const source = sourceCanvas(request, rgba);
  const boxes = request.selection
    ? [request.selection]
    : await detect(request, source, loaded.detector, candidate);
  if (!boxes.length)
    return resultSchema.parse({
      schema: VERSION,
      requestId: request.requestId,
      image: request.image,
      status: "unsupported",
      boards: [],
      warnings: [
        "The synthetic candidate found no board. Select the inner grid manually or use the baseline.",
      ],
      model: modelIdentity,
      preprocessing: preprocessingIdentity(candidate),
      timings: { totalMs: performance.now() - start },
    });
  const values = classifierInput(source, boxes);
  const tensor = new ort.Tensor("float32", values, [
    boxes.length * 64,
    3,
    96,
    96,
  ]);
  let logits: Float32Array;
  try {
    const output = await loaded.classifier.run({
      [candidate.classifier.input]: tensor,
    });
    try {
      const raw = output[candidate.classifier.output];
      if (
        !raw ||
        !(raw.data instanceof Float32Array) ||
        raw.data.length !== boxes.length * 64 * 13
      )
        throw new Error("Invalid classifier output");
      logits = Float32Array.from(raw.data);
    } finally {
      for (const value of Object.values(output)) value.dispose();
    }
  } finally {
    tensor.dispose();
  }
  const boards = boxes.map((box, boardIndex) => {
    const board = manualBoard(box);
    board.id = `candidate-${boardIndex + 1}`;
    board.geometrySource = request.selection ? "manual" : "detected";
    board.squares = Array.from({ length: 64 }, (_, squareIndex) => {
      const probs = probabilities(logits, boardIndex * 64 + squareIndex);
      let best = 0;
      for (let i = 1; i < probs.length; i++)
        if (probs[i]! > probs[best]!) best = i;
      return {
        label: LABELS[best]!,
        probabilities: probs,
        // Synthetic development confidence is not a real-page calibration.
        uncertain: true,
      };
    });
    board.warnings = [
      "Synthetic-only candidate; inspect every square and the detected grid.",
      "Detector boxes are axis-aligned and are not inner-grid refined.",
    ];
    return board;
  });
  return resultSchema.parse({
    schema: VERSION,
    requestId: request.requestId,
    image: request.image,
    status: "ok",
    boards,
    warnings: [
      "Candidate results are unqualified synthetic-development evidence only.",
      "Orientation remains unknown; image rows are preserved top to bottom.",
    ],
    model: modelIdentity,
    preprocessing: preprocessingIdentity(candidate),
    timings: { totalMs: performance.now() - start },
  });
}

self.onmessage = async (event: MessageEvent) => {
  if (event.data?.type === "configure") {
    if (!sessions) configure(event.data);
    return;
  }
  if (busy) return;
  const parsed = requestSchema.safeParse(event.data?.request);
  if (!parsed.success) return;
  const request = parsed.data;
  busy = true;
  try {
    if (!(event.data.rgba instanceof Uint8ClampedArray))
      throw new Error("Invalid input");
    self.postMessage(await recognize(request, event.data.rgba));
  } catch {
    sessions = null;
    self.postMessage({
      schema: VERSION,
      requestId: request.requestId,
      image: request.image,
      status: "error",
      boards: [],
      warnings: [
        "Local candidate recognition failed. Reload its verified files.",
      ],
      model: identity,
      preprocessing: manifest
        ? preprocessingIdentity(manifest)
        : "candidate-v2/unconfigured",
      timings: { totalMs: 0 },
    });
  } finally {
    busy = false;
  }
};
