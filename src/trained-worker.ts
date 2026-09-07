import * as ort from "onnxruntime-web/wasm";
import { verifiedAsset } from "./assets.ts";
import {
  candidateManifestSchema,
  type CandidateManifest,
} from "./candidate.ts";
import {
  classProbabilities,
  classifierTiles,
  detectorRasterFromRgba,
  sourceDetections,
} from "./candidate-runtime.ts";
import {
  identitySchema,
  LABELS,
  requestSchema,
  resultSchema,
  VERSION,
  type Board,
  type Rect,
  type Request,
} from "./contract.ts";
import {
  deduplicateGrids,
  findInnerGrid,
  rectifyGrid,
  type GridCorners,
  type RgbaRaster,
} from "./grid.ts";

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
    let classifier: ort.InferenceSession | null = null;
    let detector: ort.InferenceSession | null = null;
    try {
      [classifier, detector] = await Promise.all([
        ort.InferenceSession.create(classifierBytes, {
          executionProviders: ["wasm"],
        }),
        ort.InferenceSession.create(detectorBytes, {
          executionProviders: ["wasm"],
        }),
      ]);
      return { classifier, detector };
    } catch (error) {
      await Promise.allSettled([classifier?.release(), detector?.release()]);
      throw error;
    } finally {
      cleanup();
    }
  })();
}

async function releaseSessions(): Promise<void> {
  const pending = sessions;
  sessions = null;
  if (!pending) return;
  try {
    const loaded = await pending;
    await Promise.allSettled([
      loaded.classifier.release(),
      loaded.detector.release(),
    ]);
  } catch {
    // Failed construction releases every session it managed to create.
  }
}

function raster(request: Request, rgba: Uint8ClampedArray): RgbaRaster {
  return {
    data: new Uint8Array(rgba.buffer, rgba.byteOffset, rgba.byteLength),
    width: request.image.width,
    height: request.image.height,
  };
}

async function detect(
  request: Request,
  rgba: Uint8ClampedArray,
  session: ort.InferenceSession,
  candidate: CandidateManifest,
): Promise<Rect[]> {
  const prepared = detectorRasterFromRgba(
    rgba,
    request.image.width,
    request.image.height,
    candidate.preprocessing,
  );
  const tensor = new ort.Tensor("float32", prepared.input, [1, 3, 416, 416]);
  try {
    const output = await session.run({ [candidate.detector.input]: tensor });
    try {
      const raw = output[candidate.detector.output];
      if (!raw || !(raw.data instanceof Float32Array))
        throw new Error("Invalid detector output");
      return sourceDetections(
        raw.data,
        prepared,
        candidate.detector.proposalScoreThreshold,
        candidate.detector.nmsIou,
        candidate.refinement.maxCandidates,
      ).map((value) => value.box);
    } finally {
      for (const value of Object.values(output)) value.dispose();
    }
  } finally {
    tensor.dispose();
  }
}

function expandedRegion(
  box: Rect,
  width: number,
  height: number,
  fraction: number,
): Rect {
  const padding = Math.max(box.width, box.height) * fraction;
  const x = Math.max(0, box.x - padding);
  const y = Math.max(0, box.y - padding);
  const right = Math.min(width, box.x + box.width + padding);
  const bottom = Math.min(height, box.y + box.height + padding);
  return { x, y, width: right - x, height: bottom - y };
}

function selectionCorners(request: Request): GridCorners | null {
  const box = request.selection;
  if (!box) return null;
  const right = Math.min(request.image.width - 1, box.x + box.width);
  const bottom = Math.min(request.image.height - 1, box.y + box.height);
  return [
    { x: box.x, y: box.y },
    { x: right, y: box.y },
    { x: right, y: bottom },
    { x: box.x, y: bottom },
  ];
}

async function classify(
  source: RgbaRaster,
  corners: GridCorners,
  session: ort.InferenceSession,
  candidate: CandidateManifest,
  id: string,
  manual: boolean,
): Promise<Board> {
  const rectified = rectifyGrid(
    source,
    corners,
    3,
    candidate.refinement.outputSize,
  );
  const values = classifierTiles(rectified.data);
  const tensor = new ort.Tensor("float32", values, [64, 3, 96, 96]);
  try {
    const output = await session.run({ [candidate.classifier.input]: tensor });
    try {
      const raw = output[candidate.classifier.output];
      if (!raw || !(raw.data instanceof Float32Array))
        throw new Error("Invalid classifier output");
      const probabilities = classProbabilities(raw.data, 64);
      return {
        id,
        corners: corners.map((point) => ({ ...point })) as Board["corners"],
        geometrySource: manual ? "manual" : "detected",
        squares: probabilities.map((values) => {
          let best = 0;
          for (let index = 1; index < values.length; index++)
            if (values[index]! > values[best]!) best = index;
          return {
            label: LABELS[best]!,
            probabilities: values,
            uncertain: true,
          };
        }),
        orientation: "unknown",
        orientationEvidence: "unknown",
        warnings: [
          "Synthetic-only candidate; inspect every square and grid corner.",
        ],
      };
    } finally {
      for (const value of Object.values(output)) value.dispose();
    }
  } finally {
    tensor.dispose();
  }
}

function preprocessingIdentity(candidate: CandidateManifest): string {
  return `candidate-v3/${candidate.preprocessing}/${candidate.refinement.id}/mobilenetv3-rgb96`;
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
  const source = raster(request, rgba);
  const manual = selectionCorners(request);
  let rejected = 0;
  const grids: GridCorners[] = [];
  if (manual) grids.push(manual);
  else {
    const boxes = await detect(request, rgba, loaded.detector, candidate);
    for (const box of boxes) {
      const refined = findInnerGrid(
        source,
        expandedRegion(
          box,
          request.image.width,
          request.image.height,
          candidate.refinement.regionExpansion,
        ),
      );
      if (refined.ok) grids.push(refined.corners);
      else rejected++;
    }
  }
  if (!grids.length)
    return resultSchema.parse({
      schema: VERSION,
      requestId: request.requestId,
      image: request.image,
      status: "unsupported",
      boards: [],
      warnings: [
        rejected
          ? `Grid refinement rejected ${rejected} detector proposal${rejected === 1 ? "" : "s"}.`
          : "The synthetic candidate found no detector proposal.",
        "Supply the inner grid manually or use the unchanged baseline.",
      ],
      model: modelIdentity,
      preprocessing: preprocessingIdentity(candidate),
      timings: { totalMs: performance.now() - start },
    });
  const distinct = deduplicateGrids(grids.map((corners) => ({ corners })));
  const boards: Board[] = [];
  for (let index = 0; index < distinct.length; index++)
    boards.push(
      await classify(
        source,
        distinct[index]!.corners,
        loaded.classifier,
        candidate,
        `candidate-${index + 1}`,
        Boolean(manual),
      ),
    );
  return resultSchema.parse({
    schema: VERSION,
    requestId: request.requestId,
    image: request.image,
    status: "ok",
    boards,
    warnings: [
      "Candidate results are unqualified synthetic-development evidence only.",
      "Detector regions are proposals; only nine-line-refined grids are returned.",
      ...(rejected
        ? [
            `Grid refinement rejected ${rejected} additional proposal${rejected === 1 ? "" : "s"}.`,
          ]
        : []),
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
    await releaseSessions();
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
        : "candidate-v3/unconfigured",
      timings: { totalMs: 0 },
    });
  } finally {
    busy = false;
  }
};
