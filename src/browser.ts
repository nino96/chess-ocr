import { RecognitionClient } from "./client.ts";
import type { CandidateConfig } from "./candidate.ts";
import { resultSchema, type Result } from "./contract.ts";
import { rectifyGrid, type GridCorners } from "./grid.ts";
/** Browser bundler entry: local worker, WASM CPU, no remote backend. */
export function createBrowserClient(timeoutMs?: number): RecognitionClient {
  return new RecognitionClient(
    () =>
      new Worker(new URL("./worker.ts", import.meta.url), { type: "module" }),
    timeoutMs,
  );
}

/** Explicit local candidate entry. Model bytes are cloned for cancellation/retry workers. */
export function createCandidateBrowserClient(
  config: CandidateConfig,
  timeoutMs?: number,
): RecognitionClient {
  return new RecognitionClient(() => {
    const worker = new Worker(new URL("./trained-worker.ts", import.meta.url), {
      type: "module",
    });
    const classifier = config.classifier.slice();
    const detector = config.detector.slice();
    worker.postMessage(
      {
        type: "configure",
        manifest: config.manifest,
        identity: config.identity,
        classifier,
        detector,
      },
      [classifier.buffer, detector.buffer],
    );
    return worker;
  }, timeoutMs);
}

/**
 * Oracle/manual-grid input shared by both unchanged model clients. Localization
 * and line refinement are bypassed; both classifiers receive the identical
 * perspective-rectified raster.
 */
export function prepareManualGridInput(
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
  corners: GridCorners,
): Uint8ClampedArray {
  const rectified = rectifyGrid(
    {
      data: new Uint8Array(rgba.buffer, rgba.byteOffset, rgba.byteLength),
      width,
      height,
    },
    corners,
    4,
  );
  return new Uint8ClampedArray(rectified.data);
}

/** Restore manual-grid results to source-image geometry for paired scoring. */
export function restoreManualGridResult(
  result: Result,
  image: { width: number; height: number },
  corners: GridCorners,
): Result {
  return resultSchema.parse({
    ...result,
    image,
    boards: result.boards.map((board) => ({
      ...board,
      corners: corners.map((point) => ({ ...point })),
      geometrySource: "manual",
      warnings: [
        ...board.warnings,
        "Manual four-corner grid bypassed localization and automatic refinement.",
      ],
    })),
    preprocessing: `${result.preprocessing}/manual-grid-rgba768-bilinear-v1`,
  });
}
