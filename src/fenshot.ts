import { imageSquares } from "./squares.ts";
import {
  extractTiles,
  probsToPlacement,
  recognizeGray,
  rgbaToGray,
  type BoardCorners,
} from "@scoriiu/fenshot";
import { type InferenceSession, Tensor } from "onnxruntime-web/wasm";
import {
  VERSION,
  manualBoard,
  requestSchema,
  resultSchema,
  type Request,
  type Result,
} from "./contract.ts";
import { assets } from "./assets.ts";
export async function recognize(
  request: Request,
  rgba: Uint8ClampedArray,
  session: InferenceSession,
): Promise<Result> {
  requestSchema.parse(request);
  if (rgba.length !== request.image.width * request.image.height * 4)
    throw new Error("Invalid raster");
  const start = performance.now();
  // Match shipped browser wrapper's 1600px detection cap with browser canvas resampling.
  const scale = Math.min(
    1,
    1600 / Math.max(request.image.width, request.image.height),
  );
  let gray;
  let sx = 1,
    sy = 1;
  if (!request.selection && scale < 1) {
    const source = new OffscreenCanvas(
      request.image.width,
      request.image.height,
    );
    source
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
    const target = new OffscreenCanvas(
      Math.round(request.image.width * scale),
      Math.round(request.image.height * scale),
    );
    const ctx = target.getContext("2d")!;
    ctx.drawImage(source, 0, 0, target.width, target.height);
    gray = rgbaToGray(
      ctx.getImageData(0, 0, target.width, target.height).data,
      target.width,
      target.height,
    );
    sx = request.image.width / target.width;
    sy = request.image.height / target.height;
  } else gray = rgbaToGray(rgba, request.image.width, request.image.height);
  const reads = new Map<string, Float32Array>();
  const classify = async (corners: BoardCorners) => {
    const input = new Tensor(
      "float32",
      extractTiles(gray, corners),
      [64, 1024],
    );
    try {
      const output = await session.run({ tiles: input });
      try {
        const probs = Float32Array.from(output.probs!.data as Float32Array);
        reads.set(JSON.stringify(corners), probs);
        return probsToPlacement(probs);
      } finally {
        for (const tensor of Object.values(output)) tensor.dispose();
      }
    } finally {
      input.dispose();
    }
  };
  let corners: BoardCorners | undefined;
  if (request.selection) {
    const r = request.selection;
    corners = { x0: r.x, y0: r.y, x1: r.x + r.width, y1: r.y + r.height };
    await classify(corners);
  } else corners = (await recognizeGray(gray, classify))?.corners;
  const boards = [];
  if (corners) {
    if (
      corners.x0 < 0 ||
      corners.y0 < 0 ||
      corners.x1 > gray.width ||
      corners.y1 > gray.height
    )
      corners = undefined; // Do not convert edge-padded detections into valid source geometry.
    else {
      const board = manualBoard({
        x: corners.x0 * sx,
        y: corners.y0 * sy,
        width: (corners.x1 - corners.x0) * sx,
        height: (corners.y1 - corners.y0) * sy,
      });
      board.geometrySource = request.selection ? "manual" : "detected";
      board.squares = imageSquares(reads.get(JSON.stringify(corners))!);
      board.warnings = [
        "Uncalibrated FENShot confidence; inspect every square.",
        "Only axis-aligned grids are supported.",
      ];
      boards.push(board);
    }
  }
  return resultSchema.parse({
    schema: VERSION,
    requestId: request.requestId,
    image: request.image,
    status: boards.length ? "ok" : "unsupported",
    boards,
    warnings: boards.length
      ? ["FENShot returns at most one board. No generalized accuracy claim."]
      : ["No supported board found. Select the inner playing grid manually."],
    model: {
      name: "@scoriiu/fenshot",
      version: "0.1.4",
      sha256: assets.find((a) => a.id === "fenshot")!.sha256,
    },
    preprocessing: "fenshot-0.1.4/rgba-gray-bilinear-256/1",
    timings: { totalMs: performance.now() - start },
  });
}
