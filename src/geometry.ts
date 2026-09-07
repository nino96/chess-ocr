import { imageSchema, rectSchema, type Rect } from "./contract.ts";
export interface Detection {
  box: Rect;
  score: number;
  classId: number;
}
export function iou(a: Rect, b: Rect): number {
  const overlap =
    Math.max(0, Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x)) *
    Math.max(0, Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y));
  return overlap / (a.width * a.height + b.width * b.height - overlap);
}
export function nms(
  candidates: Detection[],
  threshold = 0.45,
  maxResults = 100,
): Detection[] {
  if (
    candidates.length > 10_000 ||
    !Number.isFinite(threshold) ||
    threshold < 0 ||
    threshold > 1 ||
    !Number.isInteger(maxResults) ||
    maxResults < 1 ||
    maxResults > 1000
  )
    throw new Error("Invalid NMS bounds");
  for (const c of candidates) {
    rectSchema.parse(c.box);
    if (
      !Number.isFinite(c.score) ||
      c.score < 0 ||
      c.score > 1 ||
      !Number.isInteger(c.classId) ||
      c.classId < 0
    )
      throw new Error("Invalid detection");
  }
  const sorted = candidates
    .map((c, i) => ({ c, i }))
    .sort((a, b) => b.c.score - a.c.score || a.i - b.i);
  const kept: Detection[] = [];
  for (const { c } of sorted) {
    if (
      !kept.some(
        (k) => k.classId === c.classId && iou(k.box, c.box) > threshold,
      )
    )
      kept.push(c);
    if (kept.length >= maxResults) break;
  }
  return kept;
}
/** YOLOX raw offsets, probabilities, and class-aware NMS. COCO classes do not denote chess boards. */
export function decodeYolox(
  raw: Float32Array,
  size = 416,
  scoreThreshold = 0.3,
  classes = 80,
  nmsThreshold = 0.45,
  maxResults = 100,
): Detection[] {
  if (
    size !== 416 ||
    !Number.isFinite(scoreThreshold) ||
    scoreThreshold < 0 ||
    scoreThreshold > 1 ||
    !Number.isInteger(classes) ||
    classes < 1 ||
    classes > 100
  )
    throw new Error("Unsupported YOLOX recipe");
  const strides = [8, 16, 32],
    count = strides.reduce((n, s) => n + (size / s) ** 2, 0);
  const columns = 5 + classes;
  if (raw.length !== count * columns) throw new Error("Invalid YOLOX tensor");
  const detections: Detection[] = [];
  let row = 0;
  for (const stride of strides)
    for (let y = 0; y < size / stride; y++)
      for (let x = 0; x < size / stride; x++, row++) {
        const offset = row * columns;
        for (let j = 0; j < columns; j++)
          if (!Number.isFinite(raw[offset + j]!))
            throw new Error("Nonfinite YOLOX output");
        const objectness = raw[offset + 4]!;
        if (objectness < 0 || objectness > 1)
          throw new Error("Invalid objectness");
        let classId = 0,
          prob = 0;
        for (let c = 0; c < classes; c++) {
          const p = raw[offset + 5 + c]!;
          if (p < 0 || p > 1) throw new Error("Invalid class probability");
          if (p > prob) {
            prob = p;
            classId = c;
          }
        }
        const score = objectness * prob;
        if (score < scoreThreshold) continue;
        const cx = (raw[offset]! + x) * stride,
          cy = (raw[offset + 1]! + y) * stride;
        const w = Math.exp(raw[offset + 2]!) * stride,
          h = Math.exp(raw[offset + 3]!) * stride;
        if (!Number.isFinite(w) || !Number.isFinite(h))
          throw new Error("Invalid YOLOX extent");
        const left = Math.max(0, cx - w / 2),
          top = Math.max(0, cy - h / 2),
          right = Math.min(size, cx + w / 2),
          bottom = Math.min(size, cy + h / 2);
        if (right > left && bottom > top)
          detections.push({
            box: { x: left, y: top, width: right - left, height: bottom - top },
            score,
            classId,
          });
      }
  return nms(detections, nmsThreshold, maxResults);
}
/** Bounded on-demand windows; no whole-book acquisition or hidden inference fan-out. */
export function pageTiles(
  width: number,
  height: number,
  tileSize = 1024,
  overlap = 128,
): Rect[] {
  imageSchema.parse({ width, height });
  if (
    !Number.isInteger(tileSize) ||
    tileSize < 128 ||
    tileSize > 2048 ||
    !Number.isInteger(overlap) ||
    overlap < 0 ||
    overlap >= tileSize
  )
    throw new Error("Invalid tiling");
  const axis = (length: number) => {
    const out = [0];
    while (out[out.length - 1]! + tileSize < length) {
      out.push(
        Math.min(length - tileSize, out[out.length - 1]! + tileSize - overlap),
      );
      if (out.length > 64) throw new Error("Too many tiles");
    }
    return out;
  };
  const xs = axis(width),
    ys = axis(height);
  if (xs.length * ys.length > 64) throw new Error("Tiling exceeds 64 windows");
  return ys.flatMap((y) =>
    xs.map((x) => ({
      x,
      y,
      width: Math.min(tileSize, width),
      height: Math.min(tileSize, height),
    })),
  );
}
