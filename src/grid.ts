import type { Rect } from "./contract.ts";

/** Deliberate CPU and allocation ceilings for the worker-safe classical pass. */
export const GRID_LIMITS = {
  maxDimension: 8192,
  maxPixels: 16_000_000,
  maxRegionSide: 1024,
  maxRegionPixels: 1_048_576,
  maxEdgeSamples: 12_000,
  maxAngles: 60,
  maxHoughBins: 2048,
  maxLinePeaks: 48,
  rectifiedSize: 768,
} as const;

export interface RgbaRaster {
  data: Uint8Array;
  width: number;
  height: number;
}
export interface Point {
  x: number;
  y: number;
}
export type GridCorners = readonly [Point, Point, Point, Point];
export type GridRejectReason =
  | "invalid-input"
  | "insufficient-line-evidence"
  | "non-orthogonal-lines"
  | "partial-or-border-grid"
  | "non-convex-grid";
export interface GridEvidence {
  lineSupport: number;
  spacingRegularity: number;
  convexity: number;
  bounds: number;
  verticalLines: readonly number[];
  horizontalLines: readonly number[];
}
export type GridResult =
  | { ok: true; corners: GridCorners; score: number; evidence: GridEvidence }
  | { ok: false; reason: GridRejectReason; score: number };

/** Preserve score order while removing detector proposals refined to one grid. */
export function deduplicateGrids<T extends { corners: GridCorners }>(
  values: readonly T[],
  maximumBoundsIou = 0.5,
): T[] {
  if (
    !Number.isFinite(maximumBoundsIou) ||
    maximumBoundsIou < 0 ||
    maximumBoundsIou > 1
  )
    throw new Error("Invalid grid overlap threshold");
  const kept: T[] = [];
  for (const value of values)
    if (
      !kept.some(
        (prior) => boundsIou(value.corners, prior.corners) > maximumBoundsIou,
      )
    )
      kept.push(value);
  return kept;
}

function boundsIou(left: GridCorners, right: GridCorners): number {
  const bounds = (corners: GridCorners) => ({
    x0: Math.min(...corners.map((point) => point.x)),
    y0: Math.min(...corners.map((point) => point.y)),
    x1: Math.max(...corners.map((point) => point.x)),
    y1: Math.max(...corners.map((point) => point.y)),
  });
  const a = bounds(left);
  const b = bounds(right);
  const intersection =
    Math.max(0, Math.min(a.x1, b.x1) - Math.max(a.x0, b.x0)) *
    Math.max(0, Math.min(a.y1, b.y1) - Math.max(a.y0, b.y0));
  const area = (box: ReturnType<typeof bounds>) =>
    Math.max(0, box.x1 - box.x0) * Math.max(0, box.y1 - box.y0);
  return intersection / Math.max(1e-9, area(a) + area(b) - intersection);
}

interface Family {
  angle: number;
  normalX: number;
  normalY: number;
  rhos: number[];
  support: number;
  regularity: number;
}

/**
 * Finds the nine-by-nine inner playing grid inside an already-expanded detector
 * region. It samples at most GRID_LIMITS.maxRegionPixels source pixels and has
 * no Canvas, DOM, or platform image dependency.
 */
export function findInnerGrid(raster: RgbaRaster, region: Rect): GridResult {
  if (!validRaster(raster) || !validRect(region, raster.width, raster.height))
    return { ok: false, reason: "invalid-input", score: 0 };
  const step = Math.max(
    1,
    Math.ceil(region.width / GRID_LIMITS.maxRegionSide),
    Math.ceil(region.height / GRID_LIMITS.maxRegionSide),
    Math.ceil(
      Math.sqrt((region.width * region.height) / GRID_LIMITS.maxRegionPixels),
    ),
  );
  const left = Math.ceil(region.x),
    top = Math.ceil(region.y);
  const right = Math.floor(region.x + region.width - 1);
  const bottom = Math.floor(region.y + region.height - 1);
  if (right - left < 24 || bottom - top < 24)
    return { ok: false, reason: "insufficient-line-evidence", score: 0 };
  const edges: Array<{ x: number; y: number; strength: number }> = [];
  // Sparse Sobel samples: enough for line voting while capping Hough work.
  for (let y = top + step; y < bottom - step; y += step)
    for (let x = left + step; x < right - step; x += step) {
      const gx = gray(raster, x + step, y) - gray(raster, x - step, y);
      const gy = gray(raster, x, y + step) - gray(raster, x, y - step);
      const strength = Math.abs(gx) + Math.abs(gy);
      if (strength >= 70) edges.push({ x, y, strength });
    }
  if (edges.length < 180)
    return { ok: false, reason: "insufficient-line-evidence", score: 0 };
  edges.sort((a, b) => b.strength - a.strength || a.y - b.y || a.x - b.x);
  const selected = edges.slice(0, GRID_LIMITS.maxEdgeSamples);
  const families: Family[] = [];
  const diagonal = Math.hypot(region.width, region.height);
  const bins = Math.min(
    GRID_LIMITS.maxHoughBins,
    Math.max(96, Math.ceil(diagonal / step) + 1),
  );
  for (let degree = 0; degree < 180; degree += 3) {
    const angle = (degree * Math.PI) / 180;
    const nx = -Math.sin(angle),
      ny = Math.cos(angle);
    const counts = new Uint16Array(bins);
    for (const edge of selected) {
      const rho =
        (edge.x - region.x) * nx + (edge.y - region.y) * ny + diagonal;
      const bin = Math.round((rho / (2 * diagonal)) * (bins - 1));
      if (bin >= 0 && bin < bins && counts[bin]! < 65535)
        counts[bin] = counts[bin]! + 1;
    }
    const peaks: Array<{ rho: number; count: number }> = [];
    for (let i = 1; i < bins - 1; i++) {
      const count = counts[i]!;
      if (count >= 8 && count >= counts[i - 1]! && count > counts[i + 1]!)
        peaks.push({ rho: (i / (bins - 1)) * 2 * diagonal - diagonal, count });
    }
    peaks.sort((a, b) => b.count - a.count || a.rho - b.rho);
    const separated: typeof peaks = [];
    const minimumDistance = Math.max(
      2 * step,
      Math.min(region.width, region.height) / 16,
    );
    for (const peak of peaks) {
      if (
        separated.every(
          (other) => Math.abs(other.rho - peak.rho) >= minimumDistance,
        )
      )
        separated.push(peak);
      if (separated.length === GRID_LIMITS.maxLinePeaks) break;
    }
    const family = regularFamily(separated, selected.length);
    if (family) families.push({ ...family, angle, normalX: nx, normalY: ny });
  }
  if (families.length < 2)
    return { ok: false, reason: "insufficient-line-evidence", score: 0 };
  let best: { a: Family; b: Family; orthogonality: number } | undefined;
  for (let i = 0; i < families.length; i++)
    for (let j = i + 1; j < families.length; j++) {
      const a = families[i]!,
        b = families[j]!;
      const delta = Math.abs(Math.sin(a.angle - b.angle)); // 1 means perpendicular.
      if (delta < Math.cos((15 * Math.PI) / 180)) continue;
      const value = a.support * a.regularity * b.support * b.regularity * delta;
      if (
        !best ||
        value >
          best.a.support *
            best.a.regularity *
            best.b.support *
            best.b.regularity *
            best.orthogonality
      )
        best = { a, b, orthogonality: delta };
    }
  if (!best) return { ok: false, reason: "non-orthogonal-lines", score: 0 };
  const { a, b } = best;
  const globalRho = (family: Family, rho: number) =>
    rho + family.normalX * region.x + family.normalY * region.y;
  const raw = [
    intersect(a, globalRho(a, a.rhos[0]!), b, globalRho(b, b.rhos[0]!)),
    intersect(a, globalRho(a, a.rhos[8]!), b, globalRho(b, b.rhos[0]!)),
    intersect(a, globalRho(a, a.rhos[8]!), b, globalRho(b, b.rhos[8]!)),
    intersect(a, globalRho(a, a.rhos[0]!), b, globalRho(b, b.rhos[8]!)),
  ];
  if (raw.some((p) => !p))
    return { ok: false, reason: "non-convex-grid", score: 0 };
  const corners = clockwise(raw as Point[]);
  const area = polygonArea(corners);
  if (area <= 16 * step * step)
    return { ok: false, reason: "non-convex-grid", score: 0 };
  const clearance = Math.min(
    ...corners.flatMap((p) => [
      p.x - region.x,
      region.x + region.width - p.x,
      p.y - region.y,
      region.y + region.height - p.y,
    ]),
  );
  const requiredClearance = Math.max(
    2 * step,
    Math.min(region.width, region.height) * 0.01,
  );
  const bounds = Math.max(0, Math.min(1, clearance / requiredClearance));
  const baseScore =
    ((((a.support + b.support) / 2) * (a.regularity + b.regularity)) / 2) *
    best.orthogonality;
  if (clearance < requiredClearance)
    return {
      ok: false,
      reason: "partial-or-border-grid",
      score: baseScore * bounds,
    };
  const evidence: GridEvidence = {
    lineSupport: (a.support + b.support) / 2,
    spacingRegularity: (a.regularity + b.regularity) / 2,
    convexity: 1,
    bounds,
    verticalLines: [...a.rhos],
    horizontalLines: [...b.rhos],
  };
  return { ok: true, corners, score: baseScore * bounds, evidence };
}

function regularFamily(
  peaks: Array<{ rho: number; count: number }>,
  edgeCount: number,
): Omit<Family, "angle" | "normalX" | "normalY"> | undefined {
  if (peaks.length < 9) return undefined;
  let best: { rhos: number[]; support: number; regularity: number } | undefined;
  for (let start = 0; start < peaks.length; start++)
    for (let end = start + 1; end < peaks.length; end++) {
      const spacing = Math.abs(peaks[end]!.rho - peaks[start]!.rho) / 8;
      if (spacing < 2) continue;
      const chosen: typeof peaks = [];
      let residual = 0;
      for (let n = 0; n < 9; n++) {
        const expected =
          peaks[start]!.rho +
          Math.sign(peaks[end]!.rho - peaks[start]!.rho) * spacing * n;
        let nearest: { rho: number; count: number } | undefined;
        for (const peak of peaks)
          if (
            !nearest ||
            Math.abs(peak.rho - expected) < Math.abs(nearest.rho - expected)
          )
            nearest = peak;
        if (
          !nearest ||
          Math.abs(nearest.rho - expected) > spacing * 0.22 ||
          chosen.includes(nearest)
        )
          break;
        chosen.push(nearest);
        residual += Math.abs(nearest.rho - expected) / spacing;
      }
      if (chosen.length !== 9) continue;
      chosen.sort((x, y) => x.rho - y.rho);
      const regularity = Math.max(0, 1 - residual / (9 * 0.22));
      const support = Math.min(
        1,
        chosen.reduce((sum, p) => sum + p.count, 0) / (edgeCount * 0.3),
      );
      if (!best || support * regularity > best.support * best.regularity)
        best = { rhos: chosen.map((p) => p.rho), support, regularity };
    }
  return best;
}

function intersect(
  a: Family,
  ar: number,
  b: Family,
  br: number,
): Point | undefined {
  const determinant = a.normalX * b.normalY - a.normalY * b.normalX;
  if (Math.abs(determinant) < 1e-6) return undefined;
  return {
    x: (ar * b.normalY - a.normalY * br) / determinant,
    y: (a.normalX * br - ar * b.normalX) / determinant,
  };
}
function clockwise(points: Point[]): GridCorners {
  const cx = points.reduce((n, p) => n + p.x, 0) / 4,
    cy = points.reduce((n, p) => n + p.y, 0) / 4;
  points.sort(
    (a, b) => Math.atan2(a.y - cy, a.x - cx) - Math.atan2(b.y - cy, b.x - cx),
  );
  let first = 0;
  for (let i = 1; i < 4; i++)
    if (
      points[i]!.y < points[first]!.y ||
      (points[i]!.y === points[first]!.y && points[i]!.x < points[first]!.x)
    )
      first = i;
  return [
    points[first]!,
    points[(first + 1) % 4]!,
    points[(first + 2) % 4]!,
    points[(first + 3) % 4]!,
  ];
}
function polygonArea(points: readonly Point[]): number {
  let sum = 0;
  for (let i = 0; i < 4; i++) {
    const p = points[i]!,
      q = points[(i + 1) % 4]!;
    sum += p.x * q.y - p.y * q.x;
  }
  return sum / 2;
}
function gray(raster: RgbaRaster, x: number, y: number): number {
  const i = (y * raster.width + x) * 4;
  return (
    (77 * raster.data[i]! +
      150 * raster.data[i + 1]! +
      29 * raster.data[i + 2]!) >>
    8
  );
}
function validRaster(r: RgbaRaster): boolean {
  return (
    Number.isInteger(r.width) &&
    Number.isInteger(r.height) &&
    r.width > 0 &&
    r.height > 0 &&
    r.width <= GRID_LIMITS.maxDimension &&
    r.height <= GRID_LIMITS.maxDimension &&
    r.width * r.height <= GRID_LIMITS.maxPixels &&
    r.data.length === r.width * r.height * 4
  );
}
function validRect(r: Rect, width: number, height: number): boolean {
  return (
    Number.isFinite(r.x) &&
    Number.isFinite(r.y) &&
    Number.isFinite(r.width) &&
    Number.isFinite(r.height) &&
    r.x >= 0 &&
    r.y >= 0 &&
    r.width > 0 &&
    r.height > 0 &&
    r.x + r.width <= width &&
    r.y + r.height <= height
  );
}

export interface RectifiedRaster {
  data: Uint8Array;
  width: number;
  height: number;
  channels: 3 | 4;
}
/** Maps every rectified output pixel through the inverse homography with bilinear sampling. */
export function rectifyGrid(
  raster: RgbaRaster,
  corners: GridCorners,
  channels: 3 | 4 = 3,
  size: number = GRID_LIMITS.rectifiedSize,
): RectifiedRaster {
  if (
    !validRaster(raster) ||
    !validCorners(corners, raster.width, raster.height) ||
    !Number.isInteger(size) ||
    size < 1 ||
    size > GRID_LIMITS.rectifiedSize ||
    (channels !== 3 && channels !== 4)
  )
    throw new Error("Invalid rectification input");
  const h = inverseHomography(corners, size);
  const data = new Uint8Array(size * size * channels);
  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++) {
      const w = h[6]! * x + h[7]! * y + h[8]!;
      const sx = (h[0]! * x + h[1]! * y + h[2]!) / w,
        sy = (h[3]! * x + h[4]! * y + h[5]!) / w;
      const out = (y * size + x) * channels;
      sampleBilinear(raster, sx, sy, data, out, channels);
    }
  return { data, width: size, height: size, channels };
}
/** 3x3 destination-to-source homography in row-major order. */
export function inverseHomography(
  corners: GridCorners,
  size: number = GRID_LIMITS.rectifiedSize,
): readonly number[] {
  const destination = [
    [0, 0],
    [size - 1, 0],
    [size - 1, size - 1],
    [0, size - 1],
  ] as const;
  const matrix: number[][] = [];
  for (let i = 0; i < 4; i++) {
    const [x, y] = destination[i]!;
    const p = corners[i]!;
    matrix.push(
      [x, y, 1, 0, 0, 0, -x * p.x, -y * p.x, p.x],
      [0, 0, 0, x, y, 1, -x * p.y, -y * p.y, p.y],
    );
  }
  for (let col = 0; col < 8; col++) {
    let pivot = col;
    for (let row = col + 1; row < 8; row++)
      if (Math.abs(matrix[row]![col]!) > Math.abs(matrix[pivot]![col]!))
        pivot = row;
    if (Math.abs(matrix[pivot]![col]!) < 1e-10)
      throw new Error("Degenerate grid homography");
    [matrix[col], matrix[pivot]] = [matrix[pivot]!, matrix[col]!];
    const scale = matrix[col]![col]!;
    for (let j = col; j < 9; j++) matrix[col]![j] = matrix[col]![j]! / scale;
    for (let row = 0; row < 8; row++)
      if (row !== col) {
        const factor = matrix[row]![col]!;
        for (let j = col; j < 9; j++)
          matrix[row]![j] = matrix[row]![j]! - factor * matrix[col]![j]!;
      }
  }
  return [
    matrix[0]![8]!,
    matrix[1]![8]!,
    matrix[2]![8]!,
    matrix[3]![8]!,
    matrix[4]![8]!,
    matrix[5]![8]!,
    matrix[6]![8]!,
    matrix[7]![8]!,
    1,
  ];
}
function validCorners(
  corners: GridCorners,
  width: number,
  height: number,
): boolean {
  return (
    corners.length === 4 &&
    corners.every(
      (p) =>
        Number.isFinite(p.x) &&
        Number.isFinite(p.y) &&
        p.x >= 0 &&
        p.y >= 0 &&
        p.x < width &&
        p.y < height,
    ) &&
    polygonArea(corners) > 1e-6
  );
}
function sampleBilinear(
  r: RgbaRaster,
  x: number,
  y: number,
  out: Uint8Array,
  offset: number,
  channels: 3 | 4,
): void {
  const x0 = Math.max(0, Math.min(r.width - 1, Math.floor(x))),
    y0 = Math.max(0, Math.min(r.height - 1, Math.floor(y))),
    x1 = Math.min(r.width - 1, x0 + 1),
    y1 = Math.min(r.height - 1, y0 + 1),
    fx = Math.max(0, Math.min(1, x - x0)),
    fy = Math.max(0, Math.min(1, y - y0));
  for (let c = 0; c < channels; c++) {
    const at = (xx: number, yy: number) => r.data[(yy * r.width + xx) * 4 + c]!;
    const value =
      at(x0, y0) * (1 - fx) * (1 - fy) +
      at(x1, y0) * fx * (1 - fy) +
      at(x0, y1) * (1 - fx) * fy +
      at(x1, y1) * fx * fy;
    out[offset + c] = Math.round(value);
  }
}
