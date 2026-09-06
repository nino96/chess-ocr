import { z } from "zod";

export const VERSION = "chess-ocr/1" as const;
export const LIMITS = {
  bytes: 20 * 1024 * 1024,
  pixels: 16_000_000,
  dimension: 8192,
  timeoutMs: 30_000,
} as const;
export const LABELS = [
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
] as const;
export const labelSchema = z.enum(LABELS);
export type Label = z.infer<typeof labelSchema>;
const finite = z.number().finite();
const dimension = finite.int().positive().max(LIMITS.dimension);
export const imageSchema = z
  .object({ width: dimension, height: dimension })
  .strict()
  .refine(
    (i) => i.width * i.height <= LIMITS.pixels,
    "Decoded image is too large",
  );
export const rectSchema = z
  .object({
    x: finite.nonnegative(),
    y: finite.nonnegative(),
    width: finite.positive(),
    height: finite.positive(),
  })
  .strict();
export type Rect = z.infer<typeof rectSchema>;
export const orientationSchema = z.enum([
  "unknown",
  "white-bottom",
  "black-bottom",
]);
export type Orientation = z.infer<typeof orientationSchema>;
export const pointSchema = z
  .object({ x: finite.nonnegative(), y: finite.nonnegative() })
  .strict();
const hash = z.string().regex(/^[a-f0-9]{64}$/);
export const identitySchema = z
  .object({
    name: z.string().min(1).max(120),
    version: z.string().min(1).max(120),
    sha256: hash,
  })
  .strict();
export const squareSchema = z
  .object({
    label: labelSchema.nullable(),
    probabilities: z.array(finite.min(0).max(1)).length(13).nullable(),
    uncertain: z.boolean(),
  })
  .strict()
  .superRefine((s, ctx) => {
    if (
      s.probabilities &&
      Math.abs(s.probabilities.reduce((a, b) => a + b, 0) - 1) > 1e-4
    )
      ctx.addIssue({
        code: "custom",
        message: "Probabilities must sum to one",
      });
    if (!s.uncertain && (!s.label || !s.probabilities))
      ctx.addIssue({
        code: "custom",
        message: "Missing evidence must remain uncertain",
      });
  });
export const boardSchema = z
  .object({
    id: z.string().min(1).max(100),
    // Clockwise TL, TR, BR, BL, in original raster pixels. Squares are row-major in this image frame.
    corners: z.tuple([pointSchema, pointSchema, pointSchema, pointSchema]),
    geometrySource: z.enum(["manual", "detected"]),
    squares: z.array(squareSchema).length(64),
    orientation: orientationSchema,
    orientationEvidence: z.enum(["unknown", "user", "model"]),
    warnings: z.array(z.string().max(300)).max(30),
  })
  .strict();
export const requestSchema = z
  .object({
    schema: z.literal(VERSION),
    requestId: z.string().min(1).max(100),
    image: imageSchema,
    selection: rectSchema.nullable(),
  })
  .strict()
  .superRefine((r, ctx) => {
    if (
      r.selection &&
      (r.selection.x + r.selection.width > r.image.width ||
        r.selection.y + r.selection.height > r.image.height)
    )
      ctx.addIssue({
        code: "custom",
        message: "Selection is outside the image",
      });
  });
export const resultSchema = z
  .object({
    schema: z.literal(VERSION),
    requestId: z.string().min(1).max(100),
    image: imageSchema,
    status: z.enum(["ok", "unsupported", "error"]),
    boards: z.array(boardSchema).max(16),
    warnings: z.array(z.string().max(300)).max(30),
    model: identitySchema.nullable(),
    preprocessing: z.string().min(1).max(120),
    timings: z.object({ totalMs: finite.nonnegative() }).strict(),
  })
  .strict()
  .superRefine((r, ctx) => {
    if (r.status === "ok" && r.boards.length === 0)
      ctx.addIssue({ code: "custom", message: "Success needs a board" });
    for (const b of r.boards) {
      if (b.corners.some((p) => p.x > r.image.width || p.y > r.image.height))
        ctx.addIssue({
          code: "custom",
          message: "Geometry is outside the image",
        });
      for (let i = 0; i < 4; i++) {
        const a = b.corners[i]!,
          c = b.corners[(i + 1) % 4]!,
          d = b.corners[(i + 2) % 4]!;
        if ((c.x - a.x) * (d.y - c.y) - (c.y - a.y) * (d.x - c.x) <= 0)
          ctx.addIssue({
            code: "custom",
            message: "Grid must be convex and clockwise",
          });
      }
    }
  });
export type Request = z.infer<typeof requestSchema>;
export type Result = z.infer<typeof resultSchema>;
export type Board = z.infer<typeof boardSchema>;
export function manualBoard(rect: Rect): Board {
  return boardSchema.parse({
    id: "selection",
    geometrySource: "manual",
    corners: [
      { x: rect.x, y: rect.y },
      { x: rect.x + rect.width, y: rect.y },
      { x: rect.x + rect.width, y: rect.y + rect.height },
      { x: rect.x, y: rect.y + rect.height },
    ],
    squares: Array.from({ length: 64 }, () => ({
      label: null,
      probabilities: null,
      uncertain: true,
    })),
    orientation: "unknown",
    orientationEvidence: "unknown",
    warnings: [],
  });
}
