import { z } from "zod";

// Metadata-only boundary; factual public status still requires human/agent review.

const publicUrl = z
  .string()
  .url()
  .refine((value) => {
    const url = new URL(value);
    return (
      url.protocol === "https:" &&
      !url.username &&
      !url.password &&
      !url.search &&
      !url.hash &&
      !["localhost", "127.0.0.1", "[::1]"].includes(url.hostname)
    );
  }, "Public HTTPS URL without credentials, query or fragment required");
const token = z.string().regex(/^[a-zA-Z0-9_-]{1,100}$/);
const record = z
  .object({
    id: token,
    kind: z.enum(["piece-svg", "pdf", "license-evidence"]),
    url: publicUrl,
    revision: z.string().min(1).max(200),
    sha256: z.string().regex(/^[a-f0-9]{64}$/),
    bytes: z.number().int().positive(),
    max_bytes: z
      .number()
      .int()
      .positive()
      .max(512 * 1024 * 1024),
    attribution: z.string().min(1).max(1000),
    license: z.string().min(1).max(500),
    evidence_ids: z.array(token).max(20),
    pages: z
      .array(z.number().int().positive().max(100000))
      .max(2000)
      .optional(),
    split: z.enum(["train", "dev", "qualification", "regression"]).optional(),
    lineage: z.array(token).min(1).max(20).optional(),
    conditions: z.array(token).min(1).max(20).optional(),
  })
  .strict();

export const publicProvenance = z
  .object({
    schema: z.literal("chess-ocr-public-provenance/1"),
    public: z.literal(true),
    review_date: z.iso.date(),
    scope: z.literal("public-input-reacquisition-only"),
    records: z.array(record).min(1).max(2000),
  })
  .strict()
  .superRefine((value, ctx) => {
    const ids = new Set(value.records.map((r) => r.id));
    if (ids.size !== value.records.length)
      ctx.addIssue({ code: "custom", message: "Duplicate public resource ID" });
    for (const r of value.records) {
      if (
        r.bytes > r.max_bytes ||
        r.evidence_ids.some((id) => !ids.has(id)) ||
        (r.kind !== "license-evidence" && r.evidence_ids.length === 0) ||
        (r.pages && new Set(r.pages).size !== r.pages.length)
      )
        ctx.addIssue({
          code: "custom",
          message: "Invalid public resource bounds/evidence",
        });
    }
  });
