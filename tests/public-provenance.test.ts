import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
// Tooling schema is shared with the pre-commit payload guard.
import { publicProvenance } from "../scripts/public-provenance.ts";

const registry = JSON.parse(
  readFileSync("provenance/public-bootstrap.json", "utf8"),
);

test("reviewed public provenance is bounded and complete for the acquired inputs", () => {
  assert.equal(publicProvenance.safeParse(registry).success, true);
  for (const path of [
    "provenance/public-historical-increment.json",
    "provenance/public-wikibook-increment.json",
  ])
    assert.equal(
      publicProvenance.safeParse(JSON.parse(readFileSync(path, "utf8")))
        .success,
      true,
    );
  assert.equal(
    registry.records.filter((r: { kind: string }) => r.kind === "piece-svg")
      .length,
    36,
  );
  assert.equal(
    registry.records.filter((r: { kind: string }) => r.kind === "pdf").length,
    1,
  );
});

test("public provenance rejects private flags, unknown fields and credential URLs", () => {
  for (const update of [
    { public: false },
    { local_path: "/private/example.pdf" },
    { reviewer: "example-person" },
  ]) {
    assert.equal(
      publicProvenance.safeParse({ ...registry, ...update }).success,
      false,
    );
  }
  for (const url of [
    "file:///private/example.pdf",
    "https://name:secret@example.org/a",
    "https://example.org/a?token=secret",
    "https://localhost/a",
  ]) {
    const value = structuredClone(registry);
    value.records[0].url = url;
    assert.equal(publicProvenance.safeParse(value).success, false);
  }
  const extra = structuredClone(registry);
  extra.records[0].local_path = "/private/example.pdf";
  assert.equal(publicProvenance.safeParse(extra).success, false);
});

test("public provenance rejects changed hashes, duplicate IDs and dangling evidence", () => {
  for (const update of [
    { sha256: "not-a-sha256" },
    { bytes: 999999999 },
    { evidence_ids: ["missing-record"] },
  ]) {
    const value = structuredClone(registry);
    Object.assign(value.records[0], update);
    assert.equal(publicProvenance.safeParse(value).success, false);
  }
  const duplicate = structuredClone(registry);
  duplicate.records.push(duplicate.records[0]);
  assert.equal(publicProvenance.safeParse(duplicate).success, false);
});
