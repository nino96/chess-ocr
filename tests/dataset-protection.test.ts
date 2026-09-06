import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { resolve, join } from "node:path";
import { spawnSync } from "node:child_process";

test("payload protection rejects renamed dataset metadata outside ignored storage", () => {
  const folder = mkdtempSync(resolve("dataset-protection-test-"));
  try {
    writeFileSync(
      join(folder, "renamed.json"),
      JSON.stringify({
        schema: "chess-ocr-dataset/1",
        originalSyntheticTest: true,
      }),
    );
    const result = spawnSync(process.execPath, ["scripts/protect.mjs"], {
      encoding: "utf8",
    });
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /dataset records must remain local/);
  } finally {
    rmSync(folder, { recursive: true });
  }
});
