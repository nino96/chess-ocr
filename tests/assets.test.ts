// Adapted from chess-reader c2ece9a apps/web/src/recognition/assets.test.ts:
// retain the installed-byte provenance test; use node:test and shared manifest.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createHash } from "node:crypto";
test("installed npm model and runtime bytes match pinned provenance", async () => {
  const manifest = JSON.parse(
    await readFile(new URL("../assets.lock.json", import.meta.url), "utf8"),
  ) as { assets: { source: string; sha256: string; bytes: number }[] };
  for (const asset of manifest.assets) {
    const bytes = await readFile(
      new URL("../" + asset.source, import.meta.url),
    );
    assert.equal(bytes.length, asset.bytes);
    assert.equal(
      createHash("sha256").update(bytes).digest("hex"),
      asset.sha256,
    );
  }
});
