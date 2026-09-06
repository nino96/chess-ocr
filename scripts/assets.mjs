import { readFile } from "node:fs/promises";
import { createHash } from "node:crypto";
const manifest = JSON.parse(await readFile("assets.lock.json", "utf8"));
for (const asset of manifest.assets) {
  const bytes = await readFile(asset.source);
  if (
    bytes.length !== asset.bytes ||
    createHash("sha256").update(bytes).digest("hex") !== asset.sha256
  )
    throw new Error(`Asset identity mismatch: ${asset.id}`);
}
console.log(`Verified ${manifest.assets.length} local assets.`);
