import { execFileSync } from "node:child_process";
import { readFile, lstat } from "node:fs/promises";
const paths = execFileSync(
  "git",
  ["ls-files", "-z", "--cached", "--others", "--exclude-standard"],
  { encoding: "utf8" },
)
  .split("\0")
  .filter(Boolean);
const forbidden =
  /(^|\/)(data|datasets|cache|work|artifacts|models|weights|checkpoints|runs|outputs|node_modules|dist|\.venv)(\/|$)|\.(onnx|pt|pth|ckpt|safetensors|wasm|pdf|epub|djvu|png|jpe?g|webp|tiff?|bmp|svg|ttf|otf|woff2?|np[yz]|pkl|pickle|bin|zip|tar|gz|7z|pem|key)$/i;
const failures = [];
for (const path of paths) {
  if (/^fixtures\/synthetic\//.test(path)) {
    failures.push(
      `${path}: fixture exceptions require a hash/provenance manifest check (not yet implemented)`,
    );
    continue;
  }
  if (
    forbidden.test(path) ||
    (/(^|\/)\.env($|\.)/.test(path) && !path.endsWith(".env.example"))
  )
    failures.push(path);
  const stat = await lstat(path);
  if (stat.isSymbolicLink() || stat.size > 1024 * 1024)
    failures.push(`${path}: symlink or oversized source`);
  else if ((await readFile(path)).includes(0))
    failures.push(`${path}: binary payload`);
}
if (failures.length)
  throw new Error(
    `Forbidden tracked/candidate payloads:\n${failures.join("\n")}`,
  );
console.log(
  `Payload protection passed for ${paths.length} tracked/candidate files.`,
);
