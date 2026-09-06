import { readdir, readFile } from "node:fs/promises";
const failures = [];
async function walk(path) {
  for (const entry of await readdir(path, { withFileTypes: true })) {
    const p = `${path}/${entry.name}`;
    if (entry.isDirectory()) await walk(p);
    else if (/\.(ts|mjs|css|html)$/.test(p)) {
      const content = await readFile(p, "utf8");
      if (!content.endsWith("\n")) failures.push(`${p}: missing final newline`);
      content.split("\n").forEach((line, i) => {
        if (/[\t ]+$/.test(line))
          failures.push(`${p}:${i + 1}: trailing whitespace`);
      });
      if (
        path.startsWith("src") &&
        /\bconsole\.(log|debug|info|warn|error)\s*\(/.test(content)
      )
        failures.push(`${p}: runtime logging prohibited`);
      if (
        path.startsWith("src") &&
        /\beval\s*\(|\bnew Function\s*\(/.test(content)
      )
        failures.push(`${p}: dynamic code evaluation prohibited`);
    }
  }
}
for (const dir of ["src", "scripts", "tests"]) await walk(dir);
if (failures.length) throw new Error(failures.join("\n"));
console.log(
  "Source hygiene/privacy lint passed; TypeScript check supplies strict semantic checks.",
);
