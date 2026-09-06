import { readFile, readdir, access } from "node:fs/promises";
import { resolve, dirname } from "node:path";
const files = ["README.md", "PLAN.md"];
async function walk(dir) {
  for (const e of await readdir(dir, { withFileTypes: true })) {
    const p = `${dir}/${e.name}`;
    if (e.isDirectory()) await walk(p);
    else if (e.name.endsWith(".md")) files.push(p);
  }
}
await walk("docs");
let count = 0;
for (const file of files) {
  const content = await readFile(file, "utf8");
  if (/[^\S\n]+\n/.test(content))
    throw new Error(`${file}: trailing whitespace`);
  for (const match of content.matchAll(/\[[^\]]*\]\(([^)]+)\)/g)) {
    const link = match[1];
    if (/^(https?:|mailto:|#)/.test(link)) continue;
    await access(resolve(dirname(file), link.split("#")[0]));
    count++;
  }
}
console.log(`Documentation links passed (${count} local targets).`);
