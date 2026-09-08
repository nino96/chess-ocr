import { readFile, readdir, access } from "node:fs/promises";
import { resolve, dirname, relative, sep } from "node:path";
const files = ["README.md", "PLAN.md", "AGENTS.md"];
const payloadRoots = [
  "work",
  "artifacts",
  "cache",
  "models",
  "weights",
  "checkpoints",
  "runs",
  "outputs",
  "data",
  "datasets",
];
async function walk(dir) {
  for (const e of await readdir(dir, { withFileTypes: true })) {
    const p = `${dir}/${e.name}`;
    if (e.isDirectory()) await walk(p);
    else if (e.name.endsWith(".md")) files.push(p);
  }
}
await walk("docs");
function slug(text) {
  return text
    .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
    .toLowerCase()
    .replace(/[^\w\s-]/g, "")
    .trim()
    .replace(/\s/g, "-");
}
function headingSlugs(content) {
  const slugs = new Set();
  const seen = new Map();
  let fenced = false;
  for (const line of content.split("\n")) {
    if (/^\s*(```|~~~)/.test(line)) {
      fenced = !fenced;
      continue;
    }
    if (fenced) continue;
    const heading = /^ {0,3}#{1,6}\s+(.*?)\s*#*\s*$/.exec(line);
    if (!heading) continue;
    const base = slug(heading[1]);
    if (!base) continue;
    const n = seen.get(base) ?? 0;
    seen.set(base, n + 1);
    slugs.add(n === 0 ? base : `${base}-${n}`);
  }
  return slugs;
}
function distance(a, b) {
  const row = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i++) {
    let prev = row[0];
    row[0] = i;
    for (let j = 1; j <= b.length; j++) {
      const next = Math.min(
        row[j] + 1,
        row[j - 1] + 1,
        prev + (a[i - 1] === b[j - 1] ? 0 : 1),
      );
      prev = row[j];
      row[j] = next;
    }
  }
  return row[b.length];
}
function closest(anchor, slugs) {
  let best = null;
  let score = Infinity;
  for (const candidate of slugs) {
    const d = distance(anchor, candidate);
    if (d < score) {
      score = d;
      best = candidate;
    }
  }
  return best;
}
const slugCache = new Map();
async function slugsFor(path) {
  if (!slugCache.has(path))
    slugCache.set(path, headingSlugs(await readFile(path, "utf8")));
  return slugCache.get(path);
}
let count = 0;
let anchors = 0;
const failures = [];
const warnings = [];
for (const file of files) {
  const content = await readFile(file, "utf8");
  if (/[^\S\n]+\n/.test(content))
    throw new Error(`${file}: trailing whitespace`);
  slugCache.set(file, headingSlugs(content));
  for (const match of content.matchAll(/\[[^\]]*\]\(([^)]+)\)/g)) {
    const link = match[1];
    if (/^(https?:|mailto:)/.test(link)) continue;
    const hash = link.indexOf("#");
    const target = hash === -1 ? link : link.slice(0, hash);
    const anchor = hash === -1 ? "" : link.slice(hash + 1);
    let path = file;
    if (target) {
      path = resolve(dirname(file), target);
      const rel = relative(process.cwd(), path);
      const root = rel.split(sep)[0];
      if (payloadRoots.includes(root)) {
        warnings.push(
          `${file}: ${link} points into gitignored payload root '${root}/'`,
        );
        continue;
      }
      await access(path);
      count++;
    }
    if (!anchor) continue;
    if (!path.endsWith(".md")) continue;
    const decoded = decodeURIComponent(anchor).toLowerCase();
    const slugs = await slugsFor(path);
    anchors++;
    if (!slugs.has(decoded)) {
      const suggestion = closest(decoded, slugs);
      failures.push(
        `${file}: ${link} -> no heading '#${decoded}'${
          suggestion ? ` (closest: '#${suggestion}')` : ""
        }`,
      );
    }
  }
}
for (const warning of warnings) console.warn(`Warning: ${warning}`);
if (failures.length)
  throw new Error(`Broken anchors:\n  ${failures.join("\n  ")}`);
console.log(
  `Documentation links passed (${count} local targets, ${anchors} anchors).`,
);
