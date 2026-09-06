import { readdir, readFile, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
const files = [];
async function walk(dir) {
  for (const item of await readdir(dir, { withFileTypes: true })) {
    const path = `${dir}/${item.name}`;
    if (item.isDirectory()) await walk(path);
    else if (item.name !== "sw.js") {
      const bytes = await readFile(path);
      files.push({
        url: path.replace(/^dist/, ""),
        sha256: createHash("sha256").update(bytes).digest("hex"),
      });
    }
  }
}
await walk("dist");
files.sort((a, b) => a.url.localeCompare(b.url));
const version = createHash("sha256")
  .update(JSON.stringify(files))
  .digest("hex");
await writeFile(
  "dist/sw.js",
  `const CACHE = 'chess-ocr-${version}';
const FILES = ${JSON.stringify(files)};
self.addEventListener('install', event => event.waitUntil((async () => {
  const cache = await caches.open(CACHE);
  try {
    for (const file of FILES) {
      const response = await fetch(file.url, {cache:'no-store', redirect:'error', credentials:'omit'});
      if (!response.ok) throw new Error('Offline asset unavailable');
      const bytes = await response.arrayBuffer();
      const digest = [...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(b=>b.toString(16).padStart(2,'0')).join('');
      if (digest !== file.sha256) throw new Error('Offline integrity failure');
      await cache.put(file.url, new Response(bytes, {headers:response.headers}));
    }
  } catch (error) { await caches.delete(CACHE); throw error; }
  await self.skipWaiting();
})()));
self.addEventListener('activate', event => event.waitUntil(self.clients.claim()));
self.addEventListener('fetch', event => {
  const url = new URL(event.request.url);
  if (event.request.method !== 'GET' || url.origin !== self.location.origin) {
    event.respondWith(Promise.resolve(new Response('Offline-only application', {status:403}))); return;
  }
  const key = url.pathname === '/' ? '/index.html' : url.pathname;
  event.respondWith((async () => (await (await caches.open(CACHE)).match(key)) || new Response('Asset not in verified offline build', {status:404}))());
});
`,
);
console.log(
  `Offline build ${version}: ${files.length} integrity-bound assets.`,
);
