import { createServer } from "vite";
import { chromium, firefox, webkit } from "@playwright/test";
import { readFile, writeFile, mkdir, readdir } from "node:fs/promises";
import { createHash } from "node:crypto";
import { platform, arch, hostname } from "node:os";
const mode = process.argv[2];
if (!["benchmark", "parity"].includes(mode))
  throw new Error("Use benchmark or parity");
const server = await createServer({
  server: {
    host: "127.0.0.1",
    port: 4175,
    strictPort: true,
    hmr: false,
    watch: {
      ignored: [
        "**/work/**",
        "**/cache/**",
        "**/artifacts/**",
        "**/data/**",
        "**/.venv/**",
      ],
    },
  },
  logLevel: "error",
});
await server.listen();
const codeHashes = {};
for (const dir of ["src", "tests"]) {
  for (const file of await readdir(dir)) {
    if (!/\.(ts|html|css)$/.test(file)) continue;
    const path = `${dir}/${file}`;
    codeHashes[path] = createHash("sha256")
      .update(await readFile(path))
      .digest("hex");
  }
}
for (const path of [
  "assets.lock.json",
  "pnpm-lock.yaml",
  "pnpm-workspace.yaml",
  "scripts/evaluate.mjs",
  "vite.config.ts",
]) {
  codeHashes[path] = createHash("sha256")
    .update(await readFile(path))
    .digest("hex");
}
const report = {
  codeHashes,
  date: new Date().toISOString(),
  environment: {
    platform: platform(),
    arch: arch(),
    host: hostname(),
    node: process.version,
  },
  mode,
  browsers: {},
};
try {
  for (const [name, engine] of Object.entries({ chromium, firefox, webkit })) {
    const browser = await engine.launch();
    try {
      const page = await browser.newPage();
      await page.goto("http://127.0.0.1:4175/tests/runtime.html");
      await page.waitForFunction(() => !!window.runtimeHarness);
      if (mode === "benchmark")
        report.browsers[name] = await page.evaluate(() =>
          window.runtimeHarness.benchmark(),
        );
      else {
        const inputs = JSON.parse(
          await readFile("work/native/parity-input-manifest.json", "utf8"),
        );
        report.browsers[name] = [];
        for (const [modelName, file] of [
          ["mobilenet", "mobilenetv3_small_100.lamb_in1k-224"],
          ["yolox", "yolox_nano-coco-416-raw"],
        ]) {
          const artifact = JSON.parse(
            await readFile(`artifacts/native/${file}.manifest.json`, "utf8"),
          );
          const inputHash = createHash("sha256")
            .update(await readFile(artifact.input_manifest))
            .digest("hex");
          if (inputHash !== artifact.input_manifest_sha256)
            throw new Error("Native input manifest mismatch");
          const result = await page.evaluate(
            ({ artifact, source, modelName }) =>
              window.runtimeHarness.parity(artifact, source, modelName),
            { artifact, source: inputs.input, modelName },
          );
          report.browsers[name].push({
            ...result,
            modelSha256: artifact.model_sha256,
            inputManifestSha256: inputHash,
          });
        }
      }
      report.browsers[name] = {
        browserVersion: browser.version(),
        results: report.browsers[name],
      };
    } finally {
      await browser.close();
    }
  }
} finally {
  await server.close();
  await mkdir("work/evidence", { recursive: true });
  await writeFile(
    `work/evidence/${mode}-${report.date.replace(/[:.]/g, "-")}.json`,
    JSON.stringify(report, null, 2) + "\n",
  );
}
console.log(JSON.stringify(report, null, 2));
if (mode === "parity")
  for (const browser of Object.values(report.browsers))
    for (const result of browser.results) {
      if (result.inputMaxAbs > 1e-6 || result.outputMaxAbs > 1e-3)
        throw new Error(
          `Parity failed for ${result.modelName}: input ${result.inputMaxAbs}, output ${result.outputMaxAbs}`,
        );
    }
if (mode === "benchmark")
  for (const browser of Object.values(report.browsers))
    if (browser.results.status !== "ok" || !browser.results.cancelled)
      throw new Error("Runtime benchmark failed");
