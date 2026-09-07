import { spawn } from "node:child_process";
import { resolve } from "node:path";
import { chromium, firefox, webkit } from "@playwright/test";

const repository = resolve(import.meta.dirname, "..");
const [manifest, classifier, detector] = process.argv
  .slice(2)
  .map((path) => (path ? resolve(repository, path) : path));
if (!manifest || !classifier || !detector)
  throw new Error(
    "Usage: candidate-browser-verify CANDIDATE_MANIFEST CLASSIFIER_ONNX DETECTOR_ONNX",
  );
const origin = "http://127.0.0.1:4174";
const server = spawn(
  process.execPath,
  [
    "node_modules/vite/bin/vite.js",
    "preview",
    "--host",
    "127.0.0.1",
    "--port",
    "4174",
    "--strictPort",
  ],
  { cwd: repository, stdio: ["ignore", "ignore", "inherit"] },
);

async function ready() {
  for (let attempt = 0; attempt < 100; attempt++) {
    if (server.exitCode !== null) throw new Error("Preview server exited");
    try {
      const response = await fetch(origin);
      if (response.ok) return;
    } catch {
      // Bounded startup polling only.
    }
    await new Promise((resolveDelay) => setTimeout(resolveDelay, 50));
  }
  throw new Error("Preview server did not become ready");
}

async function syntheticPng(page) {
  const values = await page.evaluate(async () => {
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = 256;
    const context = canvas.getContext("2d");
    if (!context) throw new Error("canvas unavailable");
    context.fillStyle = "white";
    context.fillRect(0, 0, 256, 256);
    for (let row = 0; row < 8; row++)
      for (let column = 0; column < 8; column++) {
        context.fillStyle = (row + column) % 2 ? "#444" : "#ddd";
        context.fillRect(32 + column * 24, 32 + row * 24, 24, 24);
      }
    context.strokeStyle = "black";
    context.lineWidth = 2;
    for (let line = 0; line <= 8; line++) {
      context.beginPath();
      context.moveTo(32 + line * 24, 32);
      context.lineTo(32 + line * 24, 224);
      context.stroke();
      context.beginPath();
      context.moveTo(32, 32 + line * 24);
      context.lineTo(224, 32 + line * 24);
      context.stroke();
    }
    const blob = await new Promise((resolveBlob) =>
      canvas.toBlob(resolveBlob, "image/png"),
    );
    if (!blob) throw new Error("PNG encode failed");
    return [...new Uint8Array(await blob.arrayBuffer())];
  });
  return Buffer.from(values);
}

async function loadCandidate(page) {
  await page.locator(".candidate-loader > summary").click();
  await page.locator("#candidate-manifest").setInputFiles(manifest);
  await page.locator("#candidate-classifier").setInputFiles(classifier);
  await page.locator("#candidate-detector").setInputFiles(detector);
  await page.waitForFunction(
    () => !document.querySelector("#load-candidate")?.disabled,
  );
  await page.locator("#load-candidate").click();
  await page.locator("#status").waitFor({ state: "visible" });
  await page.waitForFunction(() =>
    document.querySelector("#status")?.textContent?.startsWith("Loaded "),
  );
}

async function prepareReference(page, png, mode, queue = true) {
  if (queue) {
    const before = await page.locator("#diagnostic-status").textContent();
    await page.locator("#diagnostic-files").setInputFiles({
      name: "original-synthetic-board.png",
      mimeType: "image/png",
      buffer: png,
    });
    await page.waitForFunction(
      (previous) =>
        document.querySelector("#diagnostic-status")?.textContent !== previous,
      before,
    );
    const queueStatus = await page.locator("#diagnostic-status").textContent();
    if (!queueStatus?.includes("Queue prepared"))
      throw new Error(`Queue preparation failed: ${queueStatus}`);
  }
  await page.locator("#diagnostic-mode").selectOption(mode);
  await page.locator("#diagnostic-reference-kind").selectOption("board");
  for (const [index, value] of [32, 32, 224, 32, 224, 224, 32, 224].entries())
    await page.locator(`#diagnostic-corner-${index}`).fill(String(value));
  await page.getByRole("button", { name: "Save reference" }).click();
  if (!(await page.locator("#diagnostic-result").isHidden()))
    throw new Error("Model result was exposed before the paired run");
}

async function runMode(page) {
  const before = await page.locator("#diagnostic-status").textContent();
  await page.getByRole("button", { name: "Run paired comparison" }).click();
  await page.waitForFunction(
    (previous) =>
      document.querySelector("#diagnostic-status")?.textContent !== previous,
    before,
    { timeout: 5_000 },
  );
  if (
    (await page.locator("#diagnostic-status").textContent())?.startsWith(
      "Decoding one local image",
    )
  )
    await page.waitForFunction(
      () =>
        !document
          .querySelector("#diagnostic-status")
          ?.textContent?.startsWith("Decoding one local image"),
      undefined,
      { timeout: 120_000 },
    );
  const status = await page.locator("#diagnostic-status").textContent();
  if (!status?.includes("Paired local result saved"))
    throw new Error(`Paired browser run failed: ${status}`);
  const output = await page.locator("#diagnostic-result").textContent();
  if (!output?.includes("v2:") || !output.includes("FENShot:"))
    throw new Error("Paired metrics are missing");
}

await ready();
const reports = [];
try {
  for (const [name, launcher] of Object.entries({
    chromium,
    firefox,
    webkit,
  })) {
    const browser = await launcher.launch();
    try {
      const context = await browser.newContext();
      const page = await context.newPage();
      const external = [];
      page.on("request", (request) => {
        if (
          !request.url().startsWith(origin) &&
          !request.url().startsWith("blob:")
        )
          external.push(request.url());
      });
      await page.goto(origin);
      await page.waitForFunction(() =>
        document
          .querySelector("#offline")
          ?.textContent?.includes("Offline ready"),
      );
      const png = await syntheticPng(page);
      await loadCandidate(page);
      await prepareReference(page, png, "automatic");
      const automaticStart = performance.now();
      await runMode(page);
      const automaticMs = performance.now() - automaticStart;
      await context.setOffline(true);
      await prepareReference(page, png, "manual-grid", false);
      const manualStart = performance.now();
      await runMode(page);
      const manualMs = performance.now() - manualStart;
      if (external.length)
        throw new Error(`External requests observed: ${external.join(", ")}`);
      reports.push({
        browser: name,
        automaticMs,
        manualMs,
        externalRequests: 0,
      });
      await context.close();
    } finally {
      await browser.close();
    }
  }
  process.stdout.write(
    `${JSON.stringify({ schema: "chess-ocr-candidate-browser-verification/1", reports })}\n`,
  );
} finally {
  server.kill("SIGTERM");
}
