import { test, expect, type Page } from "@playwright/test";
import { createHash } from "node:crypto";
import { LABELS } from "../../src/contract.ts";
async function loadSynthetic(page: Page) {
  // Original procedural test raster, generated in memory; no downloaded art or files.
  const bytes = await page.evaluate(() => {
    const c = document.createElement("canvas");
    c.width = 320;
    c.height = 320;
    const ctx = c.getContext("2d")!;
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, 320, 320);
    for (let r = 0; r < 8; r++)
      for (let f = 0; f < 8; f++) {
        ctx.fillStyle = (r + f) % 2 ? "#555" : "#ddd";
        ctx.fillRect(32 + f * 32, 32 + r * 32, 32, 32);
      }
    return Array.from(
      Uint8Array.from(atob(c.toDataURL("image/png").split(",")[1]!), (c) =>
        c.charCodeAt(0),
      ),
    );
  });
  await page.locator("#file").setInputFiles({
    name: "original-synthetic.png",
    mimeType: "image/png",
    buffer: Buffer.from(bytes),
  });
  await expect(page.locator("#status")).toContainText("Image ready");
  for (const [key, value] of Object.entries({
    x: 32,
    y: 32,
    width: 256,
    height: 256,
  }))
    await page.locator(`#${key}`).fill(String(value));
  await page.getByRole("button", { name: "Apply selection" }).click();
}
test("manual selection, actual WASM inference, edits and cancellation", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  const external: string[] = [];
  await page.goto("/");
  const origin = new URL(page.url()).origin;
  page.on("request", (r) => {
    if (!r.url().startsWith(origin) && !r.url().startsWith("blob:"))
      external.push(r.url());
  });
  await expect(page.locator("#offline")).toContainText("Offline ready");
  await loadSynthetic(page);
  const first = page.locator("#board select").first();
  await first.selectOption("K");
  await page.locator("#orientation").selectOption("white-bottom");
  await page.getByRole("button", { name: "Read selection" }).click();
  await expect(page.locator("#status")).toContainText("Result ready", {
    timeout: 45_000,
  });
  await expect(first).toHaveValue("K");
  await expect(page.locator("#orientation")).toHaveValue("white-bottom");
  await expect(page.locator("#board select")).toHaveCount(64);
  await page.evaluate(() => {
    (document.getElementById("recognize") as HTMLButtonElement).click();
    (document.getElementById("cancel") as HTMLButtonElement).click();
  });
  await expect(page.locator("#status")).toContainText("cancelled");
  await expect(first).toHaveValue("K");
  expect(external).toEqual([]);
  expect(errors).toEqual([]);
});
test("invalid image and invalid selection fail honestly; keyboard editing works", async ({
  page,
}) => {
  await page.goto("/");
  await page.locator("#file").setInputFiles({
    name: "corrupt.png",
    mimeType: "image/png",
    buffer: Buffer.from("bad input"),
  });
  await expect(page.locator("#status")).toContainText("Could not open");
  await loadSynthetic(page);
  await page.locator("#width").fill("9000");
  await page.getByRole("button", { name: "Apply selection" }).click();
  await expect(page.locator("#status")).toContainText("inside the source");
  const first = page.locator("#board select").first();
  await first.focus();
  await page.keyboard.press("Alt+ArrowRight");
  await expect(page.locator("#board select").nth(1)).toBeFocused();
});
test("corrupt model bytes fail integrity checks; a retry recovers without losing edits", async ({
  browser,
}) => {
  const context = await browser.newContext({ serviceWorkers: "block" });
  const page = await context.newPage();
  await page.route("**/assets/*.onnx", (route) =>
    route.fulfill({
      status: 200,
      body: "corrupt",
      contentType: "application/octet-stream",
    }),
  );
  await page.goto("/");
  await loadSynthetic(page);
  await page.locator("#board select").first().selectOption("Q");
  await page.getByRole("button", { name: "Read selection" }).click();
  await expect(page.locator("#status")).toContainText(
    "Local recognition failed",
  );
  await expect(page.locator("#board select").first()).toHaveValue("Q");
  await page.unroute("**/assets/*.onnx");
  await page.getByRole("button", { name: "Read selection" }).click();
  await expect(page.locator("#status")).toContainText("Result ready");
  await expect(page.locator("#board select").first()).toHaveValue("Q");
  await context.close();
});
test("blank page gives unsupported detection; pointer selection preserves source pixel coordinates", async ({
  page,
}) => {
  await page.goto("/");
  await loadSynthetic(page);
  const c = page.locator("#source");
  const bounds = await c.boundingBox();
  expect(bounds).not.toBeNull();
  await page.mouse.move(
    bounds!.x + bounds!.width / 10,
    bounds!.y + bounds!.height / 10,
  );
  await page.mouse.down();
  await page.mouse.move(
    bounds!.x + bounds!.width * 0.9,
    bounds!.y + bounds!.height * 0.9,
  );
  await page.mouse.up();
  await expect(page.locator("#x")).toHaveValue("32");
  await expect(page.locator("#width")).toHaveValue("256");
  const data = await page.evaluate(() => {
    const c = document.createElement("canvas");
    c.width = 256;
    c.height = 256;
    const x = c.getContext("2d")!;
    x.fillStyle = "white";
    x.fillRect(0, 0, 256, 256);
    return Array.from(
      Uint8Array.from(atob(c.toDataURL().split(",")[1]!), (c) =>
        c.charCodeAt(0),
      ),
    );
  });
  await page.locator("#file").setInputFiles({
    name: "blank.png",
    mimeType: "image/png",
    buffer: Buffer.from(data),
  });
  await expect(page.locator("#status")).toContainText("Image ready");
  await page.getByRole("button", { name: "Find a board" }).click();
  await expect(page.locator("#status")).toContainText(
    "No supported board found",
  );
});
test("touch activation and touch-sized editor work in a narrow viewport", async ({
  browser,
}) => {
  const context = await browser.newContext({
    viewport: { width: 420, height: 800 },
    hasTouch: true,
  });
  const page = await context.newPage();
  await page.goto("/");
  await loadSynthetic(page);
  await page.locator("#width").fill("240");
  const apply = page.getByRole("button", { name: "Apply selection" });
  await apply.scrollIntoViewIfNeeded();
  const box = (await apply.boundingBox())!;
  await page.touchscreen.tap(box.x + box.width / 2, box.y + box.height / 2);
  await expect(page.locator("#status")).toContainText("Selection ready");
  const first = page.locator("#board select").first();
  await first.selectOption("N");
  await expect(first).toHaveValue("N");
  expect((await first.boundingBox())!.width).toBeGreaterThanOrEqual(44);
  await context.close();
});

test("configured remote candidate loads only on explicit action and is not persisted", async ({
  browser,
}) => {
  const context = await browser.newContext({ serviceWorkers: "block" });
  const page = await context.newPage();
  const classifier = Buffer.from("remote browser classifier fixture");
  const detector = Buffer.from("remote browser detector fixture");
  const hash = (bytes: Buffer) =>
    createHash("sha256").update(bytes).digest("hex");
  const manifest = Buffer.from(
    JSON.stringify({
      schema: "chess-ocr-candidate-bundle/3",
      name: "remote browser fixture",
      version: "1",
      qualification: "synthetic-development-only",
      preprocessing: "yolox-rgb-imagenet-v2",
      classifier: {
        sha256: hash(classifier),
        bytes: classifier.byteLength,
        input: "tiles",
        output: "logits",
        inputShape: ["squares", 3, 96, 96],
        outputShape: ["squares", 13],
        labels: LABELS,
      },
      detector: {
        sha256: hash(detector),
        bytes: detector.byteLength,
        input: "images",
        output: "predictions",
        inputShape: [1, 3, 416, 416],
        proposalScoreThreshold: 0.01,
        calibratedAcceptanceThreshold: 1,
        nmsIou: 0.65,
      },
      refinement: {
        id: "nine-line-grid-refiner-v1",
        implementationSha256: "c".repeat(64),
        regionExpansion: 0.12,
        outputSize: 768,
        classifierTileSize: 96,
        maxCandidates: 4,
      },
    }),
  );
  const requested: string[] = [];
  await page.route("**/", async (route) => {
    const response = await route.fetch();
    const body = (await response.text()).replace(
      "</head>",
      '<meta name="chess-ocr-remote-candidate" content="available"></head>',
    );
    await route.fulfill({ response, body });
  });
  for (const [role, bytes] of Object.entries({
    manifest,
    classifier,
    detector,
  }))
    await page.route(`**/__ocr_candidate/${role}`, async (route) => {
      requested.push(role);
      expect(route.request().headers()["x-chess-ocr-candidate"]).toBe("load");
      await route.fulfill({
        body: bytes,
        contentType:
          role === "manifest" ? "application/json" : "application/octet-stream",
        headers: { "Content-Length": String(bytes.byteLength) },
      });
    });
  await page.goto("/");
  const origin = new URL(page.url()).origin;
  const external: string[] = [];
  page.on("request", (request) => {
    if (!request.url().startsWith(origin) && !request.url().startsWith("blob:"))
      external.push(request.url());
  });
  await page.locator(".candidate-loader > summary").click();
  const load = page.getByRole("button", {
    name: "Load configured remote candidate",
  });
  await expect(load).toBeVisible();
  expect(requested).toEqual([]);
  await load.click();
  await expect(page.locator("#status")).toContainText(
    "Loaded remote browser fixture 1",
  );
  await expect(page.locator("#backend")).toHaveValue("candidate");
  expect(requested.sort()).toEqual(["classifier", "detector", "manifest"]);
  await page.reload();
  await expect(page.locator("#backend")).toHaveValue("baseline");
  expect(
    await page
      .locator("#backend option[value=candidate]")
      .evaluate((option: HTMLOptionElement) => option.disabled),
  ).toBe(true);
  expect(requested).toHaveLength(3);
  expect(external).toEqual([]);
  await context.close();
});
