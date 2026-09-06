import { test } from "node:test";
import assert from "node:assert/strict";
import { readFile, mkdtemp, rm, mkdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { chromium } from "@playwright/test";

const template = await readFile("python/dataset_review.html", "utf8");

test("offline review edits image-relative labels, exports versioned decisions and validates geometry", async () => {
  const browser = await chromium.launch();
  const folder = await mkdtemp(join(tmpdir(), "chess-ocr-review-"));
  try {
    const page = await browser.newPage({ acceptDownloads: true });
    const errors = [];
    const requests = [];
    page.on("pageerror", (e) => errors.push(e.message));
    page.on("request", (r) => requests.push(r.url()));
    await page.route("**/*", (route) => route.abort());
    const image = await page.evaluate(() => {
      const canvas = document.createElement("canvas");
      canvas.width = canvas.height = 160;
      const ctx = canvas.getContext("2d");
      for (let i = 0; i < 64; i++) {
        ctx.fillStyle = `rgb(${i * 3},${255 - i * 3},${i})`;
        ctx.fillRect((i % 8) * 20, Math.floor(i / 8) * 20, 20, 20);
      }
      return canvas.toDataURL("image/png");
    });
    const payload = {
      schema: "chess-ocr-dataset-review/1",
      sample_id: "original-test-1",
      revision: 4,
      image_sha256: "a".repeat(64),
      image_data_url: image,
      width: 160,
      height: 160,
      kind: "boards",
      boards: [
        {
          corners: [
            [0, 0],
            [160, 0],
            [160, 160],
            [0, 160],
          ],
          labels: Array(64).fill("."),
          orientation: "unknown",
        },
      ],
    };
    await page.setContent(
      template.replace(
        "__PAYLOAD_BASE64__",
        Buffer.from(JSON.stringify(payload)).toString("base64"),
      ),
    );
    await page.locator("#app").waitFor({ state: "visible" });
    assert.equal(await page.locator("#labels select").count(), 64);
    await page.getByLabel("Square a8", { exact: true }).selectOption("K");
    await page.getByLabel("Square a8", { exact: true }).focus();
    await page.keyboard.press("Alt+ArrowRight");
    assert.equal(
      await page.locator(":focus").getAttribute("aria-label"),
      "Square b8",
    );
    assert.equal(
      await page.locator("#square-name").textContent(),
      "Image row 1, column 2",
    );
    const color = await page
      .locator("#square-crop")
      .evaluate((canvas) =>
        Array.from(canvas.getContext("2d").getImageData(80, 80, 1, 1).data),
      );
    assert.deepEqual(color, [3, 252, 1, 255]);
    await page
      .getByLabel("Reviewer identity", { exact: true })
      .fill("test-reviewer");
    await page.getByLabel("I am a human reviewer", { exact: true }).check();
    await page.locator("#export").click();
    assert.match(await page.locator("#status").textContent(), /complete-page/);
    await page.locator("#complete-page").check();
    await page.getByLabel("TL x coordinate", { exact: true }).fill("159");
    await page.getByLabel("TL x coordinate", { exact: true }).press("Tab");
    await page.locator("#export").click();
    assert.match(
      await page.locator("#status").textContent(),
      /incomplete or invalid/,
    );
    await page.getByLabel("TL x coordinate", { exact: true }).fill("0");
    await page.getByLabel("TL x coordinate", { exact: true }).press("Tab");
    const downloadPromise = page.waitForEvent("download");
    await page.locator("#export").click();
    const download = await downloadPromise;
    const file = join(folder, "proposal.json");
    await download.saveAs(file);
    const proposal = JSON.parse(await readFile(file, "utf8"));
    assert.equal(proposal.revision, 4);
    assert.equal(proposal.labels, undefined);
    assert.equal(proposal.boards[0].labels[0], "K");
    assert.equal(proposal.boards[0].labels[1], ".");
    assert.equal(proposal.image_sha256, payload.image_sha256);
    assert.equal(proposal.human, true);
    assert.ok(proposal.elapsed_seconds > 0);
    assert.equal(proposal.image_data_url, undefined);
    await mkdir("work/evidence", { recursive: true });
    await page.screenshot({
      path: "work/evidence/dataset-review-original.png",
      fullPage: true,
    });
    assert.deepEqual(errors, []);
    assert.deepEqual(requests, []);
    await page.setContent(
      template.replace(
        "__PAYLOAD_BASE64__",
        Buffer.from(
          JSON.stringify({ ...payload, revision: "stale-string" }),
        ).toString("base64"),
      ),
    );
    await page.locator("#error").waitFor({ state: "visible" });
    assert.equal(await page.locator("#app").isVisible(), false);
  } finally {
    await browser.close();
    await rm(folder, { recursive: true, force: true });
  }
});
