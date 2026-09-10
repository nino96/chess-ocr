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
      const pixels = ctx.createImageData(160, 160);
      for (let y = 0; y < 160; y++) {
        for (let x = 0; x < 160; x++) {
          const offset = (y * 160 + x) * 4;
          pixels.data[offset] = x;
          pixels.data[offset + 1] = y;
          pixels.data[offset + 2] = (x + y) % 256;
          pixels.data[offset + 3] = 255;
        }
      }
      ctx.putImageData(pixels, 0, 0);
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
      proposals: [
        {
          runId: "b".repeat(64),
          providers: [
            { id: "fenshot-localizer-v1" },
            { id: "fenshot-labeler-v1" },
          ],
          boards: [
            {
              corners: [
                [0, 0],
                [160, 0],
                [160, 160],
                [0, 160],
              ],
              labels: Array(64).fill("."),
              orientation: "white-bottom",
              probabilities: Array(64).fill(null),
              uncertain: Array(64).fill(true),
            },
          ],
        },
        {
          runId: "c".repeat(64),
          providers: [
            { id: "classical-grid-v1" },
            { id: "fenshot-labeler-v1" },
          ],
          boards: [
            {
              corners: [
                [0, 0],
                [160, 0],
                [160, 160],
                [0, 160],
              ],
              labels: Array(64).fill("P"),
              orientation: "unknown",
              probabilities: Array.from({ length: 64 }, () => [
                0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              ]),
              uncertain: Array(64).fill(false),
            },
            {
              corners: [
                [20, 20],
                [140, 20],
                [140, 140],
                [20, 140],
              ],
              labels: Array(64).fill("N"),
              orientation: "unknown",
              probabilities: Array.from({ length: 64 }, () => [
                0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              ]),
              uncertain: Array(64).fill(false),
            },
          ],
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
    const displayedBoxes = await page.evaluate(() => {
      const image = document.querySelector("#page").getBoundingClientRect();
      const overlay = document
        .querySelector("#overlay")
        .getBoundingClientRect();
      return {
        image: [image.left, image.top, image.width, image.height],
        overlay: [overlay.left, overlay.top, overlay.width, overlay.height],
      };
    });
    displayedBoxes.image.forEach((value, index) =>
      assert.ok(
        Math.abs(value - displayedBoxes.overlay[index]) < 0.5,
        `overlay box differs from image at coordinate ${index}`,
      ),
    );
    const unicodeBounds = await page.evaluate(() => {
      const board = document
          .querySelector("#unicode-board")
          .getBoundingClientRect(),
        pieces = [...document.querySelectorAll("#unicode-board .piece")];
      return {
        width: board.width,
        contained: pieces.every((piece) => {
          const box = piece.getBoundingClientRect(),
            fontSize = Number.parseFloat(getComputedStyle(piece).fontSize);
          return (
            box.left >= board.left - 0.5 &&
            box.right <= board.right + 0.5 &&
            box.top >= board.top - 0.5 &&
            box.bottom <= board.bottom + 0.5 &&
            fontSize <= box.width
          );
        }),
      };
    });
    assert.ok(unicodeBounds.width <= 322);
    assert.equal(unicodeBounds.contained, true);
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
    await page.waitForFunction(() => window.scrollY > 0);
    const stickyGeometry = await page.locator(".geometry-card").boundingBox();
    assert.ok(stickyGeometry);
    assert.ok(stickyGeometry.y >= 15 && stickyGeometry.y < 18);
    assert.ok(stickyGeometry.y + stickyGeometry.height > 100);
    await page.evaluate(() => window.scrollTo(0, 0));
    assert.equal(await page.locator("#labels select").count(), 64);
    assert.match(
      await page.locator("#proposal-status").textContent(),
      /autofilled/,
    );
    assert.equal(await page.locator("#labels .uncertain").count(), 64);
    assert.equal(await page.locator("#piece-palette button").count(), 13);
    await page
      .getByRole("button", {
        name: "Start optional 5-minute session",
        exact: true,
      })
      .click();
    assert.equal(await page.locator("#session-progress").isVisible(), true);
    await page.getByRole("button", { name: "Zoom in", exact: true }).click();
    assert.equal(
      await page.locator("#zoom").evaluate((el) => el.value),
      "125%",
    );
    await page.getByRole("button", { name: "Fit", exact: true }).click();
    assert.equal(
      await page.locator("#zoom").evaluate((el) => el.value),
      "100%",
    );
    await page
      .getByRole("button", { name: "Use image edges", exact: true })
      .click();
    const overlayBox = await page.locator("#overlay").boundingBox();
    assert.ok(overlayBox);
    await page.mouse.move(
      overlayBox.x + overlayBox.width - 2,
      overlayBox.y + 2,
    );
    await page.mouse.down();
    await page.mouse.move(
      overlayBox.x + overlayBox.width - 12,
      overlayBox.y + 12,
    );
    await page.mouse.up();
    await page.locator("#geometry summary").click();
    assert.ok(
      Number(
        await page.getByLabel("TR x coordinate", { exact: true }).inputValue(),
      ) < 160,
    );
    assert.ok(
      Number(
        await page.getByLabel("TR y coordinate", { exact: true }).inputValue(),
      ) > 0,
    );
    await page.getByRole("button", { name: "Undo", exact: true }).click();
    await page.locator("#overlay").focus();
    await page.keyboard.press("1");
    await page.keyboard.press("Shift+ArrowRight");
    assert.equal(
      await page.getByLabel("TL x coordinate", { exact: true }).inputValue(),
      "10",
    );
    await page.getByRole("button", { name: "Undo", exact: true }).click();
    assert.equal(
      await page.getByLabel("TL x coordinate", { exact: true }).inputValue(),
      "0",
    );
    await page.getByLabel("Square a8", { exact: true }).selectOption("K");
    await page.getByRole("button", { name: "Undo", exact: true }).click();
    assert.equal(
      await page.getByLabel("Square a8", { exact: true }).inputValue(),
      ".",
    );
    await page.getByRole("button", { name: "Redo", exact: true }).click();
    assert.equal(
      await page.getByLabel("Square a8", { exact: true }).inputValue(),
      "K",
    );
    await page.getByLabel("Square a8", { exact: true }).focus();
    await page.keyboard.press("q");
    assert.equal(
      await page.getByLabel("Square a8", { exact: true }).inputValue(),
      "q",
    );
    assert.equal(
      await page.locator(":focus").getAttribute("aria-label"),
      "Square a8",
    );
    await page.keyboard.press("Tab");
    assert.equal(
      await page.locator(":focus").getAttribute("aria-label"),
      "Square b8",
    );
    await page.keyboard.press("Shift+Tab");
    await page.getByRole("button", { name: "Undo", exact: true }).click();
    await page.locator("#proposal-choice").selectOption("1");
    assert.equal(
      await page.getByLabel("Square a8", { exact: true }).inputValue(),
      "K",
    );
    assert.match(
      await page.locator("#proposal-status").textContent(),
      /preserved/,
    );
    await page
      .getByRole("button", { name: "Add proposed board 2", exact: true })
      .click();
    assert.equal(
      await page.getByRole("button", { name: "Board 2", exact: true }).count(),
      1,
    );
    await page.getByRole("button", { name: "Undo", exact: true }).click();
    assert.equal(
      await page.getByRole("button", { name: "Board 2", exact: true }).count(),
      0,
    );
    page.once("dialog", (dialog) => dialog.accept());
    await page.getByLabel("I am a human reviewer", { exact: true }).check();
    await page.locator("#complete-page").check();
    await page
      .getByRole("button", { name: "Use proposed board 1", exact: true })
      .click();
    assert.equal(
      await page.getByLabel("Square a8", { exact: true }).inputValue(),
      "P",
    );
    assert.equal(
      await page
        .getByLabel("I am a human reviewer", { exact: true })
        .isChecked(),
      false,
    );
    assert.equal(await page.locator("#complete-page").isChecked(), false);
    assert.equal(await page.locator("#elapsed").textContent(), "0");
    await page.getByRole("button", { name: "Undo", exact: true }).click();
    assert.equal(
      await page.getByLabel("Square a8", { exact: true }).inputValue(),
      "K",
    );
    await page.getByLabel("I am a human reviewer", { exact: true }).uncheck();
    await page.locator("#complete-page").uncheck();
    for (const [label, value] of [
      ["TL x coordinate", "70"],
      ["TL y coordinate", "40"],
      ["TR x coordinate", "90"],
      ["TR y coordinate", "40"],
      ["BR y coordinate", "120"],
      ["BL y coordinate", "120"],
    ]) {
      await page.getByLabel(label, { exact: true }).fill(value);
      await page.getByLabel(label, { exact: true }).press("Tab");
    }
    await page.getByLabel("Square b1", { exact: true }).focus();
    const perspectiveColor = await page
      .locator("#square-crop")
      .evaluate((canvas) =>
        Array.from(canvas.getContext("2d").getImageData(80, 80, 1, 1).data),
      );
    [45, 92, 137, 255].forEach((expected, index) =>
      assert.ok(Math.abs(perspectiveColor[index] - expected) <= 2),
    );
    const sourceCropAlpha = await page.locator("#crop").evaluate((canvas) => {
      const context = canvas.getContext("2d");
      return [
        context.getImageData(160, 20, 1, 1).data[3],
        context.getImageData(160, 160, 1, 1).data[3],
      ];
    });
    assert.deepEqual(sourceCropAlpha, [0, 255]);
    await page
      .getByRole("button", { name: "Use image edges", exact: true })
      .click();
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
    [30, 10, 40, 255].forEach((expected, index) =>
      assert.ok(Math.abs(color[index] - expected) <= 2),
    );
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
