import { test } from "node:test";
import assert from "node:assert/strict";
import { spawn, spawnSync } from "node:child_process";
import { chromium } from "@playwright/test";

test("connected review saves drafts, recovers reloads, rejects conflicts, accepts one human and archives reset", async () => {
  const build = spawnSync(
    process.execPath,
    [
      "node_modules/typescript/bin/tsc",
      "--project",
      "tsconfig.dataset-app.json",
    ],
    { stdio: "pipe" },
  );
  assert.equal(
    build.status,
    0,
    build.stderr.toString() + build.stdout.toString(),
  );
  const fixture = spawn(
    process.env.DATASET_PYTHON || "work/dataset-venv/bin/python",
    [
      "-u",
      "-c",
      `
import signal
import time
from python.test_dataset_pipeline import PipelineTests
from python import dataset_server as server
fixture = PipelineTests()
fixture.setUp()
def stop(*_):
    raise KeyboardInterrupt()
signal.signal(signal.SIGTERM, stop)
try:
    fixture.sample('first', 'train', 'first')
    fixture.sample('second', 'dev', 'second')
    fixture.sample('third', 'dev', 'third')
    with server.p.connect() as db:
        db.execute("INSERT INTO duplicates VALUES (?,?,?,NULL)", ('first-1', 'second-1', 'perceptual'))
        db.execute("INSERT INTO duplicates VALUES (?,?,?,NULL)", ('first-1', 'third-1', 'exact'))
    server.initialize()
    class Candidate:
        public_identity = {'name': 'test-candidate', 'version': '1', 'qualification': 'synthetic-development-only'}
        calls = 0
        def recognize(self, _image):
            time.sleep(.5)
            self.calls += 1
            return {'boards': [{'corners': [[10, 10], [150, 10], [150, 150], [10, 150]],
                                'labels': list('K' + '.' * 63), 'orientation': 'unknown'}],
                    'model': self.public_identity,
                    'warning': f'Synthetic-only test proposal {self.calls}.'}
    app = server.Server(('127.0.0.1', 0), Candidate())
    print(app.server_port, flush=True)
    try:
        app.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.server_close()
finally:
    fixture.tearDown()
`,
    ],
    { stdio: ["ignore", "pipe", "pipe"] },
  );
  let browser;
  let stderr = "";
  fixture.stderr.on("data", (part) => {
    stderr += part.toString();
  });
  try {
    const port = await new Promise((resolve, reject) => {
      const timer = setTimeout(
        () =>
          reject(new Error("Synthetic review server did not start: " + stderr)),
        15_000,
      );
      fixture.once("exit", () => {
        clearTimeout(timer);
        reject(new Error("Synthetic server exited: " + stderr));
      });
      fixture.stdout.once("data", (part) => {
        clearTimeout(timer);
        resolve(Number(part.toString().trim()));
      });
    });
    const origin = `http://127.0.0.1:${port}`;
    browser = await chromium.launch();
    const context = await browser.newContext();
    await context.addInitScript(() => {
      const revoke = URL.revokeObjectURL.bind(URL);
      window.__duplicateReleases = 0;
      URL.revokeObjectURL = (url) => {
        window.__duplicateReleases += 1;
        revoke(url);
      };
    });
    const page = await context.newPage();
    const errors = [];
    const external = [];
    page.on("pageerror", (e) => errors.push(e.message));
    context.on("request", (r) => {
      if (
        !r.url().startsWith(origin) &&
        !r.url().startsWith("data:") &&
        !r.url().startsWith("blob:")
      )
        external.push(r.url());
    });
    await page.goto(origin);
    await page.locator(".page-card").first().waitFor();
    assert.equal(await page.locator(".page-card").count(), 3);
    await page.locator("#duplicate-workspace").waitFor({ state: "visible" });
    await page.getByText("Pair 1 of 2", { exact: true }).waitFor();
    await page.locator("#duplicate-image-a[src^='blob:']").waitFor();
    await page.locator("#duplicate-image-b[src^='blob:']").waitFor();
    await page
      .locator("#duplicate-image-a")
      .evaluate((image) => image.decode());
    await page
      .locator("#duplicate-image-b")
      .evaluate((image) => image.decode());
    assert.match(
      await page.locator("#duplicate-image-a").getAttribute("src"),
      /^blob:/,
    );
    assert.match(
      await page.locator("#duplicate-label-a").textContent(),
      /Document 1 · page 1 · train/,
    );
    assert.match(
      await page.locator("#duplicate-label-b").textContent(),
      /Document 2 · page 1 · dev/,
    );
    await page.getByRole("button", { name: "200%", exact: true }).click();
    assert.equal(
      await page.locator("#duplicate-workspace").getAttribute("data-zoom"),
      "200",
    );
    assert.deepEqual(
      await page.locator("#duplicate-image-a").evaluate((image) => ({
        naturalWidth: image.naturalWidth,
        renderedWidth: image.width,
      })),
      { naturalWidth: 160, renderedWidth: 320 },
    );
    await page.getByRole("button", { name: "100%", exact: true }).click();
    assert.deepEqual(
      await page.locator("#duplicate-image-a").evaluate((image) => ({
        naturalWidth: image.naturalWidth,
        renderedWidth: image.width,
      })),
      { naturalWidth: 160, renderedWidth: 160 },
    );
    await page.locator(".duplicate-comparison").press("ArrowRight");
    await page.getByText("Pair 2 of 2", { exact: true }).waitFor();
    await page.locator("#duplicate-image-a[src^='blob:']").waitFor();
    await page
      .locator("#duplicate-image-a")
      .evaluate((image) => image.decode());
    assert.ok(await page.evaluate(() => window.__duplicateReleases >= 2));
    assert.equal(
      await page
        .getByRole("button", { name: "Different artwork / pages", exact: true })
        .isDisabled(),
      true,
    );
    await page
      .getByRole("button", { name: "Previous pair", exact: true })
      .click();
    await page.getByText("Pair 1 of 2", { exact: true }).waitFor();
    const releasesBeforeDecision = await page.evaluate(
      () => window.__duplicateReleases,
    );
    let failQueueRefresh = true;
    await page.route("**/api/queue", (route) => {
      if (failQueueRefresh) return route.abort();
      return route.continue();
    });
    page.once("dialog", (dialog) => void dialog.accept());
    await page
      .getByRole("button", { name: "Different artwork / pages", exact: true })
      .click();
    await page.locator("#duplicate-workspace").waitFor({ state: "hidden" });
    assert.equal(
      await page.locator("#duplicate-image-a").getAttribute("src"),
      null,
    );
    assert.ok(
      await page.evaluate(
        (before) => window.__duplicateReleases >= before + 2,
        releasesBeforeDecision,
      ),
    );
    failQueueRefresh = false;
    await page
      .getByRole("button", { name: "Refresh status", exact: true })
      .click();
    await page.getByText("Pair 1 of 1", { exact: true }).waitFor();
    await page.unroute("**/api/queue");
    await page
      .getByRole("button", { name: "Review next page", exact: true })
      .click();
    let editor = page.frameLocator("#editor");
    await editor.locator("#app").waitFor({ state: "visible" });
    await page
      .getByRole("button", { name: "Generate model proposal", exact: true })
      .click();
    assert.equal(
      await page.locator("#editor").evaluate((element) => element.inert),
      true,
    );
    await editor
      .getByText(/Synthetic-only test proposal 1.*human inspection/)
      .waitFor();
    assert.equal(
      await page.locator("#editor").evaluate((element) => element.inert),
      false,
    );
    assert.equal(
      await editor.getByLabel("Square a8", { exact: true }).inputValue(),
      "K",
    );
    const rereadId = "d".repeat(64);
    await page.route("**/api/reread/start", (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          schema: "chess-ocr-board-reread/1",
          state: "starting",
          request_id: rereadId,
        }),
      }),
    );
    await page.route(`**/api/reread/${rereadId}`, (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ state: "complete" }),
      }),
    );
    await page.route("**/api/reread/apply", (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          schema: "chess-ocr-board-reread/1",
          state: "applicable",
          result: {
            boardIndex: 0,
            corners: [
              [10, 10],
              [150, 10],
              [150, 150],
              [10, 150],
            ],
            labels: Array(64).fill("N"),
            probabilities: Array(64).fill(null),
            uncertain: Array(64).fill(true),
          },
        }),
      }),
    );
    await editor
      .locator("#reread-labeler option")
      .nth(1)
      .waitFor({ state: "attached" });
    await editor
      .getByRole("button", { name: "Re-read this board", exact: true })
      .click();
    await editor
      .getByText(/label-only proposal is ready/i, { exact: false })
      .waitFor();
    await editor
      .getByRole("button", { name: "Apply re-read proposal", exact: true })
      .click();
    assert.equal(
      await editor.getByLabel("Square a8", { exact: true }).inputValue(),
      "N",
    );
    assert.equal(
      await editor
        .getByLabel("I am a human reviewer", { exact: true })
        .isChecked(),
      false,
    );
    await editor.getByRole("button", { name: "Undo", exact: true }).click();
    assert.equal(
      await editor.getByLabel("Square a8", { exact: true }).inputValue(),
      "K",
    );
    await page.unroute("**/api/reread/start");
    await page.unroute(`**/api/reread/${rereadId}`);
    await page.unroute("**/api/reread/apply");
    assert.equal(
      await editor
        .getByLabel("I am a human reviewer", { exact: true })
        .isChecked(),
      false,
    );
    await editor.getByLabel("Square b8", { exact: true }).selectOption("Q");
    await editor.getByLabel("I am a human reviewer", { exact: true }).check();
    await editor.locator("#complete-page").check();
    page.once("dialog", (dialog) => void dialog.accept());
    await page
      .getByRole("button", { name: "Generate model proposal", exact: true })
      .click();
    await editor
      .getByText(/Synthetic-only test proposal 2.*human inspection/)
      .waitFor();
    assert.equal(
      await editor.getByLabel("Square b8", { exact: true }).inputValue(),
      ".",
    );
    assert.equal(
      await editor
        .getByLabel("I am a human reviewer", { exact: true })
        .isChecked(),
      false,
    );
    assert.equal(await editor.locator("#complete-page").isChecked(), false);
    await editor.getByLabel("Square a8", { exact: true }).selectOption("K");
    await editor
      .getByLabel("Reviewer identity", { exact: true })
      .fill("owner-test");
    await editor.getByLabel("I am a human reviewer", { exact: true }).check();
    await page.getByText("Draft saved on GX10", { exact: true }).waitFor();
    await page.reload();
    await page.locator(".page-card").first().click();
    editor = page.frameLocator("#editor");
    await page.getByText("Saved draft restored", { exact: true }).waitFor();
    assert.equal(
      await editor.getByLabel("Square a8", { exact: true }).inputValue(),
      "K",
    );
    assert.equal(
      await editor
        .getByLabel("Reviewer identity", { exact: true })
        .inputValue(),
      "owner-test",
    );

    const other = await context.newPage();
    await other.goto(origin);
    await other.locator(".page-card").first().click();
    await other.getByText("Saved draft restored", { exact: true }).waitFor();
    await editor.getByLabel("Square b8", { exact: true }).selectOption("Q");
    await page.getByText("Draft saved on GX10", { exact: true }).waitFor();
    await other
      .frameLocator("#editor")
      .getByLabel("Square b8", { exact: true })
      .selectOption("R");
    await other
      .getByText("draft changed in another tab; reload before saving", {
        exact: true,
      })
      .waitFor();
    await other.close();

    await page.route("**/api/draft", (route) => route.abort());
    await editor.getByLabel("Square c8", { exact: true }).selectOption("N");
    await page
      .getByText("Draft not saved — retry before leaving", { exact: true })
      .waitFor();
    await page
      .getByRole("button", { name: "Back to pages", exact: true })
      .click();
    assert.equal(await page.locator("#editor-section").isVisible(), true);
    await page.unroute("**/api/draft");
    await page
      .getByRole("button", { name: "Retry saving draft", exact: true })
      .click();
    await page.getByText("Draft saved on GX10", { exact: true }).waitFor();
    await editor.locator("#complete-page").check();
    let dropped = false;
    await page.route("**/api/review", async (route) => {
      if (!dropped) {
        dropped = true;
        const committed = await route.fetch();
        assert.equal(committed.status(), 200);
        await route.abort();
      } else await route.continue();
    });
    await editor
      .getByRole("button", { name: "Submit review & next", exact: true })
      .click();
    await page.getByText("Document 2 · page 1", { exact: true }).waitFor();
    await editor.locator("#app").waitFor({ state: "visible" });
    assert.equal(
      await editor
        .getByLabel("Reviewer identity", { exact: true })
        .inputValue(),
      "owner-test",
    );
    await editor
      .getByRole("button", { name: "No board on this page", exact: true })
      .click();
    await editor.locator("#complete-page").check();
    await editor
      .getByRole("button", { name: "Submit review & next", exact: true })
      .click();
    await page.getByText("Document 3 · page 1", { exact: true }).waitFor();
    await editor.locator("#app").waitFor({ state: "visible" });
    await editor
      .getByLabel("Page kind", { exact: true })
      .selectOption("partial");
    await editor.locator("#complete-page").check();
    await editor
      .getByRole("button", { name: "Submit review & next", exact: true })
      .click();
    await page.getByText(/3 \/ 3 pages accepted/).waitFor();
    assert.match(
      await page.locator("#summary").textContent(),
      /3 \/ 3 pages accepted/,
    );
    assert.match(
      await page.locator("#summary").textContent(),
      /3 \/ 20 review decisions/,
    );
    await page.locator("#filter").selectOption("all");
    assert.equal(await page.locator(".page-card.accepted").count(), 3);
    const touchContext = await browser.newContext({
      viewport: { width: 390, height: 844 },
      hasTouch: true,
      isMobile: true,
    });
    const touchPage = await touchContext.newPage();
    await touchPage.goto(origin);
    await touchPage
      .locator("#duplicate-workspace")
      .waitFor({ state: "visible" });
    assert.equal(
      await touchPage
        .locator(".duplicate-comparison")
        .evaluate(
          (element) =>
            getComputedStyle(element).gridTemplateColumns.split(" ").length,
        ),
      1,
    );
    await touchPage.getByRole("button", { name: "100%", exact: true }).tap();
    assert.equal(
      await touchPage.locator("#duplicate-workspace").getAttribute("data-zoom"),
      "100",
    );
    await touchPage.locator("#filter").selectOption("all");
    await touchPage.locator(".page-card").nth(1).tap();
    const touchEditor = touchPage.frameLocator("#editor");
    await touchEditor
      .getByRole("button", { name: "No board on this page", exact: true })
      .tap();
    await touchPage.getByText("Draft saved on GX10", { exact: true }).waitFor();
    assert.equal(
      await touchEditor.getByLabel("Page kind", { exact: true }).inputValue(),
      "negative",
    );
    await touchContext.close();
    page.once("dialog", (dialog) => void dialog.accept());
    await page
      .getByRole("button", { name: "Mark duplicate", exact: true })
      .click();
    await page
      .getByText("Duplicate decision saved.", { exact: true })
      .waitFor();
    await page.locator("#duplicate-workspace").waitFor({ state: "hidden" });
    await page.getByText("Start over", { exact: true }).click();
    await page
      .getByRole("button", { name: "Start over…", exact: true })
      .click();
    await page
      .getByRole("button", { name: "Archive and start over", exact: true })
      .click();
    await page
      .getByText("type START OVER exactly to reset the dataset", {
        exact: true,
      })
      .waitFor();
    await page
      .getByLabel("Type START OVER", { exact: true })
      .fill("START OVER");
    await page
      .getByRole("button", { name: "Archive and start over", exact: true })
      .click();
    await page.locator("#reset-dialog").waitFor({ state: "hidden" });
    await page.getByText(/0 \/ 0 pages accepted/).waitFor();
    assert.match(
      await page.locator("#summary").textContent(),
      /0 \/ 0 pages accepted/,
    );
    assert.match(
      await page.locator("#summary").textContent(),
      /3 \/ 20 review decisions/,
    );
    assert.match(await page.locator("#message").textContent(), /recoverable/);
    await page.getByText("Archives", { exact: true }).click();
    await page.locator(".archive-entry").waitFor();
    assert.match(await page.locator(".archive-entry").textContent(), /MiB/);
    await page
      .getByRole("button", { name: "Delete archive…", exact: true })
      .click();
    await page.locator("#archive-delete-cancel").click();
    assert.equal(await page.locator(".archive-entry").count(), 1);
    await page
      .getByRole("button", { name: "Delete archive…", exact: true })
      .click();
    await page
      .getByRole("button", { name: "Permanently delete archive", exact: true })
      .click();
    await page
      .getByText("type DELETE exactly to delete this archive", { exact: true })
      .waitFor();
    await page.getByLabel("Type DELETE", { exact: true }).fill("DELETE");
    await page
      .getByRole("button", { name: "Permanently delete archive", exact: true })
      .click();
    await page.locator("#archive-delete-dialog").waitFor({ state: "hidden" });
    await page.getByText("No archives.", { exact: true }).waitFor();
    assert.equal(await page.locator(".archive-entry").count(), 0);
    assert.match(
      await page.locator("#summary").textContent(),
      /3 \/ 20 review decisions/,
    );
    await page.reload();
    await page.getByText("Archives", { exact: true }).click();
    await page.getByText("No archives.", { exact: true }).waitFor();
    assert.deepEqual(errors, []);
    assert.deepEqual(external, []);
  } finally {
    if (browser) await browser.close();
    fixture.kill("SIGTERM");
    await new Promise((resolve) => fixture.once("exit", resolve));
  }
});
