import { test, expect } from "@playwright/test";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { resolve, extname } from "node:path";
test("verified application reloads and executes WASM after its origin server is stopped", async ({
  page,
}) => {
  const server = createServer(async (req, res) => {
    const path = decodeURIComponent(
        new URL(req.url!, "http://localhost").pathname,
      ),
      file = resolve("dist", "." + (path === "/" ? "/index.html" : path));
    if (!file.startsWith(resolve("dist") + "/")) {
      res.writeHead(403).end();
      return;
    }
    try {
      const bytes = await readFile(file);
      res.setHeader(
        "Content-Type",
        (
          {
            ".html": "text/html",
            ".js": "text/javascript",
            ".mjs": "text/javascript",
            ".css": "text/css",
            ".wasm": "application/wasm",
            ".json": "application/json",
          } as Record<string, string>
        )[extname(file)] ?? "application/octet-stream",
      );
      res.end(bytes);
    } catch {
      res.writeHead(404).end();
    }
  });
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  const address = server.address();
  if (!address || typeof address === "string")
    throw new Error("No local origin");
  const origin = `http://127.0.0.1:${address.port}`;
  const external: string[] = [];
  page.on("request", (r) => {
    if (!r.url().startsWith(origin) && !r.url().startsWith("blob:"))
      external.push(r.url());
  });
  try {
    await page.goto(origin);
    await expect(page.locator("#offline")).toContainText("Offline ready");
    await page.waitForFunction(
      () => navigator.serviceWorker.controller !== null,
    );
    server.closeAllConnections();
    await new Promise<void>((r, j) => server.close((e) => (e ? j(e) : r())));
    // No network service exists at the application's origin for navigation, worker or model fetches.
    await page.reload();
    await expect(page.locator("#offline")).toContainText("Offline ready");
    const bytes = await page.evaluate(() => {
      const c = document.createElement("canvas");
      c.width = 256;
      c.height = 256;
      const ctx = c.getContext("2d")!;
      ctx.fillStyle = "#aaa";
      ctx.fillRect(0, 0, 256, 256);
      return Array.from(
        Uint8Array.from(atob(c.toDataURL().split(",")[1]!), (c) =>
          c.charCodeAt(0),
        ),
      );
    });
    await page.locator("#file").setInputFiles({
      name: "original-solid.png",
      mimeType: "image/png",
      buffer: Buffer.from(bytes),
    });
    await expect(page.locator("#status")).toContainText("Image ready");
    await page.getByRole("button", { name: "Read selection" }).click();
    await expect(page.locator("#status")).toContainText("Result ready");
    expect(external).toEqual([]);
  } finally {
    if (server.listening) {
      server.closeAllConnections();
      await new Promise<void>((r) => server.close(() => r()));
    }
  }
});
