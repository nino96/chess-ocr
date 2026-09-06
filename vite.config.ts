import { defineConfig } from "vite";
import { readFile } from "node:fs/promises";
import manifest from "./assets.lock.json";
export default defineConfig({
  publicDir: false,
  worker: { format: "es" },
  plugins: [
    {
      name: "verbatim-local-runtime-assets",
      configureServer(server) {
        // Development transforms JavaScript modules. Serve the hash-verified ORT
        // bootstrap verbatim, just as a production static asset is served.
        server.middlewares.use((request, response, next) => {
          const asset = manifest.assets.find(
            (a) => request.url === `/__ocr_assets/${a.id}`,
          );
          if (!asset) {
            next();
            return;
          }
          if (request.method !== "GET") {
            response.statusCode = 405;
            response.end();
            return;
          }
          void readFile(asset.source)
            .then((bytes) => {
              response.setHeader(
                "Content-Type",
                asset.id === "ort-mjs"
                  ? "text/javascript"
                  : "application/octet-stream",
              );
              response.setHeader("Content-Length", bytes.length);
              response.end(bytes);
            })
            .catch(() => {
              response.statusCode = 404;
              response.end();
            });
        });
      },
    },
  ],
  server: {
    host: "127.0.0.1",
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
  preview: { host: "127.0.0.1" },
});
