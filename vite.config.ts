import { defineConfig, type Plugin, type UserConfig } from "vite";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import manifest from "./assets.lock.json";

export default defineConfig(async ({ command, isPreview }) => {
  const remoteCandidate = process.env.CHESS_OCR_REMOTE_CANDIDATE;
  if (
    remoteCandidate !== undefined &&
    remoteCandidate !== "0" &&
    remoteCandidate !== "1"
  )
    throw new Error("CHESS_OCR_REMOTE_CANDIDATE must be 0 or 1");
  const runtimeAssets: Plugin = {
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
  };
  const plugins: Plugin[] = [runtimeAssets];
  if (remoteCandidate === "1") {
    if (command !== "serve" || isPreview)
      throw new Error(
        "Configured remote candidates are available only from the loopback development server",
      );
    const { createRemoteCandidatePlugin } = await import(
      "./scripts/remote-candidate-server.ts"
    );
    plugins.push(
      await createRemoteCandidatePlugin(resolve(import.meta.dirname)),
    );
  }
  const config: UserConfig = {
    publicDir: false,
    worker: { format: "es" },
    plugins,
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
  };
  return config;
});
