import { RecognitionClient } from "./client.ts";
/** Browser bundler entry: local worker, WASM CPU, no remote backend. */
export function createBrowserClient(timeoutMs?: number): RecognitionClient {
  return new RecognitionClient(
    () =>
      new Worker(new URL("./worker.ts", import.meta.url), { type: "module" }),
    timeoutMs,
  );
}
