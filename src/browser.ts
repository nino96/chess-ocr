import { RecognitionClient } from "./client.ts";
import type { CandidateConfig } from "./candidate.ts";
/** Browser bundler entry: local worker, WASM CPU, no remote backend. */
export function createBrowserClient(timeoutMs?: number): RecognitionClient {
  return new RecognitionClient(
    () =>
      new Worker(new URL("./worker.ts", import.meta.url), { type: "module" }),
    timeoutMs,
  );
}

/** Explicit local candidate entry. Model bytes are cloned for cancellation/retry workers. */
export function createCandidateBrowserClient(
  config: CandidateConfig,
  timeoutMs?: number,
): RecognitionClient {
  return new RecognitionClient(() => {
    const worker = new Worker(new URL("./trained-worker.ts", import.meta.url), {
      type: "module",
    });
    const classifier = config.classifier.slice();
    const detector = config.detector.slice();
    worker.postMessage(
      {
        type: "configure",
        manifest: config.manifest,
        identity: config.identity,
        classifier,
        detector,
      },
      [classifier.buffer, detector.buffer],
    );
    return worker;
  }, timeoutMs);
}
