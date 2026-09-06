import * as ort from "onnxruntime-web/wasm";
import { verifiedAsset } from "./assets.ts";
import { recognize } from "./fenshot.ts";
import { requestSchema, VERSION } from "./contract.ts";
let session: Promise<ort.InferenceSession> | null = null;
function load(): Promise<ort.InferenceSession> {
  session ??= (async () => {
    const [model, mjs, wasm] = await Promise.all(
      ["fenshot", "ort-mjs", "ort-wasm"].map(verifiedAsset),
    );
    const mjsUrl = URL.createObjectURL(
      new Blob([mjs!], { type: "text/javascript" }),
    );
    const wasmUrl = URL.createObjectURL(
      new Blob([wasm!], { type: "application/wasm" }),
    );
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = { mjs: mjsUrl, wasm: wasmUrl };
    try {
      return await ort.InferenceSession.create(model!, {
        executionProviders: ["wasm"],
      });
    } finally {
      URL.revokeObjectURL(mjsUrl);
      URL.revokeObjectURL(wasmUrl);
    }
  })();
  return session;
}
let busy = false;
self.onmessage = async (event: MessageEvent) => {
  if (busy) return;
  const parsed = requestSchema.safeParse(event.data?.request);
  if (!parsed.success) return;
  const request = parsed.data;
  busy = true;
  try {
    if (!(event.data.rgba instanceof Uint8ClampedArray))
      throw new Error("Invalid input");
    self.postMessage(await recognize(request, event.data.rgba, await load()));
  } catch {
    session = null;
    self.postMessage({
      schema: VERSION,
      requestId: request.requestId,
      image: request.image,
      status: "error",
      boards: [],
      warnings: ["Local recognition failed. Verify local assets or retry."],
      model: null,
      preprocessing: "fenshot-0.1.4/rgba-gray-bilinear-256/1",
      timings: { totalMs: 0 },
    });
  } finally {
    busy = false;
  }
};
