import * as ort from "onnxruntime-web/wasm";
import { verifiedAsset } from "../src/assets.ts";
self.onmessage = async (event: MessageEvent) => {
  try {
    const { model, input, shape } = event.data as {
      model: Uint8Array<ArrayBuffer>;
      input: Float32Array<ArrayBuffer>;
      shape: number[];
    };
    const [mjs, wasm] = await Promise.all([
      verifiedAsset("ort-mjs"),
      verifiedAsset("ort-wasm"),
    ]);
    const mjsUrl = URL.createObjectURL(
        new Blob([mjs], { type: "text/javascript" }),
      ),
      wasmUrl = URL.createObjectURL(
        new Blob([wasm], { type: "application/wasm" }),
      );
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = { mjs: mjsUrl, wasm: wasmUrl };
    const start = performance.now();
    const session = await ort.InferenceSession.create(model, {
      executionProviders: ["wasm"],
    });
    const initMs = performance.now() - start,
      tensor = new ort.Tensor("float32", input, shape),
      times: number[] = [];
    let output = new Float32Array();
    for (let i = 0; i < 6; i++) {
      const begin = performance.now(),
        result = await session.run({ [session.inputNames[0]!]: tensor });
      times.push(performance.now() - begin);
      output = Float32Array.from(
        result[session.outputNames[0]!]!.data as Float32Array,
      );
      for (const t of Object.values(result)) t.dispose();
    }
    tensor.dispose();
    await session.release();
    URL.revokeObjectURL(mjsUrl);
    URL.revokeObjectURL(wasmUrl);
    self.postMessage({ output, initMs, times });
  } catch {
    self.postMessage({ error: "Native model WASM execution failed" });
  }
};
