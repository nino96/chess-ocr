import { RecognitionClient } from "../src/client.ts";
import { VERSION } from "../src/contract.ts";
import { mobileCrop, mobileNormalize, yolox416 } from "../src/preprocess.ts";
import { decodeYolox, pageTiles } from "../src/geometry.ts";
interface Artifact {
  model: string;
  model_sha256: string;
  output: string;
  output_sha256: string;
  shape: number[];
  preprocessing: { tensor_path: string; sha256: string; shape: number[] };
}
async function bytes(
  path: string,
  hash: string,
): Promise<Uint8Array<ArrayBuffer>> {
  const response = await fetch("/" + path);
  if (!response.ok) throw new Error("Missing parity artifact");
  const b = new Uint8Array(await response.arrayBuffer());
  const actual = [...new Uint8Array(await crypto.subtle.digest("SHA-256", b))]
    .map((v) => v.toString(16).padStart(2, "0"))
    .join("");
  if (actual !== hash) throw new Error("Parity artifact hash mismatch");
  return b;
}
function difference(a: ArrayLike<number>, b: ArrayLike<number>) {
  if (a.length !== b.length) throw new Error("Shape mismatch");
  let max = 0;
  for (let i = 0; i < a.length; i++) {
    if (!Number.isFinite(a[i]!) || !Number.isFinite(b[i]!))
      throw new Error("Nonfinite parity output");
    max = Math.max(max, Math.abs(a[i]! - b[i]!));
  }
  return max;
}
async function parity(
  artifact: Artifact,
  source: { path: string; sha256: string },
  modelName: string,
) {
  const raw = await bytes(source.path, source.sha256),
    input =
      modelName === "yolox"
        ? yolox416(raw)
        : mobileNormalize(mobileCrop(raw, 416, 416));
  const referenceBytes = await bytes(
      artifact.preprocessing.tensor_path,
      artifact.preprocessing.sha256,
    ),
    ref = new Float32Array(referenceBytes.buffer),
    inputMaxAbs = difference(input, ref);
  const model = await bytes(artifact.model, artifact.model_sha256),
    nativeBytes = await bytes(artifact.output, artifact.output_sha256),
    expected = new Float32Array(nativeBytes.buffer);
  const result = await new Promise<{
    output: Float32Array;
    initMs: number;
    times: number[];
  }>((resolve, reject) => {
    const w = new Worker(new URL("./parity-worker.ts", import.meta.url), {
        type: "module",
      }),
      timer = setTimeout(() => {
        w.terminate();
        reject(new Error("Parity timed out"));
      }, 60_000);
    w.onerror = () => {
      clearTimeout(timer);
      w.terminate();
      reject(new Error("Parity worker failed"));
    };
    w.onmessage = (e) => {
      clearTimeout(timer);
      w.terminate();
      if (e.data.error) reject(new Error(e.data.error));
      else resolve(e.data);
    };
    w.postMessage({ model, input, shape: artifact.preprocessing.shape }, [
      model.buffer,
      input.buffer,
    ]);
  });
  const outputMaxAbs = difference(result.output, expected);
  const decoded = modelName === "yolox" ? decodeYolox(result.output) : [];
  return {
    modelName,
    inputMaxAbs,
    outputMaxAbs,
    initMs: result.initMs,
    inferenceMs: result.times,
    outputElements: result.output.length,
    decodedDetections: decoded.length,
  };
}
function raster(width: number, height: number): Uint8ClampedArray {
  const a = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const v = (Math.floor(y / 32) + Math.floor(x / 32)) % 2 ? 80 : 220,
        o = (y * width + x) * 4;
      a[o] = a[o + 1] = a[o + 2] = v;
      a[o + 3] = 255;
    }
  return a;
}
async function benchmark() {
  const make = () =>
    new RecognitionClient(
      () =>
        new Worker(new URL("../src/worker.ts", import.meta.url), {
          type: "module",
        }),
    );
  const request = {
    schema: VERSION,
    requestId: "benchmark",
    image: { width: 256, height: 256 },
    selection: { x: 0, y: 0, width: 256, height: 256 },
  };
  const cold: number[] = [],
    warm: number[] = [];
  let status = "";
  for (let i = 0; i < 3; i++) {
    const client = make(),
      start = performance.now();
    status = (
      await client.recognize(
        { ...request, requestId: crypto.randomUUID() },
        raster(256, 256),
      )
    ).status;
    cold.push(performance.now() - start);
    client.cancel();
  }
  const client = make();
  await client.recognize(request, raster(256, 256));
  for (let i = 0; i < 10; i++) {
    const start = performance.now();
    await client.recognize(
      { ...request, requestId: crypto.randomUUID() },
      raster(256, 256),
    );
    warm.push(performance.now() - start);
  }
  const tiles = pageTiles(1600, 1200),
    tilingStart = performance.now();
  for (const tile of tiles)
    await client.recognize(
      {
        ...request,
        requestId: crypto.randomUUID(),
        image: { width: tile.width, height: tile.height },
        selection: { x: 0, y: 0, width: tile.width, height: tile.height },
      },
      raster(tile.width, tile.height),
    );
  const tiledMs = performance.now() - tilingStart,
    start = performance.now();
  const pending = client.recognize(
    { ...request, requestId: crypto.randomUUID() },
    raster(256, 256),
  );
  client.cancel();
  let cancelled = false;
  try {
    await pending;
  } catch {
    cancelled = true;
  }
  return {
    status,
    coldMs: cold,
    warmMs: warm,
    tiling: { page: [1600, 1200], windows: tiles.length, elapsedMs: tiledMs },
    cancellationMs: performance.now() - start,
    cancelled,
    heap:
      (performance as Performance & { memory?: { usedJSHeapSize: number } })
        .memory?.usedJSHeapSize ?? null,
  };
}
Object.assign(window, { runtimeHarness: { parity, benchmark } });
