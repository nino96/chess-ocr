// Asset imports adapted from nino96/chess-reader c2ece9a, recognition/assets.ts.
// Source provenance and intentional contract differences: docs/provenance/reuse.md.
import modelUrl from "@scoriiu/fenshot/model/chess-tiles-v2.onnx?url";
import ortWasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.wasm?url";
import ortMjsUrl from "onnxruntime-web/ort-wasm-simd-threaded.mjs?url";
import manifest from "../assets.lock.json";
const urls: Record<string, string> = {
  fenshot: modelUrl,
  "ort-wasm": ortWasmUrl,
  "ort-mjs": ortMjsUrl,
};
export const assets = manifest.assets.map((asset) => ({
  ...asset,
  url: import.meta.env.DEV ? `/__ocr_assets/${asset.id}` : urls[asset.id]!,
}));
export async function verifiedAsset(
  id: string,
): Promise<Uint8Array<ArrayBuffer>> {
  const asset = assets.find((a) => a.id === id);
  if (!asset) throw new Error("Unknown asset");
  const response = await fetch(asset.url, {
    credentials: "omit",
    redirect: "error",
  });
  if (
    !response.ok ||
    Number(response.headers.get("content-length")) > asset.bytes
  )
    throw new Error("Asset unavailable");
  const reader = response.body?.getReader();
  if (!reader) throw new Error("Asset unavailable");
  const bytes = new Uint8Array(asset.bytes);
  let offset = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (offset + value.length > bytes.length)
        throw new Error("Asset size mismatch");
      bytes.set(value, offset);
      offset += value.length;
    }
  } finally {
    await reader.cancel();
  }
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  const actual = [...digest]
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
  if (offset !== asset.bytes || actual !== asset.sha256)
    throw new Error("Asset integrity check failed");
  return bytes;
}
