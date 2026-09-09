import { classifierTiles } from "../src/candidate-runtime.ts";
import { rectifyGrid, type GridCorners } from "../src/grid.ts";
import { decodeRaster, inspectRaster } from "../src/image.ts";

const encode = (value: Uint8Array): string => {
  let binary = "";
  for (let offset = 0; offset < value.length; offset += 0x8000)
    binary += String.fromCharCode(...value.subarray(offset, offset + 0x8000));
  return btoa(binary);
};

const bytes = (value: Float32Array): Uint8Array =>
  new Uint8Array(value.buffer, value.byteOffset, value.byteLength);

const decode = (value: string): Uint8Array<ArrayBuffer> => {
  const binary = atob(value);
  const result = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index++)
    result[index] = binary.charCodeAt(index);
  return result;
};

async function inference(
  model: Uint8Array<ArrayBuffer>,
  input: Float32Array<ArrayBuffer>,
): Promise<Float32Array<ArrayBuffer>> {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL("./parity-worker.ts", import.meta.url), {
      type: "module",
    });
    const timeout = setTimeout(() => {
      worker.terminate();
      reject(new Error("Classifier audit inference timed out"));
    }, 60_000);
    worker.onerror = () => {
      clearTimeout(timeout);
      worker.terminate();
      reject(new Error("Classifier audit worker failed"));
    };
    worker.onmessage = (event) => {
      clearTimeout(timeout);
      worker.terminate();
      if (event.data.error) reject(new Error(event.data.error));
      else resolve(event.data.output as Float32Array<ArrayBuffer>);
    };
    worker.postMessage({ model, input, shape: [64, 3, 96, 96] }, [
      model.buffer,
      input.buffer,
    ]);
  });
}

async function run(
  encoded: string,
  corners: GridCorners,
  modelBytes: string,
): Promise<{
  width: number;
  height: number;
  rgba: string;
  grid: string;
  tensor: string;
  logits: string;
}> {
  const imageBytes = decode(encoded);
  const info = inspectRaster(imageBytes);
  const bitmap = await decodeRaster(
    new File([imageBytes], "private-audit-input", { type: info.mime }),
  );
  try {
    if (bitmap.width !== info.width || bitmap.height !== info.height)
      throw new Error("Decoded dimensions disagree with the container");
    const canvas = document.createElement("canvas");
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) throw new Error("Canvas unavailable");
    context.drawImage(bitmap, 0, 0);
    const clamped = context.getImageData(
      0,
      0,
      bitmap.width,
      bitmap.height,
    ).data;
    const rgba = new Uint8Array(
      clamped.buffer,
      clamped.byteOffset,
      clamped.byteLength,
    );
    const grid = rectifyGrid(
      { data: rgba, width: bitmap.width, height: bitmap.height },
      corners,
      3,
      768,
    ).data;
    const tensor = classifierTiles(grid);
    const logits = await inference(
      decode(modelBytes),
      Float32Array.from(tensor),
    );
    return {
      width: bitmap.width,
      height: bitmap.height,
      rgba: encode(rgba),
      grid: encode(grid),
      tensor: encode(bytes(tensor)),
      logits: encode(bytes(logits)),
    };
  } finally {
    bitmap.close();
  }
}

Object.assign(window, { classifierAuditHarness: { run } });

declare global {
  interface Window {
    classifierAuditHarness: { run: typeof run };
  }
}
