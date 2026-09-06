import { imageSchema } from "./contract.ts";
function cubic(x: number): number {
  x = Math.abs(x);
  return x < 1
    ? (1.5 * x - 2.5) * x * x + 1
    : x < 2
      ? ((-0.5 * x + 2.5) * x - 4) * x + 2
      : 0;
}
/** Separable antialiased bicubic, half-pixel centers and 22-bit coefficients.
 * Pinned native Pillow parity is tested before this is used for model inputs. */
export function resizeBicubic(
  rgb: Uint8Array,
  width: number,
  height: number,
  outWidth: number,
  outHeight: number,
): Uint8Array {
  imageSchema.parse({ width, height });
  imageSchema.parse({ width: outWidth, height: outHeight });
  if (rgb.length !== width * height * 3) throw new Error("Invalid RGB raster");
  const coeffs = (input: number, output: number) =>
    Array.from({ length: output }, (_, i) => {
      const scale = input / output,
        filterScale = Math.max(1, scale),
        center = (i + 0.5) * scale,
        support = 2 * filterScale;
      const first = Math.max(0, Math.trunc(center - support + 0.5)),
        last = Math.min(input, Math.trunc(center + support + 0.5));
      const weights = Array.from({ length: last - first }, (_, j) =>
        cubic((first + j - center + 0.5) / filterScale),
      );
      const sum = weights.reduce((a, b) => a + b, 0);
      return {
        first,
        weights: weights.map((w) =>
          Math.trunc((w / sum) * 4194304 + (w < 0 ? -0.5 : 0.5)),
        ),
      };
    });
  const horizontal = coeffs(width, outWidth),
    vertical = coeffs(height, outHeight),
    temp = new Uint8Array(outWidth * height * 3),
    out = new Uint8Array(outWidth * outHeight * 3);
  const byte = (sum: number) =>
    Math.max(0, Math.min(255, Math.floor(sum / 4194304)));
  for (let y = 0; y < height; y++)
    for (let x = 0; x < outWidth; x++)
      for (let c = 0; c < 3; c++) {
        const kernel = horizontal[x]!;
        let sum = 2097152;
        kernel.weights.forEach(
          (w, k) => (sum += rgb[(y * width + kernel.first + k) * 3 + c]! * w),
        );
        temp[(y * outWidth + x) * 3 + c] = byte(sum);
      }
  for (let y = 0; y < outHeight; y++)
    for (let x = 0; x < outWidth; x++)
      for (let c = 0; c < 3; c++) {
        const kernel = vertical[y]!;
        let sum = 2097152;
        kernel.weights.forEach(
          (w, k) =>
            (sum += temp[((kernel.first + k) * outWidth + x) * 3 + c]! * w),
        );
        out[(y * outWidth + x) * 3 + c] = byte(sum);
      }
  return out;
}
export function mobileCrop(
  rgb: Uint8Array,
  width: number,
  height: number,
): Uint8Array {
  const scale = 256 / Math.min(width, height),
    w = Math.floor(width * scale),
    h = Math.floor(height * scale),
    resized = resizeBicubic(rgb, width, height, w, h);
  const left = Math.round((w - 224) / 2),
    top = Math.round((h - 224) / 2),
    out = new Uint8Array(224 * 224 * 3);
  for (let y = 0; y < 224; y++)
    out.set(
      resized.subarray(
        ((top + y) * w + left) * 3,
        ((top + y) * w + left + 224) * 3,
      ),
      y * 224 * 3,
    );
  return out;
}
export function mobileNormalize(rgb: Uint8Array): Float32Array {
  if (rgb.length !== 224 * 224 * 3) throw new Error("Invalid MobileNet crop");
  const out = new Float32Array(3 * 224 * 224),
    mean = [0.485, 0.456, 0.406],
    std = [0.229, 0.224, 0.225];
  for (let c = 0; c < 3; c++)
    for (let i = 0; i < 224 * 224; i++)
      out[c * 224 * 224 + i] = Math.fround(
        Math.fround(
          Math.fround(rgb[i * 3 + c]! / 255) - Math.fround(mean[c]!),
        ) / Math.fround(std[c]!),
      );
  return out;
}
export function yolox416(rgb: Uint8Array): Float32Array {
  if (rgb.length !== 416 * 416 * 3)
    throw new Error("Expected exact 416 RGB runtime probe");
  const out = new Float32Array(rgb.length);
  for (let c = 0; c < 3; c++)
    for (let i = 0; i < 416 * 416; i++)
      out[c * 416 * 416 + i] = rgb[i * 3 + 2 - c]!;
  return out;
}
