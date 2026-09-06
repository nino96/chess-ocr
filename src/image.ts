import { LIMITS, imageSchema } from "./contract.ts";
/** Read dimensions before invoking a decoder. Only PNG and JPEG raster containers are admitted. */
export function inspectRaster(bytes: Uint8Array): {
  width: number;
  height: number;
  mime: string;
} {
  if (bytes.length > LIMITS.bytes || bytes.length < 24)
    throw new Error("Image is empty, corrupt, or exceeds 20 MiB");
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if ([137, 80, 78, 71, 13, 10, 26, 10].every((v, i) => bytes[i] === v)) {
    if (view.getUint32(8) !== 13 || view.getUint32(12) !== 0x49484452)
      throw new Error("Invalid PNG header");
    return {
      ...imageSchema.parse({
        width: view.getUint32(16),
        height: view.getUint32(20),
      }),
      mime: "image/png",
    };
  }
  if (bytes[0] === 255 && bytes[1] === 216) {
    let offset = 2;
    while (offset + 4 <= bytes.length) {
      if (bytes[offset++] !== 255) throw new Error("Invalid JPEG marker");
      while (bytes[offset] === 255) offset++;
      const marker = bytes[offset++]!;
      if (marker === 0xd9 || marker === 0xda) break;
      if (marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7)) continue;
      if (offset + 2 > bytes.length) break;
      const length = view.getUint16(offset);
      if (length < 2 || offset + length > bytes.length)
        throw new Error("Invalid JPEG segment");
      if ([0xc0, 0xc1, 0xc2].includes(marker)) {
        if (length < 8) throw new Error("Invalid JPEG frame");
        return {
          ...imageSchema.parse({
            height: view.getUint16(offset + 3),
            width: view.getUint16(offset + 5),
          }),
          mime: "image/jpeg",
        };
      }
      offset += length;
    }
  }
  throw new Error("Use a supported PNG or JPEG image");
}
export async function decodeRaster(file: File): Promise<ImageBitmap> {
  if (file.size > LIMITS.bytes) throw new Error("Image exceeds 20 MiB");
  const bytes = new Uint8Array(await file.arrayBuffer());
  const info = inspectRaster(bytes);
  const bitmap = await createImageBitmap(
    new Blob([bytes], { type: info.mime }),
  );
  try {
    imageSchema.parse({ width: bitmap.width, height: bitmap.height });
  } catch {
    bitmap.close();
    throw new Error("Decoded image exceeds limits");
  }
  return bitmap;
}
