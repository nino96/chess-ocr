/** Conservative procedural print-like pixel effects; not empirical print calibration. */
const MAX_PIXELS = 4_000_000;
const VARIANTS = new Set(["blank", "paper", "faded", "soft"]);
const keys = [
  "variant",
  "seed",
  "paper_noise",
  "ink_fade",
  "illumination",
  "blur_radius",
];

const assert = (value, message) => {
  if (!value) throw new Error(message);
};
const clamp = (value) => Math.max(0, Math.min(255, value));
const mix = (value) => {
  value = (value + 0x6d2b79f5) >>> 0;
  value = Math.imul(value ^ (value >>> 15), value | 1);
  value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
  return ((value ^ (value >>> 14)) >>> 0) / 2 ** 32;
};
const seedNumber = (seed, index) => {
  const input = `${seed}:${index}`;
  let value = 2166136261;
  for (let offset = 0; offset < input.length; offset++) {
    value ^= input.charCodeAt(offset);
    value = Math.imul(value, 16777619);
  }
  return value >>> 0;
};

/** Return a small, deterministic effect recipe. `blank` deliberately changes no pixels. */
export function degradationRecipe(seed, index) {
  assert(
    (typeof seed === "string" || Number.isInteger(seed)) &&
      Number.isInteger(index) &&
      index >= 0,
    "seed and index must be bounded deterministic values",
  );
  const variant = ["blank", "paper", "faded", "soft"][index % 4];
  const recipe = {
    variant,
    seed: seedNumber(seed, index),
    paper_noise: variant === "blank" ? 0 : variant === "paper" ? 1.2 : 0.7,
    ink_fade:
      variant === "blank"
        ? 0
        : variant === "faded"
          ? 0.06
          : variant === "soft"
            ? 0.035
            : 0.015,
    illumination: variant === "blank" ? 0 : variant === "faded" ? 0.04 : 0.02,
    blur_radius: variant === "soft" ? 1 : 0,
  };
  validateDegradation(recipe);
  return recipe;
}

/** Validate the exact, intentionally conservative degradation schema. */
export function validateDegradation(config) {
  assert(
    config && typeof config === "object" && !Array.isArray(config),
    "config must be an object",
  );
  assert(
    Object.keys(config).length === keys.length &&
      keys.every((key) => Object.hasOwn(config, key)),
    "config has unknown or missing fields",
  );
  assert(VARIANTS.has(config.variant), "config variant invalid");
  assert(
    Number.isInteger(config.seed) &&
      config.seed >= 0 &&
      config.seed <= 0xffffffff,
    "config seed invalid",
  );
  for (const key of ["paper_noise", "ink_fade", "illumination"])
    assert(Number.isFinite(config[key]), `config ${key} invalid`);
  assert(
    config.paper_noise >= 0 && config.paper_noise <= 3,
    "paper_noise must be 0-3",
  );
  assert(
    config.ink_fade >= 0 && config.ink_fade <= 0.08,
    "ink_fade must be 0-.08",
  );
  assert(
    config.illumination >= 0 && config.illumination <= 0.06,
    "illumination must be 0-.06",
  );
  assert(
    config.blur_radius === 0 || config.blur_radius === 1,
    "blur_radius must be 0 or 1",
  );
  if (config.variant === "blank")
    assert(
      config.paper_noise === 0 &&
        config.ink_fade === 0 &&
        config.illumination === 0 &&
        config.blur_radius === 0,
      "blank must be identity",
    );
  return config;
}

function blurRgb(source, width, height) {
  const horizontal = new Uint8ClampedArray(source.length);
  const output = new Uint8ClampedArray(source.length);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const target = (y * width + x) * 4;
      for (let channel = 0; channel < 3; channel++) {
        let total = 0,
          weight = 0;
        for (let dx = -1; dx <= 1; dx++) {
          const sourceX = Math.max(0, Math.min(width - 1, x + dx));
          total += source[(y * width + sourceX) * 4 + channel];
          weight++;
        }
        horizontal[target + channel] = total / weight;
      }
      horizontal[target + 3] = source[target + 3];
    }
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const target = (y * width + x) * 4;
      for (let channel = 0; channel < 3; channel++) {
        let total = 0,
          weight = 0;
        for (let dy = -1; dy <= 1; dy++) {
          const sourceY = Math.max(0, Math.min(height - 1, y + dy));
          total += horizontal[(sourceY * width + x) * 4 + channel];
          weight++;
        }
        output[target + channel] = total / weight;
      }
      output[target + 3] = source[target + 3];
    }
  return output;
}

/** Apply deterministic RGB-only effects; source geometry and alpha bytes are preserved. */
export function degradePixels(pixels, width, height, config) {
  assert(
    pixels instanceof Uint8ClampedArray,
    "pixels must be Uint8ClampedArray",
  );
  assert(
    Number.isInteger(width) &&
      Number.isInteger(height) &&
      width > 0 &&
      height > 0 &&
      width * height <= MAX_PIXELS,
    "dimensions must be positive and at most four million pixels",
  );
  assert(
    pixels.length === width * height * 4,
    "pixel length does not match RGBA dimensions",
  );
  validateDegradation(config);
  const output = new Uint8ClampedArray(pixels);
  if (config.variant === "blank") return output;
  for (let pixel = 0; pixel < width * height; pixel++) {
    const offset = pixel * 4,
      x = pixel % width,
      y = Math.floor(pixel / width);
    const wave =
      Math.sin(
        (x / Math.max(1, width - 1)) * Math.PI * 1.7 + config.seed * 0.00001,
      ) *
        0.5 +
      Math.cos((y / Math.max(1, height - 1)) * Math.PI * 1.3) * 0.5;
    const illumination = 1 + config.illumination * wave;
    const noise = (mix(config.seed ^ pixel) - 0.5) * 2 * config.paper_noise;
    for (let channel = 0; channel < 3; channel++) {
      const value = pixels[offset + channel];
      output[offset + channel] = clamp(
        (value + (255 - value) * config.ink_fade) * illumination + noise,
      );
    }
  }
  if (config.blur_radius === 1) {
    const blurred = blurRgb(output, width, height);
    // A full 3-tap blur failed smallest-board label preservation. Admit only
    // this fixed mild blend; do not silently retain severe ambiguous targets.
    for (let i = 0; i < output.length; i++)
      if (i % 4 !== 3) output[i] = output[i] * 0.75 + blurred[i] * 0.25;
  }
  return output;
}
