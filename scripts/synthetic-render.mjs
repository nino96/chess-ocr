/** Deterministic, offline page renderer for the reviewed synthetic bootstrap. */
import { createHash } from "node:crypto";
import {
  access,
  lstat,
  mkdir,
  readFile,
  rename,
  writeFile,
} from "node:fs/promises";
import { dirname, isAbsolute, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";
import {
  perspectiveRecipe,
  projectPoint,
  validatePerspective,
  perspectiveCss,
} from "./synthetic-perspective.mjs";
import {
  degradationRecipe,
  validateDegradation,
} from "./synthetic-degradation.mjs";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const DEFAULT_ASSET_ROOT = resolve(ROOT, "work/dataset/bootstrap/assets");
const PROVENANCE = resolve(ROOT, "provenance/public-bootstrap.json");
const MAX_PAGES = 64;
const MAX_SVG_BYTES = 256 * 1024;
const PIECES = "PNBRQKpnbrqk";
const EMPTY = ".";
const SETS = ["chessnut", "fantasy", "rhosgfx"];

const hash = (value) => createHash("sha256").update(value).digest("hex");
const pad = (index) => String(index).padStart(6, "0");
const rng = (seed, index) => {
  let state = Number.parseInt(hash(`${seed}:${index}`).slice(0, 8), 16) >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 2 ** 32;
  };
};
const assert = (value, message) => {
  if (!value) throw new Error(message);
};

function put(labels, square, piece) {
  const file = square.charCodeAt(0) - 97;
  const rank = Number(square[1]);
  labels[(8 - rank) * 8 + file] = piece;
}
function labelsFrom(entries) {
  const labels = Array(64).fill(EMPTY);
  for (const [square, piece] of entries) put(labels, square, piece);
  return labels;
}
function opening(index) {
  // A short original legal opening sequence; teaching positions below are not claimed legal.
  const states = [
    [
      ["a1", "R"],
      ["b1", "N"],
      ["c1", "B"],
      ["d1", "Q"],
      ["e1", "K"],
      ["f1", "B"],
      ["g1", "N"],
      ["h1", "R"],
      ["a2", "P"],
      ["b2", "P"],
      ["c2", "P"],
      ["d2", "P"],
      ["e2", "P"],
      ["f2", "P"],
      ["g2", "P"],
      ["h2", "P"],
      ["a8", "r"],
      ["b8", "n"],
      ["c8", "b"],
      ["d8", "q"],
      ["e8", "k"],
      ["f8", "b"],
      ["g8", "n"],
      ["h8", "r"],
      ["a7", "p"],
      ["b7", "p"],
      ["c7", "p"],
      ["d7", "p"],
      ["e7", "p"],
      ["f7", "p"],
      ["g7", "p"],
      ["h7", "p"],
    ],
    [
      ["a1", "R"],
      ["b1", "N"],
      ["c1", "B"],
      ["d1", "Q"],
      ["e1", "K"],
      ["f1", "B"],
      ["g1", "N"],
      ["h1", "R"],
      ["a2", "P"],
      ["b2", "P"],
      ["c2", "P"],
      ["d2", "P"],
      ["f2", "P"],
      ["g2", "P"],
      ["h2", "P"],
      ["e4", "P"],
      ["a8", "r"],
      ["b8", "n"],
      ["c8", "b"],
      ["d8", "q"],
      ["e8", "k"],
      ["f8", "b"],
      ["g8", "n"],
      ["h8", "r"],
      ["a7", "p"],
      ["b7", "p"],
      ["c7", "p"],
      ["d7", "p"],
      ["f7", "p"],
      ["g7", "p"],
      ["h7", "p"],
      ["e5", "p"],
    ],
    [
      ["a1", "R"],
      ["b1", "N"],
      ["c1", "B"],
      ["d1", "Q"],
      ["e1", "K"],
      ["f1", "B"],
      ["h1", "R"],
      ["a2", "P"],
      ["b2", "P"],
      ["c2", "P"],
      ["d2", "P"],
      ["f2", "P"],
      ["g2", "P"],
      ["h2", "P"],
      ["e4", "P"],
      ["f3", "N"],
      ["a8", "r"],
      ["b8", "n"],
      ["c8", "b"],
      ["d8", "q"],
      ["e8", "k"],
      ["f8", "b"],
      ["g8", "n"],
      ["h8", "r"],
      ["a7", "p"],
      ["b7", "p"],
      ["c7", "p"],
      ["d7", "p"],
      ["f7", "p"],
      ["g7", "p"],
      ["h7", "p"],
      ["e5", "p"],
    ],
  ];
  return labelsFrom(states[index % states.length]);
}
function teaching(kind, random) {
  const labels = Array(64).fill(EMPTY);
  const count =
    kind === "empty" ? 0 : kind === "sparse" ? 8 : kind === "medium" ? 22 : 42;
  const offset = Math.floor(random() * PIECES.length);
  // Cycle types with a seeded offset so sparse boards can include every class.
  for (let n = 0; n < count; n++) {
    let square;
    do square = Math.floor(random() * 64);
    while (labels[square] !== EMPTY);
    labels[square] = PIECES[(n + offset) % PIECES.length];
  }
  return labels;
}
function board(id, set, labels, orientation, x, y, size, positionKind, parent) {
  return {
    id,
    set,
    labels: orientation === "black-bottom" ? [...labels].reverse() : labels,
    corners: [
      [x, y],
      [x + size, y],
      [x + size, y + size],
      [x, y + size],
    ],
    orientation,
    position_kind: positionKind,
    parent,
  };
}

/** Return a JSON-safe page recipe. Labels are image-relative, top-left to bottom-right. */
export function makeRecipe(seed, index) {
  assert(
    Number.isInteger(index) && index >= 0 && index < 1_000_000,
    "index must be a bounded non-negative integer",
  );
  assert(
    typeof seed === "string" || Number.isInteger(seed),
    "seed must be a string or integer",
  );
  const random = rng(seed, index),
    variant = index % 4,
    set = SETS[Math.floor(index / 3) % SETS.length];
  const positionKind =
    // Deliberately capped at five percent: only three original, checked opening states exist.
    index % 20 === 0
      ? "legal-opening"
      : index % 29 === 28
        ? "empty"
        : ["sparse", "medium", "dense"][index % 3];
  const labels =
    positionKind === "legal-opening"
      ? opening(Math.floor(index / 20))
      : teaching(positionKind, random);
  const orientation = index % 7 === 0 ? "black-bottom" : "white-bottom";
  const width = variant === 0 ? 612 : 792,
    height = variant === 0 ? 792 : 612;
  let boards;
  if (variant === 0)
    boards = [
      board(
        "board-0",
        set,
        labels,
        orientation,
        156,
        178,
        300,
        positionKind,
        `opening-${Math.floor(index / 20) % 3}`,
      ),
    ];
  else if (variant === 1)
    boards = [
      board(
        "board-0",
        set,
        labels,
        orientation,
        68,
        110,
        480,
        positionKind,
        `opening-${Math.floor(index / 20) % 3}`,
      ),
    ];
  else if (variant === 2)
    boards = [
      board(
        "board-0",
        set,
        labels,
        orientation,
        62,
        126,
        270,
        positionKind,
        `teaching-${index}-0`,
      ),
      board(
        "board-1",
        SETS[(index + 1) % SETS.length],
        teaching("sparse", random),
        "white-bottom",
        460,
        280,
        210,
        "sparse",
        `teaching-${index}-1`,
      ),
    ];
  else
    boards = [
      board(
        "board-0",
        set,
        labels,
        orientation,
        236,
        118,
        330,
        positionKind,
        `teaching-${index}-0`,
      ),
      board(
        "board-1",
        SETS[(index + 2) % SETS.length],
        teaching("dense", random),
        "black-bottom",
        62,
        450,
        150,
        "dense",
        `teaching-${index}-1`,
      ),
    ];
  const shearX = index % 6 === 0 ? 0.024 : index % 6 === 3 ? -0.018 : 0;
  for (const item of boards) {
    if (item.position_kind !== "legal-opening")
      item.parent = `teaching-${seed}-${index}-${item.id}`;
  }
  const rotationDeg = index % 8 === 5 ? 1.25 : index % 8 === 7 ? -1.25 : 0;
  const radians = (rotationDeg * Math.PI) / 180;
  if (shearX || rotationDeg) {
    for (const item of boards) {
      item.corners = item.corners.map(([x, y]) => {
        const shearedX = x + shearX * y;
        const dx = shearedX - width / 2,
          dy = y - height / 2;
        return [
          width / 2 + dx * Math.cos(radians) - dy * Math.sin(radians),
          height / 2 + dx * Math.sin(radians) + dy * Math.cos(radians),
        ];
      });
    }
  }
  if (index % 11 === 10) boards = [];
  const partial = boards.length > 0 && index % 37 === 36;
  const perspective = perspectiveRecipe(width, height, index);
  if (perspective)
    for (const board of boards) {
      board.render_corners = board.corners;
      board.corners = board.corners.map((p) => projectPoint(perspective, p));
    }
  return {
    schema: "chess-ocr-synthetic-page/1",
    index,
    seed,
    width,
    height,
    kind: boards.length === 0 ? "negative" : partial ? "partial" : "boards",
    layout:
      boards.length > 1
        ? "multiple-boards"
        : variant === 1
          ? "large-board"
          : "small-board",
    boards: partial ? [] : boards,
    unsupported_boards: partial ? boards : [],
    condition: {
      captions: true,
      coordinates: index % 2 === 0,
      background: Math.floor(index / 9) % 3 === 0 ? "cream" : "paper",
      hatched: index % 4 === 1,
      square_style: index % 5 === 2 ? "color" : "grayscale",
      border: index % 3 === 0 ? "double" : "single",
      affine: { shear_x: shearX, rotation_deg: rotationDeg },
      perspective,
      degradation: degradationRecipe(seed, Math.floor(index / 4)),
      negative: boards.length === 0,
      partial,
    },
  };
}

export function validateRecipe(recipe, fidelity = false) {
  assert(recipe && typeof recipe === "object", "recipe must be an object");
  validatePerspective(
    recipe.condition.perspective ?? null,
    recipe.width,
    recipe.height,
    recipe.index,
  );
  validateDegradation(recipe.condition.degradation);
  if (!fidelity)
    assert(
      Object.entries(
        degradationRecipe(recipe.seed, Math.floor(recipe.index / 4)),
      ).every(([key, value]) => recipe.condition.degradation[key] === value),
      "changed degradation recipe",
    );
  assert(
    ["boards", "negative", "partial"].includes(recipe.kind),
    "page kind invalid",
  );
  assert(
    Array.isArray(recipe.unsupported_boards || []) &&
      (recipe.unsupported_boards || []).length <= 4,
    "unsupported board count",
  );
  assert(
    recipe.kind === "boards" || recipe.boards.length === 0,
    "non-board page has usable targets",
  );
  assert(
    recipe.kind === "partial" || !(recipe.unsupported_boards || []).length,
    "unexpected unsupported targets",
  );
  assert(
    Number.isInteger(recipe.index) && recipe.index >= 0,
    "recipe index invalid",
  );
  assert(
    Number.isInteger(recipe.width) &&
      recipe.width >= 200 &&
      recipe.width <= 2400,
    "recipe width invalid",
  );
  assert(
    Number.isInteger(recipe.height) &&
      recipe.height >= 200 &&
      recipe.height <= 2400,
    "recipe height invalid",
  );
  assert(
    Array.isArray(recipe.boards) &&
      recipe.boards.length <= 4 &&
      (recipe.boards.length > 0 ||
        ["negative", "partial"].includes(recipe.kind)),
    "recipe boards invalid",
  );
  for (const b of [...recipe.boards, ...(recipe.unsupported_boards || [])]) {
    assert(/^[a-z0-9-]{1,40}$/.test(b.id), "board id invalid");
    assert(SETS.includes(b.set), "unapproved piece set");
    assert(
      Array.isArray(b.labels) &&
        b.labels.length === 64 &&
        b.labels.every(
          (piece) =>
            typeof piece === "string" && /^[.PNBRQKpnbrqk]$/.test(piece),
        ),
      "board labels invalid",
    );
    assert(
      ["white-bottom", "black-bottom", "unknown"].includes(b.orientation),
      "orientation invalid",
    );
    assert(
      Array.isArray(b.corners) &&
        b.corners.length === 4 &&
        b.corners.every(
          (p) => Array.isArray(p) && p.length === 2 && p.every(Number.isFinite),
        ),
      "corners invalid",
    );
    const renderCorners = b.render_corners || b.corners;
    if (recipe.condition.perspective) {
      assert(
        Array.isArray(b.render_corners) &&
          b.render_corners.length === 4 &&
          b.render_corners.every(
            (p) =>
              Array.isArray(p) && p.length === 2 && p.every(Number.isFinite),
          ),
        "invalid render corners",
      );
      assert(
        b.corners.every((p, i) =>
          p.every(
            (v, j) =>
              Math.abs(
                v -
                  projectPoint(recipe.condition.perspective, renderCorners[i])[
                    j
                  ],
              ) < 1e-7,
          ),
        ),
        "projected corners mismatch",
      );
    } else assert(!b.render_corners, "unexpected render corners");
    const [tl, tr, br, bl] = renderCorners;
    assert(
      renderCorners.every(
        ([x, y]) => x >= 0 && y >= 0 && x <= recipe.width && y <= recipe.height,
      ),
      "source board outside page",
    );
    assert(
      b.corners.every(
        ([x, y]) => x >= 0 && y >= 0 && x <= recipe.width && y <= recipe.height,
      ),
      "board outside page",
    );
    const cross =
      (tr[0] - tl[0]) * (bl[1] - tl[1]) - (tr[1] - tl[1]) * (bl[0] - tl[0]);
    assert(cross >= 64 * 64, "degenerate or inverted grid");
    assert(
      Math.abs(br[0] - tr[0] - bl[0] + tl[0]) < 1e-7 &&
        Math.abs(br[1] - tr[1] - bl[1] + tl[1]) < 1e-7,
      "only affine parallelograms supported",
    );
  }
}
function safePath(root, candidate) {
  const result = resolve(root, candidate);
  assert(
    result === root || result.startsWith(root + sep),
    "asset path escapes root",
  );
  return result;
}
async function approvedGlyphs(assetRoot) {
  const provenance = JSON.parse(await readFile(PROVENANCE, "utf8"));
  const approved = new Map(
    provenance.records
      .filter((r) => r.kind === "piece-svg")
      .map((r) => [r.id, r]),
  );
  const glyphs = {};
  for (const set of SETS)
    for (const piece of [
      "bB",
      "bK",
      "bN",
      "bP",
      "bQ",
      "bR",
      "wB",
      "wK",
      "wN",
      "wP",
      "wQ",
      "wR",
    ]) {
      const record = approved.get(`${set}-${piece}`);
      assert(record, `missing public allowlist ${set}-${piece}`);
      const path = safePath(assetRoot, `${set}/${piece}.svg`);
      for (let parent = dirname(path); ; parent = dirname(parent)) {
        assert(!(await lstat(parent)).isSymbolicLink(), "symlink asset parent");
        if (parent === dirname(parent)) break;
      }
      const info = await lstat(path);
      assert(
        !info.isSymbolicLink() &&
          info.isFile() &&
          info.size <= MAX_SVG_BYTES &&
          info.size === record.bytes,
        `unsafe SVG ${set}-${piece}`,
      );
      const bytes = await readFile(path);
      const text = bytes.toString("utf8");
      assert(
        hash(bytes) === record.sha256,
        `SVG hash mismatch ${set}-${piece}`,
      );
      assert(
        !/<!(?:doctype|entity)\b|<(?:script|iframe|object|embed|image|foreignObject|animate)\b|\bon\w+\s*=|(?:href|xlink:href)\s*=\s*["'](?!#)|url\(\s*["']?(?!#)/i.test(
          text,
        ),
        `unsafe SVG markup ${set}-${piece}`,
      );
      glyphs[`${set}:${piece}`] =
        `data:image/svg+xml;base64,${bytes.toString("base64")}`;
    }
  return glyphs;
}
function drawingScript(recipe, glyphs) {
  return `(${function ({ recipe, glyphs }) {
    const c = document.querySelector("canvas"),
      x = c.getContext("2d");
    c.width = recipe.width;
    c.height = recipe.height;
    x.fillStyle =
      recipe.condition.background === "cream" ? "#f7f0dc" : "#faf9f5";
    x.fillRect(0, 0, c.width, c.height);
    x.fillStyle = "#24211c";
    x.font = "600 18px system-ui,sans-serif";
    x.fillText("Chess study diagram", 32, 52);
    if (recipe.kind === "negative") {
      x.strokeStyle = "#777";
      x.lineWidth = 1;
      for (let row = 0; row <= 5; row++) {
        x.beginPath();
        x.moveTo(80, 180 + row * 32);
        x.lineTo(440, 180 + row * 32);
        x.stroke();
      }
      for (let col = 0; col <= 4; col++) {
        x.beginPath();
        x.moveTo(80 + col * 90, 180);
        x.lineTo(80 + col * 90, 340);
        x.stroke();
      }
      x.fillStyle = "#555";
      for (let row = 0; row < 5; row++)
        for (let col = 0; col < 4; col++)
          x.fillRect(
            90 + col * 90,
            190 + row * 32,
            42 + ((row + col) % 3) * 10,
            3,
          );
    }
    const load = (src) =>
      new Promise((ok, no) => {
        const i = new Image();
        i.onload = () => ok(i);
        i.onerror = no;
        i.src = src;
      });
    const key = (piece) =>
      piece === piece.toUpperCase() ? "w" + piece : "b" + piece.toUpperCase();
    // Decode before changing shared canvas state. Concurrent boards must never
    // interleave transforms while awaiting image decoding.
    return Promise.all(
      Object.entries(glyphs).map(async ([key, src]) => [key, await load(src)]),
    ).then((entries) => {
      const decoded = Object.fromEntries(entries);
      for (const b of [
        ...recipe.boards,
        ...(recipe.unsupported_boards || []),
      ]) {
        const [tl, tr, br, bl] = b.corners,
          side = Math.hypot(tr[0] - tl[0], tr[1] - tl[1]),
          cell = side / 8;
        x.save();
        x.setTransform(
          (tr[0] - tl[0]) / 8,
          (tr[1] - tl[1]) / 8,
          (bl[0] - tl[0]) / 8,
          (bl[1] - tl[1]) / 8,
          tl[0],
          tl[1],
        );
        x.fillStyle = "#5b5145";
        x.font = `${14 / cell}px system-ui,sans-serif`;
        x.fillText("Diagram", 0, -0.35);
        for (let row = 0; row < 8; row++)
          for (let col = 0; col < 8; col++) {
            const color = recipe.condition.square_style === "color";
            x.fillStyle =
              (row + col) % 2
                ? color
                  ? "#b58863"
                  : "#aaaaaa"
                : color
                  ? "#f0d9b5"
                  : "#f7f7f7";
            x.fillRect(col, row, 1, 1);
            if (recipe.condition.hatched && (row + col) % 2) {
              x.save();
              x.beginPath();
              x.rect(col, row, 1, 1);
              x.clip();
              x.strokeStyle = "rgba(30,30,30,.28)";
              x.lineWidth = 1 / cell;
              x.beginPath();
              for (let h = -1; h <= 1; h += 0.18) {
                x.moveTo(col + h, row + 1);
                x.lineTo(col + h + 1, row);
              }
              x.stroke();
              x.restore();
            }
          }
        b.labels.forEach((p, n) => {
          if (p === ".") return;
          const img = decoded[b.set + ":" + key(p)];
          const col = n % 8,
            row = (n / 8) | 0;
          x.drawImage(img, col + 0.04, row + 0.04, 0.92, 0.92);
        });
        x.strokeStyle = "#29241f";
        x.lineWidth = (recipe.condition.border === "double" ? 2 : 1) / cell;
        x.strokeRect(0, 0, 8, 8);
        if (recipe.condition.coordinates) {
          x.fillStyle = "#45392f";
          x.font = `${Math.max(9, cell * 0.16) / cell}px system-ui`;
          for (let n = 0; n < 8; n++) {
            const reverse = b.orientation === "black-bottom";
            x.fillText("abcdefgh"[reverse ? 7 - n : n], n + 0.45, 8.32);
            x.fillText(String(reverse ? n + 1 : 8 - n), -0.32, n + 0.6);
          }
        }
        if (recipe.kind === "partial") {
          x.fillStyle =
            recipe.condition.background === "cream" ? "#f7f0dc" : "#faf9f5";
          x.fillRect(-0.1, 5.5, 8.5, 3);
        }
        x.restore();
      }
    });
  }})(${JSON.stringify({ recipe, glyphs })})`;
}

/** Render no more than 64 recipe pages without network access. */
export async function renderBatch({
  assetRoot = DEFAULT_ASSET_ROOT,
  recipes,
  outputDir,
  fidelity = false,
}) {
  assert(
    Array.isArray(recipes) && recipes.length > 0 && recipes.length <= MAX_PAGES,
    `recipes must contain 1-${MAX_PAGES} pages`,
  );
  assert(
    typeof outputDir === "string" && outputDir.length > 0,
    "outputDir required",
  );
  recipes.forEach((recipe) => validateRecipe(recipe, fidelity));
  const resolvedRoot = resolve(assetRoot);
  await access(resolvedRoot);
  const glyphs = await approvedGlyphs(resolvedRoot);
  const degradationSource = (
    await readFile(
      new URL("./synthetic-degradation.mjs", import.meta.url),
      "utf8",
    )
  ).replaceAll("export function", "function");
  await mkdir(outputDir, { recursive: true });
  for (let parent = resolve(outputDir); ; parent = dirname(parent)) {
    assert(!(await lstat(parent)).isSymbolicLink(), "symlink output parent");
    if (parent === dirname(parent)) break;
  }
  const browser = await chromium.launch({ headless: true });
  const results = [];
  try {
    for (const recipe of recipes) {
      const page = await browser.newPage({
        viewport: { width: recipe.width, height: recipe.height },
        deviceScaleFactor: 1,
      });
      await page.route("**/*", (route) => route.abort());
      await page.setContent(
        `<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'"><style>body{margin:0}#output{width:${recipe.width}px;height:${recipe.height}px;overflow:hidden;background:white}canvas{transform-origin:0 0;transform:${perspectiveCss(recipe.condition.perspective)}}</style><div id="output"><canvas></canvas></div>`,
        { waitUntil: "domcontentloaded", timeout: 30_000 },
      );
      const drawingRecipe = structuredClone(recipe);
      for (const b of [
        ...drawingRecipe.boards,
        ...(drawingRecipe.unsupported_boards || []),
      ])
        b.corners = b.render_corners || b.corners;
      await page.evaluate(drawingScript(drawingRecipe, glyphs));
      await page.evaluate(
        `(() => {${degradationSource}\nconst c=document.querySelector('canvas'), x=c.getContext('2d'); const im=x.getImageData(0,0,c.width,c.height); im.data.set(degradePixels(im.data,c.width,c.height,${JSON.stringify(recipe.condition.degradation)})); x.putImageData(im,0,0);})()`,
      );
      const png = await page
        .locator("#output")
        .screenshot({ type: "png", timeout: 30_000 });
      await page.close();
      const filename = `page-${pad(recipe.index)}.png`,
        destination = resolve(outputDir, filename);
      assert(
        destination.startsWith(resolve(outputDir) + sep),
        "output path escapes outputDir",
      );
      const temporary = `${destination}.tmp-${process.pid}-${Date.now()}`;
      await writeFile(temporary, png);
      await rename(temporary, destination);
      results.push({
        index: recipe.index,
        file: filename,
        sha256: hash(png),
        bytes: png.length,
        recipe,
      });
    }
  } finally {
    await browser.close();
  }
  return results;
}

async function main(argv) {
  const [command, ...args] = argv;
  if (command === "recipes") {
    const [seed, startText, countText] = args;
    const start = Number(startText),
      count = Number(countText);
    assert(
      seed !== undefined &&
        Number.isInteger(start) &&
        start >= 0 &&
        Number.isInteger(count) &&
        count > 0 &&
        count <= MAX_PAGES,
      `usage: node scripts/synthetic-render.mjs recipes <seed> <start> <count 1-${MAX_PAGES}>`,
    );
    process.stdout.write(
      JSON.stringify(
        Array.from({ length: count }, (_, offset) =>
          makeRecipe(seed, start + offset),
        ),
      ) + "\n",
    );
    return;
  }
  if (command === "render") {
    const [recipeFile, outputDir, assetRoot] = args;
    assert(
      recipeFile && outputDir && args.length <= 3,
      "usage: node scripts/synthetic-render.mjs render <recipes.json> <output-dir> [asset-root]",
    );
    const info = await lstat(recipeFile);
    assert(
      !info.isSymbolicLink() && info.isFile() && info.size <= 2 * 1024 * 1024,
      "unsafe recipe file",
    );
    const recipes = JSON.parse(await readFile(recipeFile, "utf8"));
    assert(Array.isArray(recipes), "recipes JSON must be an array");
    const records = await renderBatch({
      recipes,
      outputDir,
      ...(assetRoot ? { assetRoot } : {}),
    });
    process.stdout.write(
      JSON.stringify(
        records.map(({ file, sha256, ...record }) => ({
          ...record,
          image: file,
          image_sha256: sha256,
        })),
      ) + "\n",
    );
    return;
  }
  if (command === "assets") {
    assert(
      args.length <= 1,
      "usage: node scripts/synthetic-render.mjs assets [asset-root]",
    );
    const glyphs = await approvedGlyphs(resolve(args[0] || DEFAULT_ASSET_ROOT));
    process.stdout.write(
      JSON.stringify({
        verified: Object.keys(glyphs).sort(),
        count: Object.keys(glyphs).length,
      }) + "\n",
    );
    return;
  }
  throw new Error(
    "usage: node scripts/synthetic-render.mjs <recipes|render|assets> ...",
  );
}
if (
  process.argv[1] &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url)
)
  main(process.argv.slice(2)).catch((error) => {
    process.stderr.write(`${error.message}\n`);
    process.exitCode = 1;
  });
