// Independent DOM/CSS layout control for the canvas production renderer.
// Uses the same approved source bytes, but no production drawing/key/layout code.
import { readFile, mkdir, writeFile } from "node:fs/promises";
import { chromium } from "@playwright/test";
import { createHash } from "node:crypto";
import { makeRecipe, renderBatch } from "./synthetic-render.mjs";

const out = "work/dataset/bootstrap/fidelity";
await mkdir(out, { recursive: true });
const sha = (bytes) => createHash("sha256").update(bytes).digest("hex");
const generation = {
  renderer_sha256: sha(await readFile("scripts/synthetic-render.mjs")),
  control_sha256: sha(await readFile("scripts/synthetic-fidelity.mjs")),
};
const pieces = [
  ".",
  "P",
  "N",
  "B",
  "R",
  "Q",
  "K",
  "p",
  "n",
  "b",
  "r",
  "q",
  "k",
];
const ids = [
  null,
  "wP",
  "wN",
  "wB",
  "wR",
  "wQ",
  "wK",
  "bP",
  "bN",
  "bB",
  "bR",
  "bQ",
  "bK",
];
const recipes = [];
for (const set of ["chessnut", "fantasy", "rhosgfx"]) {
  for (const piece of pieces) {
    const recipe = makeRecipe(20260907, 1);
    Object.assign(recipe, {
      index: 100000 + recipes.length,
      width: 960,
      height: 960,
      kind: "boards",
    });
    recipe.condition = {
      ...recipe.condition,
      coordinates: false,
      hatched: false,
      square_style: "grayscale",
      partial: false,
    };
    recipe.boards = [
      {
        id: "control",
        set,
        labels: Array(64).fill(piece),
        corners: [
          [96, 96],
          [864, 96],
          [864, 864],
          [96, 864],
        ],
        orientation: "unknown",
        position_kind: "teaching-calibration",
        parent: `calibration-${set}-${pieces.indexOf(piece)}`,
      },
    ];
    recipes.push(recipe);
  }
}
// A deterministic mixed subset covers multi-board transforms, hatching, rotations,
// both orientations and sizes. Partial pages are rendered but not classifier truth.
for (let i = 0; i < 40; i++) recipes.push(makeRecipe(20260907, i));
const records = [];
for (let i = 0; i < recipes.length; i += 32) {
  records.push(
    ...(await renderBatch({
      recipes: recipes.slice(i, i + 32),
      outputDir: out,
    })),
  );
}
const browser = await chromium.launch();
const controls = [];
try {
  const page = await browser.newPage({
    viewport: { width: 900, height: 900 },
    deviceScaleFactor: 1,
  });
  await page.route("**/*", (r) => r.abort());
  for (const record of records) {
    if (record.recipe.kind !== "boards") continue;
    for (let b = 0; b < record.recipe.boards.length; b++) {
      const board = record.recipe.boards[b];
      const side = Math.round(
        Math.hypot(
          board.corners[1][0] - board.corners[0][0],
          board.corners[1][1] - board.corners[0][1],
        ),
      );
      const cell = side / 8;
      const cells = [];
      for (let i = 0; i < 64; i++) {
        const glyph = ids[pieces.indexOf(board.labels[i])];
        const dark = (Math.floor(i / 8) + (i % 8)) % 2 === 1;
        const color = record.recipe.condition.square_style === "color";
        const bg = dark
          ? color
            ? "#b58863"
            : "#aaaaaa"
          : color
            ? "#f0d9b5"
            : "#f7f7f7";
        const data = glyph
          ? (
              await readFile(
                `work/dataset/bootstrap/assets/${board.set}/${glyph}.svg`,
              )
            ).toString("base64")
          : null;
        const x = (i % 8) * cell,
          y = Math.floor(i / 8) * cell;
        let hatch = "";
        if (dark && record.recipe.condition.hatched) {
          const lines = [];
          for (let h = -1; h <= 1; h += 0.18)
            lines.push(
              `<path d="M ${x + h * cell} ${y + cell} L ${x + (h + 1) * cell} ${y}"/>`,
            );
          hatch = `<defs><clipPath id="clip${i}"><rect x="${x}" y="${y}" width="${cell}" height="${cell}"/></clipPath></defs><g clip-path="url(#clip${i})" stroke="rgb(30,30,30)" stroke-opacity=".28" stroke-width="1">${lines.join("")}</g>`;
        }
        cells.push(
          `<rect x="${x}" y="${y}" width="${cell}" height="${cell}" fill="${bg}"/>${hatch}${data ? `<image x="${x + cell * 0.04}" y="${y + cell * 0.04}" width="${cell * 0.92}" height="${cell * 0.92}" href="data:image/svg+xml;base64,${data}"/>` : ""}`,
        );
      }
      await page.setContent(
        `<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'"><svg id="grid" xmlns="http://www.w3.org/2000/svg" width="${side}" height="${side}" viewBox="0 0 ${side} ${side}">${cells.join("")}</svg>`,
      );
      await page.locator("image").evaluateAll((images) =>
        Promise.all(
          images.map(async (im) => {
            const image = new Image();
            image.src = im.getAttribute("href");
            await image.decode();
          }),
        ),
      );
      const filename = `control-${record.recipe.index}-${b}.png`;
      await page.locator("#grid").screenshot({ path: `${out}/${filename}` });
      controls.push({ index: record.recipe.index, board: b, image: filename });
    }
  }
  // Same immutable inputs in a different batch/process must reconstruct exactly.
} finally {
  await browser.close();
}
const repeated = await renderBatch({
  recipes: recipes.slice(0, 3),
  outputDir: `${out}/rebuild`,
});
if (repeated.some((r, i) => r.sha256 !== records[i].sha256))
  throw new Error("Rebuild differs");
await writeFile(
  `${out}/records.json`,
  JSON.stringify(
    records.map(({ file, sha256, ...r }) => ({
      ...r,
      image: file,
      image_sha256: sha256,
    })),
  ),
);
await writeFile(`${out}/controls.json`, JSON.stringify(controls));
await writeFile(`${out}/generation.json`, JSON.stringify(generation));
console.log(
  JSON.stringify({
    pages: records.length,
    independent_controls: controls.length,
    rebuild_pages: repeated.length,
    state: "pixel-check-pending",
  }),
);
