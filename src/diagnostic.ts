import type { CandidateConfig } from "./candidate.ts";
import {
  createBrowserClient,
  createCandidateBrowserClient,
  prepareManualGridInput,
  restoreManualGridResult,
} from "./browser.ts";
import type { RecognitionClient } from "./client.ts";
import {
  LABELS,
  VERSION,
  type Board,
  type Label,
  type Request,
  type Result,
} from "./contract.ts";
import {
  PRIVATE_EVALUATION_VERSION,
  compareResult,
  privateEvaluationSchema,
  referenceSchema,
  summarizeEvaluation,
  type PrivateEvaluation,
  type Reference,
} from "./evaluation.ts";
import { decodeRaster, inspectRaster } from "./image.ts";
import {
  rectifyGrid,
  type GridCorners,
  type Point,
  type RgbaRaster,
} from "./grid.ts";

const SENSITIVE_WARNING =
  "SENSITIVE LOCAL EVIDENCE: contains image hashes, geometry, positions, and raw model results; do not publish";
const MAX_QUEUE_BYTES = 100 * 1024 * 1024;

async function digest(bytes: Uint8Array): Promise<string> {
  const copy = Uint8Array.from(bytes);
  return [...new Uint8Array(await crypto.subtle.digest("SHA-256", copy.buffer))]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

export async function sha256(file: Blob): Promise<string> {
  return digest(new Uint8Array(await file.arrayBuffer()));
}

/** Imports are usable only after the operator reselects every hash-bound input. */
export async function matchingImportFiles(
  session: PrivateEvaluation,
  files: File[],
): Promise<boolean> {
  const expected = new Map(
    session.entries.map((entry) => [entry.index, entry]),
  );
  if (files.length !== expected.size) return false;
  const actual = await Promise.all(files.map(sha256));
  return [...expected.values()]
    .sort((left, right) => left.index - right.index)
    .every(
      (entry, index) =>
        actual[index] === entry.image.sha256 &&
        files[index]!.size === entry.image.bytes,
    );
}

type QueueItem = {
  bytes: Uint8Array<ArrayBuffer>;
  mime: string;
  sha256: string;
  width: number;
  height: number;
};
type RunMode = "automatic" | "manual-grid";
type Entry = PrivateEvaluation["entries"][number];
type Draft = Omit<Entry, "results"> & { results?: Entry["results"] };
type MutableCorners = [Point, Point, Point, Point];
type ReferenceDraft = {
  kind: Reference["kind"];
  corners: MutableCorners;
  orientation: "unknown" | "white-bottom" | "black-bottom";
  labels: Label[];
};

const CORNER_NAMES = ["top-left", "top-right", "bottom-right", "bottom-left"];
const PIECE_SYMBOLS: Record<Label, string> = {
  empty: "·",
  P: "♙",
  N: "♘",
  B: "♗",
  R: "♖",
  Q: "♕",
  K: "♔",
  p: "♟",
  n: "♞",
  b: "♝",
  r: "♜",
  q: "♛",
  k: "♚",
};

export function imageEdgeCorners(
  width: number,
  height: number,
): MutableCorners {
  return [
    { x: 0, y: 0 },
    { x: width - 1, y: 0 },
    { x: width - 1, y: height - 1 },
    { x: 0, y: height - 1 },
  ];
}

export function gridSegments(corners: GridCorners): Array<[Point, Point]> {
  const lerp = (a: Point, b: Point, amount: number): Point => ({
    x: a.x + (b.x - a.x) * amount,
    y: a.y + (b.y - a.y) * amount,
  });
  const segments: Array<[Point, Point]> = [];
  for (let index = 0; index <= 8; index++) {
    const amount = index / 8;
    segments.push([
      lerp(corners[0], corners[3], amount),
      lerp(corners[1], corners[2], amount),
    ]);
    segments.push([
      lerp(corners[0], corners[1], amount),
      lerp(corners[3], corners[2], amount),
    ]);
  }
  return segments;
}

export function closestBoard(
  result: Result,
  reference: Reference,
): Board | null {
  if (!result.boards.length) return null;
  if (reference.kind !== "board") return result.boards[0]!;
  const error = (board: Board) =>
    board.corners.reduce((sum, point, index) => {
      const expected = reference.corners[index]!;
      return sum + Math.hypot(point.x - expected.x, point.y - expected.y);
    }, 0);
  return result.boards.reduce((best, board) =>
    error(board) < error(best) ? board : best,
  );
}

export type DisplayBoard = {
  board: Board;
  index: number;
  role: "scored" | "additional" | "false-return";
};

/** Keep every returned board visible while identifying the one used for scoring. */
export function displayBoards(
  result: Result,
  reference: Reference,
): DisplayBoard[] {
  const scored =
    reference.kind === "board" ? closestBoard(result, reference) : null;
  return result.boards.map((board, index) => ({
    board,
    index,
    role:
      board === scored
        ? "scored"
        : reference.kind === "board"
          ? "additional"
          : "false-return",
  }));
}

const byId = <T extends HTMLElement>(id: string): T =>
  document.getElementById(id) as T;
const download = (name: string, value: unknown): void => {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(value, null, 2)], { type: "application/json" }),
  );
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};

export function mountDiagnostic(
  candidateForRun: () => CandidateConfig | null,
): void {
  const root = byId<HTMLElement>("diagnostic");
  if (!root) return;
  const labelContainer = byId("diagnostic-labels");
  const labelTemplate = byId<HTMLTemplateElement>("diagnostic-label-template");
  for (let index = 0; index < 64; index++) {
    const fragment = labelTemplate.content.cloneNode(true) as DocumentFragment;
    const select = fragment.querySelector("select")!;
    select.id = `diagnostic-label-${index}`;
    select.setAttribute(
      "aria-label",
      `Reference row ${Math.floor(index / 8) + 1}, column ${(index % 8) + 1}`,
    );
    fragment.querySelector("span")!.textContent =
      `${Math.floor(index / 8) + 1},${(index % 8) + 1}`;
    labelContainer.append(fragment);
  }
  const status = (message: string): void => {
    byId("diagnostic-status").textContent = message;
  };
  let queue: QueueItem[] = [];
  let selected = 0;
  let entries: Draft[] = [];
  let imported: PrivateEvaluation | null = null;
  let running = false;
  let activeBitmap: ImageBitmap | null = null;
  let activeBitmapIndex = -1;
  let referenceDrafts = new Map<number, ReferenceDraft>();
  let cornerPlacements = new Map<number, number>();
  let cornerPlacement = 0;
  let draggingCorner: number | null = null;
  let selectedCorner = 0;
  let visualGeneration = 0;
  let fenshotClient: RecognitionClient | null = null;
  let v2Client: { identity: string; client: RecognitionClient } | null = null;
  let mainThreadHeapPeakBytes: number | null = null;
  const releaseActiveBitmap = (): void => {
    activeBitmap?.close();
    activeBitmap = null;
    activeBitmapIndex = -1;
  };
  const decodeActive = async (item: QueueItem): Promise<ImageBitmap> => {
    if (activeBitmap && activeBitmapIndex === selected) return activeBitmap;
    releaseActiveBitmap();
    activeBitmap = await decodeRaster(
      new File([item.bytes.buffer], "local-evaluation-input", {
        type: item.mime,
      }),
    );
    activeBitmapIndex = selected;
    return activeBitmap;
  };
  const sampleHeap = (): number | null => {
    const value = (
      performance as Performance & { memory?: { usedJSHeapSize: number } }
    ).memory?.usedJSHeapSize;
    if (value !== undefined)
      mainThreadHeapPeakBytes = Math.max(mainThreadHeapPeakBytes ?? 0, value);
    return value ?? null;
  };

  const sourceCanvas = byId<HTMLCanvasElement>("diagnostic-source");
  const sourceContext = sourceCanvas.getContext("2d")!;
  const activeDraft = (): ReferenceDraft | null =>
    referenceDrafts.get(selected) ?? null;
  const syncCornerInputs = (corners: GridCorners): void => {
    corners
      .flatMap((point) => [point.x, point.y])
      .forEach((value, index) => {
        byId<HTMLInputElement>(`diagnostic-corner-${index}`).value = String(
          Math.round(value),
        );
      });
  };
  const syncReferenceControls = (): void => {
    const draft = activeDraft();
    if (!draft) return;
    byId<HTMLSelectElement>("diagnostic-reference-kind").value = draft.kind;
    byId<HTMLSelectElement>("diagnostic-orientation").value = draft.orientation;
    syncCornerInputs(draft.corners);
    draft.labels.forEach((label, index) => {
      byId<HTMLSelectElement>(`diagnostic-label-${index}`).value = label;
    });
  };
  const scalePoint = (point: Point, scaleX: number, scaleY: number): Point => ({
    x: point.x * scaleX,
    y: point.y * scaleY,
  });
  const strokeGrid = (
    context: CanvasRenderingContext2D,
    corners: GridCorners,
    scaleX: number,
    scaleY: number,
    handles: boolean,
    color = "#096ccc",
    label?: string,
  ): void => {
    context.save();
    context.strokeStyle = color;
    context.fillStyle = color;
    context.lineWidth = Math.max(2, context.canvas.width / 320);
    for (const [from, to] of gridSegments(corners)) {
      const start = scalePoint(from, scaleX, scaleY);
      const end = scalePoint(to, scaleX, scaleY);
      context.beginPath();
      context.moveTo(start.x, start.y);
      context.lineTo(end.x, end.y);
      context.stroke();
    }
    if (handles)
      corners.forEach((point, index) => {
        const scaled = scalePoint(point, scaleX, scaleY);
        const radius = Math.max(7, context.canvas.width / 80);
        context.beginPath();
        context.arc(scaled.x, scaled.y, radius, 0, Math.PI * 2);
        context.fill();
        context.fillStyle = "white";
        context.font = `${Math.max(11, radius)}px sans-serif`;
        context.textAlign = "center";
        context.textBaseline = "middle";
        context.fillText(String(index + 1), scaled.x, scaled.y);
        context.fillStyle = color;
      });
    if (label) {
      const anchor = scalePoint(corners[0], scaleX, scaleY);
      const size = Math.max(18, context.canvas.width / 24);
      context.fillStyle = color;
      context.fillRect(anchor.x, anchor.y, size, size);
      context.fillStyle = "white";
      context.font = `bold ${Math.max(12, size * 0.65)}px sans-serif`;
      context.textAlign = "center";
      context.textBaseline = "middle";
      context.fillText(label, anchor.x + size / 2, anchor.y + size / 2);
    }
    context.restore();
  };
  const drawDiagnosticSource = (bitmap: ImageBitmap): void => {
    const maximumWidth = 1000;
    const scale = Math.min(1, maximumWidth / bitmap.width);
    sourceCanvas.width = Math.max(1, Math.round(bitmap.width * scale));
    sourceCanvas.height = Math.max(1, Math.round(bitmap.height * scale));
    sourceContext.drawImage(
      bitmap,
      0,
      0,
      sourceCanvas.width,
      sourceCanvas.height,
    );
    const draft = activeDraft();
    if (draft?.kind === "board")
      strokeGrid(
        sourceContext,
        draft.corners,
        sourceCanvas.width / bitmap.width,
        sourceCanvas.height / bitmap.height,
        true,
      );
  };
  const rgbaRaster = (bitmap: ImageBitmap): RgbaRaster => {
    const canvas = document.createElement("canvas");
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) throw new Error("Canvas is unavailable");
    context.drawImage(bitmap, 0, 0);
    const data = context.getImageData(0, 0, bitmap.width, bitmap.height).data;
    return {
      data: new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
      width: bitmap.width,
      height: bitmap.height,
    };
  };
  const drawOverlay = (
    canvas: HTMLCanvasElement,
    bitmap: ImageBitmap,
    boards: readonly DisplayBoard[],
  ): void => {
    const scale = Math.min(1, 480 / bitmap.width, 360 / bitmap.height);
    canvas.width = Math.max(1, Math.round(bitmap.width * scale));
    canvas.height = Math.max(1, Math.round(bitmap.height * scale));
    const context = canvas.getContext("2d")!;
    context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    for (const display of boards)
      strokeGrid(
        context,
        display.board.corners,
        canvas.width / bitmap.width,
        canvas.height / bitmap.height,
        false,
        display.role === "scored" ? "#096ccc" : "#b4512d",
        String(display.index + 1),
      );
  };
  const drawRectified = (
    canvas: HTMLCanvasElement,
    raster: RgbaRaster,
    corners: GridCorners,
  ): void => {
    const size = 256;
    const rectified = rectifyGrid(raster, corners, 4, size);
    canvas.width = size;
    canvas.height = size;
    canvas
      .getContext("2d")!
      .putImageData(
        new ImageData(new Uint8ClampedArray(rectified.data), size, size),
        0,
        0,
      );
  };

  const reference = (): Reference => {
    const draft = activeDraft();
    if (!draft) throw new Error("Choose a diagnostic image first.");
    if (draft.kind !== "board")
      return referenceSchema.parse({ kind: draft.kind });
    if (cornerPlacement < 4)
      throw new Error(
        "Click all four grid corners before saving the reference.",
      );
    return referenceSchema.parse({
      kind: "board",
      corners: draft.corners,
      orientation: draft.orientation,
      labels: draft.labels,
    });
  };
  const existing = (index: number, mode: RunMode) =>
    entries.find((entry) => entry.index === index && entry.mode === mode);
  const mode = (): RunMode =>
    byId<HTMLSelectElement>("diagnostic-mode").value as RunMode;
  const boardView = (
    values: readonly (Label | null)[],
    expected?: readonly Label[],
  ): HTMLElement => {
    const board = document.createElement("div");
    board.className = "diagnostic-board";
    board.setAttribute("role", "img");
    board.setAttribute("aria-label", "Board pieces in image row order");
    values.forEach((label, index) => {
      const square = document.createElement("span");
      square.className = `diagnostic-square ${
        (Math.floor(index / 8) + (index % 8)) % 2 ? "dark" : ""
      } ${expected && label !== expected[index] ? "wrong" : ""}`;
      square.textContent = label ? PIECE_SYMBOLS[label] : "?";
      square.title = `Row ${Math.floor(index / 8) + 1}, column ${(index % 8) + 1}: ${label ?? "unknown"}`;
      board.append(square);
    });
    return board;
  };
  const resultCard = (
    title: string,
    result: Result,
    reference: Reference,
    bitmap: ImageBitmap,
    raster: RgbaRaster,
  ): HTMLElement => {
    const card = document.createElement("article");
    card.className = "diagnostic-result-card";
    const heading = document.createElement("h4");
    heading.textContent = title;
    card.append(heading);
    const boards = displayBoards(result, reference);
    const state = document.createElement("p");
    state.textContent = boards.length
      ? `${result.status}; ${result.boards.length} board${result.boards.length === 1 ? "" : "s"} returned`
      : `${result.status}; no board returned`;
    card.append(state);
    if (result.warnings.length) {
      const warnings = document.createElement("p");
      warnings.className = "diagnostic-warning";
      warnings.textContent = result.warnings.join(" ");
      card.append(warnings);
    }
    const overlayCaption = document.createElement("p");
    overlayCaption.textContent =
      boards.length > 1
        ? reference.kind === "board"
          ? "All returned grids: blue is scored; orange grids are additional false returns"
          : "All orange grids are false returns"
        : boards.length === 1
          ? reference.kind === "board"
            ? "Returned grid over the source"
            : "False return over the source"
          : "Source image; no grid returned";
    card.append(overlayCaption);
    const overlay = document.createElement("canvas");
    overlay.className = "diagnostic-overlay";
    overlay.setAttribute("aria-label", `${title} grid over source image`);
    drawOverlay(overlay, bitmap, boards);
    card.append(overlay);
    for (const display of boards) {
      const returned = document.createElement("section");
      returned.className = "diagnostic-returned-board";
      const returnedHeading = document.createElement("h5");
      returnedHeading.textContent =
        display.role === "scored"
          ? `Board ${display.index + 1} · scored against your reference`
          : display.role === "additional"
            ? `Board ${display.index + 1} · additional false return`
            : `Board ${display.index + 1} · false return`;
      returned.append(returnedHeading);
      const rectifiedCaption = document.createElement("p");
      rectifiedCaption.textContent = "Rectified view from the returned grid";
      returned.append(rectifiedCaption);
      const rectified = document.createElement("canvas");
      rectified.className = "diagnostic-rectified";
      rectified.setAttribute(
        "aria-label",
        `${title} board ${display.index + 1} rectified from returned grid`,
      );
      drawRectified(rectified, raster, display.board.corners);
      returned.append(rectified);
      const piecesCaption = document.createElement("p");
      piecesCaption.textContent =
        "Predicted pieces; red cells disagree with you";
      returned.append(piecesCaption);
      const expected =
        display.role === "scored" && reference.kind === "board"
          ? reference.labels
          : undefined;
      returned.append(
        boardView(
          display.board.squares.map((square) => square.label),
          expected,
        ),
      );
      if (display.board.warnings.length) {
        const warnings = document.createElement("p");
        warnings.className = "diagnostic-warning";
        warnings.textContent = display.board.warnings.join(" ");
        returned.append(warnings);
      }
      card.append(returned);
    }
    return card;
  };
  const renderVisuals = async (): Promise<void> => {
    const generation = ++visualGeneration;
    const item = queue[selected];
    const panel = byId<HTMLElement>("diagnostic-image-panel");
    if (!item) {
      panel.hidden = true;
      return;
    }
    panel.hidden = false;
    try {
      const bitmap = await decodeActive(item);
      if (generation !== visualGeneration) return;
      drawDiagnosticSource(bitmap);
      const entry = existing(selected, mode());
      if (!entry?.results) return;
      const cards = byId("diagnostic-result-cards");
      cards.replaceChildren();
      if (entry.reference.kind === "board") {
        const referenceCard = document.createElement("article");
        referenceCard.className = "diagnostic-result-card";
        const heading = document.createElement("h4");
        heading.textContent = "Your reference";
        const caption = document.createElement("p");
        caption.textContent = "Expected pieces in source-image row order";
        referenceCard.append(
          heading,
          caption,
          boardView(entry.reference.labels),
        );
        cards.append(referenceCard);
      }
      const raster = rgbaRaster(bitmap);
      cards.append(
        resultCard(
          "v2 candidate",
          entry.results.v2.result,
          entry.reference,
          bitmap,
          raster,
        ),
        resultCard(
          "FENShot",
          entry.results.fenshot.result,
          entry.reference,
          bitmap,
          raster,
        ),
      );
    } catch (error) {
      status(
        error instanceof Error
          ? error.message
          : "Could not draw diagnostic image.",
      );
    }
  };
  const render = (): void => {
    const item = queue[selected];
    byId("diagnostic-current").textContent = item
      ? `${selected + 1}/${queue.length}: ${item.width}×${item.height}, SHA-256 ${item.sha256.slice(0, 12)}…`
      : "No local images queued.";
    byId<HTMLButtonElement>("diagnostic-save-reference").disabled =
      !item || running;
    byId<HTMLButtonElement>("diagnostic-previous").disabled =
      !item || running || selected === 0;
    byId<HTMLButtonElement>("diagnostic-next").disabled =
      !item || running || selected === queue.length - 1;
    byId<HTMLButtonElement>("diagnostic-run").disabled =
      !item || !existing(selected, mode()) || running;
    const complete =
      entries.length > 0 && entries.every((entry) => entry.results);
    byId<HTMLButtonElement>("diagnostic-export-private").disabled = !complete;
    byId<HTMLButtonElement>("diagnostic-export-summary").disabled = !complete;
    const draft = activeDraft();
    const isBoard = draft?.kind === "board";
    byId<HTMLElement>("diagnostic-corner-details").hidden = !isBoard;
    byId<HTMLElement>("diagnostic-labels").hidden = !isBoard;
    byId<HTMLSelectElement>("diagnostic-orientation").disabled = !isBoard;
    byId<HTMLButtonElement>("diagnostic-restart-corners").disabled = !isBoard;
    byId<HTMLButtonElement>("diagnostic-use-image-edges").disabled = !isBoard;
    byId("diagnostic-corner-help").textContent = !isBoard
      ? "This reference does not require board corners."
      : cornerPlacement < 4
        ? `Click corner ${cornerPlacement + 1}: ${CORNER_NAMES[cornerPlacement]}.`
        : `Grid ready. Drag a numbered corner to adjust it. For keyboard adjustment, press 1–4 to choose a corner, then use arrow keys; corner ${selectedCorner + 1} is selected.`;
    const result = existing(selected, mode());
    const output = byId<HTMLElement>("diagnostic-result");
    output.hidden = !result?.results;
    if (result?.results) {
      const v2 = compareResult(result.reference, result.results.v2.result);
      const fenshot = compareResult(
        result.reference,
        result.results.fenshot.result,
      );
      byId("diagnostic-result-metrics").textContent =
        `v2: ${JSON.stringify(v2)}\nFENShot: ${JSON.stringify(fenshot)}`;
    }
    void renderVisuals();
  };
  const device = () => ({
    label: byId<HTMLInputElement>("diagnostic-device").value.trim(),
    os: byId<HTMLInputElement>("diagnostic-os").value.trim(),
    browser: byId<HTMLInputElement>("diagnostic-browser").value.trim(),
    browserVersion: byId<HTMLInputElement>(
      "diagnostic-browser-version",
    ).value.trim(),
    cpu: byId<HTMLInputElement>("diagnostic-cpu").value.trim(),
    memoryGiB:
      Number(byId<HTMLInputElement>("diagnostic-memory").value) || null,
    mainThreadHeapPeakBytes,
    peakMemoryMethod: byId<HTMLInputElement>(
      "diagnostic-memory-method",
    ).value.trim(),
  });
  const session = (): PrivateEvaluation => {
    const candidate = candidateForRun();
    if (
      !candidate ||
      candidate.manifest.preprocessing !== "yolox-rgb-imagenet-v2"
    )
      throw new Error(
        "Load the verified v2 candidate through the baseline candidate loader first.",
      );
    const fenshot = entries.find((entry) => entry.results)?.results?.fenshot
      .result.model;
    if (!fenshot) throw new Error("FENShot did not return a model identity.");
    const orderedIndices = [
      ...new Set(entries.map((entry) => entry.index)),
    ].sort((left, right) => left - right);
    const normalized = new Map(
      orderedIndices.map((index, normalizedIndex) => [index, normalizedIndex]),
    );
    return privateEvaluationSchema.parse({
      schema: PRIVATE_EVALUATION_VERSION,
      sensitive: true,
      warning: SENSITIVE_WARNING,
      createdAt: new Date().toISOString(),
      device: device(),
      models: {
        v2: candidate.identity,
        fenshot,
      },
      entries: entries
        .toSorted(
          (left, right) =>
            left.index - right.index || left.mode.localeCompare(right.mode),
        )
        .map((entry) => {
          if (!entry.results)
            throw new Error(
              "Run every saved reference before exporting evidence.",
            );
          return {
            ...entry,
            index: normalized.get(entry.index)!,
            results: entry.results,
          };
        }),
    });
  };
  const saveReference = (): void => {
    const item = queue[selected];
    if (!item) return;
    try {
      const value = reference();
      const currentMode = mode();
      if (currentMode === "manual-grid" && value.kind !== "board")
        throw new Error(
          "Manual-grid mode requires a complete board reference.",
        );
      entries = entries.filter(
        (entry) => entry.index !== selected || entry.mode !== currentMode,
      );
      // Results deliberately do not exist until the reference is independently saved.
      entries.push({
        index: selected,
        image: {
          width: item.width,
          height: item.height,
          bytes: item.bytes.byteLength,
          sha256: item.sha256,
        },
        mode: currentMode,
        reference: value,
      });
      status(
        `Reference saved locally. Run the paired ${currentMode} comparison when ready.`,
      );
      render();
    } catch (error) {
      status(
        error instanceof Error ? error.message : "Reference is incomplete.",
      );
    }
  };
  const pairedRun = async (): Promise<void> => {
    const item = queue[selected];
    const currentMode = mode();
    const draft = existing(selected, currentMode);
    const candidate = candidateForRun();
    if (
      !item ||
      !draft ||
      !candidate ||
      candidate.manifest.preprocessing !== "yolox-rgb-imagenet-v2"
    ) {
      status("Reference and verified v2 candidate are required.");
      return;
    }
    running = true;
    render();
    status(
      "Decoding one local image and running v2 and unchanged FENShot sequentially on identical RGBA pixels…",
    );
    try {
      const bitmap = await decodeActive(item);
      const canvas = document.createElement("canvas");
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
      const context = canvas.getContext("2d", { willReadFrequently: true });
      if (!context) throw new Error("Canvas is unavailable");
      context.drawImage(bitmap, 0, 0);
      const rgba = context.getImageData(0, 0, bitmap.width, bitmap.height).data;
      const manualCorners =
        currentMode === "manual-grid" && draft.reference.kind === "board"
          ? draft.reference.corners
          : null;
      const runRgba = manualCorners
        ? prepareManualGridInput(
            rgba,
            bitmap.width,
            bitmap.height,
            manualCorners,
          )
        : rgba;
      const runImage = manualCorners
        ? { width: 768, height: 768 }
        : { width: bitmap.width, height: bitmap.height };
      const request = (requestId: string): Request => ({
        schema: VERSION,
        requestId,
        image: runImage,
        selection: manualCorners
          ? { x: 0, y: 0, width: 768, height: 768 }
          : null,
      });
      const candidateIdentity = candidate.identity.sha256;
      const v2Cold = v2Client?.identity !== candidateIdentity;
      if (v2Cold) {
        v2Client?.client.cancel();
        v2Client = {
          identity: candidateIdentity,
          client: createCandidateBrowserClient(candidate),
        };
      }
      const currentV2 = v2Client;
      if (!currentV2) throw new Error("Could not initialize v2 client");
      const fenshotCold = !fenshotClient;
      fenshotClient ??= createBrowserClient();
      const heap = () => sampleHeap();
      sampleHeap();
      const startV2 = performance.now();
      let v2 = await currentV2.client.recognize(
        request(crypto.randomUUID()),
        new Uint8ClampedArray(runRgba),
      );
      const v2Ms = performance.now() - startV2;
      const startFenshot = performance.now();
      let fenshot = await fenshotClient.recognize(
        request(crypto.randomUUID()),
        new Uint8ClampedArray(runRgba),
      );
      const fenshotMs = performance.now() - startFenshot;
      if (manualCorners) {
        const originalImage = { width: bitmap.width, height: bitmap.height };
        v2 = restoreManualGridResult(v2, originalImage, manualCorners);
        fenshot = restoreManualGridResult(
          fenshot,
          originalImage,
          manualCorners,
        );
      }
      entries = entries.filter((entry) => entry !== draft);
      entries.push({
        ...draft,
        results: {
          v2: {
            result: v2,
            cold: v2Cold,
            totalMs: v2Ms,
            workerHeapBytes: heap(),
          },
          fenshot: {
            result: fenshot,
            cold: fenshotCold,
            totalMs: fenshotMs,
            workerHeapBytes: heap(),
          },
        },
      });
      status(
        "Paired local result saved in memory. Export private evidence or privacy-safe aggregate summary.",
      );
    } catch (error) {
      status(error instanceof Error ? error.message : "Paired run failed.");
    } finally {
      running = false;
      render();
    }
  };

  const invalidateReference = (): void => {
    entries = entries.filter(
      (entry) => entry.index !== selected || entry.mode !== mode(),
    );
  };
  const sourcePoint = (event: PointerEvent): Point | null => {
    const item = queue[selected];
    if (!item) return null;
    const bounds = sourceCanvas.getBoundingClientRect();
    return {
      x: Math.round(
        Math.max(
          0,
          Math.min(
            item.width - 1,
            ((event.clientX - bounds.left) * item.width) / bounds.width,
          ),
        ),
      ),
      y: Math.round(
        Math.max(
          0,
          Math.min(
            item.height - 1,
            ((event.clientY - bounds.top) * item.height) / bounds.height,
          ),
        ),
      ),
    };
  };
  const updateCorner = (index: number, point: Point): void => {
    const draft = activeDraft();
    if (!draft || draft.kind !== "board") return;
    draft.corners[index] = point;
    selectedCorner = index;
    invalidateReference();
    syncCornerInputs(draft.corners);
    if (activeBitmap) drawDiagnosticSource(activeBitmap);
  };
  const nearestCorner = (event: PointerEvent): number | null => {
    const draft = activeDraft();
    const point = sourcePoint(event);
    const item = queue[selected];
    if (!draft || !point || !item) return null;
    const bounds = sourceCanvas.getBoundingClientRect();
    const threshold =
      30 * Math.max(item.width / bounds.width, item.height / bounds.height);
    let nearest = 0;
    for (let index = 1; index < 4; index++)
      if (
        Math.hypot(
          draft.corners[index]!.x - point.x,
          draft.corners[index]!.y - point.y,
        ) <
        Math.hypot(
          draft.corners[nearest]!.x - point.x,
          draft.corners[nearest]!.y - point.y,
        )
      )
        nearest = index;
    return Math.hypot(
      draft.corners[nearest]!.x - point.x,
      draft.corners[nearest]!.y - point.y,
    ) <= threshold
      ? nearest
      : null;
  };

  sourceCanvas.addEventListener("pointerdown", (event) => {
    const point = sourcePoint(event);
    const draft = activeDraft();
    if (!point || draft?.kind !== "board") return;
    if (cornerPlacement < 4) {
      draggingCorner = cornerPlacement;
      updateCorner(cornerPlacement, point);
    } else {
      draggingCorner = nearestCorner(event);
      if (draggingCorner === null) {
        status("Drag one of the four numbered corner handles.");
        return;
      }
      selectedCorner = draggingCorner;
    }
    sourceCanvas.classList.add("is-dragging");
    sourceCanvas.setPointerCapture(event.pointerId);
    event.preventDefault();
  });
  sourceCanvas.addEventListener("pointermove", (event) => {
    if (draggingCorner === null) return;
    const point = sourcePoint(event);
    if (point) updateCorner(draggingCorner, point);
  });
  const finishCornerDrag = (): void => {
    if (draggingCorner === null) return;
    if (cornerPlacement < 4) cornerPlacement++;
    cornerPlacements.set(selected, cornerPlacement);
    draggingCorner = null;
    sourceCanvas.classList.remove("is-dragging");
    status(
      cornerPlacement < 4
        ? `Now click the ${CORNER_NAMES[cornerPlacement]} corner.`
        : "Grid ready. Drag any numbered corner for fine adjustment, then check the 64 reference squares.",
    );
    render();
  };
  sourceCanvas.addEventListener("pointerup", finishCornerDrag);
  sourceCanvas.addEventListener("pointercancel", finishCornerDrag);
  sourceCanvas.addEventListener("keydown", (event) => {
    const numeric = Number(event.key);
    if (numeric >= 1 && numeric <= 4) {
      selectedCorner = numeric - 1;
      render();
      return;
    }
    const delta = {
      ArrowLeft: { x: -1, y: 0 },
      ArrowRight: { x: 1, y: 0 },
      ArrowUp: { x: 0, y: -1 },
      ArrowDown: { x: 0, y: 1 },
    }[event.key];
    const draft = activeDraft();
    const item = queue[selected];
    if (!delta || draft?.kind !== "board" || !item) return;
    event.preventDefault();
    const amount = event.shiftKey ? 10 : 1;
    const point = draft.corners[selectedCorner]!;
    updateCorner(selectedCorner, {
      x: Math.max(0, Math.min(item.width - 1, point.x + delta.x * amount)),
      y: Math.max(0, Math.min(item.height - 1, point.y + delta.y * amount)),
    });
    cornerPlacement = 4;
    cornerPlacements.set(selected, cornerPlacement);
    render();
  });
  byId("diagnostic-restart-corners").addEventListener("click", () => {
    const draft = activeDraft();
    const item = queue[selected];
    if (!draft || !item) return;
    draft.corners = imageEdgeCorners(item.width, item.height);
    cornerPlacement = 0;
    cornerPlacements.set(selected, cornerPlacement);
    selectedCorner = 0;
    invalidateReference();
    syncCornerInputs(draft.corners);
    status("Click the top-left inner-grid corner.");
    render();
  });
  byId("diagnostic-use-image-edges").addEventListener("click", () => {
    const draft = activeDraft();
    const item = queue[selected];
    if (!draft || !item) return;
    draft.corners = imageEdgeCorners(item.width, item.height);
    cornerPlacement = 4;
    cornerPlacements.set(selected, cornerPlacement);
    selectedCorner = 0;
    invalidateReference();
    syncCornerInputs(draft.corners);
    status("Image edges selected as the inner grid. Drag a corner if needed.");
    render();
  });
  byId<HTMLSelectElement>("diagnostic-reference-kind").addEventListener(
    "change",
    (event) => {
      const draft = activeDraft();
      if (!draft) return;
      draft.kind = (event.target as HTMLSelectElement)
        .value as Reference["kind"];
      invalidateReference();
      render();
    },
  );
  byId<HTMLSelectElement>("diagnostic-orientation").addEventListener(
    "change",
    (event) => {
      const draft = activeDraft();
      if (!draft) return;
      draft.orientation = (event.target as HTMLSelectElement)
        .value as ReferenceDraft["orientation"];
      invalidateReference();
      render();
    },
  );
  const setReferenceLabel = (index: number, label: Label): void => {
    const draft = activeDraft();
    if (!draft) return;
    draft.labels[index] = label;
    invalidateReference();
    render();
  };
  for (let index = 0; index < 64; index++) {
    const select = byId<HTMLSelectElement>(`diagnostic-label-${index}`);
    select.addEventListener("change", (event) => {
      setReferenceLabel(
        index,
        (event.target as HTMLSelectElement).value as Label,
      );
    });
    select.addEventListener("keydown", (event) => {
      if (event.altKey) {
        const delta = {
          ArrowLeft: -1,
          ArrowRight: 1,
          ArrowUp: -8,
          ArrowDown: 8,
        }[event.key];
        if (delta === undefined) return;
        event.preventDefault();
        labelContainer
          .querySelectorAll("select")
          [Math.max(0, Math.min(63, index + delta))]?.focus();
        return;
      }
      const label = event.key === "." ? "empty" : event.key;
      if (!LABELS.includes(label as Label)) return;
      event.preventDefault();
      select.value = label;
      setReferenceLabel(index, label as Label);
    });
  }
  for (let index = 0; index < 8; index++)
    byId<HTMLInputElement>(`diagnostic-corner-${index}`).addEventListener(
      "change",
      (event) => {
        const draft = activeDraft();
        const item = queue[selected];
        const value = Number((event.target as HTMLInputElement).value);
        if (!draft || !item || !Number.isFinite(value)) return;
        const corner = Math.floor(index / 2);
        const point = { ...draft.corners[corner]! };
        if (index % 2 === 0)
          point.x = Math.max(0, Math.min(item.width - 1, value));
        else point.y = Math.max(0, Math.min(item.height - 1, value));
        updateCorner(corner, point);
        cornerPlacement = 4;
        cornerPlacements.set(selected, cornerPlacement);
        render();
      },
    );

  byId<HTMLInputElement>("diagnostic-files").addEventListener(
    "change",
    (event) => {
      void (async () => {
        const files = [...((event.target as HTMLInputElement).files ?? [])];
        if (!files.length || files.length > 20)
          return status("Choose from 1 to 20 PNG/JPEG files.");
        try {
          if (files.reduce((sum, file) => sum + file.size, 0) > MAX_QUEUE_BYTES)
            throw new Error("The compressed queue exceeds its 100 MiB bound.");
          const prepared: QueueItem[] = [];
          for (const file of files) {
            const bytes = new Uint8Array(await file.arrayBuffer());
            const dimensions = inspectRaster(bytes);
            prepared.push({
              bytes,
              sha256: await digest(bytes),
              ...dimensions,
            });
          }
          queue = prepared;
          referenceDrafts = new Map(
            prepared.map((item, index) => [
              index,
              {
                kind: "board" as const,
                corners: imageEdgeCorners(item.width, item.height),
                orientation: "unknown" as const,
                labels: Array<Label>(64).fill("empty"),
              },
            ]),
          );
          releaseActiveBitmap();
          selected = 0;
          cornerPlacements = new Map(prepared.map((_, index) => [index, 0]));
          cornerPlacement = 0;
          selectedCorner = 0;
          entries = [];
          syncReferenceControls();
          status(
            "Queue prepared. Files stay only in this page memory; references must be saved before recognition.",
          );
          render();
        } catch (error) {
          status(
            error instanceof Error ? error.message : "Could not prepare queue.",
          );
        }
      })();
    },
  );
  byId("diagnostic-previous").addEventListener("click", () => {
    releaseActiveBitmap();
    selected = Math.max(0, selected - 1);
    cornerPlacement = cornerPlacements.get(selected) ?? 0;
    selectedCorner = 0;
    syncReferenceControls();
    render();
  });
  byId("diagnostic-next").addEventListener("click", () => {
    releaseActiveBitmap();
    selected = Math.min(queue.length - 1, selected + 1);
    cornerPlacement = cornerPlacements.get(selected) ?? 0;
    selectedCorner = 0;
    syncReferenceControls();
    render();
  });
  byId("diagnostic-save-reference").addEventListener("click", saveReference);
  byId("diagnostic-mode").addEventListener("change", render);
  byId("diagnostic-run").addEventListener("click", () => void pairedRun());
  byId("diagnostic-export-private").addEventListener("click", () => {
    try {
      download("private-evaluation.json", session());
    } catch (error) {
      status(error instanceof Error ? error.message : "Private export failed.");
    }
  });
  byId("diagnostic-export-summary").addEventListener("click", () => {
    try {
      download("evaluation-summary.json", summarizeEvaluation(session()));
    } catch (error) {
      status(error instanceof Error ? error.message : "Summary export failed.");
    }
  });
  byId<HTMLInputElement>("diagnostic-import").addEventListener(
    "change",
    (event) => {
      void (async () => {
        try {
          imported = privateEvaluationSchema.parse(
            JSON.parse(
              (await (event.target as HTMLInputElement).files?.[0]?.text()) ||
                "",
            ),
          );
          status(
            "Private evidence parsed. Reselect its inputs in the original exported order to verify SHA-256 before it can be restored.",
          );
        } catch {
          status("Private import is not a valid sensitive evaluation export.");
        }
      })();
    },
  );
  byId<HTMLInputElement>("diagnostic-reselect").addEventListener(
    "change",
    (event) => {
      void (async () => {
        if (!imported) return status("Select a private export first.");
        const files = [...((event.target as HTMLInputElement).files ?? [])];
        if (!(await matchingImportFiles(imported, files)))
          return status(
            "Reselected files do not exactly match the private export hashes and sizes.",
          );
        const items: QueueItem[] = [];
        for (const file of files) {
          const bytes = new Uint8Array(await file.arrayBuffer());
          const dimensions = inspectRaster(bytes);
          items.push({
            bytes,
            sha256: await digest(bytes),
            ...dimensions,
          });
        }
        queue = items;
        releaseActiveBitmap();
        entries = imported.entries;
        referenceDrafts = new Map(
          items.map((item, index) => {
            const restored = imported?.entries.find(
              (entry) => entry.index === index,
            )?.reference;
            return [
              index,
              restored?.kind === "board"
                ? {
                    kind: restored.kind,
                    corners: restored.corners.map((point) => ({
                      ...point,
                    })) as MutableCorners,
                    orientation: restored.orientation,
                    labels: [...restored.labels],
                  }
                : {
                    kind: restored?.kind ?? ("board" as const),
                    corners: imageEdgeCorners(item.width, item.height),
                    orientation: "unknown" as const,
                    labels: Array<Label>(64).fill("empty"),
                  },
            ];
          }),
        );
        selected = 0;
        cornerPlacements = new Map(
          items.map((_, index) => [
            index,
            imported?.entries.some(
              (entry) =>
                entry.index === index && entry.reference.kind === "board",
            )
              ? 4
              : 0,
          ]),
        );
        cornerPlacement = cornerPlacements.get(selected) ?? 0;
        selectedCorner = 0;
        syncReferenceControls();
        status(
          "Private evidence restored locally after matching-file verification.",
        );
        render();
      })();
    },
  );
  render();
}
