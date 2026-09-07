import "./style.css";
import {
  LABELS,
  VERSION,
  type Rect,
  requestSchema,
  type Label,
  orientationSchema,
  manualBoard,
} from "./contract.ts";
import { Editor } from "./editor.ts";
import {
  createBrowserClient,
  createCandidateBrowserClient,
} from "./browser.ts";
import { loadCandidateFiles, type CandidateConfig } from "./candidate.ts";
import { decodeRaster } from "./image.ts";
const el = <T extends HTMLElement>(id: string): T =>
  document.getElementById(id) as T;
const canvas = el<HTMLCanvasElement>("source"),
  ctx = canvas.getContext("2d", { willReadFrequently: true })!;
const editor = new Editor();
let client = createBrowserClient();
let candidate: CandidateConfig | null = null;
let backend: "baseline" | "candidate" = "baseline";
let bitmap: ImageBitmap | null = null,
  selection: Rect | null = null,
  generation = 0,
  busy = false;
const status = (text: string): void => {
  el("status").textContent = text;
};
function controls(): void {
  el<HTMLButtonElement>("detect").disabled = !bitmap || busy;
  el<HTMLButtonElement>("recognize").disabled = !bitmap || !selection || busy;
  el<HTMLButtonElement>("cancel").disabled = !busy;
  el<HTMLFieldSetElement>("bounds").disabled = !bitmap;
  el<HTMLButtonElement>("export").disabled = !editor.board;
  el<HTMLButtonElement>("load-candidate").disabled =
    busy ||
    !el<HTMLInputElement>("candidate-manifest").files?.[0] ||
    !el<HTMLInputElement>("candidate-classifier").files?.[0] ||
    !el<HTMLInputElement>("candidate-detector").files?.[0];
}
function draw(): void {
  if (!bitmap) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    return;
  }
  ctx.drawImage(bitmap, 0, 0);
  const corners = editor.board?.corners;
  if (corners) {
    ctx.strokeStyle = "#096ccc";
    ctx.lineWidth = Math.max(2, canvas.width / 250);
    ctx.beginPath();
    corners.forEach((p, i) =>
      i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y),
    );
    ctx.closePath();
    ctx.stroke();
  }
}
function renderBoard(): void {
  const container = el("board");
  container.replaceChildren();
  for (let i = 0; i < 64; i++) {
    const square = editor.board?.squares[i];
    const wrap = document.createElement("div");
    wrap.className = `square ${(Math.floor(i / 8) + (i % 8)) % 2 ? "dark" : ""} ${!square || square.uncertain ? "uncertain" : ""} ${editor.isEdited(i) ? "edited" : ""}`;
    const select = document.createElement("select");
    select.setAttribute(
      "aria-label",
      `Row ${Math.floor(i / 8) + 1}, column ${(i % 8) + 1}${editor.isEdited(i) ? ", corrected" : !square || square.uncertain ? ", uncertain" : ""}`,
    );
    select.disabled = !editor.board;
    for (const [value, name] of [
      ["", "?"],
      ...LABELS.map((l) => [l, l === "empty" ? "·" : l]),
    ]) {
      const option = document.createElement("option");
      option.value = value!;
      option.textContent = name!;
      select.append(option);
    }
    select.value = square?.label ?? "";
    select.addEventListener("change", () => {
      if (!select.value) {
        select.value = editor.board?.squares[i]?.label ?? "";
        return;
      }
      editor.edit(i, select.value as Label);
      wrap.classList.add("edited");
      select.setAttribute(
        "aria-label",
        `Row ${Math.floor(i / 8) + 1}, column ${(i % 8) + 1}, corrected`,
      );
      placement();
    });
    select.addEventListener("keydown", (event) => {
      // Alt + arrows navigate the grid; ordinary arrows retain native select behavior.
      if (!event.altKey) return;
      const delta = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -8, ArrowDown: 8 }[
        event.key
      ];
      if (delta === undefined) return;
      event.preventDefault();
      container
        .querySelectorAll("select")
        [Math.max(0, Math.min(63, i + delta))]?.focus();
    });
    wrap.append(select);
    container.append(wrap);
  }
  el<HTMLSelectElement>("orientation").value =
    editor.board?.orientation ?? "unknown";
  placement();
  controls();
}
function placement(): void {
  el("placement").textContent =
    editor.placement() ?? "Placement needs all squares and an orientation.";
}
function stop(): void {
  client.cancel();
  editor.invalidate();
  busy = false;
  generation++;
  controls();
}
function setSelection(rect: Rect): void {
  if (!bitmap) return;
  requestSchema.parse({
    schema: VERSION,
    requestId: "selection",
    image: { width: bitmap.width, height: bitmap.height },
    selection: rect,
  });
  stop();
  selection = rect;
  editor.reset(manualBoard(rect));
  for (const key of ["x", "y", "width", "height"] as const)
    el<HTMLInputElement>(key).value = String(rect[key]);
  renderBoard();
  draw();
  status("Selection ready. Read it locally or enter the pieces yourself.");
}
el<HTMLInputElement>("file").addEventListener("change", async (event) => {
  const file = (event.target as HTMLInputElement).files?.[0];
  if (!file) return;
  stop();
  const token = generation;
  bitmap?.close();
  bitmap = null;
  selection = null;
  editor.reset();
  renderBoard();
  draw();
  controls();
  status("Decoding local image…");
  try {
    const decoded = await decodeRaster(file);
    if (token !== generation) {
      decoded.close();
      return;
    }
    bitmap = decoded;
    canvas.width = decoded.width;
    canvas.height = decoded.height;
    el("empty").hidden = true;
    setSelection({ x: 0, y: 0, width: decoded.width, height: decoded.height });
    status("Image ready. Find a board or select its inner playing grid.");
  } catch {
    if (token === generation)
      status(
        "Could not open image. Use a valid PNG/JPEG, at most 20 MiB, 16 million pixels and 8192 pixels per side.",
      );
  }
});
el("select").addEventListener("click", () => {
  try {
    setSelection({
      x: Number(el<HTMLInputElement>("x").value),
      y: Number(el<HTMLInputElement>("y").value),
      width: Number(el<HTMLInputElement>("width").value),
      height: Number(el<HTMLInputElement>("height").value),
    });
  } catch {
    status(
      "Selection must have positive dimensions and stay inside the source image.",
    );
  }
});
let anchor: { x: number; y: number } | null = null;
function point(event: PointerEvent) {
  const r = canvas.getBoundingClientRect();
  return {
    x: Math.round(
      Math.max(
        0,
        Math.min(
          canvas.width,
          ((event.clientX - r.left) * canvas.width) / r.width,
        ),
      ),
    ),
    y: Math.round(
      Math.max(
        0,
        Math.min(
          canvas.height,
          ((event.clientY - r.top) * canvas.height) / r.height,
        ),
      ),
    ),
  };
}
canvas.addEventListener("pointerdown", (event) => {
  if (!bitmap) return;
  anchor = point(event);
  canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener("pointerup", (event) => {
  if (!anchor) return;
  const end = point(event),
    start = anchor;
  anchor = null;
  if (end.x === start.x || end.y === start.y) return;
  setSelection({
    x: Math.min(start.x, end.x),
    y: Math.min(start.y, end.y),
    width: Math.abs(end.x - start.x),
    height: Math.abs(end.y - start.y),
  });
});
canvas.addEventListener("pointercancel", () => {
  anchor = null;
});
async function run(manual: boolean): Promise<void> {
  if (!bitmap) return;
  const token = generation,
    requestId = crypto.randomUUID();
  const request = requestSchema.parse({
    schema: VERSION,
    requestId,
    image: { width: bitmap.width, height: bitmap.height },
    selection: manual ? selection : null,
  });
  // The source is redrawn without overlays before extracting pixels.
  ctx.drawImage(bitmap, 0, 0);
  const rgba = ctx.getImageData(0, 0, bitmap.width, bitmap.height).data;
  draw();
  editor.begin(requestId);
  busy = true;
  controls();
  status(
    `Recognizing locally with ${backend === "candidate" ? "the synthetic candidate" : "FENShot"} on WASM CPU… You can keep editing or cancel.`,
  );
  try {
    const result = await client.recognize(request, rgba);
    if (token !== generation) return;
    const applied = editor.apply(result);
    renderBoard();
    draw();
    if (result.boards.length && !applied) {
      status(
        "The detected grid differs from your edited grid. Your edits are preserved; select the new grid explicitly to start over.",
      );
      return;
    }
    status(
      `${result.status === "ok" ? `Result ready in ${Math.round(result.timings.totalMs)} ms. Review every square. ` : ""}${result.warnings.join(" ")}`,
    );
  } catch (error) {
    if (token === generation)
      status(error instanceof Error ? error.message : "Recognition failed");
  } finally {
    if (token === generation) {
      busy = false;
      controls();
    }
  }
}
el("detect").addEventListener("click", () => void run(false));
el("recognize").addEventListener("click", () => void run(true));
for (const id of [
  "candidate-manifest",
  "candidate-classifier",
  "candidate-detector",
])
  el(id).addEventListener("change", controls);
el("load-candidate").addEventListener("click", () => {
  void (async () => {
    try {
      const manifest = el<HTMLInputElement>("candidate-manifest").files?.[0];
      const classifier = el<HTMLInputElement>("candidate-classifier")
        .files?.[0];
      const detector = el<HTMLInputElement>("candidate-detector").files?.[0];
      if (!manifest || !classifier || !detector) return;
      status("Verifying local candidate files…");
      const loaded = await loadCandidateFiles(manifest, classifier, detector);
      stop();
      candidate = loaded;
      backend = "candidate";
      client = createCandidateBrowserClient(candidate);
      const select = el<HTMLSelectElement>("backend");
      select.options[1]!.disabled = false;
      select.value = "candidate";
      status(
        `Loaded ${candidate.manifest.name} ${candidate.manifest.version}. It is synthetic-only and unqualified; every square will remain marked for review.`,
      );
    } catch (error) {
      status(
        error instanceof Error
          ? error.message
          : "Could not load the local candidate files.",
      );
    } finally {
      controls();
    }
  })();
});
el("backend").addEventListener("change", () => {
  const selected = el<HTMLSelectElement>("backend").value;
  stop();
  if (selected === "candidate" && candidate) {
    backend = "candidate";
    client = createCandidateBrowserClient(candidate);
    status(
      "Synthetic candidate selected. Its outputs are unqualified and require full review.",
    );
  } else {
    backend = "baseline";
    client = createBrowserClient();
    el<HTMLSelectElement>("backend").value = "baseline";
    status("FENShot baseline selected. Review every result.");
  }
});
el("cancel").addEventListener("click", () => {
  stop();
  status("Recognition cancelled. Your edits are preserved.");
});
el("orientation").addEventListener("change", () => {
  editor.orient(
    orientationSchema.parse(el<HTMLSelectElement>("orientation").value),
  );
  placement();
});
el("export").addEventListener("click", () => {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(editor.export(), null, 2)], {
      type: "application/json",
    }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = "edited-position.json";
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
renderBoard();
if (import.meta.env.PROD && "serviceWorker" in navigator) {
  void navigator.serviceWorker
    .register("/sw.js")
    .then(async () => {
      await navigator.serviceWorker.ready;
      el("offline").textContent =
        "Offline ready · baseline assets cached · candidate files stay local and are loaded per session · no uploads";
    })
    .catch(() => {
      el("offline").textContent =
        "Offline cache unavailable. Keep this local server running.";
    });
}
