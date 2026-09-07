import type { CandidateConfig } from "./candidate.ts";
import {
  createBrowserClient,
  createCandidateBrowserClient,
  prepareManualGridInput,
  restoreManualGridResult,
} from "./browser.ts";
import type { RecognitionClient } from "./client.ts";
import { VERSION, type Label, type Request } from "./contract.ts";
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

  const labels = (): Label[] =>
    Array.from({ length: 64 }, (_, index) => {
      const value = byId<HTMLSelectElement>(`diagnostic-label-${index}`).value;
      return (value || "empty") as Label;
    });
  const reference = (): Reference => {
    const kind = byId<HTMLSelectElement>("diagnostic-reference-kind").value;
    if (kind !== "board") return referenceSchema.parse({ kind });
    const numbers = Array.from({ length: 8 }, (_, index) =>
      Number(byId<HTMLInputElement>(`diagnostic-corner-${index}`).value),
    );
    return referenceSchema.parse({
      kind: "board",
      corners: [
        { x: numbers[0]!, y: numbers[1]! },
        { x: numbers[2]!, y: numbers[3]! },
        { x: numbers[4]!, y: numbers[5]! },
        { x: numbers[6]!, y: numbers[7]! },
      ],
      orientation: byId<HTMLSelectElement>("diagnostic-orientation").value,
      labels: labels(),
    });
  };
  const existing = (index: number, mode: RunMode) =>
    entries.find((entry) => entry.index === index && entry.mode === mode);
  const mode = (): RunMode =>
    byId<HTMLSelectElement>("diagnostic-mode").value as RunMode;
  const render = (): void => {
    const item = queue[selected];
    byId("diagnostic-current").textContent = item
      ? `${selected + 1}/${queue.length}: ${item.width}×${item.height}, SHA-256 ${item.sha256.slice(0, 12)}…`
      : "No local images queued.";
    byId<HTMLButtonElement>("diagnostic-save-reference").disabled =
      !item || running;
    byId<HTMLButtonElement>("diagnostic-run").disabled =
      !item || !existing(selected, mode()) || running;
    const complete =
      entries.length > 0 && entries.every((entry) => entry.results);
    byId<HTMLButtonElement>("diagnostic-export-private").disabled = !complete;
    byId<HTMLButtonElement>("diagnostic-export-summary").disabled = !complete;
    const result = existing(selected, mode());
    const output = byId("diagnostic-result");
    output.hidden = !result?.results;
    if (result?.results) {
      const v2 = compareResult(result.reference, result.results.v2.result);
      const fenshot = compareResult(
        result.reference,
        result.results.fenshot.result,
      );
      output.textContent = `v2: ${JSON.stringify(v2)}\nFENShot: ${JSON.stringify(fenshot)}`;
    }
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
          releaseActiveBitmap();
          selected = 0;
          entries = [];
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
    render();
  });
  byId("diagnostic-next").addEventListener("click", () => {
    releaseActiveBitmap();
    selected = Math.min(queue.length - 1, selected + 1);
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
        selected = 0;
        status(
          "Private evidence restored locally after matching-file verification.",
        );
        render();
      })();
    },
  );
  render();
}
