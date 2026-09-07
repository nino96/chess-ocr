// Local dataset dashboard. The server validates every persisted annotation again.
type Page = {
  id: string;
  source: string;
  page: number;
  revision: number;
  accepted: number;
  draft: number;
  proposal_count: number;
  deferred_reason: string | null;
  review_state: string | null;
};
type Source = {
  id: string;
  label: string;
  split: string;
  selected_pages: number;
};
type Queue = {
  sources: Source[];
  pages: Page[];
  duplicates: { a: string; b: string; reason: string }[];
  status: Record<string, unknown>;
};
type Draft = {
  sample_id: string;
  revision: number;
  image_sha256: string;
  state: Record<string, unknown>;
};
type Archive = {
  id: string;
  bytes: number | null;
  version: string | null;
  error: string | null;
};
const el = <T extends HTMLElement>(id: string) => {
  const value = document.getElementById(id);
  if (!value) throw new Error("Missing application control");
  return value as T;
};
const input = (id: string) => el<HTMLInputElement>(id);
const frame = el<HTMLIFrameElement>("editor");
let data: Queue | undefined;
let current: Page | undefined;
let draft: Draft | undefined;
let draftVersion = 0;
let generation = 0;
let persisted = 0;
let saving: Promise<void> | undefined;
let timer: ReturnType<typeof setTimeout> | undefined;
let ready = false;
let busy = false;
let limit = 50;
let refreshSequence = 0;
let editorSession = 0;
let editorToken = "";
let submitting = false;
let profile: { reviewer: string; human: boolean } | undefined;
let selectedArchive: Archive | undefined;
const note = (message: string) => {
  el("message").textContent = message;
};
const saved = (message: string) => {
  el("save-status").textContent = message;
};
function object(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
async function api(
  path: string,
  body?: unknown,
): Promise<Record<string, unknown>> {
  const controller = new AbortController();
  const timeout = setTimeout(
    () => controller.abort(),
    path === "/api/ingest" ? 240_000 : 30_000,
  );
  try {
    const response = await fetch(path, {
      ...(body === undefined
        ? {}
        : {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              "X-Dataset-Request": "1",
            },
            body: JSON.stringify(body),
          }),
      signal: controller.signal,
      cache: "no-store",
    });
    const result: unknown = await response.json();
    if (!object(result)) throw new Error("Invalid server response");
    if (!response.ok)
      throw new Error(
        typeof result.error === "string" ? result.error : "Request failed",
      );
    return result;
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError")
      throw new Error(
        "Request timed out. Edits remain in this tab; retry saving or refresh status to check whether the operation completed.",
        { cause: "network" },
      );
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}
function queueResponse(value: Record<string, unknown>): Queue {
  if (
    value.schema !== "chess-ocr-dataset-app/1" ||
    !Array.isArray(value.pages) ||
    !Array.isArray(value.sources) ||
    !Array.isArray(value.duplicates) ||
    !object(value.status)
  )
    throw new Error("Invalid queue response");
  const id = (x: unknown) =>
    typeof x === "string" && /^[a-zA-Z0-9_-]{1,80}$/.test(x);
  if (
    !value.pages.every(
      (x: unknown) =>
        object(x) &&
        id(x.id) &&
        id(x.source) &&
        Number.isInteger(x.page) &&
        Number.isInteger(x.revision) &&
        [0, 1].includes(Number(x.accepted)) &&
        [0, 1].includes(Number(x.draft)) &&
        Number.isInteger(x.proposal_count) &&
        Number(x.proposal_count) >= 0 &&
        (x.deferred_reason === null || typeof x.deferred_reason === "string") &&
        (x.review_state === null || typeof x.review_state === "string"),
    ) ||
    !value.sources.every(
      (x: unknown) =>
        object(x) &&
        id(x.id) &&
        typeof x.label === "string" &&
        typeof x.split === "string" &&
        Number.isInteger(x.selected_pages),
    ) ||
    !value.duplicates.every(
      (x: unknown) =>
        object(x) && id(x.a) && id(x.b) && typeof x.reason === "string",
    )
  )
    throw new Error("Invalid queue records");
  return value as unknown as Queue;
}
function filtered(): Page[] {
  const doc = input("document").value,
    filter = input("filter").value;
  return (data?.pages ?? []).filter(
    (p) =>
      (!doc || doc === p.source) &&
      (filter === "all" ||
        Boolean(p.accepted) === (filter === "accepted") ||
        (filter === "proposal" && p.proposal_count > 0) ||
        (filter === "deferred" && p.deferred_reason !== null) ||
        (filter === "ambiguous" &&
          ["partial", "unsupported"].includes(p.review_state ?? ""))),
  );
}
function label(page: Page): string {
  return `${data?.sources.find((s) => s.id === page.source)?.label ?? "Document"} · page ${page.page}`;
}
function button(text: string, action: () => Promise<void>): HTMLButtonElement {
  const b = document.createElement("button");
  b.textContent = text;
  b.addEventListener("click", () => {
    void run(action);
  });
  return b;
}
function renderQueue(): void {
  const pages = filtered();
  el("queue-count").textContent =
    `${pages.length} matching pages of ${data?.pages.length ?? 0} total · ${data?.sources.map((s) => `${s.label}: ${s.split}, ${s.selected_pages} selected pages`).join("; ") ?? ""}`;
  el<HTMLButtonElement>("review-next").disabled = pages.length === 0;
  el("pages").replaceChildren(
    ...pages.slice(0, limit).map((p) => {
      const b = button(
        `${label(p)} · ${p.accepted ? "Accepted" : p.draft ? "Draft" : "Needs review"}`,
        () => openPage(p),
      );
      b.className = `page-card ${p.accepted ? "accepted" : ""}`;
      const img = document.createElement("img");
      img.loading = "lazy";
      img.src = `/thumbnail/${p.id}`;
      img.alt = `Preview of ${label(p)}`;
      b.prepend(img);
      return b;
    }),
  );
  el("more").hidden = pages.length <= limit;
  el("duplicates").replaceChildren(
    ...(data?.duplicates ?? []).map((pair) => {
      const row = document.createElement("div");
      row.className = "panel";
      const text = document.createElement("p");
      text.textContent = `Possible duplicate (${pair.reason})`;
      row.append(text);
      for (const [name, id] of [
        ["Inspect first page", pair.a],
        ["Inspect second page", pair.b],
      ] as const) {
        row.append(
          button(name, async () => {
            const page = data?.pages.find((p) => p.id === id);
            if (page) await openPage(page);
          }),
        );
      }
      for (const decision of ["distinct", "duplicate"]) {
        const b = button(
          decision === "distinct"
            ? "Different artwork / pages"
            : "Mark duplicate",
          async () => {
            await api("/api/duplicate", { a: pair.a, b: pair.b, decision });
            await refresh();
            note("Duplicate decision saved.");
          },
        );
        b.disabled =
          decision === "distinct" &&
          ["exact", "board-exact"].includes(pair.reason);
        row.append(b);
      }
      return row;
    }),
  );
}
async function refresh(): Promise<void> {
  const sequence = ++refreshSequence;
  const result = queueResponse(await api("/api/queue"));
  if (sequence !== refreshSequence) return;
  data = result;
  const selected = input("document").value;
  el("document").replaceChildren(
    new Option("All documents", ""),
    ...data.sources.map((s) => new Option(s.label, s.id)),
  );
  input("document").value = data.sources.some((s) => s.id === selected)
    ? selected
    : "";
  const s = data.status;
  const worker = object(s.worker) ? s.worker.state : "unknown";
  const jobs = object(s.jobs)
    ? Object.entries(s.jobs)
        .map(([state, count]) => `${count} ${state}`)
        .join(", ")
    : "";
  const budget = object(s.budget) ? s.budget : {};
  el("summary").textContent =
    `${String(s.accepted_pages)} / ${String(s.pages)} pages accepted · job: ${String(worker)}${jobs ? ` (${jobs})` : ""} · ${String(s.review_decisions)} / ${String(budget.review_limit)} review decisions used`;
  renderQueue();
  await refreshArchives();
  void refreshProposalControls();
}
function proposalOptions(id: string, values: unknown[]): void {
  const select = el<HTMLSelectElement>(id);
  const previous = select.value;
  select.replaceChildren(
    new Option("Select a provider", ""),
    ...values.flatMap((value) => {
      if (!object(value) || typeof value.id !== "string") return [];
      const title = typeof value.label === "string" ? value.label : value.id;
      return [new Option(title, value.id)];
    }),
  );
  select.value = previous;
}
async function refreshProposalControls(): Promise<void> {
  try {
    const result = await api("/api/providers");
    const providers = Array.isArray(result.providers) ? result.providers : [];
    const localizers = providers.filter(
      (p) => object(p) && p.capability === "localization",
    );
    const labelers = providers.filter(
      (p) => object(p) && p.capability === "labels",
    );
    const selectedLocalizer = el<HTMLSelectElement>("proposal-localizer").value;
    const selectedLabeler = el<HTMLSelectElement>("proposal-labeler").value;
    proposalOptions("proposal-localizer", localizers);
    proposalOptions("proposal-labeler", labelers);
    if (object(result.defaults)) {
      const localizer = result.defaults.localization;
      const labeler = result.defaults.labels;
      if (!selectedLocalizer && typeof localizer === "string")
        el<HTMLSelectElement>("proposal-localizer").value = localizer;
      if (!selectedLabeler && typeof labeler === "string")
        el<HTMLSelectElement>("proposal-labeler").value = labeler;
    }
    const status = await api("/api/proposals/status");
    const latest =
      Array.isArray(status.runs) && object(status.runs[0])
        ? status.runs[0]
        : undefined;
    el("proposal-run-status").textContent = latest
      ? `Proposal status: ${String(latest.state)} · ${String(latest.completed)} / ${String(latest.pages)} pages`
      : "Proposal providers ready; no run has started.";
  } catch {
    // Proposal infrastructure is deliberately optional during local review.
    el("proposal-run-status").textContent =
      "No proposal service available; manual review remains fully usable.";
  }
}
function archiveLabel(archive: Archive): string {
  const stamp = archive.id.slice(0, 16);
  const date = `${stamp.slice(0, 4)}-${stamp.slice(4, 6)}-${stamp.slice(6, 8)} ${stamp.slice(9, 11)}:${stamp.slice(11, 13)}:${stamp.slice(13, 15)} UTC`;
  return `${date} · ${archive.id.slice(17)} · ${archive.bytes === null ? "size unavailable" : `${(archive.bytes / 1048576).toLocaleString(undefined, { maximumFractionDigits: 2 })} MiB`}`;
}
async function refreshArchives(): Promise<void> {
  try {
    const response = await api("/api/archives");
    if (
      !Array.isArray(response.archives) ||
      !response.archives.every(
        (a: unknown) =>
          object(a) &&
          typeof a.id === "string" &&
          /^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{12}$/.test(a.id) &&
          (a.bytes === null ||
            (typeof a.bytes === "number" &&
              Number.isSafeInteger(a.bytes) &&
              a.bytes >= 0)) &&
          (a.version === null ||
            (typeof a.version === "string" &&
              /^[0-9a-f]{64}$/.test(a.version))) &&
          (a.error === null || typeof a.error === "string"),
      )
    )
      throw new Error("Invalid archive list");
    const archives = response.archives as Archive[];
    el("archive-status").textContent = archives.length
      ? `${archives.length} archived datasets`
      : "No archives.";
    el("archives").replaceChildren(
      ...archives.map((archive) => {
        const row = document.createElement("div");
        row.className = "panel archive-entry";
        const text = document.createElement("p");
        text.textContent = archiveLabel(archive);
        row.append(text);
        if (archive.error) {
          const error = document.createElement("p");
          error.textContent = archive.error;
          row.append(error);
        }
        const remove = button("Delete archive…", async () => {
          selectedArchive = archive;
          input("archive-delete-confirmation").value = "";
          el("archive-delete-target").textContent = archiveLabel(archive);
          el("archive-delete-error").textContent = "";
          el<HTMLDialogElement>("archive-delete-dialog").showModal();
        });
        remove.className = "danger";
        remove.disabled = archive.version === null || archive.error !== null;
        row.append(remove);
        return row;
      }),
    );
  } catch (error) {
    el("archives").replaceChildren();
    el("archive-status").textContent =
      error instanceof Error ? error.message : "Could not load archives.";
  }
}
async function flush(): Promise<void> {
  clearTimeout(timer);
  if (saving) await saving;
  if (!draft || generation === persisted) return;
  const snapshot = draft,
    seq = generation,
    session = editorSession;
  saving = (async () => {
    saved("Saving draft…");
    const result = await api("/api/draft", {
      ...snapshot,
      version: draftVersion,
    });
    if (!Number.isInteger(result.version))
      throw new Error("Invalid draft response");
    if (session !== editorSession) return;
    draftVersion = Number(result.version);
    persisted = seq;
    saved(generation === persisted ? "Draft saved on GX10" : "Unsaved changes");
  })();
  try {
    await saving;
  } finally {
    saving = undefined;
  }
  if (generation !== persisted) await flush();
}
async function openPage(page: Page): Promise<void> {
  if (current && !ready)
    throw new Error(
      "Wait for the editor to finish loading, or reload the app if it failed.",
    );
  await flush();
  current = page;
  draft = undefined;
  generation = persisted = 0;
  ready = false;
  editorSession++;
  el("queue").hidden = true;
  el("editor-section").hidden = false;
  el("editing").textContent = label(page);
  saved("Loading editor…");
  editorToken = crypto.randomUUID();
  frame.src = `/review/${page.id}?session=${editorToken}`;
  el("editor-section").scrollIntoView();
}
async function navigate(delta: number): Promise<void> {
  const pages =
    data?.pages.filter(
      (p) => !input("document").value || p.source === input("document").value,
    ) ?? [];
  const next = pages[pages.findIndex((p) => p.id === current?.id) + delta];
  if (next) await openPage(next);
}
function closeEditor(): void {
  current = undefined;
  draft = undefined;
  ready = false;
  editorSession++;
  el("editor-section").hidden = true;
  el("queue").hidden = false;
  frame.removeAttribute("src");
}
async function run(action: () => Promise<void>): Promise<void> {
  if (busy) {
    note("Another action is still running.");
    return;
  }
  busy = true;
  try {
    await action();
  } catch (e) {
    note(e instanceof Error ? e.message : "Operation failed");
  } finally {
    busy = false;
  }
}
window.addEventListener("message", (event: MessageEvent<unknown>) => {
  if (
    event.origin !== location.origin ||
    event.source !== frame.contentWindow ||
    !object(event.data) ||
    !current
  )
    return;
  const message = event.data;
  if (message.sample_id !== current.id || message.session !== editorToken)
    return;
  if (
    message.type === "ready" &&
    Number.isInteger(message.draft_version) &&
    Number.isInteger(message.revision) &&
    Number(message.revision) >= 0
  ) {
    // The page may have been reviewed in another tab since the queue was read.
    current.revision = Number(message.revision);
    ready = true;
    draftVersion = Number(message.draft_version);
    saved(message.restored ? "Saved draft restored" : "Ready to review");
    if (profile)
      frame.contentWindow?.postMessage(
        { type: "profile", ...profile },
        location.origin,
      );
    // Older servers deliberately have no proposal route. Keep review usable.
    void api(`/api/proposals/${encodeURIComponent(current.id)}`)
      .then((proposal) => {
        if (current && message.sample_id === current.id && frame.contentWindow)
          frame.contentWindow.postMessage(
            { type: "proposal", proposal },
            location.origin,
          );
        frame.contentWindow?.postMessage(
          { type: "review-extensions" },
          location.origin,
        );
      })
      .catch(() => undefined);
  } else if (
    message.type === "draft" &&
    message.revision === current.revision &&
    object(message.state) &&
    typeof message.image_sha256 === "string" &&
    !submitting
  ) {
    if (
      typeof message.state.reviewer === "string" &&
      typeof message.state.human === "boolean"
    )
      profile = {
        reviewer: message.state.reviewer,
        human: message.state.human,
      };
    draft = {
      sample_id: current.id,
      revision: current.revision,
      image_sha256: message.image_sha256,
      state: message.state,
    };
    generation++;
    saved("Unsaved changes");
    clearTimeout(timer);
    timer = setTimeout(() => {
      void flush().catch((e: unknown) => {
        saved("Draft not saved — retry before leaving");
        note(e instanceof Error ? e.message : "Save failed");
      });
    }, 400);
  } else if (
    message.type === "submit" &&
    message.revision === current.revision &&
    object(message.proposal)
  ) {
    if (busy) {
      note("Another action is running. Submit again when it finishes.");
      frame.contentWindow?.postMessage(
        { type: "submission-finished" },
        location.origin,
      );
      return;
    }
    const proposal = message.proposal;
    submitting = true;
    frame.inert = true;
    void run(async () => {
      try {
        await flush();
        const submission = { ...proposal, draft_version: draftVersion };
        try {
          await api("/api/review", submission);
        } catch (firstError) {
          // A response can be lost after the transaction commits. The identical
          // request is idempotent, so one bounded retry recovers that case.
          try {
            await api("/api/review", submission);
          } catch {
            if (
              firstError instanceof TypeError ||
              (firstError instanceof Error && firstError.cause === "network")
            ) {
              closeEditor();
              throw new Error(
                "Connection lost after saving your draft. Refresh status and reopen the page to continue; the review may already have been accepted.",
              );
            }
            throw firstError;
          }
        }
        const old = current?.id;
        draft = undefined;
        generation = persisted = 0;
        closeEditor();
        await refresh();
        const pending = filtered().filter((p) => !p.accepted && p.id !== old);
        note("Review accepted and saved. No second reviewer is required.");
        if (pending[0]) await openPage(pending[0]);
      } finally {
        submitting = false;
        frame.inert = false;
        frame.contentWindow?.postMessage(
          { type: "submission-finished" },
          location.origin,
        );
      }
    });
  } else if (
    message.type === "defer" &&
    message.revision === current.revision &&
    typeof message.reason === "string" &&
    typeof message.image_sha256 === "string"
  ) {
    void run(async () => {
      await flush();
      await api("/api/defer", {
        sample_id: current?.id,
        revision: current?.revision,
        image_sha256: message.image_sha256,
        reason: message.reason,
        proposal_run:
          message.proposal_run === null ||
          typeof message.proposal_run === "string"
            ? message.proposal_run
            : null,
        elapsed_seconds:
          typeof message.elapsed_seconds === "number"
            ? message.elapsed_seconds
            : 0,
      });
      note("Page deferred. It remains available under Deferred.");
      closeEditor();
      await refresh();
    });
  }
});
window.addEventListener("beforeunload", (event) => {
  if (generation !== persisted || submitting) {
    event.preventDefault();
    event.returnValue = "";
  }
});
el("refresh").addEventListener("click", () => {
  void run(refresh);
});
el("retry-save").addEventListener("click", () => {
  void run(flush);
});
el("review-next").addEventListener("click", () => {
  void run(async () => {
    const page = filtered()[0];
    if (page) await openPage(page);
  });
});
el("proposal-start").addEventListener("click", () => {
  void run(async () => {
    const localizer = el<HTMLSelectElement>("proposal-localizer").value;
    const labeler = el<HTMLSelectElement>("proposal-labeler").value;
    if (!localizer || !labeler)
      throw new Error(
        "Choose both a localizer and labeler before starting proposals.",
      );
    const result = await api("/api/proposals/start", {
      localizer,
      labeler,
      scope: el<HTMLSelectElement>("proposal-scope").value,
      max_pages: Number(input("proposal-max-pages").value),
    });
    el("proposal-run-status").textContent =
      `Proposal status: ${String(result.state ?? "started")}`;
  });
});
el("proposal-stop").addEventListener("click", () => {
  void run(async () => {
    const result = await api("/api/proposals/stop", {});
    el("proposal-run-status").textContent =
      `Proposal status: ${String(result.state ?? "stopped")}`;
  });
});
for (const id of ["document", "filter"])
  el(id).addEventListener("change", () => {
    limit = 50;
    renderQueue();
  });
el("more").addEventListener("click", () => {
  limit += 50;
  renderQueue();
});
el("previous").addEventListener("click", () => {
  void run(() => navigate(-1));
});
el("next").addEventListener("click", () => {
  void run(() => navigate(1));
});
el("back").addEventListener("click", () => {
  void run(async () => {
    await flush();
    closeEditor();
    await refresh();
  });
});
for (const b of document.querySelectorAll<HTMLButtonElement>("[data-action]"))
  b.addEventListener("click", () => {
    void run(async () => {
      await flush();
      note("Working…");
      const result = await api("/api/action", { action: b.dataset.action });
      if (b.dataset.action === "validate") {
        const errors = Array.isArray(result.errors) ? result.errors : [];
        note(
          errors.length
            ? `Validation needs attention: ${errors.join(", ")}. Review the queue and possible duplicates.`
            : "Validation passed. Candidate dataset export is available.",
        );
      } else {
        note(
          `Job status: ${String(result.state)}. Use Refresh status to check progress.`,
        );
      }
      await refresh();
    });
  });
el("ingest").addEventListener("click", () => {
  void run(async () => {
    await flush();
    note("Inspecting inbox PDFs within the configured budget…");
    const result = await api("/api/ingest", {
      group: input("group").value.trim(),
      split: input("split").value,
      reviewer: input("ingest-reviewer").value.trim(),
      pages_per_pdf: Number(input("page-count").value),
      approve_local_use: input("authorize").checked,
    });
    note(
      `Inbox processed (${String(result.state ?? "ready")}). Choose Start / resume rendering to render the selected pages.`,
    );
    await refresh();
  });
});
const dialog = el<HTMLDialogElement>("reset-dialog");
el("reset").addEventListener("click", () => {
  input("reset-confirmation").value = "";
  dialog.showModal();
});
el("reset-cancel").addEventListener("click", () => dialog.close());
el("reset-confirm").addEventListener("click", () => {
  void run(async () => {
    await flush();
    const result = await api("/api/reset", {
      confirmation: input("reset-confirmation").value,
    });
    closeEditor();
    dialog.close();
    await refresh();
    note(
      `Dataset cleared. Previous data is recoverable in ${String(result.archive)}. Inbox PDFs and cumulative budget usage were retained.`,
    );
  });
});
void run(refresh);
el("refresh-archives").addEventListener("click", () => {
  void run(refreshArchives);
});
const archiveDialog = el<HTMLDialogElement>("archive-delete-dialog");
el("archive-delete-cancel").addEventListener("click", () =>
  archiveDialog.close(),
);
el("archive-delete-confirm").addEventListener("click", () => {
  void run(async () => {
    if (!selectedArchive) return;
    const confirm = el<HTMLButtonElement>("archive-delete-confirm");
    confirm.disabled = true;
    try {
      await flush();
      await api("/api/archive-delete", {
        id: selectedArchive.id,
        version: selectedArchive.version,
        confirmation: input("archive-delete-confirmation").value,
      });
      archiveDialog.close();
      selectedArchive = undefined;
      await refresh();
      note(
        "Archive permanently deleted. Its recovery copy is gone; active data, inbox and cumulative usage were retained.",
      );
    } catch (error) {
      el("archive-delete-error").textContent =
        error instanceof Error
          ? error.message
          : "Deletion failed. Refresh archives to check remaining files.";
    } finally {
      confirm.disabled = false;
    }
  });
});
