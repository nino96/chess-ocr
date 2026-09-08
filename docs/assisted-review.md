# Assisted dataset review

This is the issue #2 design and operating contract for machine-assisted review.
It implements proposal tooling; it does not make a model prediction ground truth,
complete the source-diverse dataset, or qualify recognition.

## Recorded decisions — 2026-09-07

- Board localization and 64-square labeling are separate capabilities. The
  dashboard selects one registered provider for each, so a localizer can be
  compared or replaced without coupling it to a label model.
- Provider selection uses immutable, validated
  `chess-ocr-provider-manifest/1` records. A manifest selects a reviewed runtime
  identifier and binds its implementation/model SHA-256, preprocessing identity
  and resource limits. It cannot name executable plugin code, a URL, or a remote
  service.
- The default pair is the unchanged FENShot 0.1.4 localizer and tile classifier.
  `classical-grid-v1` is the second localization candidate for the bounded
  comparison. FENShot localization is bound to the installed detector/preprocessor
  code; labeling is bound to the exact ONNX bytes.
- A valid proposal may prefill only a new, untouched draft. Loading or switching
  a proposal after human interaction preserves the current draft and displays
  the candidate for comparison. Replacing an edited board is an explicit,
  confirmed per-board action, remains undoable, and clears the human-review
  declarations and accumulated review time until the changed draft is inspected.
- Proposal jobs can use TRAIN pages, or prospectively assigned development pages
  for the bounded promotion comparison. They never enumerate qualification,
  change annotations, accept pages, infer orientation, legalize a position, or
  invent FEN fields. One human pixel review independent of the proposal remains
  the only acceptance path; a second reviewer is optional.
- The classical provider receives only a small nonqualification smoke screen; it
  is a cheap diagnostic/fallback, not a promotion contender. The meaningful
  identical-input human comparison is the issue #3 candidate versus unchanged
  FENShot, within the existing ceiling of 100 nonqualification pages/two CPU hours.
  Promotion uses human time, missed/false boards, corner displacement and piece
  corrections—not model confidence or successful examples alone.

## Provider and result contracts

Built-in provider IDs are:

| Capability   | Provider               | Runtime                                |
| ------------ | ---------------------- | -------------------------------------- |
| Localization | `fenshot-localizer-v1` | unchanged FENShot grid detector        |
| Localization | `classical-grid-v1`    | deterministic multi-grid evidence      |
| Labels       | `fenshot-labeler-v1`   | unchanged FENShot ONNX tile classifier |

`chess-ocr-onnx-localizer-v1` and `chess-ocr-onnx-labeler-v1` are executable fixed
adapters. They accept only explicitly registered manifests under ignored `work/`
or `artifacts/`, separately verify detector and classifier bytes, bind tensor
contracts, thresholds, preprocessing, refinement identity and limits, and create
two WASM sessions in the detached runner. They are not dynamic module loaders.
Changing an artifact or the refiner requires a new immutable provider ID.

`chess-ocr-dataset-proposal/1` binds each result to run ID, sample ID, sample
revision, source-image SHA-256, both complete manifests and measured runtime.
Boards keep original-image four-corner geometry, 64 image-row-major labels,
per-square 13-class probability evidence where available, uncertainty flags and
unknown orientation. Only results matching the current revision and image hash
are exposed to the editor.

Training remains owned by issue #3. These adapters expose only draft proposals;
they do not alter annotations or satisfy the real-development gate.

## Bounded job lifecycle

List providers, start a detached run, inspect it, request a bounded stop, and
resume a stopped run with:

```sh
pnpm run dataset proposals providers
pnpm run dataset proposals start --localizer fenshot-localizer-v1 --labeler fenshot-labeler-v1 --scope train-pending --max-pages 20
pnpm run dataset proposals status
pnpm run dataset proposals stop
pnpm run dataset proposals resume RUN_ID
```

Use `--after-repair` only after diagnosing a `needs-repair` run. Every attempt
reserves 45 CPU seconds in the existing dataset ledger, including failed attempts.
A run is capped at 100 pages and two hours. Work is persisted after each page;
inference runs outside the dataset writer lock, with short transactions for
attempt/result records. Status is resumable local state, so no AI turn needs to
poll it. The dashboard exposes the same start/status/stop surface.

The start command also accepts `dev-pending`, `dev-all`, or `accepted-dev` after
development sources are prospectively assigned. No command accepts qualification.

Proposal inputs, intermediate RGBA, results, manifests for local models and review
metrics remain under ignored dataset storage/SQLite. Reset refuses a live proposal
job, archives proposal staging with the dataset, clears run/result/review-assistance
records, and retains registered immutable provider manifests.

## Review workspace and evidence

The focused dashboard filters pending, proposed, deferred, ambiguous/partial,
accepted, or all pages. In the editor a reviewer can zoom, drag or keyboard-adjust
corners, use the piece palette, move among squares with the keyboard, undo/redo,
inspect uncertainty highlighting, switch proposal pairs without losing edits,
mark corrections, and defer a page with a bounded reason. An optional five-minute
session is a timebox only; it does not auto-submit or weaken complete-page review.

A label-only **re-read after changing the grid** is intentionally not exposed in
this increment. FENShot's unchanged tile preprocessor accepts an axis-aligned box,
not an arbitrary reviewed quadrilateral; silently applying it after a perspective
corner edit would misrepresent the pixels being classified. Add that action only
with a reviewed rectification-capable label adapter and the same stale-request,
budget and edit-preservation tests. Reviewers can still switch complete stored
proposals and edit labels manually.

The accepted review stores no probability payload in the human annotation.
Separate local assistance evidence records active time, the proposal run,
one-to-one board matches at polygon IoU 0.5, missed and false boards, mean corner
movement in pixels and square widths, piece corrections and touched controls.
Deferral never accepts a page. There is no manual baseline to compare against:
the pages behind the earlier timing were destroyed in
[the test-isolation incident](archive/dataset-incident-2026-09-07.md), so no
paired accuracy comparison can be reconstructed from it.

The one bounded diagnostic run of both current localization paths is recorded in
[the 2026-09-07 assisted-review smoke record](archive/assisted-review-smoke-2026-09-07.md).
Its result was negative for both paths; neither is promoted on it.
