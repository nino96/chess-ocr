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
  confirmed per-board action and remains undoable.
- Proposal jobs can use TRAIN pages only. They never enumerate qualification,
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

| Capability | Provider | Runtime |
| --- | --- | --- |
| Localization | `fenshot-localizer-v1` | unchanged FENShot grid detector |
| Localization | `classical-grid-v2` (current; v1 retained for old runs) | deterministic multi-grid evidence |
| Labels | `fenshot-labeler-v1` | unchanged FENShot ONNX tile classifier |

`chess-ocr-onnx-localizer-v1` and `chess-ocr-onnx-labeler-v1` are reserved
validated runtime identifiers for issue #3 integration. They are not dynamically
loaded and are not advertised until an adapter exists. Registering a future
manifest requires a local ignored manifest under `work/` or `artifacts/`, an
artifact in one of those approved roots, a matching model/artifact SHA-256 and a
unique immutable provider ID. The adapter must then be reviewed in source and
added to the fixed runtime registry.

`chess-ocr-dataset-proposal/1` binds each result to run ID, sample ID, sample
revision, source-image SHA-256, both complete manifests and measured runtime.
Boards keep original-image four-corner geometry, 64 image-row-major labels,
per-square 13-class probability evidence where available, uncertainty flags and
unknown orientation. Only results matching the current revision and image hash
are exposed to the editor.

The issue #3 branch began from merge commit
`977d3ab40187203d43a2c485fd1a3adc89e3e174`, which includes issue #2 commit
`8155a3e0e2bd2672267b097cc72e87a0298cc899`. Training remains owned by issue #3.
After its model and exact preprocessing/output contract are merged, integrate it
by adding the corresponding fixed adapter and manifest tests here; do not copy
changes into or edit the concurrent training worktree from issue #2.

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
Deferral never accepts a page. The prior manual baseline—12 decisions in
2,581.248 seconds—has no surviving page-level truth, so it is context only and
cannot serve as a paired accuracy comparison.

## Bounded diagnostic result — 2026-09-07

A four-page TRAIN smoke run on `capablanca-1921-20` through
`capablanca-1921-23` was inspected against the source page pixels. It is
negative evidence for both current localization paths, not a provider promotion
comparison:

| Page | Visible board | FENShot proposal | Classical proposal |
| --- | --- | --- | --- |
| 20 | none | false board over the text body | false text-region grid |
| 21 | one board near the upper page | false lower-page grid; missed the board | two false lower-page grids; missed the board |
| 22 | one board near the lower page | false upper-page grid; missed the board | unsupported; missed the board |
| 23 | none | false text-region grid | three false text-region grids |

The labels are consequently not useful because they are extracted from the
wrong crops. The result does not justify more human time choosing between
FENShot and classical. Keep both as diagnostic/baseline evidence only and make
the issue #3 model-versus-FENShot comparison the next meaningful gate. The
source pages and proposal payloads remain local ignored evidence; no image,
crop, label or model artifact is committed.

Implementation and synthetic tests establish contracts, isolation, stale-result
handling and editing behavior. Per the 2026-09-07 owner supersession, no
classical-versus-FENShot human winner selection is required: FENShot remains the
baseline and classical remains diagnostic. The pending promotion evidence is the
issue #3 model versus FENShot on identical real development pages after its fixed
adapter is merged. The new model is not assumed to win.
