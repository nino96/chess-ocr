# Deterministic synthetic dataset

Issue #2's local renderer uses only the three approved SVG designs in
[public bootstrap provenance](../provenance/public-bootstrap.json). It never
fetches assets. The installed FENShot package contains recognition runtime/model
files but no generator source; this renderer is original code, not a broad upstream
asset downloader. No new dependencies or artwork payloads are tracked.

Bulk launch requires the current local fidelity report to pass. The owner waived
recovery after the [dataset incident](dataset-incident-2026-09-07.md); lost human
labels are not part of this corpus. `pnpm run synthetic status` reports actual
persisted job state, not an inferred completion or recognition result.

## Commands

Use the existing locked dataset Python environment and pinned pnpm setup. From
the repository directory:

```sh
node scripts/synthetic-render.mjs assets
node scripts/synthetic-fidelity.mjs
work/dataset-venv/bin/python python/synthetic_fidelity.py
pnpm run synthetic init
pnpm run synthetic start
pnpm run synthetic status
pnpm run synthetic stop
pnpm run synthetic start
```

`init` requires complete, current independent fidelity evidence. It freezes the
recipe, code, lock, Chromium executable, font selection/configuration, and asset
provenance identities. It reserves compute and storage in the existing project
ledger without refunding prior attempts. Do not initialize again to retry. `start`
resumes complete verified batches and confirms worker startup; `stop` takes effect
at the current bounded batch boundary. `run` is the foreground diagnostic command
and honors an existing stop marker. Runtime/code changes reject the frozen job.

Operational state is in ignored `work/dataset/synthetic/`: `state.json`, immutable
batch records, `frozen.json`, `recipes.json`, `coverage.json`, and PNGs. Attempt
charges are also recorded durably in the dataset SQLite database so rolling back
a state snapshot cannot refund compute. Keep the database with the corpus.
The generation job does not make rights decisions, download new sources, run
training, or turn model proposals into human acceptance.

## Scope and fidelity

[Seed v1 recipe](../recipes/synthetic-seed-v1.json) requests 12,000 pages, bounded
by 20,000 board equivalents, including unsupported partial stress examples.
Training boards, negatives, and quarantined partials are counted separately.
The renderer includes grayscale/colored/hatch squares, sparse/medium/dense/empty
teaching positions, three original legal opening states, both orientations,
multiple board sizes/layouts, captions/coordinates, modest rotation and affine
shear, bounded true projective whole-page perspective, tables/text as non-board
negatives, and explicitly unsupported partials. Conservative print-like effects
include paper noise, ink fading, uneven illumination and mild separable blur.
Full-strength three-tap blur failed the smallest-board class-preservation audit;
it is excluded from usable seed labels. The admitted candidate uses a fixed 25%
blurred/75% unblurred blend and must independently pass the same audit.
These procedural effects are not empirically calibrated scans. It does not yet
cover severe rotation, ink dropout/bleed-through/JPEG, arrows, borderless grids,
or broad legal PGN diversity.
These remain coverage gaps; this seed is not described as “all kinds of boards.”

Synthetic labels use image-relative row-major `.PNBRQKpnbrqk`, never an upstream
class order or inferred side-to-move. Negative and partial records have no usable
board targets; partial geometry is retained separately in `unsupported_boards`.
Random teaching positions are not labeled legal. Opening/position repetition and
joint source/condition exposure are reported explicitly. Repetition is not new
independent evidence and must be capped in a later frozen training sampler.

The fidelity harness compares production canvas pages against independently laid
out source-glyph controls, checks every design/piece on both square backgrounds
and all 64 positions, verifies actual candidate tensor ordering, and reconstructs
a bounded subset exactly. Failed reports remain local. Small-board rasterization
must be compared at native scale with the same whole-page affine rasterization
and rectification. The independent SVG control derives its transform from recorded
corners rather than invoking production canvas drawing. Intended recipes alone
cannot validate pixels. Calibration/mixed MAE ceilings remain 3/18 respectively.
An independent Python pixel-effect implementation precedes reference perspective
warping; it does not call the production JavaScript function. Twelve additional
150-pixel audit boards cover each design/effect and all classes on both square
backgrounds. Strict positive nearest-class margins must hold both for intended
effects versus clean controls and production pixels versus effected controls.
This is synthetic label-preservation evidence, not real recognition accuracy.
Agent visual inspection supplements executable checks; it is not human truth.

`python/synthetic_training.py` lazily derives 768-pixel grids, 96-pixel RGB tiles,
ImageNet-normalized NCHW tensors and normalized detector targets. This is the
existing candidate preprocessing, not demonstrated parity with a promoted
browser recognizer. Callers must verify the corpus gate and identities; the
loader honestly describes its labels as recipe-derived. No tensors are stored
for every synthetic board by default.

## Resources and reconstruction

Measured prelaunch validation (2026-09-07): 91 pages, 106 independent controls,
6,784 tile comparisons, 2,496 clean calibration identity checks, and 768 effect
audit squares (1,536 strict class-margin comparisons) passed. Maximum calibration
MAE was 0.4775 under 3; maximum other-control MAE was 17.1213 under 18. Minimum
effect class margin was 7,552 summed byte-error units. These are synthetic
renderer checks, not recognition accuracy on real documents.

Renderer SHA-256: `d16ee83978e8b886c28e9a3b433b13760cd8d08f56c38c109eb3922d1f499827`.
Raw local report SHA-256: `470d45c903845828f1ffbb47f9351b3c6236a5dd5b23b4b94d7a51bbcd7b117c`.
The report and failed predecessors remain under ignored
`work/dataset/bootstrap/fidelity/`; no generated image is published.
One 64-page production-supervisor batch passed in 26.88 seconds on this host.
All 28 JavaScript tests, 59 Python tests, source/privacy/provenance checks and
the offline build passed. Browser inference is unchanged; a new end-to-end
recognition evaluation and training/browser parity remain unrun.

Each renderer batch is confined to one CPU, a 120-second process-group deadline,
bounded decoded page dimensions, a 4 GiB monitored aggregate RSS ceiling, and a
16 MiB individual output-file limit. Both concurrent dataset workers account for
reserved storage and preserve the owner's 30% filesystem-free-space floor.
The independent timeout supervisor also bounds a renderer if its Python parent
is interrupted. Local budget increases require a recorded reason under the
[owner's discretionary allocation](budget.md), not a silent new sweep.

A fresh checkout can reconstruct recipes and, given exact allowed SVG bytes and
the captured runtime/font environment, a deterministic synthetic corpus. Exact
PNG identity across different Chromium/font/platform environments is not claimed.
Public source metadata is sufficient to identify inputs, not to publish their
payloads or reproduce unshared human annotation history.

Checks:

```sh
node --experimental-strip-types --test tests/synthetic-render.test.ts
work/dataset-venv/bin/python -m unittest python/test_synthetic_job.py
```

Dataset readiness still requires source-diverse real TRAIN and separately reserved
development/qualification groups, human pixel confirmation of real truth,
cross-split leakage investigation, usable assisted review, measured baseline
proposals, and the remaining training-interface/qualification gates. A synthetic
volume or passing tooling checks is not recognition improvement.
