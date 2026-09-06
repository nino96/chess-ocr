# Deterministic synthetic dataset

Issue #2's local renderer uses only the three approved SVG designs in
[public bootstrap provenance](../provenance/public-bootstrap.json). It never
fetches assets. The installed FENShot package contains recognition runtime/model
files but no generator source; this renderer is original code, not a broad upstream
asset downloader. No new dependencies or artwork payloads are tracked.

Current launch status: **blocked by failed fidelity controls and an unresolved
[dataset recovery incident](dataset-incident-2026-09-07.md)**. No bulk synthetic
job has been launched. The commands below are implemented interfaces, not a claim
that the current seed gate passes.

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
shear, tables/text as non-board negatives, and explicitly unsupported partials.
It does not yet cover true projective perspective, severe rotation, realistic
ink damage/blur/JPEG, arrows, borderless grids, or broad legal PGN diversity.
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
must be compared at native scale; intended recipes alone cannot validate pixels.
Agent visual inspection supplements executable checks; it is not human truth.

`python/synthetic_training.py` lazily derives 768-pixel grids, 96-pixel RGB tiles,
ImageNet-normalized NCHW tensors and normalized detector targets. This is the
existing candidate preprocessing, not demonstrated parity with a promoted
browser recognizer. Callers must verify the corpus gate and identities; the
loader honestly describes its labels as recipe-derived. No tensors are stored
for every synthetic board by default.

## Resources and reconstruction

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
