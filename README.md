# Chess OCR

Training and evaluation of printed 2D chess-diagram recognition, with a standalone
TypeScript contract and a small offline browser demo. This is not an ebook reader,
chess engine or generalized recognition claim.

## Run the browser baseline

Use **Node 24.19.0 / pnpm 11.11.0** (see `.node-version` and `package.json`):

```sh
corepack enable
pnpm install --frozen-lockfile
pnpm run setup
pnpm run dev
```

Open the loopback URL Vite prints. Load a PNG/JPEG page, select the inner grid
by dragging or entering pixel bounds, then choose **Read selection**. **Find a
board** runs unchanged FENShot automatic localization; failures remain explicit.
Edit any of the 64 squares and choose orientation before exporting placement.
Uppercase letters are white pieces, lowercase black, and `·` is empty.
Alt+arrow keys move between square editors; native select keys change a piece.
Files remain in memory, and no input is uploaded or added to training.

For offline reload, use a production build:

```sh
pnpm run build
pnpm run preview
```

Wait for **Offline ready**. The verified application, worker, model and runtime
are then cached. Development mode does not install an offline service worker.
The browser cache retains application assets, not your images or edits; download
edited JSON before closing/reloading a session. Runtime is local WASM CPU.

## Checks and evaluation

```sh
pnpm run check                 # strict types, source hygiene and payload protection
pnpm test                      # contract, preprocessing, lifecycle and geometry tests
pnpm run build                 # production demo with integrity-bound offline cache
pnpm exec playwright install chromium
pnpm run test:smoke            # one real Chromium WASM integration check
pnpm run eval                  # synthetic runtime distributions; requires all 3 browsers
```

Routine CI runs source checks/build plus the Chromium smoke for affected browser
paths. It does not run an extensive browser matrix on training changes.
For an affected browser/runtime integration gate, explicitly run:

```sh
pnpm exec playwright install chromium firefox webkit
pnpm run test:browser
```

This includes offline reload with the origin server shut down, asset corruption
and retry, unsupported input, selection, keyboard editing and cancellation.
These checks are browser-engine evidence on the available host, not physical
macOS/Windows/iPad qualification. CI configuration is not proof a remote job ran.

## Native models and parity

[Native runtime setup](docs/native-runtime.md) provides the Python 3.12 Linux
ARM64 lock, artifact-specific provenance, safe loading and exact export commands.
Native COCO YOLOX-Nano and ImageNet MobileNetV3 heads stay unchanged. They are
runtime/export probes, not chess-trained alternatives to FENShot.

After preparing their documented local artifacts:

```sh
pnpm run test:parity
```

The parity harness checks artifact hashes, independently reproduces preprocessing
in JavaScript, and executes both ONNX graphs in WASM workers. Missing artifacts
or numeric disagreement fail the command. `pnpm run eval` measures only original
procedural synthetic inputs and writes raw timing evidence to ignored
`work/evidence/`; it is not an accuracy benchmark. Dataset qualification is #2/#3.

## Library contract

The source entry is [src/index.ts](src/index.ts). Version `chess-ocr/1` validates
bounded raster dimensions/selection, clockwise original-image corners, 64
image-relative labels and 13-class probabilities, orientation evidence/unknown,
warnings, model/preprocessing identity and timings. Image coordinates refer to
the browser-decoded raster (including its JPEG orientation handling).

`RecognitionClient` owns a single reusable worker; cancellation/timeouts terminate
it, and subsequent requests recreate it. `Editor` preserves corrections across
retries and late/out-of-order results. A changed grid cannot relocate corrections
silently. The model's original probability evidence remains alongside explicit
user corrections. Side-to-move, castling, en passant and counters are not inferred.

## Status and boundaries

Issue [#1](https://github.com/nino96/chess-ocr/issues/1) implements the runnable
baseline; see [evidence and remaining gates](docs/issue-1-evidence.md).
[#2](https://github.com/nino96/chess-ocr/issues/2) owns real source-diverse data,
[#3](https://github.com/nino96/chess-ocr/issues/3) owns qualified offline training,
and [#4](https://github.com/nino96/chess-ocr/issues/4) owns optional server work.

FENShot 0.1.4 comes directly from npm. Its core and model are unchanged; confidence
is uncalibrated, automatic detection returns at most one axis-aligned board, and
manual selection does not demonstrate automatic localization. Unsupported skew,
misses and source diversity remain real limitations. The native runtime probes
have no demonstrated chess accuracy advantage.

[Reuse review](docs/reuse.md) records the chess-reader components and evidence
used here. [Artifact review](docs/artifacts.md) preserves third-party attribution
and exact hashes. Original repository source is licensed under the MIT License;
third-party packages, models and notices retain their separate terms.

Read [AGENTS.md](AGENTS.md) and [PLAN.md](PLAN.md) before contributing. Keep
originals, datasets, weights, exports and generated runs under ignored
`data/`, `cache/`, `work/` or `artifacts/`. Never commit payloads, private positions
or credentials. No paid service, telemetry, runtime CDN or server upload is part
of this baseline.

## Dataset collection (issue #2)

The current [synthetic-first kickoff](docs/dataset-kickoff.md) starts asset
collection and an audited synthetic seed alongside real-source acquisition.
It replaces bulk manual labeling as the collection strategy. Generation now has
an executable fidelity gate; annotation assistance remains pending its comparison
and workflow checks.
Same-split duplicate candidates are retained with an export audit, not a required
human task. Cross-split leakage remains blocking.

Start with the [operator workflow](docs/operator-workflow.md) for the sequence
from feasibility collection to larger-budget approval and planned model training,
including your review/approval checkpoints and the agent's responsibilities.

The [local dataset pipeline](docs/dataset-pipeline.md) accepts PDFs placed in
ignored `work/dataset/inbox/`, uses explicit resource limits, and runs resumable
background acquisition/rendering with `pnpm run dataset start`, `status` and
`stop`. For review, start the primary local dashboard with:

```sh
pnpm run dataset serve --port 8766
```

It binds to loopback only. In VS Code use **Ports** → **Forward a Port** → `8766`
→ **Open Browser**. The dashboard keeps server-side drafts, presents the queue and
page thumbnails, and accepts a page after one human pixel review; model or agent
proposals cannot accept it. `ingest` records explicit local-use authorization and
a conservative source/artwork group. Validation and hashed train/dev exports are
implemented; no real collection or recognition qualification is claimed.
Use **Archives** to see dated archive sizes and permanently delete an old recovery
copy after confirmation. Active data, inbox PDFs and cumulative usage are retained.
Reviewed public-source provenance is tracked for
[reproducibility](docs/reproducibility.md). Private source details, operational
history, downloaded assets and generated datasets remain local and ignored.

The [synthetic renderer and job controller](docs/synthetic-dataset.md) implement
deterministic recipes, independent fidelity gating, bounded generation, lazy
training inputs and resumable `pnpm run synthetic start/status/stop` commands.
Bulk use requires current fidelity evidence. Coverage and recognition limitations
remain explicit. The [public dataset screen](docs/public-dataset-review.md)
distinguishes physical-board datasets from the printed-page collection.

Dataset work suffered a
[test-isolation and recovery incident](docs/dataset-incident-2026-09-07.md).
The reviewed seed was lost; the owner waived recovery and authorized continuation.
Rebuilt real labels are unverified proposals. See the synthetic status command and
current fidelity evidence below the pipeline documentation; no dataset-readiness
or recognition gain is claimed.

## First native training bootstrap (issue #3)

The [frozen training decision](docs/issue-3-training-plan.md) authorizes one
synthetic-only MobileNetV3 classifier plus YOLOX-Nano detector schedule. It is
diagnostic and may assist annotation; it is not real-page qualification and does
not replace the shipped FENShot baseline.

Create the issue #3 worktree from merged `origin/main`, commit reviewed training
code, then initialize it with explicit read-only roots for the completed corpus
and admitted native artifacts:

```sh
pnpm run training -- init \
  --dataset-root /absolute/path/to/chess-ocr/work/dataset/synthetic \
  --native-root /absolute/path/to/chess-ocr \
  --overlay-root /absolute/path/to/ignored/training-overlay
pnpm run training -- start
pnpm run training -- status
pnpm run training -- stop
```

Initialization rejects dirty code, stale corpus/model hashes, an unavailable
pinned container, unsafe paths and insufficient free space. Before allocating a
GPU, `start` runs the frozen container's CPU-only input/dependency/output validation;
failure is retained without a GPU charge. It returns after the bounded background
supervisor is live. Checkpoints, curves, exports and raw logs remain under ignored
`work/training/`; do not publish them without the separate artifact rights review
required by this repository.

Commands without `--run-root` always address the default
`work/training/synthetic-bootstrap-v1` directory; they do not discover the
newest repaired attempt. If initialization uses a repaired directory such as
`work/training/synthetic-bootstrap-v1-repair-3`, pass that same `--run-root` to
every later `status`, `start`, and `stop` command.

Status reports the current run and a budget summary by default. GPU budget is
measured in GPU-seconds: one second while a scheduled training container owns
the GPU. It includes preflight and optimization segments, and a retry consumes
the same frozen reservation. The summary shows total capacity, consumed time,
remaining time, and each segment's allocation. Add `--history` when the full
attempt ledger is needed for audit or diagnosis.

If a completed classifier checkpoint needs to be reused after an export-only
failure, initialize a fresh detector-only run without `--prior-run`. This gives
the new run its own frozen reservation while retaining the old run as evidence:

```sh
pnpm run training -- init --detector-only \
  --classifier-checkpoint /absolute/path/to/repair-3/classifier/checkpoint-010000.pt \
  --dataset-root /absolute/path/to/chess-ocr/work/dataset/synthetic \
  --native-root /absolute/path/to/chess-ocr \
  --overlay-root /absolute/path/to/ignored/training-overlay \
  --run-root work/training/synthetic-bootstrap-v1-detector-1
pnpm run training -- start --run-root work/training/synthetic-bootstrap-v1-detector-1
```

The controller verifies the checkpoint schema, completed schedule, selected-candidate
evidence and hash, exports it with native-to-ONNX parity before detector optimization,
marks the classifier complete without classifier GPU charges, and then runs the
detector schedule. It reuses retained development evidence instead of repeating a
full CPU evaluation. Choose the new reservation in the reviewed
recipe before initialization; omitting `--prior-run` is what makes it fresh.
