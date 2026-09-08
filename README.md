# Chess OCR

Training and evaluation of printed 2D chess-diagram recognition, with a standalone
TypeScript contract and a small offline browser demo. What this project does and
does not claim is stated once, in
[scope and standing claims](docs/scope-and-claims.md).

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
For an opt-in candidate bundle served through an SSH tunnel, follow
[the remote candidate instructions](docs/local-candidate.md#load-through-a-remote-ssh-tunnel).

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

[Native runtime setup](docs/native-runtime.md) owns the Python 3.12 Linux ARM64
lock, artifact provenance, safe loading and the exact export commands. After
preparing the local artifacts it documents:

```sh
pnpm run test:parity
```

The parity harness checks artifact hashes, independently reproduces preprocessing
in JavaScript, and executes both ONNX graphs in WASM workers. Missing artifacts
or numeric disagreement fail the command.

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
misses and source diversity remain real limitations.

[Reuse review](docs/provenance/reuse.md) records the chess-reader components and
evidence used here. [Artifact review](docs/provenance/artifacts.md) preserves
third-party attribution and exact hashes. Original repository source is licensed
under the MIT License; third-party packages, models and notices retain their
separate terms.

[AGENTS.md](AGENTS.md) owns the contribution rules: commit hygiene, the ignored
payload roots, and what must never be committed. Read it and [PLAN.md](PLAN.md)
before contributing.

## Documentation

| Start here                                     | For                                                                  |
| ---------------------------------------------- | -------------------------------------------------------------------- |
| [operator workflow](docs/operator-workflow.md) | The stage-by-stage narrative and your review checkpoints.            |
| [architecture](docs/architecture.md)           | Pipeline shape, vocabulary, diagnostic modes, subsystem-to-file map. |
| [dataset pipeline](docs/dataset-pipeline.md)   | The local dataset CLI and review dashboard.                          |
| [training runbook](docs/training-runbook.md)   | The GPU training controller, its gates and stop conditions.          |
| [reproducibility](docs/reproducibility.md)     | What a fresh clone can and cannot rebuild.                           |

[docs/index.md](docs/index.md) is the full map of every document and its kind.
