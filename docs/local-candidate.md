# Test a local trained candidate

The browser demo and dataset dashboard keep their existing default behavior.
FENShot remains the browser default. V2 appears in the dataset provider registry
only after its ignored immutable manifests are explicitly registered.

The training checkpoints are native `.pt` recovery artifacts. The UIs load the
corresponding ONNX exports, bound by a small manifest that records exact bytes,
SHA-256 values, tensor names, label order and detector thresholds. No model,
checkpoint or generated manifest is committed or published.

Two detectors are retained. V1 is `legacy-bgr-div255-v1`, synthetic-only and
uncalibrated; preserve its old diagnostic bundle unchanged. V2 is
`yolox-rgb-imagenet-v2`, completed all 9,000 updates, and remains
synthetic-development-only. The executable browser/provider integration uses
schema 3 because it binds the shared refiner, tensor shapes, separate proposal
and acceptance thresholds, and resource limits. Older bundles are rejected.
Neither model is qualified: see
[scope and standing claims](scope-and-claims.md#synthetic-evidence-is-not-recognition-accuracy).

## Prepare the ignored manifest

From the repository/worktree containing the run:

The bundle carries two distinct thresholds, and what each one means is explained
in the training runbook under
[proposal threshold versus acceptance threshold](training-runbook.md#proposal-threshold-versus-acceptance-threshold).
For this run the calibrated acceptance threshold is `1.0` and the proposal
threshold is `0.01`. Prepare the hash-bound v2 bundle and its two immutable
ignored provider manifests with both values:

```sh
pnpm run candidate -- prepare \
  --run-root work/training/synthetic-bootstrap-v2-detector \
  --output work/candidates/synthetic-bootstrap-v2-detector-v3.json \
  --provider-output-directory work/candidates/providers \
  --preprocessing yolox-rgb-imagenet-v2 \
  --proposal-score-threshold 0.01 \
  --calibrated-score-threshold 1.0
```

Both thresholds remain labeled `synthetic-development-only`. Preparing the
bundle copies no model bytes and changes no retained run artifact. Automatic v2
uses low-threshold regions only as inputs to the deterministic nine-line refiner;
unrefined regions are never returned as accepted boards.

For this run the three local inputs are:

```text
work/candidates/synthetic-bootstrap-v2-detector-v3.json
work/training/synthetic-bootstrap-v2-detector/classifier/selected.onnx
work/training/synthetic-bootstrap-v2-detector/detector/selected.onnx
```

## Browser test UI

Start the demo as described in
[run the browser baseline](../README.md#run-the-browser-baseline), expand **Test a
trained local candidate**, select those three files, and choose **Verify and load
candidate**.
The app checks model sizes and hashes before creating the local WASM worker. Use
**Recognition backend** to switch back to FENShot without reloading the page.

Candidate results preserve source-image coordinates and row order. Orientation
remains unknown. All 64 squares stay visibly uncertain because synthetic
development confidence is not real-page calibration. Automatic results require
nine-line refinement; low-evidence and partial grids are rejected. The paired
diagnostic accepts at most 20 inputs and 100 MiB compressed, keeps only the active
image decoded, hides results until the reference is saved, and supports automatic
and manual four-corner modes. Its private export is sensitive local evidence;
only the aggregate summary is publication-safe.

### Load through a remote SSH tunnel

When the repository and candidate artifacts are on an SSH host but the demo is
open in a laptop browser, start the loopback development server with the fixed v2
bundle explicitly enabled:

```sh
CHESS_OCR_REMOTE_CANDIDATE=1 pnpm run dev
```

The protected endpoint requires a Linux SSH host so it can bind every opened file
descriptor to its approved managed root. On another host, use the file-picker
fallback.

Forward the printed loopback port through SSH and open that forwarded loopback
URL. Expand **Test a trained local candidate** and choose **Load configured remote
candidate**. The server configuration exposes only the three paths listed under
[Prepare the ignored manifest](#prepare-the-ignored-manifest). It fails startup
if a path leaves its managed `work/candidates` or `work/training` root, contains a
symbolic link, exceeds its role bound, disagrees with the schema or hashes, or has
an incompatible ONNX tensor contract.

The endpoint exists only for this opt-in development-server session. It binds to
loopback, accepts only fixed manifest/classifier/detector roles from the active
same-origin session, and returns `Cache-Control: no-store`. The browser requests
the bytes only after the button is pressed, repeats its existing manifest and
hash checks, and keeps the verified bundle in memory for that page session. A
reload returns to FENShot and does not reload the bundle automatically. The bytes
travel through the SSH tunnel for local browser WASM inference; this is neither
remote inference nor model publication. The three file pickers remain the
fallback.

## Dataset dashboard

Register the generated provider manifests, then list their immutable IDs:

`candidate prepare` names each manifest after the short model and refiner hashes,
so the exact filenames differ per run. List the directory and register what is
there:

```sh
ls work/candidates/providers/     # e.g. v2-localizer-2bdc13f5-2c5cff3b.json
pnpm run dataset proposals register work/candidates/providers/v2-localizer-<det>-<refiner>.json
pnpm run dataset proposals register work/candidates/providers/v2-labeler-<cls>-<refiner>.json
pnpm run dataset proposals providers
```

Use the dataset environment setup from [the pipeline guide](dataset-pipeline.md).
The detached runner verifies both model hashes and the refiner implementation,
creates separate bounded WASM sessions, and releases both on success or failure.
Use `dev-pending`, `dev-all`, or `accepted-dev` only after a held-out development
membership exists. Qualification is never a proposal scope.

Verify the actual retained artifacts in Node WASM and all production browser
engines with:

```sh
pnpm run candidate:verify work/candidates/providers/v2-localizer-<det>-<refiner>.json \
  work/candidates/providers/v2-labeler-<cls>-<refiner>.json
pnpm run build
pnpm run candidate:verify:browser work/candidates/synthetic-bootstrap-v2-detector-v3.json \
  work/training/synthetic-bootstrap-v2-detector/classifier/selected.onnx \
  work/training/synthetic-bootstrap-v2-detector/detector/selected.onnx
```

When a page is open, **Generate model proposal** runs both models locally and
offers geometry and labels as an editable draft. It never marks a page negative,
never submits a review and never checks the human or complete-page declarations.
Replacing an existing draft requires confirmation and clears both declarations
and accumulated review time, so the changed proposal needs a fresh inspection.
The same revision/image-hash checks used by draft saving run before and after
inference. The editor is inert while the bounded local proposal request runs;
generation checks discard a result if its draft or page nevertheless changes.

These interfaces are diagnostic and annotation-assistance tools. Their output is
not qualification evidence, and the
[one-human-pixel-review rule](scope-and-claims.md#the-one-human-pixel-review-rule)
still governs acceptance of any dataset page.
