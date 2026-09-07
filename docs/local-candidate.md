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
Neither model is qualified; saturated synthetic metrics are not real-page
generalization.

## Prepare the ignored manifest

From the repository/worktree containing the run:

The corrected post-hoc audit admits partial pages as no-valid-board cases. No
nonzero-recall threshold met its false-positive limit, so the exact calibrated
threshold is `1.0`. The lower `0.01` value feeds region proposals to the refiner;
it is not an accepted-board threshold. Prepare the hash-bound v2 bundle and its
two immutable ignored provider manifests with both values:

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

Start the usual demo with `pnpm run dev`, expand **Test a trained local
candidate**, select those three files, and choose **Verify and load candidate**.
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

## Dataset dashboard

Register the generated provider manifests, then list their immutable IDs:

```sh
pnpm run dataset proposals register work/candidates/providers/v2-localizer-DET-HASH-REFINER-HASH.json
pnpm run dataset proposals register work/candidates/providers/v2-labeler-CLS-HASH-REFINER-HASH.json
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
pnpm run candidate:verify work/candidates/providers/LOCALIZER.json work/candidates/providers/LABELER.json
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
not qualification evidence, and one independent human pixel review is still
required before any dataset page is accepted.
