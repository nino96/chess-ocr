# Test a local trained candidate

The browser demo and dataset dashboard keep their existing default behavior.
FENShot remains the browser default, and the dataset dashboard has no model
unless one is explicitly supplied when its loopback server starts.

The training checkpoints are native `.pt` recovery artifacts. The UIs load the
corresponding ONNX exports, bound by a small manifest that records exact bytes,
SHA-256 values, tensor names, label order and detector thresholds. No model,
checkpoint or generated manifest is committed or published.

Two detectors are retained. V1 is `legacy-bgr-div255-v1`, synthetic-only and
uncalibrated. Its original schema-1 bundle did not identify preprocessing and was
therefore ambiguous. Corrected loaders reject schema 1 with a regeneration
instruction; its ignored local bundle was regenerated under schema 2 with the
legacy identifier for diagnostic use. V2 is `yolox-rgb-imagenet-v2`, completed
all 9,000 updates, and remains synthetic-development-only. Neither is qualified;
never relabel V1 as corrected or treat V2's saturated synthetic metrics as real
generalization.

## Prepare the ignored manifest

From the repository/worktree containing the run:

```sh
pnpm run candidate -- prepare \
  --run-root work/training/synthetic-bootstrap-v1-detector-3 \
  --output work/candidates/synthetic-bootstrap-v1-detector-3.json \
  --preprocessing legacy-bgr-div255-v1 \
  --score-threshold 0.3
```

The `0.3` threshold is an explicit exploratory value, not a calibrated real-page
operating point. Creating another manifest with a different threshold does not
retrain or modify either model. A normal completed run must have both export
manifests. The command also has one bounded recovery path for a complete detector
schedule rejected only by the former `0.0001` export cutoff: the run must retain
the terminal checkpoint, full curve, matching failed state/log and diagnosed
drift at or below `0.001`.

The corrected post-hoc audit admits partial pages as no-valid-board cases. No
nonzero-recall threshold met its false-positive limit, so the exact calibrated
threshold is `1.0`. Prepare the hash-bound v2 bundle with that value:

```sh
pnpm run candidate -- prepare \
  --run-root work/training/synthetic-bootstrap-v2-detector \
  --output work/candidates/synthetic-bootstrap-v2-detector.json \
  --preprocessing yolox-rgb-imagenet-v2 \
  --score-threshold 1.0
```

This threshold is still labeled `synthetic-development-only`. Preparing the
bundle copies no model bytes and changes no retained run artifact. It causes
automatic v2 detection to abstain; manual-grid classifier diagnostics still run.
A later full-page diagnostic may use a separately named low proposal threshold
only to feed the required nine-line grid refiner. It must retain `1.0` as the
synthetic calibrated acceptance threshold and cannot present region proposals as
accepted boards.

For this run the three local inputs are:

```text
work/candidates/synthetic-bootstrap-v1-detector-3.json
work/training/synthetic-bootstrap-v1-detector-3/classifier/selected.onnx
work/training/synthetic-bootstrap-v1-detector-3/detector/selected.onnx
```

## Browser test UI

Start the usual demo with `pnpm run dev`, expand **Test a trained local
candidate**, select those three files, and choose **Verify and load candidate**.
The app checks model sizes and hashes before creating the local WASM worker. Use
**Recognition backend** to switch back to FENShot without reloading the page.

Candidate results preserve source-image coordinates and row order. Orientation
remains unknown. All 64 squares stay visibly uncertain because synthetic
development confidence is not real-page calibration. Detector rectangles are
axis-aligned and are not yet inner-grid corner refinement.

## Dataset dashboard

Start the loopback dashboard with the optional candidate arguments:

```sh
pnpm run dataset serve --port 8766 \
  --candidate-manifest work/candidates/synthetic-bootstrap-v1-detector-3.json \
  --candidate-classifier work/training/synthetic-bootstrap-v1-detector-3/classifier/selected.onnx \
  --candidate-detector work/training/synthetic-bootstrap-v1-detector-3/detector/selected.onnx \
  --candidate-overlay-root work/training-overlay
```

Use the dataset environment setup from [the pipeline guide](dataset-pipeline.md).
The optional inference adapter imports NumPy and ONNX Runtime from the verified
training overlay; the dashboard itself still uses its small dataset environment.

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
