# Test a local trained candidate

The browser demo and dataset dashboard keep their existing default behavior.
FENShot remains the browser default, and the dataset dashboard has no model
unless one is explicitly supplied when its loopback server starts.

The training checkpoints are native `.pt` recovery artifacts. The UIs load the
corresponding ONNX exports, bound by a small manifest that records exact bytes,
SHA-256 values, tensor names, label order and detector thresholds. No model,
checkpoint or generated manifest is committed or published.

The currently retained detector is `legacy-bgr-div255-v1`, synthetic-only and
uncalibrated. Its schema-1 bundle does not identify preprocessing and is therefore
ambiguous. Corrected loaders reject it with a regeneration instruction; regenerate
the ignored bundle under schema 2 with the legacy identifier for diagnostic use.
Do not relabel the existing weights as corrected or qualified.

## Prepare the ignored manifest

From the repository/worktree containing the run:

```sh
pnpm run candidate -- prepare \
  --run-root work/training/synthetic-bootstrap-v1-detector-3 \
  --output work/candidates/synthetic-bootstrap-v1-detector-3.json \
  --score-threshold 0.3
```

The `0.3` threshold is an explicit exploratory value, not a calibrated real-page
operating point. Creating another manifest with a different threshold does not
retrain or modify either model. A normal completed run must have both export
manifests. The command also has one bounded recovery path for a complete detector
schedule rejected only by the former `0.0001` export cutoff: the run must retain
the terminal checkpoint, full curve, matching failed state/log and diagnosed
drift at or below `0.001`.

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
