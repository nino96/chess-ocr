# Understanding the v2 evaluation work

This guide explains the work delivered by PRs #10 and #11 in plain language. It
is an orientation document, not a claim that v2 is ready to replace FENShot.

## The short version

Chess OCR needs to turn a page image into the 64 visible squares of a printed
chess diagram. The repository already shipped FENShot as its baseline. A new v2
candidate was trained on synthetic pages, but synthetic accuracy alone cannot
show that it works on real books or screenshots.

The recent work did two things:

1. PR #10 preserved and corrected the audit of the completed v2 training run.
2. PR #11 made that frozen candidate testable beside unchanged FENShot in the
   browser and dataset dashboard.

No new model was trained. No private screenshot or model binary was committed.
Qualification data has not been opened.

## Vocabulary

| Term          | Meaning                                                                                                         |
| ------------- | --------------------------------------------------------------------------------------------------------------- |
| FENShot       | The unchanged 0.1.4 baseline already used by the browser demo.                                                  |
| v2 candidate  | The completed YOLOX-Nano detector plus retained MobileNetV3 classifier. It is still synthetic-development-only. |
| Detector      | Finds possible board regions on a full page.                                                                    |
| Grid refiner  | Looks inside a possible region for two line families forming a regular 8-by-8 board. It rejects weak evidence.  |
| Rectification | Warps four skewed inner-grid corners into a square 768-by-768 image.                                            |
| Classifier    | Reads the 64 rectified 96-by-96 squares as empty or one of 12 piece classes.                                    |
| Reference     | The human-entered expected result used to score both models.                                                    |
| Development   | Real, source-held-out data used to decide whether the candidate is promising.                                   |
| Qualification | A later untouched evaluation set. It must not influence development decisions.                                  |

## What PR #10 changed

PR #10 did not change the learned weights. It corrected how the existing
synthetic run is described and evaluated:

- preserved the completed v2 run and selected step 9,000;
- clamped detector average precision to the valid range from zero to one;
- treated partial and unsupported pages as pages with no valid complete board;
- reported false detections on those pages separately;
- added post-hoc evaluation that cannot train or alter checkpoints;
- reconciled GPU accounting without counting inherited attempts twice;
- recalculated the synthetic calibration result.

The corrected synthetic acceptance threshold is `1.0`: no threshold with useful
recall also met the synthetic false-positive constraint. This is an honest
warning, not evidence that the model is useless. PR #11 therefore uses `0.01`
only to produce possible detector regions for grid refinement. A detector box is
never silently treated as a trustworthy board.

## What PR #11 adds

PR #11 turns the frozen local ONNX files into a bounded evaluation candidate:

```text
decoded page pixels
        |
        v
YOLOX detector -- low-score region proposals
        |
        v
nine-line grid refiner -- reject if grid evidence is weak
        |
        v
four inner-grid corners
        |
        v
768 x 768 perspective rectification
        |
        v
64 normalized 96 x 96 square tensors
        |
        v
MobileNetV3 classifier -- 13 probabilities per square
        |
        v
editable result with original-image geometry
```

The preprocessing, detector decoding, grid geometry, rectification, tiling, and
probability conversion are shared between the browser worker and the detached
dataset proposal runner. This avoids evaluating a materially different pipeline
from the one shown in the browser.

Every candidate manifest binds the exact detector and classifier hashes, tensor
contracts, preprocessing identity, refiner source hash, thresholds, and resource
limits. Changed bytes or changed refinement code require a new immutable
manifest. Model files and generated manifests remain ignored local artifacts.

## Why there are two diagnostic modes

The paired diagnostic runs v2 and FENShot against the same decoded input and
scores both automatically against one saved human reference.

### Automatic full-page mode

This exercises the complete pipeline: localization, refinement, rectification,
and square classification. Use it with full screenshots or deliberately loose
page selections.

### Manual four-corner mode

The reviewer supplies the exact inner-grid corners. Both models receive the same
rectified grid, so localization is removed from the comparison. This isolates
classifier and domain-shift errors.

Together the modes provide useful failure attribution:

| Automatic result  | Manual-grid result | Likely failure area                                       |
| ----------------- | ------------------ | --------------------------------------------------------- |
| Wrong             | Correct            | Detector or grid refinement                               |
| Wrong             | Wrong              | Classifier/domain shift, possibly in addition to geometry |
| Both models wrong | Any                | Record each model's failure; do not hide the page.        |

## Why the reference is saved first

Model output remains hidden until the reviewer records one of these references:

- no complete board;
- partial board;
- complete board corners, orientation, and all 64 visible labels.

This prevents the model prediction from becoming the accidental source of truth.
The reviewer does not need to repair two separate outputs: the saved reference is
used to calculate both models' missed and false boards, corner displacement,
occupied/empty errors, class and color errors, confident wrong squares, and
latency.

## Trying the local diagnostic

The ignored v2 files must already exist. Their usual paths are documented in
[local candidate testing](local-candidate.md). Start the application:

```sh
pnpm install --frozen-lockfile
pnpm run setup
pnpm run dev
```

In the browser:

1. Open **Test a trained local candidate**.
2. Select the schema-3 candidate manifest, classifier ONNX, and detector ONNX.
3. Choose **Verify and load candidate**. The browser checks sizes and SHA-256
   values locally before it creates the WASM worker.
4. In **Paired local diagnostic**, choose up to 20 PNG/JPEG inputs.
5. Select automatic or manual-grid mode and save the complete human reference.
6. Choose **Run paired comparison**.
7. Record the device/browser fields and download the appropriate export.

Only the active image is decoded. Advancing releases it. There is no upload,
telemetry, background network access, or browser persistence.

## The two export formats

`chess-ocr-private-evaluation/1` is resumable local evidence. It includes file
hashes, geometry, positions, raw model results, and timings. It is sensitive,
ignored by Git, and must not be published. Import requires reselecting files
whose bytes match the recorded hashes.

`chess-ocr-evaluation-summary/1` contains aggregate counts, model identities,
browser/device information, and performance. It excludes pixels, filenames,
paths, file hashes, corners, and positions. It is the only export designed for a
publication review, and still needs inspection before publication.

## Dataset-dashboard integration

The dashboard can register separate immutable v2 localization and labeling
provider manifests. A detached, bounded local runner verifies both ONNX files,
creates two WASM sessions, and releases them on success or failure.

Generated results are proposals only. They are tied to the exact page hash and
revision, become stale when the page changes, cannot accept an annotation, and
cannot enumerate qualification. One human must inspect the complete page, every
corner, and all 64 squares before accepting truth.

The relevant operating commands are in
[assisted dataset review](assisted-review.md) and
[local candidate testing](local-candidate.md).

## Where the implementation lives

| Area                                            | Primary files                                                                                                                |
| ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| Shared candidate preprocessing/decode           | `src/candidate-runtime.ts`                                                                                                   |
| Grid search and rectification                   | `src/grid.ts`                                                                                                                |
| Candidate manifest validation                   | `src/candidate.ts`                                                                                                           |
| Browser worker                                  | `src/trained-worker.ts`                                                                                                      |
| Paired diagnostic state/UI logic                | `src/diagnostic.ts`                                                                                                          |
| Reference, private export, summary, and metrics | `src/evaluation.ts`                                                                                                          |
| Browser markup/styles                           | `index.html`, `src/style.css`                                                                                                |
| Dataset providers                               | `src/proposals/index.ts`                                                                                                     |
| Detached proposal job                           | `scripts/proposal-runner.ts`, `python/dataset_proposals.py`                                                                  |
| Bundle/provider preparation                     | `scripts/candidate.mjs`                                                                                                      |
| Actual-model verification                       | `scripts/candidate-verify.mjs`, `scripts/candidate-browser-verify.mjs`                                                       |
| Executable tests                                | `tests/*candidate*`, `tests/grid.test.ts`, `tests/evaluation.test.ts`, `tests/diagnostic.test.ts`, `tests/proposals.test.ts` |

## What has actually been demonstrated

- strict contracts, bounds, cancellation, cleanup, privacy filtering, and stale
  result handling have executable tests;
- the retained v2 ONNX files execute through Node WASM;
- the actual files execute in Chromium, Firefox, and WebKit in automatic and
  manual-grid paths with no unintended external requests;
- the normal browser and offline-reload suites still pass in all three engines.

These checks demonstrate integration, not real-world accuracy. Current timings
were measured on the GB10 development host, not the named laptop or physical
mobile devices.

## What remains unfinished

The candidate cannot be promoted yet. The next independently reviewable stages
are:

1. review and merge PR #11 only with owner approval;
2. resolve the 11 cross-split duplicate blockers and freeze a source/artwork-
   held-out real development membership;
3. review six development pages reference-first without model proposals;
4. use proposals for the remaining bounded tranche, while fully inspecting each
   page and all 64 squares;
5. lock truth, score frozen v2 and unchanged FENShot on identical pages, and
   apply the recorded promotion gates;
6. separately run six private screenshots through both diagnostic modes;
7. access qualification only after development decisions and the candidate are
   frozen.

No automatic retraining, threshold sweep, new seed, model-family change, model
publication, or qualification access is authorized by the current work.
