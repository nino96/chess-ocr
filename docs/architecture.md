# Architecture

How the recognition system is shaped: its vocabulary, its pipeline, the two
diagnostic modes, the export formats, the dataset-dashboard integration, and the
map from subsystem to source file.

This document describes the machinery. What the machinery is and is not allowed
to claim lives in [scope and claims](scope-and-claims.md).

## Vocabulary

| Term          | Meaning                                                                                                        |
| ------------- | -------------------------------------------------------------------------------------------------------------- |
| FENShot       | The unchanged 0.1.4 baseline used by the browser demo.                                                         |
| v2 candidate  | A YOLOX-Nano detector plus a MobileNetV3 classifier, trained on synthetic pages.                               |
| Detector      | Finds possible board regions on a full page.                                                                   |
| Grid refiner  | Looks inside a possible region for two line families forming a regular 8-by-8 board. It rejects weak evidence. |
| Rectification | Warps four skewed inner-grid corners into a square 768-by-768 image.                                           |
| Classifier    | Reads the 64 rectified 96-by-96 squares as empty or one of 12 piece classes.                                   |
| Reference     | The human-entered expected result used to score both models.                                                   |
| Development   | Real, source-held-out data used to decide whether a candidate is promising.                                    |
| Qualification | A later untouched evaluation set. It must not influence development decisions.                                 |

## The recognition pipeline

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

A low detector score threshold produces region proposals only. Unrefined regions
are never returned as accepted boards; a detector box is never silently treated
as a trustworthy board. The distinction between a proposal threshold and a
calibrated acceptance threshold is explained in
[the training runbook](training-runbook.md#proposal-threshold-versus-acceptance-threshold).

## Why there are two diagnostic modes

The paired diagnostic runs a candidate and FENShot against the same decoded
input and scores both automatically against one saved human reference.

### Automatic full-page mode

This exercises the complete pipeline: localization, refinement, rectification,
and square classification. Use it with full screenshots or deliberately loose
page selections.

### Manual four-corner mode

The reviewer clicks the four inner-grid corners over the source image, then can
drag each numbered handle or adjust it with the keyboard. The diagnostic draws
the full 8-by-8 grid before the reference is saved; numeric coordinates are an
optional fine-adjustment control. Both models receive the same rectified grid,
so localization is removed from the comparison. This isolates classifier and
domain-shift errors.

On wider screens the source grid and reference piece editor remain side by side.
Each square accepts the exact `.PNBRQKpnbrqk` character from the keyboard;
Alt+arrow keys move between squares.

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

The diagnostic accepts at most 20 inputs and 100 MiB compressed, and keeps only
the active image decoded — advancing releases it. There is no upload, telemetry,
background network access, or browser persistence.

After a paired run, the diagnostic shows the saved reference, both predicted
piece grids, each returned grid over the source, and a board rectified from that
returned geometry. Wrong squares are marked on the predicted grids. The private
export remains the authoritative raw result; the visual views stay in memory.

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

The dashboard can register separate immutable localization and labeling provider
manifests. A detached, bounded local runner verifies both ONNX files and the
refiner implementation, creates two WASM sessions, and releases them on success
or failure.

Generated results are proposals only. They are tied to the exact page hash and
revision, become stale when the page changes, cannot accept an annotation, and
cannot enumerate qualification. The acceptance rule is in
[scope and claims](scope-and-claims.md#the-one-human-pixel-review-rule).

The operating commands are in [assisted dataset review](assisted-review.md) and
[local candidate testing](local-candidate.md).

## Where the implementation lives

| Area                                            | Primary files                                                                                                                |
| ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| Shared candidate preprocessing/decode           | `src/candidate-runtime.ts`                                                                                                   |
| Grid search and rectification                   | `src/grid.ts`                                                                                                                |
| Candidate manifest validation                   | `src/candidate.ts`                                                                                                           |
| Session-protected remote candidate convenience  | `scripts/remote-candidate-server.ts`, `src/remote-candidate.ts`                                                              |
| Browser worker                                  | `src/trained-worker.ts`                                                                                                      |
| Paired diagnostic state/UI logic                | `src/diagnostic.ts`                                                                                                          |
| Reference, private export, summary, and metrics | `src/evaluation.ts`                                                                                                          |
| Browser markup/styles                           | `index.html`, `src/style.css`                                                                                                |
| Dataset providers                               | `src/proposals/index.ts`                                                                                                     |
| Detached proposal job                           | `scripts/proposal-runner.ts`, `python/dataset_proposals.py`                                                                  |
| Bundle/provider preparation                     | `scripts/candidate.mjs`                                                                                                      |
| Actual-model verification                       | `scripts/candidate-verify.mjs`, `scripts/candidate-browser-verify.mjs`                                                       |
| Executable tests                                | `tests/*candidate*`, `tests/grid.test.ts`, `tests/evaluation.test.ts`, `tests/diagnostic.test.ts`, `tests/proposals.test.ts` |

## FENShot baseline adapter

FENShot is the fixed control. Its unchanged `recognizeGray`, grid arbitration,
tile extraction and probability conversion run against unchanged weights in an
ORT WASM session. Artifact admission, pinning and licensing for that package are
in [browser artifact admission](provenance/artifacts.md).

The browser adapter caps automatic detection at 1600 pixels, as the shipped
wrapper does, then maps the resulting bounds back to the original decoded image.

Manual selection is a separate, explicitly named fallback that uses the shipped
tile extraction. It does not establish automatic localization success.

FENShot supplies one candidate, not a multi-board page detector. Skew and
perspective are unsupported.

The adapter reorders probabilities from bottom-up model tiles and
`1KQRBNPkqrbnp` classes into top-down image rows and the contract's class order.
Its 0.7 confidence floor is inherited and uncalibrated.

`resolveOrientation`, `inferCastling` and `placementToFen` are not used:
orientation begins unknown and unobservable FEN fields are absent. No visible
piece is repaired for legality.
