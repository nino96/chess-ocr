# Recognition delivery plan

This is the initial owner-requested direction, not measured model performance.
Live issues own acceptance criteria; update this document when reviewed decisions
change. No training, downloads, paid API or cloud deployment is authorized merely
by a proposed architecture. Maintain the assigned issue's explicit budget.

## Ownership and minimal sequence

1. [#1 Browser baseline and contract](https://github.com/nino96/chess-ocr/issues/1):
   runnable local demo, shared schema, native checkpoint identity, actual WASM
   execution, environment locks and measured runtime budgets.
2. [#2 Dataset](https://github.com/nino96/chess-ocr/issues/2): first meaningful
   real training tranche, source-held-out development and reserved qualification,
   acquisition/review/reproduction tools. Research can overlap #1; integrate
   against merged interfaces. Continue acquisition while #3 learns.
3. [#3 Offline recognition](https://github.com/nino96/chess-ocr/issues/3): train
   and deliver better page/selection-to-position behavior through the demo and
   versioned library, not a classifier-only result.
4. [#4 Optional server](https://github.com/nino96/chess-ocr/issues/4): opt-in GB10
   backend and deployment guide; full offline success is not a dependency when
   measured limits justify the fallback. Cloud spending/exposure requires approval.

The reader application remains elsewhere. This repo owns recognition, data,
training/evaluation, export and a small demo. Use one backend-neutral response:
original-image board geometry, 64 image-relative pieces/probabilities, orientation
evidence/unknown, warnings/abstention, model/preprocessing identity and timings.
Do not manufacture unobservable FEN state or correct printed diagrams for legality.

## Architecture hypotheses

| Role                       | Initial native checkpoint                    | Important boundary                                               |
| -------------------------- | -------------------------------------------- | ---------------------------------------------------------------- |
| Browser board detector     | COCO-pretrained YOLOX-Nano                   | A rectangle is not yet the exact inner playing grid              |
| Browser square classifier  | `timm/mobilenetv3_small_100.lamb_in1k`       | ImageNet features, not pretrained chess knowledge                |
| Fixed control              | Shipped FENShot 0.1.4, exact artifact review | Baseline to beat, not the architecture we must rescue            |
| Optional server recognizer | RF-DETR Small                                | Must improve measured accuracy or serve a stated runtime purpose |

Acquire exact native training weights after provenance review, preserve them
locally, and train new task heads. Pin every revision/hash/dependency. Do not
reconstruct an inference ONNX when a legitimate native checkpoint is available.
Native forward equivalence, trainability and good transfer are different claims.

YOLOX-Nano is a small established detector with documented ONNX deployment;
MobileNetV3 is a standard compact backbone. Proposed initial classifier input is
96×96 RGB with model-appropriate normalization, rather than 32×32 grayscale.
That resolution differs from the ImageNet checkpoint and must be measured.
Neither model's benchmark numbers establish chess performance or WASM speed.

Use page/selection -> candidate boards -> inner-grid refinement/rectification
-> square classifier -> editable uncertainty. Exclude coordinates/borders from
grid bounds. Handle limited scan rotation/skew; reject unsupported geometry.
Small page diagrams may require bounded overlapping page crops/scales. Measure
both misses and the extra latency. Do not preprocess whole books.

Export and exercise actual ONNX Runtime Web WASM CPU early, including detector
decode/NMS and preprocessing. Quantization or custom runtime builds are later
measured optimizations, not prerequisites for proving accuracy. WebGPU is optional.
Cache hash/version-bound model/runtime assets locally after readiness; no CDN.
Use workers, cancellation and stale-result IDs. There is no arbitrary 2 MB cap:
set measured cold/warm latency, memory and total-download budgets in #1 before
#3 qualification, without moving them after outcomes.

For fallback, sending the same model to GB10 changes compute location, not its
predictions. RF-DETR Small is a prospective higher-resolution alternative for
board and occupied-cell detection; stage full-page localization and crop
recognition as needed. Derive occupied-cell regions from reviewed grid labels
but do not label them as tight glyph boxes. Resolve duplicate/missing cells and
orientation without chess-legality repairs. Verify actual ARM64/GB10 runtime.

Primary references (verify exact artifacts at acquisition):

- [YOLOX model table and license](https://github.com/Megvii-BaseDetection/YOLOX)
- [YOLOX ONNX deployment](https://yolox.readthedocs.io/en/latest/demo/onnx_readme.html)
- [MobileNet checkpoint card](https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k)
- [ONNX Runtime Web deployment](https://onnxruntime.ai/docs/tutorials/web/deploy.html)
- [RF-DETR model/license variants](https://github.com/roboflow/rf-detr)

## Dataset: pages, geometry and labels together

Canonical sample records bind source original/rights hashes, page identity and
dimensions, inner-grid corners, 64 labels, orientation, document/edition/artwork
lineage, parent transformations and immutable review decisions. From one record
derive detector targets, exact/loose crops and actual classifier tensors. Keep
reference resolution; inspect any information lost during preprocessing.

First learning target: 300–500 REAL train boards across >=6 genuinely independent
groups, with initially >=3 development and >=3 reserved qualification groups and
about 100 reviewed boards in each where available. Those exploratory counts are
not qualification guarantees. Missing independence is not solved by more pages.
Begin substantive #3 training on the accepted tranche while collection continues.

Retain the larger reference target: 1,200 train / 240 development / 240 frozen
qualification / 120 clean regression. Any amendment must be explicit, prospective
and justified; do not quietly change gates after exposure. Labels for a reserved
pool may be prepared, but no inference outcomes are used to tune candidates.

Prioritize these source lanes by missing visual appearances:

- Authorized contemporary coach worksheets, newsletters and publisher material.
  Ask for existing independently authored material, not copies of third-party
  books or a dataset designed to imitate the held-out publisher. Written grants
  distinguish training/evaluation, crop redistribution and weight publication.
- Suitable open educational documents such as Wikibooks, with per-image rights,
  attribution and third-party exclusions reviewed; no blanket site-level grant.
- CTAN documentation and independently licensed fonts/artwork as clean typography
  and diagnostic cases. Package licensing does not clear every embedded font.
- Historical scans as one real degradation/print stratum, never the only one.

Keep training, evaluation, original/crop redistribution and model publication
decisions separate. Record exact edition/revision/URL, attribution, SHA-256,
rights evidence URL/hash/date, exclusions and unresolved permissions. Do not
require crop redistribution rights merely to admit an otherwise approved local
training source. Public/private origin, rights lane and experimental split are
independent dimensions. Unknown pretrained-data overlap remains explicit.

References: [Wikibooks media policy](https://en.wikibooks.org/wiki/Wikibooks:Copyrights),
[CTAN chessboard](https://ctan.org/pkg/chessboard),
[Lichess CC0 position exports](https://database.lichess.org/).
Lichess position licensing does not license screenshots or piece artwork.

Coverage must count independent source/design groups as well as boards/tiles:

- All 12 colored piece classes on both square backgrounds; sparse/medium/dense.
- Distinct outlines/fills and light/dark conventions; modern and historic print.
- Flat, hatch, halftone and colored boards; crisp and low-native-resolution glyphs.
- Blur, scan grain, ink spread, broken strokes, JPEG, bleed-through, uneven light.
- Coordinates, arrows, captions, margins; multiple boards and on-demand page crops.
- No-board pages, partial grids and unsupported geometry, not only successes.

Split at document/related edition/shared artwork/visual-parent lineage. Keep
exact/perceptual duplicates and all derivatives together. Treat repeated positions
as a separate audit rather than joining every common opening across books.
Exclude held-out artwork from synthetic training. Exposed chess-reader diagnostic
material cannot become fresh qualification; inherited provisional groups stay
provisional until verified.

## Annotation and synthetic fidelity

Notation, PDF glyph placement or a recognizer may propose labels. Display the
source page/crop/grid and a board freshly rendered from proposed labels together.
Review actual geometry, orientation and all 64 squares, preserve accepted edits,
and quarantine conflicts. Independent checks must cover each new glyph design,
all uncertainty and a stratified sample of accepted training data. Qualification
truth requires independent human checking. Time the first 20 reviews and report
disagreement rates and review cost; model agreement/legality are not ground truth.

Synthetic position specifications supply intended labels, not proof that a
renderer drew them correctly. Verify every renderer/design against an independent
render control, including CSS fills, piece color/order and production tensors,
before bulk generation. Use many independently designed rights-cleared sets and
PGN-derived positions, plus explicitly tagged teaching/unsupported cases. Synthetic
examples fill measured real-data gaps; no million-crop quota or fixed font count
proves generalization.

Train-only degradation should match inspected real samples: resampling, blur,
ink spread/broken strokes, grain/JPEG, lighting/contrast and modest affine/projective
misalignment. Transform geometry with whole-page transforms. Keep crop perturbation
label-preserving or mark it ambiguous/partial; arbitrary texture/color inversion
or large warps are not fidelity evidence. Count transformed examples separately.

## Adaptation and experimental decisions

Initial proposal: about half canonical real training exposure and half synthetic
or degraded exposure, with capped source repetition. This is a hypothesis to
freeze, not a universal optimum. Do not repeat a 12-board group seven times per
epoch solely to equalize it with 84 boards. Record effective class/source weights.

Train the new classifier head first, then lower-LR later backbone blocks; explicitly
control BN statistics and regularization. Inspect gradients, labels and tiny-set
fit before a substantive run. Native checkpoints avoid lost optimizer/BN state
assumptions, but ImageNet transfer may still fail; keep the FENShot baseline.

Freeze meaningful update counts/schedule, data/code/model hashes, seed(s), sampling,
optimizer, augmentation, selection, confidence and resource reservations before
training. Log full curves and per-class/condition/source behavior. Persist native
model/optimizer/scheduler/RNG state; prove resume on the actual stochastic recipe.
Do not stop successfully at a mechanics pilot or interpret budget truncation as
architecture failure. New experiments need a concrete project budget; previous
chess-reader allocations are not automatically expanded or transferred.

If transfer fails, inspect a bounded specific cause: label/rendering error,
information loss, source weighting, objective/selection mismatch, forgetting,
localization or degradation. Do not default to a bigger corpus or an architecture
search. One reviewed resolution comparison can address demonstrated information
loss; unrelated model sweeps require a separate owner decision.

## What counts as success

Report separate localization recall/false positives and corner/grid error,
exact-crop classifier errors, and the REAL end-to-end task. Detector IoU alone
does not ensure square alignment. Report all-input raw/reliable exact boards,
occupied/class/color errors, confident errors, abstention coverage, and uncertainty
across source groups. Tiny single-condition samples do not establish robustness.

Proposed retained promotion: >=5 percentage-point paired target/degraded dev gain
against unchanged FENShot on identical raw inputs, no lost baseline-correct clean
cases, no occupied/class collapse and no increase in confidently wrong boards.
Operationalize strata, calibration and minimum samples prospectively in #3.

After candidate freeze, independently checked source-held-out qualification:

> =95% reliable exact boards over ALL eligible inputs (misses/abstentions fail),
> =99.5% accuracy among confident squares with coverage reported, and zero observed
> reliable-wrong boards. Zero observed errors is not a zero-risk claim. Keep clean
> regression separate, report uncertainty, and do not tune on qualification results.

Integrate qualifying candidates through the browser demo and versioned library:
WASM CPU in Chromium/Firefox/WebKit, offline reload, no unintended network,
hash integrity, cancellation/timeouts/stale results, keyboard/touch/accessibility,
and measured runtime budgets. Physical iPad testing is explicitly deferred until
scheduled/authorized and must never be described as passed via browser emulation.

Optional server mode needs the same quality contract plus deliberate upload consent,
visible endpoint/backend, loopback default, no retained images/FENs, auth/TLS for
remote deployment, CORS, decoded-image/request/concurrency limits, backpressure,
timeout/cancellation and credential-origin protection. No cloud account/spend or
private-data upload is authorized by these issues alone.

## Reuse without importing the old project

Public tooling, reviewed metadata and historical comparisons are available in
[the chess-reader handoff at c2ece9a](https://github.com/nino96/chess-reader/tree/c2ece9a422dc573e41a2d9f9a84af7c1240c3ad3/experiments/recognition-dataset/handoff).
Inspect licenses, imports and hashes before copying a useful component. Reference
immutable historical reports rather than importing every old run or model binary.
No changes to chess-reader production, frozen corpora, PR #39 or parent #24 are
implied by this repository bootstrap.

The first user-visible delivery is an unfamiliar page becoming an editable
position with board outlines and uncertain squares. The app should consume that
capability, not depend on a particular neural-network implementation.
