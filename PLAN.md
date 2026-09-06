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
| Alternative browser board detector | Classical computer vision and grid geometry | Proposed bounded comparison in #3; no learned detector required |
| Browser square classifier  | `timm/mobilenetv3_small_100.lamb_in1k`       | ImageNet features, not pretrained chess knowledge                |
| Fixed control              | Shipped FENShot 0.1.4, exact artifact review | Baseline to beat, not the architecture we must rescue            |
| Optional server recognizer | RF-DETR Small                                | Must improve measured accuracy or serve a stated runtime purpose |
| Additional GB10 evaluation candidate | ChessQueries ViT-L/14 + square-query decoder | Unvalidated on printed pages; explicit geometry and runtime checks required |

Acquire exact native training weights after provenance review, preserve them
locally, and train new task heads. Pin every revision/hash/dependency. Do not
reconstruct an inference ONNX when a legitimate native checkpoint is available.
Native forward equivalence, trainability and good transfer are different claims.

Owner addition (2026-09-06): evaluate the unchanged ChessQueries checkpoint in
one bounded printed-diagram development screen before proposing adaptation.
[Issue #4](https://github.com/nino96/chess-ocr/issues/4) owns the pinned source and
safetensors identities, proposed GB10 venv/CUDA setup, resource reservation and
advance/defer/reject decision. Its physical-board benchmarks do not establish
printed-page quality, image-relative geometry or browser feasibility. Code and
weights list noncommercial terms requiring separate artifact review. This is an
evaluation candidate, not a selected backend or permission to start training.

Owner addition (2026-09-06): [issue #3](https://github.com/nino96/chess-ocr/issues/3)
also evaluates one bounded classical board/grid detector using line, contour and
regular 8×8-grid evidence. Historical inspiration is the creator's description of
[Chessvision.ai's computer-vision/graph detection and CNN cell classification](https://devpost.com/software/chessvision-ai),
not knowledge of its current private implementation. Removing a learned localizer
could reduce detector-training requirements; neural inference can also be
deterministic, and classical heuristics do not guarantee reliability.

Compare classical detection and YOLOX-Nano with the same frozen chess classifier,
square preprocessing, confidence policy and shared downstream grid processing
where applicable. Record unavoidable geometry-stage differences. Retain unchanged
FENShot as the complete-path control. Freeze approved development inputs, hashes,
thresholds, commands, coverage, stop/decision criteria and concrete ledger ceilings
before execution; this addition does not enlarge the budget or expose qualification.
Include degraded/broken lines, borderless grids, captions, small/multiple boards,
skew and negative/partial cases. Report paired localization/grid errors, all-input
exact boards, confident errors, clean losses and CPU latency/memory. End with an
advance/defer/reject decision under unchanged #3 gates, including actual browser
WASM CPU validation before promotion. YOLOX plus ImageNet-pretrained MobileNetV3
adapted for chess may still win. No detector is selected by this planning addition;
ChessQueries remains the separate #4 screen.

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
truth requires a human check independent of model/agent proposals; one such human
review accepts a page. Time the first 20 reviews and report review cost and
observed corrections/ambiguities honestly; do not report a disagreement rate from
one reviewer. Model agreement/legality are not ground truth.

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

## Issue #1 CI scope — owner direction, 2026-09-06

This repository's main ongoing work is OCR training and evaluation. Routine CI
runs source/type/contract/unit/payload checks and a production build. Browser
input changes receive one Chromium WASM smoke test; native-only changes do not
trigger browser tests. Firefox/WebKit, offline-origin shutdown, corrupt assets,
and accessibility/touch checks are retained as manual browser integration gates
for relevant runtime/demo changes, not an every-training-change CI matrix.
Initial multi-browser evidence is retained; it is not physical iPad evidence.


## Implemented issue #1 baseline

The repository now contains the npm FENShot control, versioned raster/geometry/
probability contract, cancellable WASM worker and small editable browser demo.
Exact native starting checkpoints were exported with their unchanged COCO/ImageNet
heads; source preprocessing and actual browser WASM parity were checked on original
synthetic inputs. See [evidence](docs/issue-1-evidence.md) for commands and limits.
No dataset training, real-diagram qualification or recognition superiority is
claimed. Required laptop runtime budgets and unavailable physical OS/device gates
remain explicit. The selected pinned GB10 native inference probe requires cuDNN
disabled to pass CPU/GPU parity; training validation remains a later gate.


## Package-manager alignment

Owner direction: use pnpm consistently with chess-reader. This repository pins
pnpm 11.11.0; `pnpm-lock.yaml` is authoritative and CI uses frozen installs.
The npm lock was imported without changing any package/version identities.
Historical evidence retains the original command names rather than relabeling
previous npm runs as pnpm runs. Current commands are in README.

## Issue #2 pipeline and owner privacy direction — 2026-09-06

Dataset implementation can proceed independently of the deferred physical-device
checks. The 2026-09-07 owner update supersedes the original blanket metadata ban:
reviewed public-source URLs, revisions, hashes, rights, selection and reconstruction
recipes belong in Git; private source details and operational/mixed records remain
ignored. Downloaded originals and generated datasets/weights are never committed.
See [public provenance and reconstruction limits](docs/reproducibility.md).

The [implemented pipeline](docs/dataset-pipeline.md) provides a local PDF inbox,
explicit local-use ingestion, bounded background rendering/acquisition, status,
stop/resume, hash/revision checks, one-human acceptance independent of model
proposals, and candidate train/dev
exports. Complete-page targets preserve negatives and exclude unsupported cases.
Artwork independence remains a reviewed property, not a count of downloaded files.
The first real tranche, verified coverage/lineage, measured human audit, approved
synthetic fidelity, and downstream #3 preprocessing parity remain undelivered.
Pipeline mechanics do not establish the dataset or recognition outcome.

## Owner decision: review dashboard and managed reset — 2026-09-06

The loopback dashboard is the primary dataset-review workflow: it presents the
local queue/page thumbnails, retains server-side drafts, submits human reviews and
advances the queue. One human pixel review independent of any model/agent proposal
accepts a matching annotation; a second review is optional for every split,
including qualification. This does not relax the frozen #3 recognition metrics or
the requirement that qualification truth be independently human checked.

The confirmation-required **Start over** action archives managed state under
ignored `work/dataset/archives/<timestamp-id>/`, retains the inbox and approved
budget, and carries cumulative acquisition/review use forward. An archive remains
storage usage; reset is not a budget refund and has no automated restore command.
The **Archives** panel lists dated sizes and supports explicitly confirmed permanent
deletion of one unchanged archive. Deletion reclaims storage without changing
active data, inbox files or cumulative usage; unsafe paths and links are rejected.
These are local workflow controls, not a real collection, qualification result or
browser-test claim.

## Planned dashboard enhancements before larger collection

Owner direction, 2026-09-06; all work stays in issue #2. The current dashboard
supports a small manual feasibility batch; successful synthetic workflow tests
do not establish large-collection throughput. The following are delivery work,
not implemented capabilities or approval for more acquisition/compute/reviews:

- **Annotation assistance:** deliver the bounded board-proposal and label-prefill
  comparison below. Measure total human time per board, missed/false boards and
  geometry/piece corrections; preserve one-human acceptance and edited labels.
- **Focused review queues:** add filters for page kind, draft/annotation issues,
  ambiguity and source/condition coverage gaps, with resumable batch progress.
  Prioritization must not silently exclude negatives, difficult pages or misses.
- **Concurrent acquisition and review:** remove the whole-job writer-lock conflict
  so rendering does not block saving reviews. Retain transactional revision guards,
  request cancellation/recovery, bounded concurrency and budget accounting. Test
  simultaneous saves, worker stop/resume, reset exclusion and stale results.
- **Append-only page selection:** allow additional explicitly selected pages of an
  admitted PDF without changing source identity, prior annotations or frozen split
  membership. Preview incremental costs, enforce remaining limits, and make retry
  idempotent; test restart/recovery and duplicate selection.
- **Acquisition control in the app:** clearly distinguish local PDF admission,
  source discovery/rights review, and execution of an admitted source queue.
  Add a UI for reviewing and admitting prepared public-source manifests with their
  pinned hashes and rights evidence, plus queued/running/failed counts, progress,
  remaining budget, actionable stop reasons and explicit repaired-job retry.
  This does not authorize an unattended crawler, new rights decisions or spending.
- **Measured scale gate:** under a recorded bounded test allocation, measure queue
  responsiveness, draft-save latency, memory and recovery with representative
  larger queues. Agree acceptable limits before the run and report distributions,
  dataset/code hashes and untested conditions; use original synthetic fixtures
  for tooling load tests and authorized real reviews for human-throughput evidence.

Use the initial manual reviews to estimate annotation effort. The owner approved
the larger bounded [kickoff allocation](docs/budget.md#issue-2-synthetic-first-kickoff--2026-09-07);
reset/archive deletion cannot replenish it. Deliver these usability/scale checks
before claiming readiness for hundreds of
boards. This roadmap does not complete issue #2's real dataset outcome.

## Decision: automatic annotation proposals after feasibility

Owner direction, 2026-09-06; owned by issue #2. After the initial manual feasibility
batch, implement automatic board proposals and piece-label prefilling before
scaling annotation to the larger collection. This is committed follow-up scope,
not an already implemented feature. No board detector has been selected.

- Use the feasibility batch's reviewed pages and measured manual-review effort
  to define one bounded comparison of a classical multi-board grid detector and
  the existing FENShot detection path. Preserve full-page negatives, missed boards,
  small/multiple boards and difficult geometry; do not select only successes.
- Choose localization separately from square-label proposals. The unchanged
  chess-trained FENShot classifier is an initial label-prefilling candidate,
  not a mandated detector or teacher. The unadapted YOLOX/ImageNet checkpoints
  are not assumed to provide useful chess annotations.
- Select the proposal workflow by total human annotation time, missed and false
  board proposals, grid corrections and piece corrections on identical pages.
  Record a measured advance/defer/reject decision rather than assuming either
  approach wins. Qualification inputs remain untouched.
- Implement bounded resumable proposal jobs, source/model/preprocessing identity,
  editable candidate outlines and labels, and preservation of human corrections
  across retries, geometry edits and late results. Provide executable tests and
  documented start/status/stop/review commands before larger-scale use.
- Predictions remain proposals: confidence cannot replace independent human
  verification or admit labels automatically. One human pixel review, independent
  of the proposal/model, accepts a matching annotation; a second review is
  optional, never mandatory. Keep all input-specific evidence
  and private-derived metadata local; commit reviewed public recipes/provenance.
  Freeze the comparison limits within an approved
  allocation; this decision does not increase the current collection/GPU budget.

Handoff gate: report the selected method, measured review-time benefit and failure
coverage, or the specific reason neither method advances. The operator workflow
must reflect the implemented behavior before the larger annotation phase begins.

## Owner decision: synthetic-first kickoff — 2026-09-07

Follow [the concrete kickoff plan](docs/dataset-kickoff.md) and its bounded local
resource ledger. Preserve existing human labels as-is; same-split duplicate
candidates stay in candidate exports with a multiplicity audit, not mandatory
human decisions. Cross-split leakage still blocks export and is the agent's
investigation task. Do not silently mark candidates distinct or alter labels.

Begin rights-reviewed asset acquisition and deterministic synthetic rendering
before the large real tranche. After per-design renderer fidelity and training
gates, #3 may run a synthetic-only bootstrap to improve proposal assistance before
the 300–500-real-board tranche exists. This prospectively supersedes the earlier
real-tranche-first learning sequence, not real-data delivery or qualification.
The later real adaptation mix remains a separately frozen hypothesis. Acquire
diverse real pages in parallel without requiring immediate manual annotation.

The owner should review short confirmation/correction batches, not manufacture
the bulk labels or adjudicate same-split similarity. Qualification still needs
one human pixel check and real source separation; synthetic counts do not replace
those gates. No training accuracy or five-minute total review guarantee is implied.
Use existing issues #2/#3 and the dataset branch; no independent workstream was added.
