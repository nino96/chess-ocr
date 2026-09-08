# Synthetic-first kickoff and dataset generation

Status: active

Owner decision, 2026-09-07. Issue
[#2](https://github.com/nino96/chess-ocr/issues/2) owns collection, annotation
assistance and synthetic generation;
[#3](https://github.com/nino96/chess-ocr/issues/3) owns learning and recognition.
The existing dataset branch continues; no new issue was added for these
increments.

## The decision

Stop treating manual page annotation as the way to manufacture the bulk training
set. Preserve the owner's accepted labels unchanged. Generate synthetic training
labels from validated rendering recipes, acquire diverse real documents in
parallel, and use proposals to turn real labeling into confirmation/correction.

Rights-reviewed asset acquisition and deterministic synthetic rendering come
before the large real tranche. After per-design renderer fidelity and training
gates, #3 may run a synthetic-only bootstrap to improve proposal assistance
before the 300–500-real-board tranche exists. This prospectively supersedes the
earlier real-tranche-first learning sequence; it is not real-data delivery, the
real-data milestone, or recognition qualification. The roughly 50/50
real/synthetic exposure hypothesis applies to later real adaptation, not to this
explicitly synthetic-only bootstrap, and the small existing real group must not
be oversampled to manufacture that ratio. Acquire diverse real pages in parallel
without requiring immediate manual annotation.

The owner should review short confirmation/correction batches, not manufacture
the bulk labels or adjudicate same-split similarity. Qualification still needs
one human pixel check and real source separation; synthetic counts do not replace
those gates. No training accuracy or five-minute total review guarantee is
implied.

### Precedents considered

[Fenify](https://github.com/notnil/fenify#training) describes generated
pretraining followed by chess-book fine-tuning; it does not solve page
localization.
[FENShot](https://github.com/scoriiu/fenshot#why-the-classifier-is-different)
describes fully synthetic classifier training and publishes its generation code.
These establish useful precedents, not independently verified performance on our
printed-page distribution, and not knowledge of other products' private training
data. Upstream fetchers and claimed asset permissions are not approved
automatically.

## Already reviewed data and duplicate policy

- Keep all accepted annotation bytes/revisions, geometry and decisions. Export
  them as candidate training data, not a fresh evaluation set.
- Same-split exact/perceptual candidates are retained with a machine-readable
  audit; no human duplicate decisions are required. Do not mark them distinct,
  and do not alter labels.
- Cross-split candidates still block export until the agent investigates
  leakage. Keep related artwork, editions and derivatives together; never train
  on held-out artwork to remove a blocker. Existing explicit exclusions remain
  historical decisions, not silently reversed.
- Training must report effective source/board repetition and cap repeated
  exposure without deleting labels. Same font/style is not duplication; a shared
  position alone does not merge unrelated artworks. Same-split repeats do not
  increase independent coverage and must not inflate evaluation certainty.

## Executable increments and exit conditions

### 1. Start asset collection and preserve the reviewed seed

Apply the [authorized ledger](../budget.md#issue-2-synthetic-first-kickoff--2026-09-07),
record preparation reservations, download only explicitly reviewed assets to
ignored storage, and pin original bytes, author notices and rights evidence.
Export the existing human-reviewed seed after the duplicate-policy tests pass.
That is useful pipeline/proposal evidence; one artwork group is not a
source-diverse training set.

Initial asset lane: three separately authored permissively licensed piece designs
from Lichess; expand toward 8–16 genuinely different reviewed designs when
coverage requires it. Inspect Wikimedia files individually, including attribution
and ShareAlike requirements; do not assume a site-wide license. Chess.com artwork
is excluded without a grant permitting this use. Never run upstream catch-all
asset downloaders, and do not mistake a code license for artwork permission.

### 2. Validate the renderer, then generate the first synthetic seed

The generator must be offline, deterministic and resumable, with persisted
start/status/stop, an explicit asset allowlist and hashes, bounded SVG/raster
decoding, a fixed seed, configuration identity and per-attempt accounting. No
source fetching in rendering or tests. Review existing public generator code for
bounded reuse before introducing dependencies; preserve the project's pnpm lock
and image-relative `.PNBRQKpnbrqk` mapping (upstream class order differs).

Before bulk generation, inspect every design's 12 pieces on both square
backgrounds and compare an independent render/extraction control. Test
white/black identity, all 64 positions, row/column order, orientation, SVG
fill/stroke and transparent background handling, geometry, retry/recovery and
changed-input rejection. A label recipe or classifier agreeing with itself is not
renderer validation.

Then launch at most **20,000 board equivalents** as the first seed. Generate
compact source/position/page recipes and images; extract training tensors lazily.
Use declared legal-position and teaching-position strata; include sparse, medium,
dense, empty and partial cases without pretending random placements are legal.
Freeze mixture and class/background exposure after fidelity inspection. Record
related positions and parents; split/group before augmentation.

Include full-page layouts, captions/coordinates, small and multiple boards and
negatives for localization, not just perfect crops. Transform image and corners
together. Calibrate modest degradation to inspected real inputs; quarantine
transforms that erase pieces or change labels. Synthetic controls are reported
separately from real development/qualification.

Exit: a hash-bound usable seed with an audited renderer and class/condition
coverage, or a concrete failure. A downloaded sprite collection is not this exit
condition.

### 3. Begin real acquisition without a manual-labeling prerequisite

In parallel with generator/proposal work, prepare an initial increment of up to
12 rights-reviewed source families and 500 selected pages within the global
ledger. Prioritize missing printed appearances: modern open teaching documents,
suitable historical scans, and typography not already represented by the initial
manual. Do not pad independence with more manuals from the same font family.

The agent owns source discovery, rights records, lineage assignment, admission,
background start and failure diagnosis. Select pages independently of recognizer
success and freeze each source's explicit selection. Source and rights decisions
are reviewed before each bounded acquisition queue, not delegated to an
unattended crawler. Start the existing worker as soon as a queue is admitted; do
not wait for all 12 families or finish all labeling first.

Reserve genuinely separate real development/qualification groups before training;
do not run proposals on fresh qualification and then use it to tune the proposal
workflow. The real collection stays unreviewed until pixel-checked; it must not
silently enter accepted training as pseudo-ground-truth.

### 4. Use assistance before asking for more manual annotation

First screen the existing chess-trained classifier on the accepted seed; training
from scratch is not a prerequisite for testing useful prefilling. Compare the
bounded classical multi-board proposal path with existing FENShot localization.
If correction burden remains high, #3 runs one frozen synthetic bootstrap with
the existing native classifier candidate under the GPU suballocation, then
measures whether it improves assistance. No model-family or seed sweep.

Deliver draggable/replaceable corners, undo, piece corrections, accept-and-next
and preservation of accepted labels across late results. Use optional
**five-minute review sessions** with batch progress. Measure total review time,
missed/false boards and corrections on identical pages, including
difficult/negative inputs. Aim for most correct boards to require confirmation
only; do not promise that a whole source-diverse dataset can be validated in five
minutes.

Synthetic boards do not need individual human labeling after renderer/design
fidelity checks. Real accepted pages still need one human pixel check; a second
reviewer remains optional. Agent/model proposals remain explicitly proposals. If
a real page needs extensive repair, defer it instead of asking for another
exhausting manual session. Keep such failures in coverage reports.

### 5. Train incrementally and evaluate honestly

A synthetic bootstrap can advance annotation assistance early. Later train on the
first useful real tranche as acquisition continues: existing target 300–500 real
boards across at least six genuine source/design groups, with separately reserved
development and qualification. These are feasibility targets, not promised
quality.

#3 must freeze hashes, preprocessing, sampling/source repetition, seed,
optimizer, complete schedule, checkpoint selection and budget before GPU
execution; test actual stochastic resume and inspect full curves. Existing
localization, reliable exact-board, confident-error and browser WASM gates remain
unchanged. Data creation does not establish recognition success. Do not merge
dependent integration before the required #1/#2 interfaces are merged.

## Resource and publication boundaries

The owner permits justified, bounded local budget increases at lead discretion;
retain prior charges and record reasons. Preserve at least 30% filesystem free
space. Paid/cloud services and asset/model publication remain unauthorized.
Reviewed public provenance is published separately from operational or
mixed/private records.

The [test-isolation incident](../archive/dataset-incident-2026-09-07.md) removed
the active reviewed seed and its recovery archive. The owner explicitly waived
recovery and authorized continuation without those labels. Preserve remaining
evidence and prior charges; do not substitute new proposals for lost human truth.
Synthetic bulk remains conditional on current independently checked fidelity
evidence. Commit reviewed changes before every unattended launch, as required by
the owner.

## Where this decision is carried out

- [The synthetic pipeline](../synthetic-dataset.md) — deterministic page recipes,
  independent renderer controls, persisted bounded start/status/stop, durable
  attempt accounting and candidate lazy training interfaces. Implementation is
  not proof of recognition improvement.
- [The historical increment and public dataset screen](../provenance/public-dataset-review.md)
  — explicit public provenance and fixed page selections. Closely related De Witt
  and Staunton artwork remains one reserved group.
- [Public provenance and reproducibility](../reproducibility.md) — what a fresh
  clone can rebuild, and what stays unpublished.
- [The dataset pipeline guide](../dataset-pipeline.md) and
  [operator workflow](../operator-workflow.md) — the commands.

Modern-document diversity, development groups, accepted real annotations,
assisted-review measurements and the full first-real-tranche outcome were
incomplete at the time of this decision.
