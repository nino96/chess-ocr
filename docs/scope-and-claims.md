# Scope and standing claims

The single home for what this repository claims, what it deliberately does not
claim, and the rules that decide when a claim may be made at all. Other documents
link here instead of restating any of it.

## Product boundary

This repository owns recognition, data, training/evaluation, export and a small
offline browser demo. The reader application lives elsewhere and consumes a
versioned contract.

- It is **not** an ebook or PDF/EPUB reader, a chess engine, an accounts system,
  a study database, or a generalized recognition claim.
- The first target is printed 2D diagrams taken from pages or selections, not
  photographs of physical 3D boards.
- The required default is offline ONNX Runtime Web WASM CPU inference in a
  worker. WebGPU is optional acceleration, never a substitute for WASM
  acceptance.
- The complete path is page/selection -> board localization -> inner-grid
  refinement/rectification -> piece placement -> editable, uncertain output.
  Exact-crop accuracy establishes neither localization nor end-to-end success.
- An optional GB10 or cloud mode is deliberate and visibly selected, never a
  silent fallback, and never uploads without explicit informed consent for that
  input and endpoint.

## What the output may and may not contain

The response is backend-neutral: original-image board geometry, 64
image-relative pieces and probabilities, orientation evidence or unknown,
warnings and abstention, model and preprocessing identity, and timings.

- Side to move, castling rights, en passant and move counters are **never**
  inferred. They are unobservable in a printed diagram, so they are absent.
- No visible piece is altered to satisfy chess legality, and no printed diagram
  is "corrected".
- Orientation may remain unknown until the user selects it. It stays unknown
  unless a later orientation model is independently validated.
- User edits survive late or out-of-order results, retries and backend switches.
  A changed grid cannot silently relocate corrections, and the model's original
  probability evidence remains alongside explicit user corrections.

## Synthetic evidence is not recognition accuracy

Synthetic label-preservation checks show that a renderer drew the labels it was
given and that degradations preserved class margins. They are not evidence of
recognition accuracy on real books or screenshots. Saturated synthetic metrics
are diagnostic, not real-page generalization.

Synthetic position specifications supply intended labels, not proof that a
renderer drew them correctly. Agent visual inspection supplements executable
checks; it is not human truth.

A synthetic-only bootstrap is an explicitly bounded experiment. It is not the
real-data delivery milestone, not recognition delivery, and not qualification.
The roughly 50/50 real/synthetic exposure hypothesis applies to later real
adaptation, not to a synthetic-only bootstrap, and the small existing real group
is never oversampled to manufacture that ratio.

## Qualification, development and diagnostic

These three words are not interchangeable, and mixing them is the fastest way to
produce a false claim.

| Term | What it is | What it may decide |
|---|---|---|
| **Diagnostic** | Tooling and evidence used to understand behavior: paired browser comparisons, proposal generation, synthetic calibration, audits | Nothing about quality. It reports, it does not qualify |
| **Development** | Real, source- and artwork-held-out data used to judge whether a candidate is promising | Whether to continue, refine, or abandon a candidate |
| **Qualification** | A later, untouched, human-checked evaluation set | Whether a candidate is good enough — and only after development decisions and the candidate are frozen |

Qualification must not influence development decisions, is never a proposal
scope, and is never enumerated by any automated job. All synthetic pages retain
TRAIN purpose; an internal train/development/calibration partition exists for
optimization, checkpoint selection and diagnostic calibration only, and is never
qualification.

Issue #2 dataset tooling is independent of physical laptop/iPad qualification. A
dataset CLI, a dashboard, or a proposal provider is tooling, not a delivered
training collection and not evidence of recognition accuracy.

## The one-human-pixel-review rule

Qualification truth requires a human check independent of model and agent
proposals. **One such human pixel review accepts a page.** A second reviewer is
optional and never mandatory.

The review inspects the source page, crop or grid beside a board freshly
rendered from the proposed labels, and covers actual geometry, orientation and
all 64 squares. Independent checks must additionally cover each new glyph
design, all uncertainty, and a stratified sample of accepted training data.

Report review cost and the corrections and ambiguities actually observed. Do not
report a disagreement rate derived from a single reviewer. Model agreement and
positional legality are not ground truth.

## Proposals are never truth

A proposal is a visible, editable draft and nothing more. Proposal jobs run on
TRAIN pages only. They never:

- enumerate qualification;
- change an existing annotation;
- accept a page or submit a review;
- mark a page negative;
- infer orientation, legalize a position, or invent FEN fields;
- satisfy the human or complete-page declarations.

A valid proposal may prefill only a new, untouched draft. Replacing an edited
board is an explicit, confirmed, undoable action that clears the declarations and
accumulated review time, because the changed proposal needs a fresh inspection.

The proposal-provider contract is in [assisted dataset review](assisted-review.md).

## The standing promotion comparison

The meaningful comparison is **the issue #3 candidate against unchanged
FENShot on identical inputs**, judged by human time, missed and false boards,
corner displacement and piece corrections — not by model confidence or by a
selection of successful examples.

A classical grid provider is a cheap diagnostic and fallback that receives only a
small nonqualification smoke screen. It is not a promotion contender.

FENShot stays the shipped browser default until all issue #3 quality, offline
WASM, runtime and browser gates pass. Promotion additionally requires separately
reviewed real development data, the bounded classical detector comparison,
shared inner-grid refinement, and paired end-to-end evidence. Fresh
human-checked qualification stays untouched until candidate freeze.

No real-data accuracy, qualified localization, skew support, or superiority over
FENShot is claimed.

## Local evaluation only

Evaluation is local. **No model release or publication is approved.**

- Checkpoints, ONNX files, generated manifests and raw run evidence stay in
  ignored storage and are never committed.
- The npm package remains explicitly private until a separate publication
  decision.
- The repository's own source is MIT; admitted third-party artifacts retain their
  separate licenses and notices, and admitting an artifact for local baseline
  execution certifies neither upstream training-image rights nor disjointness.
- Model publication and any public artifact mechanism require a separate rights
  review and owner approval.
- A private evaluation export is sensitive local evidence and must not be
  published. Only an aggregate summary export is designed for publication review,
  and it still needs inspection before publication.

## Where the boundaries are enforced

| Concern | Document |
|---|---|
| Agent and contributor rules | [../AGENTS.md](../AGENTS.md) |
| Current forward plan | [../PLAN.md](../PLAN.md) |
| Resource ceilings and charges | [budget.md](budget.md) |
| Pipeline shape and vocabulary | [architecture.md](architecture.md) |
| GPU controller procedure and gates | [training-runbook.md](training-runbook.md) |
| Proposal-provider contract | [assisted-review.md](assisted-review.md) |
| What a fresh clone can rebuild | [reproducibility.md](reproducibility.md) |
