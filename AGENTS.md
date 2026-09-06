# Chess OCR agent instructions

These are the repository-wide rules. Tool-specific instruction files must
point here rather than create competing policies.

## Read first and own the outcome

Read the assigned issue, its dependencies and acceptance criteria, README.md,
PLAN.md, and all relevant accepted architecture/evaluation decisions before
editing implementation. The issue owns delivery scope; reviewed decisions
own constraints. Surface conflicts before changing either.

- Keep four outcome-focused workstreams: browser baseline (#1), dataset (#2),
  offline recognition (#3), optional server (#4). Keep batches, seeds, bug fixes
  and incremental reports in their owning issue. Create another issue only
  for a genuinely independent outcome with owner agreement.
- Work on one assigned issue; merge required interfaces/dependencies before
  dependent integration. Independent research may proceed as stated in issues.
- A downloader, smoke test, checkpoint, completed failed run or large corpus
  does NOT complete the recognition outcome. Continue to the assigned delivery
  criterion; when blocked, report the specific incomplete outcome and authority
  needed. Never redefine success to match what happened to finish.
- Prefer bounded evidence over an architecture survey. Starting choices are
  hypotheses, not mandatory models to rescue. Escalate changes with measured
  reasons; do not silently start sweeps or add model families.
- Primary agent owns design, integration, review and final validation. Use
  subagents for useful independent research/review or disjoint bounded tasks;
  one writer per file, no worker Git/branch changes without lead coordination.
  Prefer a capable lower-cost available model for bounded work; do not pin model
  IDs or provider settings in shared policy. Delegation never widens authority.

## Product boundary

- Build a standalone recognition library with a small browser demo and an
  optional server adapter. Do not recreate a PDF/EPUB reader, chess engine,
  accounts or study database. chess-reader consumes a versioned contract.
- First target: printed 2D diagrams from pages/selections, not physical 3D boards.
- Required default: offline ONNX Runtime Web WASM CPU inference in a worker.
  WebGPU is optional acceleration, never a substitute for WASM acceptance.
- The complete path is page/selection -> board localization -> inner-grid
  refinement/rectification -> piece placement -> editable, uncertain output.
  Exact-crop accuracy cannot establish localization or end-to-end success.
- Preserve source-image geometry and 64 image-relative labels/probabilities.
  Orientation may be unknown. Do not invent side-to-move, castling, en passant
  or counters; do not alter visible pieces to satisfy chess legality.
- User edits survive late/out-of-order results, retries and backend switches.
- Optional GB10/cloud mode is deliberate and visibly selected, never silent
  fallback. No upload without explicit informed consent for that input/endpoint.
  Request authority before cloud provisioning/spending, external exposure,
  paid APIs, permission outreach or uploading private material.

## Assets, privacy and licensing

- NEVER commit downloaded training originals, PDFs/EPUBs, extracted pages,
  crops, glyph/font assets, augmented images, tensors, datasets, checkpoints,
  ONNX/native model binaries, runtime caches, credentials or generated runs.
  Keep them under ignored data/cache/work/artifacts locations. No force-add;
  no Git LFS workaround. Release assets require separate artifact-specific
  rights/privacy review and an explicitly approved publication mechanism.
- Check in source/tooling, locks, recipes, factual public annotations and
  aggregate evidence only after privacy review. Tiny ORIGINAL synthetic test
  fixtures may be tracked under fixtures/synthetic with provenance/hash/expected
  output records. Downloaded assets do not become original synthetic fixtures.
  Any other fixture exception needs explicit reviewed policy/ignore changes.
- Private diagnostic inputs never enter training without explicit approval.
  Do not publish their identities, names, paths, contents, images, derivatives,
  FENs or results in Git, issues, PRs, logs or attachments. Do not read unrelated
  local books or search their names to perform a publication check.
- Source acquisition, training/evaluation, image redistribution and model
  publication are separate decisions. Public access is not a license. Applicable
  open licenses need not explicitly say AI. Preserve attribution, notices,
  exclusions and ShareAlike/source obligations; never apply the code license
  automatically to data/fonts/weights.
- Pin exact URL/revision, byte SHA-256, license evidence and review date BEFORE
  admission. Hashes establish identity, not permission or label correctness.
  Unknown pretrained inventories/lineage remain unknown, not certified disjoint.
- No telemetry, runtime CDN or automatic third-party forwarding. No request-body,
  image, position, filename/path or credential logs. Server defaults to loopback,
  no retention; remote exposure requires authentication/TLS and explicit origins.
- Treat image/PDF/archive/config/model metadata as untrusted: bound bytes,
  dimensions, decoded pixels, time, paths, archive expansion and concurrency.
  Reject traversal/symlinks where appropriate and never load untrusted pickle.

## Data and experiments

- Real pages are the reference distribution and real diagrams belong in TRAIN.
  Sample by new design/condition coverage, not recognizer success or raw counts.
- Record document/related edition/artwork lineage, original parent, sequence,
  repeated placement and perceptual duplicates. Keep dependent images and held-out
  artwork out of training, including synthetic derivatives. Common FEN alone is
  an audit dimension, not an edge collapsing unrelated works into one group.
- Notation, PDF glyph placements, legality and model agreement propose/check
  labels; they do not prove them. Compare actual pixels with rendered labels,
  inspect every new glyph family, audit accepted labels independently, preserve
  edits, and quarantine ambiguity. Qualification truth needs independent human
  checking; describe agent-only verification honestly.
- Verify render fidelity BEFORE bulk synthesis. Transform geometry with whole
  pages; crop jitter and degradation must preserve labels or be quarantined.
  Evaluate real-only reference sets; report synthetic stress tests separately.
- Reuse valid prior evidence by input/code/model hashes. Never rewrite frozen
  experiments, alter test membership after exposure or rerun unchanged failed
  comparisons merely to demonstrate activity.
- Before GPU runs freeze dataset, recipe, effective source/class exposure,
  sampling, seed(s), schedule/update counts, checkpoint selection, confidence,
  augmentation, hashes and budgets. Record all attempts and full curves.
- Small-group oversampling and class weighting require explicit justification.
  Starting native checkpoint fidelity does not prove a learning recipe is good.
  Control normalization/BN state and verify actual image-label/tensor ordering.
- Keep one project budget ledger; approvals in another repository do not
  automatically fund new models here. Declare source/download/page/CPU/storage,
  active review and GPU ceilings, plus per-run reservations and failed attempts.
  No paid service or enlarged budget inferred from this roadmap.
- Run long acquisition as bounded resumable local jobs with persisted status
  and owner-ready start/status/stop commands. No AI polling while waiting. New
  sources and rights/label decisions are not delegated to an unattended downloader.
- Stopping for a real limit is legitimate; label it blocked/incomplete rather
  than successful recognition or architecture impossibility. Do not truncate a
  meaningful training schedule solely because a mechanics pilot completed.

## Implementation and verification

- Use strict TypeScript and validated schemas across browser/server boundaries.
  Keep preprocessing, decoding, geometry and contract tests shared/versioned.
  Heavy work is bounded/cancellable and stale results rejected by request ID.
- Native training, ONNX export and actual browser parity need separate checks.
  Preserve trainable checkpoints with optimizer/scheduler/RNG state and verify
  recovery for the real stochastic recipe, not just a deterministic special case.
- Pin dependencies and model inputs; keep Python/GPU tooling out of the browser
  runtime. Platform-specific wheel locks are not universally portable.
- Every behavior change needs executable tests, including relevant failure,
  cancellation, timeout, corrupt input, stale output, reload and recovery paths.
  Add keyboard/touch/accessibility alongside the demo, not as final cleanup.
- Tests use local synthetic/approved fixtures, never the public internet.
  Do not add green placeholders, empty suites, unconditional skips, weakened
  assertions or silently relaxed accuracy/runtime thresholds.
- Bootstrap currently has NO application scripts/locks. #1 introduces real
  setup/check/test/eval commands and CI. Never claim a command works before it
  exists. Documentation-only changes require link, whitespace and scope checks.
- Run narrow checks frequently. Full browser/runtime/security matrices occur
  at affected integration gates, not for each rejected offline seed. Record
  unrun OS/hardware gates and why; physical iPad is not simulated WebKit.
- Report commit, environment/device, commands, schema, source/model hashes and
  raw artifact references. Report distributions and coverage, not cherry-picked
  times or aggregate tile accuracy hiding confident board errors.
- Preserve owner changes and unrelated repositories. No destructive resets,
  shared-history rewrites, force pushes or bypassed checks. Review staged paths
  for payloads/private details before each commit/push.
- Keep README/PLAN and the implemented command surface current. Proposed
  functionality stays labeled proposed. Final handoff lists delivered outcome,
  exact checks/results, unrun gates, delegated work and remaining limitations.
