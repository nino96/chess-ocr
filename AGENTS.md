# Chess OCR agent instructions

These are the repository-wide rules. Tool-specific instruction files must
point here rather than create competing policies.

These rules do not restate what already has a home. Read the owning document:

| Concern | Document |
| --- | --- |
| Every document in this repository | [docs/index.md](docs/index.md) |
| Writing and organizing prose | [documentation standards](docs/documentation-standards.md) |
| Standing claims and scope boundaries | [scope and standing claims](docs/scope-and-claims.md) |
| Resource ceilings, reservations and charges | [project ledger](docs/budget.md) |
| What a fresh clone can rebuild | [public provenance and reproducibility](docs/reproducibility.md) |

Codex sandbox specifics are in [.codex/instructions.md](.codex/instructions.md).
Delegation and token-efficient execution are in
[.agents/delegation.md](.agents/delegation.md).

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
  subagents for useful independent research/review or disjoint bounded tasks, on
  the terms in [.agents/delegation.md](.agents/delegation.md). Delegation never
  widens authority.

## Product boundary

The product boundary and every standing claim are defined once in
[scope and standing claims](docs/scope-and-claims.md): what this repository is
and is not, what the output may contain, and what the words diagnostic,
development and qualification each decide. Build to that document rather than to
a summary of it. The agent-facing obligations it implies are:

- Never widen the claim to fit what finished. An optional GB10 or cloud mode is
  deliberate and visibly selected, never a silent fallback.
- Request authority before cloud provisioning or spending, external exposure,
  paid APIs, permission outreach, or uploading private material.

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
- Reviewed public-source provenance belongs in Git, not only in ignored work
  folders. Record public URLs, immutable revisions, original/evidence SHA-256
  values, licenses/attribution, selected pages, lineage, splits and reconstruction
  recipes under `provenance/` and [docs/provenance/](docs/provenance/artifacts.md).
  Never copy an operational database or mixed public/private manifest wholesale.
  Commit only explicitly public, reviewed records; strip local paths,
  account/reviewer identity, credentials and private-derived details. Public
  access is still not permission to redistribute original assets, crops, fonts,
  datasets or weights.
- Reproducibility requires more than source links, and what a clean checkout can
  actually rebuild is recorded in
  [public provenance and reproducibility](docs/reproducibility.md). Commit
  generator code, dependency locks, public selection/seed/configuration and safe
  public annotation versions as each is delivered. If exact reconstruction
  requires unshared human annotations or private inputs, say so and preserve
  those locally; never claim full reproducibility from a plan. Validate public
  manifests and keep the payload/private-record protections intact.
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
  edits, and quarantine ambiguity. Acceptance follows the
  [one-human-pixel-review rule](docs/scope-and-claims.md#the-one-human-pixel-review-rule).
  Describe agent-only verification honestly.
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
- Before an experiment, record the hypothesis, changed inputs, reusable hashed
  evidence, command, resource ceiling, completion/stop condition and next decision
  in the owning issue or ignored local status artifact as privacy permits. One
  failed comparison triggers one bounded diagnosis, not automatic new seeds,
  model families or sweeps. Additional runs need a new evidence-based reason and
  must fit the authorized budget. Preserve complete meaningful schedules.
- [The project ledger](docs/budget.md) is the one home for every ceiling,
  reservation and charge; declare and record them there rather than in another
  document. Approvals in another repository do not automatically fund new models
  here, and no paid service or enlarged budget is inferred from this roadmap.
- Run long acquisition, generation, training and evaluation as bounded resumable
  local jobs with persisted status and owner-ready start/status/stop commands. Do
  not keep an AI turn or subagent alive merely to poll logs, sleep or narrate
  progress; hand off the running job when no independent work remains and inspect
  results on completion or resumption. New sources and rights/label decisions are
  never delegated to an unattended downloader.
- **Commit reviewed code before every unattended launch.** A background job —
  acquisition, synthetic generation, training — starts only from reviewed,
  committed code. Do not launch from a dirty working tree: when a run has to be
  explained, retried or charged, the exact bytes that produced it must be
  recoverable from git, and a job that outlives the turn cannot be reconstructed
  from an uncommitted diff.
- Stopping for a real limit is legitimate; label it blocked/incomplete rather
  than successful recognition or architecture impossibility. Do not truncate a
  meaningful training schedule solely because a mechanics pilot completed.

## Implementation and verification

- Use pnpm, matching chess-reader. Pin its version in package.json, keep
  pnpm-lock.yaml authoritative, and install with --frozen-lockfile. Do not
  introduce npm/yarn lockfiles or silently switch package managers.

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
- Run narrow checks while editing; full browser/runtime/security matrices occur
  once at the affected integration gate, not for each rejected offline seed.
  Reuse unchanged valid evidence with its hashes and original command; rerun for
  relevant changes or unresolved failures. Never weaken tests, omit required
  gates or label incomplete recognition successful to reduce usage. Record unrun
  OS/hardware gates and why; physical iPad is not simulated WebKit.
- Report commit, environment/device, commands, schema, source/model hashes and
  raw artifact references. Report distributions and coverage, not cherry-picked
  times or aggregate tile accuracy hiding confident board errors.
- Preserve owner changes and unrelated repositories. No destructive resets,
  shared-history rewrites, force pushes or bypassed checks. Review staged paths
  for payloads/private details before each commit/push.
- Keep README, PLAN and the documented command surface aligned with what is
  implemented; never claim a command works before it exists. Proposed
  functionality stays labeled proposed. Every documentation change satisfies
  [documentation standards](docs/documentation-standards.md), including its rule
  that a changed rule is edited rather than annotated with a supersession note —
  the history belongs in [docs/decisions/](docs/decisions/index.md) or
  [docs/archive/](docs/archive/index.md).
- Final handoff lists the delivered outcome, exact checks and results, unrun
  gates, delegated work and remaining limitations.
