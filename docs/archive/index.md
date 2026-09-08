# Archive

**Nothing in this directory is current.** Every file here is a record: what was
observed on one host, at one commit, on one date. Records are immutable. They are
not edited to stay true, they are not corrected in place, and they must never be
read as instructions for what to do today.

A correction to a record is written in a newer record, never over the old one.

Live documentation may **cite** these files as evidence for a claim. It must not
depend on them: a reader who never opens this directory should still be able to
do the work. If you find yourself needing a fact from here in order to run
something, that fact belongs in a reference document instead — see
[documentation standards](../documentation-standards.md).

Most of the raw evidence these records point at lives under `work/`, which is
gitignored: local-only evidence, not reproducible from this repository.

| Date | Record | What it is |
| ---------- | ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| 2026-09-08 | [V2 evaluation work — PRs #10 and #11](v2-evaluation-changelog-2026-09-08.md) | Changelog for the two pull requests that corrected the v2 audit and made the frozen candidate browser-testable. |
| 2026-09-07 | [Assisted-review bounded diagnostic smoke](assisted-review-smoke-2026-09-07.md) | Four-page TRAIN smoke of FENShot and classical localization; negative for both, plus the manifest-hash migration note. |
| 2026-09-07 | [Dataset session-bootstrap prompt](dataset-next-session-2026-09-07.md) | A stale prompt for continuing dataset work in a fresh session; superseded, self-contradictory, machine-specific. |
| 2026-09-07 | [Dataset test-isolation incident](dataset-incident-2026-09-07.md) | Unittest discovery imported two copies of the pipeline; the archive-deletion test destroyed the accepted seed and its recovery archive. |
| 2026-09-07 | [Issue #3 synthetic bootstrap run log](issue-3-attempt-log.md) | The v2 corrective triage, the completed corrective detector run and its evaluation correction, and every failed preflight attempt with its charge. |
| 2026-09-07 | [Synthetic renderer prelaunch validation](synthetic-prelaunch-validation-2026-09-07.md) | The passing prelaunch fidelity figures, the renderer and report SHA-256 pins, and one host's batch timing. |
| 2026-09-06 | [Issue #1 implementation evidence](issue-1-evidence-2026-09-06.md) | The `ab60d74` parity tables, runtime distributions, CI-scope validation, the 2026-09-08 paired-browser timings and the pnpm migration. |
| 2026-09-06 | [Native ARM64 export and GB10 container run](native-arm64-run-2026-09-06.md) | Export identities and hashes for both starting checkpoints, and the pinned-container inference check that failed on the default cuDNN path. |
| 2026-09-06 | [Dataset review app evidence](dataset-review-evidence-2026-09-06.md) | Four stacked generations of the review dashboard's validation counts, from the first delivery through the assisted-review framework. |

## Where the live documents are

| Superseded record | Read instead |
| ------------------------------------------------------------------ | ---------------------------------------------------------------------------- |
| [V2 evaluation changelog](v2-evaluation-changelog-2026-09-08.md) | [Architecture](../architecture.md), [local candidate testing](../local-candidate.md) |
| [Issue #3 run log](issue-3-attempt-log.md) | [Training runbook](../training-runbook.md) |
| [Issue #1 evidence](issue-1-evidence-2026-09-06.md) | [Issue #1 outstanding gates](../issue-1-evidence.md) |
| [Native ARM64 run](native-arm64-run-2026-09-06.md) | [Native runtime](../native-runtime.md) |
| [Prelaunch validation](synthetic-prelaunch-validation-2026-09-07.md) | [Deterministic synthetic dataset](../synthetic-dataset.md) |
| [Assisted-review smoke](assisted-review-smoke-2026-09-07.md) | [Assisted dataset review](../assisted-review.md) |
| [Review app evidence](dataset-review-evidence-2026-09-06.md) | [Operator workflow](../operator-workflow.md) |
| [Session-bootstrap prompt](dataset-next-session-2026-09-07.md) | [PLAN.md](../../PLAN.md), [AGENTS.md](../../AGENTS.md) |
| [Test-isolation incident](dataset-incident-2026-09-07.md) | [Dataset pipeline](../dataset-pipeline.md), [AGENTS.md](../../AGENTS.md) |
