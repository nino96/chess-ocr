# Dataset review app evidence

2026-09-06; issue #2. Implemented against base commit
`9ae98fdae04be05384ee0eec08c0dd63e9c457e0` in the working tree. This is tooling
delivery, not dataset or recognition qualification.

The loopback dashboard provides page thumbnails, direct review submission,
saved drafts, navigation, no-board/partial/unsupported classification, existing
job and export controls, inbox admission and duplicate decisions. One human
review accepts a page; stale edits are rejected and lost-response retries do not
consume another review decision. Confirmed reset archives the active dataset,
retains the inbox and carries cumulative budget usage into the new generation.

Validation on Linux ARM64 GX10, Node 24.19.0, pnpm 11.11.0, Python 3.12.3 and
the pinned Pillow 11.1.0 dataset environment:

- `pnpm run check`: passed strict types, formatting, source/privacy and links.
- `pnpm test`: 16 passed.
- `pnpm run build`: passed.
- `work/dataset-venv/bin/python -m unittest python/test_dataset_pipeline.py python/test_dataset_reset.py python/test_dataset_server.py`: 35 passed.
- `pnpm run test:dataset-review`: both connected and standalone browser tests passed.
- `git diff --check`: passed.

Browser evidence includes reload recovery, concurrent-tab conflicts, failed-save
navigation protection, retry after a committed response is lost, one-review
acceptance, negatives and partials, confirmed reset, keyboard controls and
Chromium touch emulation. HTTP tests cover session/origin/host guards, request
limits, corrupt/symlink inputs, draft validation and stale submissions. Reset
tests cover interruption recovery, lock refusal and retained budget enforcement.
All inputs are original synthetic temporary fixtures; no real collection was reset.

Local raw output, implementation SHA-256 identities, schemas and handoff are in
ignored `work/evidence/dataset-app-validation.txt`. No model or source artifacts
changed. GitHub issues [#2](https://github.com/nino96/chess-ocr/issues/2) and
[#3](https://github.com/nino96/chess-ocr/issues/3) now explicitly require one human
review, with an optional second reviewer; recognition metrics remain unchanged.

Two bounded Terra workers implemented acceptance/reset helpers, regression tests
and documentation. The lead owned the app, integration and final review/checks.
No measured model token/quota accounting is available.

Physical devices, the owner's actual SSH browser session, and Firefox/WebKit
dataset UI remain untested. Chromium emulation is not physical iPad evidence.
Core inference is unchanged, so its full runtime matrix was not rerun. Browser
PDF upload and automatic annotation proposals remain unimplemented; archive
restoration needs operator handling. See the [operator workflow](operator-workflow.md).

## Archive management follow-up

The Archives panel now lists UTC dates and sizes, and deletes one selected archive
after the owner types `DELETE`. Metadata fingerprints reject changed selections;
directory-relative operations reject traversal and links. Deletion preserves the
active dataset, inbox and cumulative ledger. Interrupted deletions can be resumed
by refreshing the list and confirming the remaining archive. No owner archive was
deleted during implementation.

Validation on the same environment: `pnpm run check` and `git diff --check` passed;
the dataset Python command above now passes 40 tests, including unsafe paths,
stale selection, lock/reset-recovery refusal, deletion timeout and partial-failure
recovery. Both `pnpm run test:dataset-review` tests passed, including archive
listing, cancel, confirmation, deletion and reload. The lead handled this bounded
follow-up without delegation. Inference/OS/device gates remain unchanged and were
not rerun. Local hashes and handoff: `work/evidence/archive-management.txt`.

## Synthetic-first kickoff and duplicate policy — 2026-09-07

Owner requested preserving accepted labels without manual duplicate adjudication.
Same-split unresolved candidates now remain in train/dev exports with a hash-bound
`same-split-duplicate-audit.json`; they are not marked distinct. Cross-split
unresolved/confirmed pairs still block export. Existing explicit exclusions remain
unchanged. The dashboard actionable list is cross-split only; status reports
retained same-split candidates separately. Qualification gates were not relaxed.

On Linux ARM64/GX10, Node 24.19.0, pnpm 11.11.0, CPython 3.12.3 and the locked
Pillow 11.1.0 dataset environment:

- `work/dataset-venv/bin/python -m unittest python/test_dataset_pipeline.py python/test_dataset_reset.py python/test_dataset_server.py`: 43 tests passed.
- `pnpm run test:dataset-review`: both Chromium workflow tests passed.
- `pnpm run check`, `pnpm test` (16 tests), `pnpm run build`: passed.
- Final test-organization cleanup: server suite rerun, 7 tests passed; no checks removed.
- Documentation links and `git diff --check`: passed. Initial sandbox-only Git
  subprocess EPERM failures were rerun successfully with scoped host permission.

The live loopback dashboard was restarted and its queue verified. An existing
reviewed candidate export completed with annotation/review-history hashes unchanged;
all exported file hashes verified. A first sandbox-scoped detached export stopped
before completing a page; the host-level restart succeeded, retaining both attempt
reservations. Exact input identities, assets, manifests and operational outcome
remain in ignored `work/dataset/bootstrap/`, not this public evidence report.

Initial rights-reviewed asset acquisition and static SVG/contact-sheet inspection
are complete locally. This is not independent renderer/extraction fidelity, bulk
synthetic generation, trained improvement or a delivered real tranche. Those next
gates are explicit in [the kickoff plan](dataset-kickoff.md). No GPU training ran.
Core inference was unchanged; full inference/browser/physical-device matrices
were not rerun. A bounded research worker gathered primary references and a coding
worker implemented the duplicate policy/tests; the lead owned decisions, integration,
asset admission and final checks. No measured model token/quota usage is available.
