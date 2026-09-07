# Continue dataset creation in a fresh session

Current implementation update: the renderer, perspective/print effects and
independent fidelity checks are delivered; use the status commands before doing
anything else. Do not repeat acquisition of the four verified local PDFs or the
36 SVG assets. The provider-separated proposal framework and focused editor are
delivered; do not rebuild them. Remaining work includes corpus-result validation,
source-diverse real TRAIN/DEV/qualification preparation, the bounded paired
issue-#3-model-versus-FENShot assisted-review comparison, leakage investigation
and the recorded
training/evaluation gates. The older
kickoff prompt below describes the full objective, not a reason to rebuild tools.

Paste this into a new session opened in this repository:

```text
Continue issue #2 in /home/niyam-gb10/code/chess-ocr using the selected primary model.
Read AGENTS.md, README.md, PLAN.md, docs/dataset-kickoff.md, docs/budget.md,
the owning issues/dependencies, docs/reproducibility.md,
provenance/public-bootstrap.json and ignored work/dataset/bootstrap/ handoff records.
Inspect current Git/worker state once; preserve all user changes and accepted labels.
The original 12-page/16-board seed was lost in the documented test-isolation
incident. The owner waived recovery and authorized continuing without it. Never
represent replacement proposals as recovered human truth. Preserve prior charges.

The owner authorized synthetic-first dataset creation and adequate local resources.
Concrete cumulative ceilings and suballocations are in docs/budget.md; prior usage
must remain charged. Do not ask again for this recorded allocation. No paid/cloud
services, private upload, permission outreach or asset/model publication is part
of the plan. Reviewed public metadata publication is authorized.

The owner will not manually label the bulk corpus or adjudicate duplicates. Keep
same-split duplicate candidates and existing labels as-is with audit; investigate
cross-split leakage yourself. Never convert model proposals to human acceptance.

Next deliverable: implement and independently validate the deterministic, bounded,
resumable synthetic renderer using the locally pinned approved asset allowlist;
then launch the first <=20,000-board synthetic seed under recorded limits. Reuse
eligible generator components after license/security/class-order review, but never
run a catch-all upstream asset fetcher. Original assets/private metadata stay ignored;
commit reviewed public provenance, generator recipes and reproducibility evidence.
Every glyph design, color/background, geometry and pixel/label ordering must pass
fidelity checks before bulk generation. Synthetic labels then need no per-board
human entry. Keep real accepted data and qualification separate from synthetic data.

In parallel where useful, prepare source-diverse real acquisition increments under
the kickoff plan; start admitted queues without waiting for manual labeling. Do not
pad diversity with shared fonts or treat acquired pages as accepted truth.
Use the implemented FENShot/classical provider screen only for a small diagnostic
smoke; do not spend human review effort selecting a winner or expose qualification
pages. The meaningful promotion comparison is the issue #3 model versus FENShot.
Issue #3 started from merge commit
`977d3ab40187203d43a2c485fd1a3adc89e3e174`; after its model contract merges, add
its fixed localizer/labeler adapters through `docs/assisted-review.md`, without
editing its concurrent worktree from issue #2. If necessary, a frozen synthetic
bootstrap in issue #3 is authorized before the large real tranche, subject to
merged dependencies and training gates. No model/seed sweeps. Human interaction
should become optional short correction/confirmation sessions, not manual creation.

Implement, test and launch the next bounded job rather than rewriting the plan.
Do not keep AI turns/subagents alive to poll it. Hand off exact commands, hashes,
status, results/limitations and next action. Do not claim a full training corpus,
accuracy improvement or recognition qualification before its gates actually pass.
```
