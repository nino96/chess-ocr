# Operator workflow: feasibility, collection and model training

**Current direction (2026-09-07):** follow the
[synthetic-first kickoff](dataset-kickoff.md). The initial manual feasibility stage
has served its purpose. The owner approved the bounded larger allocation in
[the ledger](budget.md); the agent starts asset/real-source acquisition and
implements audited generation and assisted review. The owner is not expected to
manually create bulk labels or resolve same-split duplicates. Historical
feasibility steps below remain command reference, not a request to repeat them.

Use this guide to see what happens next and where your input is needed. The
[command reference](dataset-pipeline.md) explains each implemented command and
its limits. Run commands from the repository root on the Linux collection host.
The pipeline and its environment are already initialized in the current workspace,
and the approved feasibility budget is configured. A fresh clone requires the
setup and budget commands in the reference.

## Who does what

| Stage | You | Agent and local tools | Exit condition |
| --- | --- | --- | --- |
| Choose feasibility inputs | Supply authorized PDFs if desired; confirm their intended local use | Research public candidates, inspect rights and propose related artwork groups and initial splits | Small, explicit source/page selection |
| Acquire and render | Optionally run ingest/start yourself | Register reviewed public manifests or ingest supplied PDFs; run the bounded background worker | Pages ready for review, or a specific repair/budget blocker |
| Review feasibility pages | Annotate pixels as a human reviewer | Generate review pages; import decisions; validate revisions, geometry and duplicates | Timed review batch and honest coverage/yield assessment |
| Approve larger collection | Approve a concrete follow-up resource and human-review allocation | Estimate costs from the pilot, identify missing designs and implement missing collection features | Approved plan for a useful real training tranche |
| Build larger collection | Review assigned batches and ambiguous cases | Repeat bounded acquisition, review imports and leakage/coverage checks | Accepted train/dev data and independently reserved qualification membership |
| Train and compare models | Approve a separate training budget and review the experiment proposal | Implement #3 training/evaluation jobs, run the frozen schedule, save resumable state and compare against FENShot | Measured advance/defer/reject decision |
| Qualify and integrate | Participate in independent qualification truth checking and physical laptop/device tests | Freeze candidate, evaluate reserved inputs, verify actual offline browser inference | Existing recognition and runtime gates pass, or remain explicitly incomplete |

You do not need to run every command yourself. You can ask the agent to start,
resume, inspect or export within the approved scope. Jobs run on the local host
without an AI turn remaining open. Status is checked on demand; the current
implementation sends no notifications and does not schedule its own human reviews.

## 1. Choose a small feasibility batch

Your immediate decisions are which optional local books to contribute and who will
perform the human pixel review. Public-source discovery can be done by the agent;
you do not have to find or upload books. The current source ceiling is twelve
admitted documents, not a list of twelve selected titles and not twelve proven
independent artwork groups. Public-source lists and rights references are now
tracked after privacy review; private lists/evidence remain local.

Start with two or three candidate design families and a small, explicit page
selection. Keep enough review allowance to cover different designs and negative
or difficult pages. Do not attempt to exhaust all twelve source slots immediately.
A sample with no diagrams is useful yield evidence; it is not a training tranche.

Before ingestion, agree on a conservative family/group identifier and initial
train/dev assignment. Related books, editions and shared artwork must stay in one
split. The local inbox defaults to train and does not admit qualification directly.
Reserve distinct candidate families for later evaluation before tuning models on
them; merely leaving pages unlabelled does not prove they are independent.

The historical 20-decision feasibility limit has been replaced by the kickoff
allocation; review capacity is not a request for more manual labeling.
There is no automatic label-proposal system yet: annotation is a real human task.
One human pixel review, independent of any model/agent proposal, accepts the page.
A second reviewer is optional and never mandatory. An agent review cannot accept
or overwrite a human-accepted annotation.

## 2. Ingest and start rendering

In the app, admission and background processing are separate actions:

1. For new local PDFs, place the intended files in `work/dataset/inbox/`, open
   **Ingest PDFs from the local inbox**, fill the group/split/page count and your
   name, confirm local-use authorization, then choose **Ingest inbox PDFs**.
   This registers the sources and selected pages; it does not start rendering.
2. Choose **Start / resume rendering** in the top toolbar. Despite its label,
   this starts the common detached acquisition worker: it downloads already
   admitted public sources and renders queued PDF pages. For already ingested
   PDFs, begin with this step; re-ingestion is unnecessary.
3. Use **Refresh status** to see progress and **Stop job** to request a stop.
   The worker continues if you close the browser, SSH connection or chat while
   the GX10 stays on. The web app can be restarted separately to check progress.

There is currently no app action that discovers public sources or admits their
manifests. For that broader collection workflow, the agent must first prepare
rights-reviewed, hash-pinned source manifests and register them with `dataset add`.
The app can then start/resume the admitted acquisition queue. Starting the worker
with no pending jobs does not search for more material: it stops at a review,
source, repair or budget boundary. The missing app controls and scale work are
recorded in the [dashboard enhancement plan](../PLAN.md#planned-dashboard-enhancements-before-larger-collection).

Equivalent CLI commands for local PDFs:

For your own PDFs, place only one related family in `work/dataset/inbox/` and run:

```sh
pnpm run dataset ingest --group family-a --reviewer owner --approve-local-use --pages-per-pdf 5
pnpm run dataset start
```

The first command records your authorization for local training use, hashes the
files, copies them into managed storage and queues up to five uniformly spaced
pages per PDF. Use `--split dev` only when that family's evaluation assignment
has been chosen before ingestion. The same command handles every PDF currently
in the inbox, so remove processed inbox copies before staging another family;
the managed originals remain. Owning a PDF does not by itself establish use rights.

The page count is an initial sampling choice, not a diagram count. Select it
before admission: the current implementation cannot extend an admitted book's
page selection. Larger collection from the same book needs an implemented and
tested append-only page-extension workflow; changing IDs or editing SQLite to
work around this would lose the intended provenance guarantees.

For reviewed public inputs, the agent prepares the local pinned manifest and
rights evidence, then uses `pnpm run dataset add LOCAL_MANIFEST_FILE`. You do not
need to author that JSON. Discovery, rights review and manifest preparation are
not automated by `start`; it only processes the explicitly admitted queue.

`start` returns after launching the worker. Keep the collection host powered on.
Closing this chat does not stop it; host shutdown interrupts it. Use `start` to
resume after the host returns. No continuous AI polling is needed.

## 3. Check progress and handle stops

```sh
pnpm run dataset status
pnpm run dataset queue
pnpm run dataset stop
```

| Reported state or finding | Your next action |
| --- | --- |
| `awaiting-sources` | Supply intended PDFs or ask the agent to prepare reviewed public candidates |
| `running` / `exporting` | Let the local job proceed; inspect status whenever useful |
| `needs-review` | Begin the offline review workflow below |
| `budget-blocked` | Ask for a usage/yield report; approve a justified follow-up allocation if needed |
| `needs-repair` / quarantined jobs | Ask the agent to diagnose the local cause; after repair use `retry JOB_ID --after-repair` and `start` |
| `stopped` / interrupted heartbeat | Inspect the reason, then use `start` when ready; a live writer prevents overlap |
| Cross-split duplicate candidates | Ask the agent to investigate leakage; same-split similarities are retained with an audit and need no action |
| `export-interrupted` | Ask for the cause, then rerun export within the remaining budget |

Do not delete the database, reset reservations, change hashes to accept corruption,
or relabel a duplicate as distinct simply to make a gate pass. Excluding a bad
source preserves its history and excludes its known related component. The command
reference describes the explicit repair and exclusion commands.

## 4. Review pages in the dashboard

Start the loopback dashboard:

```sh
pnpm run dataset serve --port 8766
```

In VS Code use **Ports** → **Forward a Port** → `8766` → **Open Browser**. Work
through the queue and thumbnails: inspect the complete page, add each complete
board or choose **No board** explicitly, set inner-grid corners, then enter all
64 image-relative labels. Uppercase means white, lowercase black and `.` empty;
preserve printed pieces even for illegal teaching positions. Mark partial or
unsupported pages accurately rather than creating a false negative.

The dashboard autosaves drafts on the local server. Complete the reviewer, human
and complete-page declarations, then choose **Submit review & next**. One human
pixel review independent of model/agent proposals accepts a matching annotation;
corrections create a new revision needing a human review. A second review is
optional evidence, never an acceptance gate. Old revision/hash submissions fail
without overwriting accepted edits.

The dashboard exposes existing start/stop, validation, candidate export, inbox
ingestion with its explicit authorization checkbox, and duplicate decisions. It
does not upload PDFs: place intended files in `work/dataset/inbox/` through VS
Code/the local workspace, then ingest that existing inbox. No automatic proposal
feature exists yet. Use **Stop job** for the separate worker; `Ctrl+C` stops only
the web app. Rendering uses the same writer lock as draft saves, so stop the job
and choose **Retry saving draft** if a save is temporarily blocked.

**Start over** is a deliberate reset, not routine cleanup: it requires typing
`START OVER`, archives managed state under ignored `work/dataset/archives/`, keeps
the inbox and approved budget, and carries acquisition/review charges forward.
It is never a budget refund, archives still use storage, and recovery is an
operator-led local procedure rather than an automated restore command.

The self-contained exported HTML is an optional offline fallback; import its JSON
with `pnpm run dataset import-review REVIEW_FILE.json` under the same stale/hash
and one-human safeguards.

On a separate laptop, transfer only the intended self-contained review HTML using
a private method you choose, then return the exported JSON to the collection host.
The HTML embeds the source image and is sensitive data. This workflow requires
no remote server or public sharing. Dataset ingestion itself is currently tested
on Linux; laptop review is separate from the still-pending physical runtime tests.

The first batch establishes review effort and corrections/ambiguities. The CLI
records per-decision timing; the agent still needs to analyze the local records to
produce the feasibility report. A disagreement rate requires a separately planned
comparison review; a zero-error tiny batch is not a quality guarantee.

## 5. Decide the larger collection budget

Automatic board proposals and piece-label prefilling are committed follow-up work
after this feasibility batch and before larger-scale annotation. The
[recorded decision](../PLAN.md#decision-automatic-annotation-proposals-after-feasibility)
requires a bounded comparison based on manual-review effort; no detector has been
selected. The agent owns implementation and tests. Your role is to review the
measured benefit/failure report and approve any additional resource allocation.
This feature remains unimplemented until that work passes its handoff gate.

After the first review batch, the agent should bring you a concrete proposal with:

- Actual rendered and reviewed pages, boards per page and diagram yield by family.
- Observed artwork independence, small-glyph readability, class/condition gaps,
  duplicates and label corrections/ambiguities.
- Measured human review time, remaining review work and who will perform it.
- Storage used and projected original/page/tensor storage, plus failed attempts.
- Sources/groups and page selections for the next increment, with explicit
  download, compute, storage and review ceilings and a stop condition.
- Any required implementation changes, including append-only page extension and
  revised compute accounting, separately identified from collection operations.

The current worker charges the full 90 seconds per rendered page. Four hours
therefore allow fewer than 160 pages after other work, despite the 2,000-page
admission ceiling. Do not extrapolate actual fast render times into a larger
queue while ignoring this enforced accounting. Any accounting change needs tests
and must retain prior attempt history.

You approve the concrete follow-up allocation; the agent can then apply it with
`dataset budget`. That command sets **total cumulative ceilings**, not additional
allowances, and requires every budget field. Existing reservations/review decisions
remain counted. Do not copy an arbitrary larger number into configuration merely
to unblock a run. The 2026-09-07 kickoff approval supersedes that initial allocation;
use the current ledger rather than repeating the feasibility-budget command.

## 6. Build and export the first useful learning tranche

The proposed starting target remains 300–500 reviewed real training boards from
at least six independent groups, with separate development and reserved
qualification groups. These counts are hypotheses for useful learning, not proof
of recognition quality. Dataset delivery also needs reviewed lineage, adequate
coverage, sound geometry and independent truth checks.

Continue acquisition/review in batches while tracking missing appearances. An
expanded budget alone does not implement source extension, automated proposals,
synthetic generation, verified per-board coverage or qualification freezing.
Those outstanding features need implementation and validation as appropriate.

When the active tranche is reviewed and duplicate blockers are resolved:

```sh
pnpm run dataset validate
pnpm run dataset export-start
pnpm run dataset status
```

The current exporter creates immutable candidate train/dev snapshots, including
page targets, rectified grids and proposed 96-pixel classifier tensors. It does
not export qualification or establish production preprocessing parity. Preserve
its hashes for #3. Keep generated data and private/operational metadata outside Git;
commit reviewed public provenance and reconstruction recipes.

## 7. Approve and run model training — planned, not implemented

Issue #3 now provides the separate `pnpm run training -- init/start/status/stop`
command surface for its first frozen synthetic bootstrap. It does not run through
the dataset CLI and does not imply that later real adaptation is ready. You will
not need to invent model hyperparameters or continuously supervise logs. Before
initializing any later experiment, the agent must prepare a reviewable plan that:

1. Freeze the accepted data/splits, starting checkpoints, preprocessing, task
   heads, recipe, sampling, seed, complete schedule and checkpoint-selection rule.
2. Define the paired comparison against unchanged FENShot on identical real
   development pages, including localization, exact-board errors, confident
   errors and clean-case regressions. Keep qualification untouched.
3. Specify a concrete GPU/CPU/storage ceiling that covers the full schedule,
   validation and failure allowance. Dataset acquisition approval does not fund it.
4. Implement and test resumable checkpoints with optimizer/scheduler/RNG state,
   bounded local start/status/stop commands and actual preprocessing parity.

You approve that experiment allocation. The agent runs the bounded job and hands
back the status commands; it does not keep an AI turn open to poll training.
Afterward it presents the measured outcome and any narrowly justified diagnosis.
Further experiments require an evidence-based reason within the authorized scope.

## 8. Qualify the candidate and test the product

Only after candidate selection is frozen do we evaluate independently checked,
reserved qualification inputs. A training run, larger corpus or classifier-only
accuracy gain does not demonstrate end-to-end improvement. The existing #3
recognition gates remain in force; they are not relaxed to fit this pilot.

Your involvement here is independent truth review where assigned, approval of any
new resource allocation, and testing the actual browser experience on your named
laptop/device. The agent owns integration, native/ONNX/browser checks and reporting
all remaining limitations. Physical iPad evidence stays separate from simulated
WebKit tests. No model or dataset publication is authorized by this workflow.

## Local jobs versus GitHub workflows

[CI source/dataset checks](../.github/workflows/check.yml) and
[offline review UI checks](../.github/workflows/dataset-review.yml) validate tooling
with original synthetic fixtures. They do not run your private collection or train
models. Your actual jobs run on the collection host and use ignored local storage.
Committing or pushing tooling does not upload the inbox, source manifests, review
files, research source lists, databases, exports or model artifacts.
