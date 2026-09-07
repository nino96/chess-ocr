# Local dataset pipeline

For the step-by-step process and division of responsibilities, read the
[operator workflow](operator-workflow.md). This page is the command reference.

This is issue #2 tooling, independent of physical laptop/iPad qualification.
It is not a delivered training collection or evidence of recognition accuracy.
All originals, private-source details, operational review history and exported
tensors stay under ignored `work/dataset/`. The 2026-09-07 owner update permits
reviewed public-source provenance and reproduction recipes in Git, but not asset
payloads or mixed/private records. See [reproducibility](reproducibility.md).

## Dataset design

The target unit is an unfamiliar **page becoming an exact board**, not a correct
square sampled from a known font. Even excellent square accuracy can conceal
frequent board errors. Localization and reading need different evidence:

- Localization needs complete pages, including text, captions, small/multiple
  boards and reviewed no-board pages. Selecting only recognizer successes would
  hide misses and create an easy training distribution.
- Reading needs all 13 image-relative classes across different piece artwork,
  square backgrounds, print quality and scan conditions. Empty squares must
  retain borders, marks and noise that occur in real books.
- Geometry must identify the inner grid. A loose detector rectangle is not an
  accurate crop for 64-square classification. Keep four original-image corners
  and derive every tile and detector target from the same reviewed record.
- More pages from one book add positions and conditions, but not necessarily
  new designs. Reprints, shared fonts, translations and scan mirrors must stay
  together. Source group counts require actual artwork inspection.

Start with a small uniformly selected page batch from each candidate book and
inspect its yield, font family, native glyph size and difficult conditions.
Choose subsequent sources for missing appearances, not higher raw counts. Retain
real training pages alongside disjoint development and reserved qualification.
The 300–500-board first-learning target and larger reference targets in PLAN
remain targets, not guarantees. Native YOLOX and ImageNet checkpoints are starting
hypotheses; qualification must measure the complete path on unseen real books.

The [synthetic-first kickoff](dataset-kickoff.md) now permits an early synthetic
bootstrap before the large real tranche. It must not replace real
training pages, copy held-out artwork, or count as new independent sources. Bulk
synthesis and degradation are deliberately not implemented before renderer/design
fidelity and real-condition reviews. The current pipeline does not infer labels,
legalize positions, train models, or score qualification.

## Setup

Supported execution environment: **Linux, CPython 3.12, Pillow 11.1.0 and Poppler
24.02.0** (`pdfinfo` and `pdftoppm`, supplied by Ubuntu 24.04 `poppler-utils`).
The renderer version is checked before PDF page rendering. The wheel lock covers
Linux ARM64 and x86_64 only. macOS/Windows ingestion has not been validated;
this does not affect the separate browser device checks.

```sh
python3 -m venv work/dataset-venv
work/dataset-venv/bin/python -m pip install --require-hashes -r python/requirements-dataset-linux-cp312.txt
pnpm run dataset init
```

`DATASET_PYTHON` can select an already prepared Python executable. The CLI never
installs dependencies or downloads runtime assets automatically. `init` is
idempotent and starts with zero acquisition/compute/storage allocation.

The owner approved the following bounded kickoff allocation on 2026-09-07, and it
is configured in the current workspace. Apply it explicitly on a fresh workspace:

```sh
pnpm run dataset budget --sources 64 --pages 2000 --download-bytes 10737418240 --storage-bytes 68719476736 --cpu-seconds 720000 --review-limit 2000
```

This caps admitted sources and rendered pages, downloaded bytes, total local
workspace storage, conservatively reserved compute time, and up to 2,000 review
**decisions**. One human decision accepts a matching page; corrections are further
decisions. Failed/interrupted attempts retain full reservations. Inspect measured
review time and corrections/ambiguities; this is capacity, not assigned human work.
No paid services or permission outreach is included. The shared project ledger
separately bounds conditional #3 GPU bootstrap; it is not launched by this command.
The shared project ledger is [budget.md](budget.md); per-attempt records
and actual ceilings are persisted locally in SQLite.

The former 20-decision/four-hour feasibility limits are superseded. Accounting
still charges 90 seconds per rendered page even when an attempt finishes faster;
the new compute ceiling covers retries and subsequent preparation. A source ceiling
does not represent that many selected or rights-cleared books. Begin with the
smaller increments in the [budget ledger](budget.md), not all capacity at once.

## Paste PDFs and ingest

Put only PDFs you intend to contribute in `work/dataset/inbox/`. The command does
not inspect other folders, recurse into subfolders, extract archives, or upload.
Filenames are not used as dataset identities. The original inbox copies remain.

```sh
pnpm run dataset ingest --group my-book-family --reviewer owner --approve-local-use --pages-per-pdf 40
pnpm run dataset start
pnpm run dataset status
```

`--approve-local-use` records your declaration that you are authorized to use
these files for local training (or evaluation with `--split dev`). Owning a PDF
is not itself a rights determination. This is an owner authorization record,
not an automated license clearance; publication remains unapproved.

One invocation treats **all PDFs currently in the inbox as one conservative
related edition/artwork group**. Ingest one family at a time. Move already
processed inbox copies out before adding a different family. Originals remain
in managed storage. Re-ingesting identical bytes with the same group/split is
idempotent; changing that assignment is rejected. Related works must use the
same group even in later invocations. Unknown groups are explicitly unverified
and cannot be ingested directly as qualification. Human artwork review is still
needed before reporting verified independence.

By default, this selects up to 40 uniformly spaced pages across each PDF, without
consulting any detector. Use `--pages-per-pdf` to choose a larger bounded initial
sample. Source records and page selections are immutable; extending an already
admitted book's selection is not yet supported. Plan the selection before ingest.
This is a page sample, not automatic discovery of all diagrams in a book.

The worker renders only selected pages (maximum dimension 2400), preserving the
PDF original locally. Original image sources retain their encoded raster
orientation. Raster limits are 8192 per edge and 24 million decoded pixels;
nonstandard/ambiguous geometry stays outside accepted detector examples.
Low-resolution glyph information lost in page rendering must be evaluated before
training; the fixed rendering recipe is not proof that 2400 pixels is sufficient.

## Background operation and recovery

```sh
pnpm run dataset start                  # detached worker; also resumes interrupted jobs
pnpm run dataset status                 # aggregate counts, heartbeat, limits, next action
pnpm run dataset queue                  # local sample IDs, duplicate pairs, blocked job IDs
pnpm run dataset stop                   # persisted request; kills active acquisition/render group
pnpm run dataset retry 7 --after-repair  # one additional attempt after addressing the local cause
pnpm run dataset start
```

`run` is the foreground equivalent of `start`. One writer lock covers acquisition,
ingestion, review imports and export. Status/stop remain available. Decoder child
processes inherit the lock, so a crashed supervisor cannot start an overlapping
attempt while its child still runs. Attempts have CPU/memory/file-size/wall-time
limits; failed descendants are killed. Downloads are atomic and hash-checked,
with bounded retries/backoff (three initial attempts). Partial downloads restart
from byte zero under a new reservation; HTTP range resumption is not assumed.

Validation failures are quarantined immediately. Transient failures retry within
the attempt and resource ceilings. A manual repair authorizes one further attempt
without erasing history or enlarging the budget. Corrupt managed originals must
be restored to their pinned bytes, never accepted under a silently changed hash.
Quarantine status intentionally omits PDF content and third-party error text.
A stale heartbeat is evidence to inspect/resume, not proof a process is dead.

The worker exits at review, source, repair or budget boundaries. No AI process
needs to stay alive or poll it. A command returning successfully does not mean
a dataset is reviewed or recognition is qualified.

## Dashboard review (primary workflow)

Start the local dashboard after the dataset environment is prepared:

```sh
pnpm run dataset serve --port 8766
```

It binds only to `127.0.0.1`. In VS Code, open **Ports**, choose **Forward a
Port**, enter `8766`, then choose **Open Browser**. Forwarding is a local-editor
convenience, not permission to expose the service publicly; keep the forwarded
port private.

Use the dashboard queue and page thumbnails to open the next review. It autosaves
server-side drafts, lets you mark **No board** explicitly, and submits **Submit
review & next** only after the human declarations are complete. One human pixel
review, independent of any model/agent proposal, accepts a matching annotation;
corrections create a new revision requiring a human review. The dashboard provides
the existing start/stop, validation, candidate export, inbox ingestion (with its
explicit local-use checkbox), and duplicate inspection/resolution actions. It
does not upload PDFs: add new intended PDFs to `work/dataset/inbox/` through VS
Code/the local workspace, then ingest that existing inbox. An optional, explicitly
configured local candidate can create an editable on-demand proposal, but cannot
accept a page or set either human declaration; see
[local candidate testing](local-candidate.md).

### Assisted proposals

The dashboard has separate localizer and labeler selectors. A valid proposal
prefills only an untouched new draft. Once a reviewer interacts, loading or
switching providers preserves the draft and shows the proposal for comparison;
replacing an edited board is explicit, confirmed and undoable. Uncertain squares
are highlighted, but every square and the complete page still require human pixel
inspection. Deferral is recorded locally and never accepts a page.

The default pair is FENShot localization plus FENShot labels; the deterministic
classical grid localizer is a diagnostic/fallback for a small smoke screen. The
meaningful promotion comparison will be the issue #3 model versus FENShot. List
and operate the bounded detached jobs with:

```sh
pnpm run dataset proposals providers
pnpm run dataset proposals start --localizer fenshot-localizer-v1 --labeler fenshot-labeler-v1 --scope train-pending --max-pages 20
pnpm run dataset proposals status
pnpm run dataset proposals stop
pnpm run dataset proposals resume RUN_ID
```

Use `--scope train-all` or `accepted-train` only for the recorded diagnostic or
promotion-comparison need.
Qualification is not a valid scope. Runs are capped at 100 pages/two hours and
attempts reserve 45 CPU seconds in the existing ledger. Add `--after-repair` to
resume only after diagnosing a provider failure. Result visibility is bound to
the current sample revision and image SHA-256. Provider manifests, schemas,
issue #3 integration boundary and metric definitions are in
[assisted dataset review](assisted-review.md).

Use **Stop job** for the separate acquisition/export worker; `Ctrl+C` stops only
the web app. The app and worker share one writer lock, so active rendering can
temporarily block a draft save. Stop the job, then choose **Retry saving draft**;
your browser draft remains available for that retry.

## Optional standalone HTML fallback

```sh
pnpm run dataset queue
pnpm run dataset review SAMPLE_ID
# Open the printed local HTML path; annotate and export a proposal JSON.
# Save the exported JSON inside work/dataset/, not the repository source tree.
pnpm run dataset import-review work/dataset/REVIEW_FILE.json
```

The self-contained HTML uses no server, external fonts, telemetry or network
requests. It displays the whole source page, selected grid and editable 64-square
labels alongside Unicode label rendering. Unicode is a review aid with explicit
font limitations, not a certified synthetic renderer. One self-attested human
review of the pixels accepts a matching annotation. Corrections create a new
revision and need a human review of that revision. A second human review is
optional and never a page-acceptance prerequisite. Do not claim agent-generated
or model-generated output is human truth; “independent” here means independent of
the proposal/model, not necessarily a second reviewer.

All labels are image-relative row-major; `.` is empty, uppercase white, lowercase
black. Orientation may remain unknown. Page kind and complete-page review are
required. A page with incomplete/unsupported diagrams must be marked partial or
unsupported; it is excluded from detector training rather than exported as an
unlabelled negative. No missing side-to-move or other FEN fields are fabricated.

Imports check the original image hash and current revision. Late/stale reviews
are rejected, accepted edits are never overwritten, and all imported decisions
are retained. Validation requires at least one matching human decision. Repeated
submission of the same reviewer decision is idempotent and does not consume another
review decision. Old pending records with a matching human review are promoted
without new review time or budget. The assisted path never writes predictions
into accepted truth. The standalone fallback does not run providers, but it can
render an already embedded proposal.

### Confirmed Start over

The confirmation-required **Start over** action archives managed state under
ignored `work/dataset/archives/<timestamp-id>/`. It retains the inbox and approved
budget, while carrying cumulative acquisition reservations and review decision/time
use into the new ledger. Reset is not a budget refund: the archive still counts
against storage. No current real dataset has been reset, and there is no automated
restore command; recover an archive only through an operator-led local procedure.

Open **Archives** in the dashboard to see each archive's UTC date and size.
Choose **Delete archive…**, type `DELETE`, then choose **Permanently delete
archive**. This removes only that archive's recovery copy. It reclaims storage,
but does not refund cumulative download, compute or review usage, or touch the
active dataset and inbox. Cancel leaves the archive intact.

Deletion rejects changed selections, symlinks and paths outside the archive
directory, and cannot run while another dataset writer or reset recovery is
active. If deletion is interrupted or reaches its time limit, refresh the archive
list and confirm deletion of the remaining files. No automatic retry or restore
is performed. Uninspectable entries are disabled and need local operator cleanup.

Exact page hashes and perceptual page/rectified-board hashes propose duplicate
pairs. Perceptual similarity is not proof of duplication: inspect both local
review pages before recording a decision:

```sh
pnpm run dataset duplicate SAMPLE_A SAMPLE_B distinct
# Or: pnpm run dataset duplicate SAMPLE_A SAMPLE_B duplicate
```

Exact duplicate pixels cannot be declared distinct. Unresolved cross-split pairs
and detected cross-split duplicates block export. Same-split candidates need no
human decision: they remain retained, not declared distinct, and are recorded in
`same-split-duplicate-audit.json` with a within-split multiplicity warning. The
dashboard only lists cross-split candidates as actionable. Existing explicit
same-split duplicate decisions conservatively
exclude one entire page, even if only a board was repeated; this can discard useful
additional boards and is reported. Hash screening cannot certify all artwork is
disjoint. Manual lineage review is necessary, particularly across scan degradation.
New geometry/label revisions invalidate affected board-similarity decisions.

If review reveals a cross-split duplicate or unsuitable source, exclude its known
related lineage component without rewriting the original assignment or history:

```sh
pnpm run dataset exclude-source SOURCE_ID --reason duplicate-lineage
```

Excluded sources remain on disk and count against the admission/storage budget,
but leave the active review/export set. This decision is recorded permanently;
there is no automatic reassignment into another split. Inspect any other sources
with potentially shared artwork before claiming the remaining splits independent.

## Validation and export

```sh
pnpm run dataset validate
pnpm run dataset export-start           # detached, deterministic train/dev snapshot
pnpm run dataset status
```

`export` runs synchronously. `stop` also interrupts export between boards; restart
export to rebuild its deterministic partial directory. Completed snapshots are
immutable and identified by recipe/input/source/annotation/code hashes. A repeated
completed export is rejected with a reuse instruction. Original byte/source rights
integrity, label geometry, review history, unfinished acquisition and duplicate
checks run before export. Qualification and regression are never exported by this
command. Dataset-delivery readiness remains false pending the outstanding human,
coverage, lineage and training-interface gates.

Exports include page images and normalized one-class YOLO text boxes, canonical
page/board records, perspective-rectified 768-pixel grids, and 64×3×96×96
little-endian float32 ImageNet-normalized candidate tensors. The class sequence
is `.PNBRQKpnbrqk`; geometry and labels share image-relative ordering. There is
one page-level target file covering all annotated complete boards. Negative pages
have empty target files; partial/unsupported pages are represented in records but
have no detector target/image pair.

The 96-pixel tensor recipe is **proposed #3 preprocessing**, not the unchanged
224-pixel native runtime probe and not yet a promoted browser preprocessing path.
Actual native/browser parity must be checked in #3. Loose-selection augmentation,
production preprocessing integration, calibrated degradation, approved synthesis,
a qualification freeze/export protocol and the real training tranche remain open.

Coverage reports counts by split, declared connected lineage components, artwork
tags, page kind, class and checkerboard parity, plus missing class/parity cells.
Parity does not prove which printed background is light/dark. Source condition
tags are declarations, not machine-verified per-board coverage. Repeated FENs do
not create lineage edges. Report independent groups and board-level errors in #3,
not an aggregate tile score or source count alone.

## Checks

```sh
work/dataset-venv/bin/python -m unittest python/test_dataset_pipeline.py python/test_dataset_reset.py python/test_dataset_server.py python/test_dataset_proposals.py
pnpm run check
pnpm test
pnpm run build
pnpm run test:dataset-review            # requires installed Chromium
```

Tests generate original synthetic rasters/PDFs in temporary storage and make no
internet requests. Their passing demonstrates tool mechanics, not dataset quality.
No downloaded originals, their metadata, or research source lists are test fixtures.

## Repository workflows

The operational sequence is setup/budget → inbox ingestion → background rendering
→ page annotation and independent review → cross-split leakage checks → validation
→ immutable train/dev export. Use `status` and `queue` between stages; `stop`,
`start`, `retry --after-repair` and `exclude-source` cover the recovery paths above.

[Repository checks](../.github/workflows/check.yml) run source/privacy checks,
unit tests, the build and Python dataset tests, including isolated PDF rendering.
[Dataset review UI checks](../.github/workflows/dataset-review.yml) run the offline
Chromium annotation test when the pipeline, review interface or its test changes.
These CI workflows use original procedural fixtures; they do not ingest your
books, run private dataset jobs, train models or publish dataset artifacts.
Remote CI execution is separate from the local passing checks.
