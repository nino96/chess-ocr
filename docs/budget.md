# Project local resource ledger

The single home for resource ceilings, reservations and charges. Other documents
link here rather than restate a figure. What the spending may and may not be used
to claim is stated once in [scope and standing claims](scope-and-claims.md).

## Owner update — local budget discretion, 2026-09-07

The owner authorizes the lead to increase local CPU/GPU, acquisition, review and
storage ceilings when measured need justifies a reasonable bounded increment.
The figures below are planning allocations, not mandatory terminal stops. Record
the reason and new ceiling prospectively in this ledger and local operational
state; retain every prior charge and failed attempt. This is not permission for
unbounded sweeps, extravagant allocations, paid/cloud services, external exposure
or publication of asset/model payloads.

**Keep at least 30% of the dataset filesystem's total capacity free.** Check usable
free bytes before each acquisition/render/generation batch, including worst-case
temporary output. A discretionary budget increase cannot waive this free-space
floor. Stop with resumable state if the floor would be crossed; do not delete
owner data automatically to make space.

2026-09-07 extension: reserve one additional local CPU hour and 1 GiB temporary
fidelity storage (within the existing cumulative limits) for the newly requested
print-degradation and projective-page checks. The additional design/class/effect
controls validate legibility and geometry before bulk synthesis; no GPU run or
new model comparison is included. Retain earlier failed-control charges.

Modern real-layout increment: 32 MiB download/300 CPU-second preparation reservation
for the fixed 2006 Chess Wikibook PDF and rights record, with per-file caps of
16 MiB/2 MiB and 120-second transfer limits. Fixed 24-page TRAIN selection is
charged separately by the existing page worker. No new cumulative ceiling needed.

2026-09-06 implementation reservation; no paid service, provisioning, training,
external exposure or private upload. Operational experiment detail is persisted
in ignored `work/issue-1-status.md` and `work/native/status.md`.

| Work                               | Download ceiling | Local disk ceiling | Compute ceiling                   |
| ---------------------------------- | ---------------- | ------------------ | --------------------------------- |
| Browser dependencies/checks        | 1 GiB            | 3 GiB              | 30 CPU minutes; zero GPU training |
| Native checkpoint/export/env probe | 4 GiB            | 8 GiB              | 25 CPU minutes; 5 GPU minutes     |

The native reservation includes failed attempts and the lead's one <=60-second
cuDNN diagnosis. There was one export per starting model, no seed/model sweep,
and no training. Browser installations downloaded approximately 498 MiB in total.
Measured current local sizes: `node_modules` 243,024,865 bytes; `cache/native`
26,102,698; `work/native` 1,941,783,671; `artifacts/native` 15,105,509 (these sizes
precede the final small report/check files). Native wheelhouse/environment and
checkpoint acquisition stayed within the reservation; the container already
existed and was not pulled. No raw assets are checked in.

Failure record: missing pinned browser executables (installed); image-wrapper
schema rejection (fixed); cancellation test timing race (fixed); WebKit network
emulation internal error (verified actual server shutdown instead); interrupted
Vite parity harness reload (disabled watching); default cuDNN GPU mismatch
(non-cuDNN forward agreed); Vite watched the native venv (excluded ignored
workspaces); dev middleware transformed the MJS bootstrap (serve locked bytes
verbatim in development). Each failure led to a bounded relevant check, not
new training or an architecture search. Retain these outcomes with the raw
local reports when resuming; do not present failed runs as recognition success.

The issue #1 reservation does not authorize #2/#3 training, new source acquisition, model
families, enlarged downloads or any paid resources. Named-laptop measurements
need access to that device; no runtime promotion budget has been silently set
from the GB10 CPU measurements. No model token/quota measurements are available.

## Issue #2 local tooling and approved collection reservation

Owner requested a resumable dataset pipeline and local PDF inbox. Tooling checks
use original generated temporary fixtures only: no source acquisition, GPU run,
paid service or corpus publication. A separate dataset venv reuses the existing
pinned Pillow wheel locally. Reserve up to 15 CPU minutes and 256 MiB for tooling
checks; this is implementation work, not a bulk collection allocation.

The owner approved the first collection reservation on 2026-09-06: 12 sources, 2,000 selected pages,
2 GiB downloads, 8 GiB dataset storage, 4 hours conservatively reserved compute,
and an initial 20 review decisions; zero GPU/paid work. The approved ceilings
are applied to the local ledger. Fresh workspaces start at zero; use the documented
`dataset budget` command to apply this allocation. Source-specific rights and
local-use authorization remain separate admission requirements. The local SQLite
reservation ledger counts failed/interrupted attempts without refunds; it does
not reuse issue #1 or another repository's allocation. Private metadata and
operational records stay local; reviewed public provenance is now tracked under
the 2026-09-07 owner update in [reproducibility](reproducibility.md).

### Feasibility limits of this allocation

This is an initial acquisition/review feasibility allocation, not a demonstrated
budget for the complete first learning tranche or model improvement. Twelve is
an admission ceiling; no twelve-book source list has been selected or admitted.
Under the
[one-human-pixel-review rule](scope-and-claims.md#the-one-human-pixel-review-rule),
20 decisions can accept at most 20 unchanged pages; corrections consume further
decisions. The worker charges the full
90-second reservation for every rendered page, even on successful fast attempts.
Four hours therefore covers at most 160 page-render attempts before acquisition,
inspection and export charges; the 2,000-page ceiling is not achievable within
that compute allocation under current accounting. No GPU training is included.

Use the first reviews to measure diagram yield, review time, corrections/ambiguity,
artwork independence and rendering cost, and report them as
[scope and standing claims](scope-and-claims.md#the-one-human-pixel-review-rule)
requires. A complete learning tranche requires
an evidence-based follow-up allocation and separately budgeted training; approval
of these initial limits does not imply either. These feasibility limits are now
superseded by the owner-authorized kickoff below.

## Issue #2 synthetic-first kickoff — 2026-09-07

The owner approved the resources needed to start dataset creation, including a
synthetic seed to reduce manual work. Convert that approval into bounded local
increments rather than unbounded acquisition. No paid/cloud services, new external
exposure, permission outreach or private uploads are needed or reserved.

The dataset ledger's new **cumulative**, not additional, ceilings are 64 source
records, 2,000 selected real pages, 10 GiB downloads, 64 GiB dataset disk,
720,000 conservatively reserved compute seconds (200 hours), and 2,000 review
decisions. Preserve all prior charges. The review ceiling is capacity, not a request
for the owner to perform 2,000 reviews; offer optional five-minute review sessions.
The compute ceiling covers up to three 90-second attempts per real page
(150 hours) plus acquisition, export and synthetic preparation; it is not a plan
to burn 200 hours or weaken per-job limits.

Initial suballocations within those totals:

| Increment                     | Bound and stop condition                                                                                                                                                               |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Asset preparation             | Up to 16 reviewed piece designs; first download only 3 sets/36 SVGs; 16 MiB request reservation and 1,200 compute seconds for initial evidence/download/inspection                     |
| Renderer fidelity and tooling | Up to 2 CPU hours and 1 GiB; validate every admitted design, both backgrounds, all 12 pieces and independent pixel/label ordering before bulk synthesis                                |
| Synthetic seed                | Up to 20,000 board equivalents, initially 3 audited designs; 8 CPU hours and 16 GiB; retain compact page/position recipes and PNGs, generate tensors lazily                            |
| Real acquisition increment    | Up to 12 newly reviewed source families and 500 selected pages initially; source/lineage decisions before download; later increments by measured coverage benefit within global limits |
| Proposal comparison           | One bounded identical-input screen, up to 100 non-qualification pages, 2 CPU hours; assess misses and review-time benefit, not just confidence                                         |

Issue #3 may reserve up to **8 local GB10 GPU hours** for one frozen synthetic
bootstrap schedule plus its recovery check and bounded diagnosis, under this owner
approval. The 2026-09-07 owner update permits that schedule to train both the
pinned MobileNetV3 classifier and YOLOX-Nano detector; it does not add another
seed, model family, sweep or GPU time. No training starts until its data/renderer
checks, merged dependencies, exact recipe and GPU attempt ledger are ready. This
is not permission for seeds, model-family sweeps or truncating a meaningful
schedule to fit a mechanics test.
Keep this GPU reservation separate from the CPU acquisition ledger; record actual
attempts in the same project's local status artifact. No GPU time is spent by the
kickoff acquisition itself. If the measured schedule cannot fit, resize the plan
prospectively and report it rather than silently overrun.

The joint schedule further reserves at most four CPU hours and 16 GiB of ignored
training output while preserving the existing 30% free-space floor. Its allocation
is 20 GPU minutes for required preflight/recovery evidence, 100 minutes for the
classifier, 340 minutes for the detector and 20 minutes for one bounded diagnosis.
Unused time is not authority for another experiment.

The training controller reports this reservation in GPU-seconds. One GPU-second
is one second of wall time while a scheduled training container owns the GPU; it
is not a count of optimizer updates. Preflight, classifier and detector segments
each have a frozen allocation, retries consume the same segment allocation, and
the normal status view reports capacity, consumed and remaining seconds. Use
`status --history` for the complete attempt ledger.

### Corrective detector v2 reservation and actual usage — 2026-09-07

The retained run histories contain inherited attempts, so summing each run's
`gpu_seconds_charged` would double-count the same work. Deduplicating attempts by
their immutable start timestamp across all retained state files gives **9,271.1
cumulative unique GPU-seconds consumed** (9,271.076973802876 unrounded) in ten
attempt segments. This includes failed preflights, classifier attempts and the
completed detector optimization followed by export failure.

Reserve at most 600 additional preflight GPU-seconds plus 5,400 detector
GPU-seconds for the first corrected v2 run. That run completed using 13.6674
preflight plus 3,211.7127 detector GPU-seconds. Across every retained run,
`pnpm run training -- ledger --training-root work/training` deduplicates copied
legacy attempts by immutable start time and new attempts by ledger ID: **12,496.5
cumulative unique GPU-seconds consumed** (12,496.457073617727 unrounded) in 12
attempt segments. This is within the existing 28,800-second project allocation.
Retain the 24,000 CPU-second future-run ceiling and 16 GiB output ceiling; this
does not authorize a classifier rerun, new seed, model family or sweep. Ordinary
status remains current-run-only, `status --history` owns one run's inherited
history, and `ledger` owns the cross-run cumulative accounting.

The concrete sequence and acceptance boundaries are in
[dataset kickoff](decisions/2026-09-07-synthetic-first-kickoff.md). Reviewed public inventories/URLs/hashes and
selection recipes are tracked; private details and operational job state stay ignored.
