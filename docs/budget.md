# Project local resource ledger

2026-09-06 implementation reservation; no paid service, provisioning, training,
external exposure or private upload. Operational experiment detail is persisted
in ignored `work/issue-1-status.md` and `work/native/status.md`.

| Work | Download ceiling | Local disk ceiling | Compute ceiling |
| --- | --- | --- | --- |
| Browser dependencies/checks | 1 GiB | 3 GiB | 30 CPU minutes; zero GPU training |
| Native checkpoint/export/env probe | 4 GiB | 8 GiB | 25 CPU minutes; 5 GPU minutes |

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
not reuse issue #1 or another repository's allocation. Sources, metadata and
per-input reports stay local under the owner's stricter privacy instruction.

### Feasibility limits of this allocation

This is an initial acquisition/review feasibility allocation, not a demonstrated
budget for the complete first learning tranche or model improvement. Twelve is
an admission ceiling; no twelve-book source list has been selected or admitted.
One human review accepts a matching page, so 20 decisions can accept at most 20
unchanged pages; corrections consume further decisions. The worker charges the full
90-second reservation for every rendered page, even on successful fast attempts.
Four hours therefore covers at most 160 page-render attempts before acquisition,
inspection and export charges; the 2,000-page ceiling is not achievable within
that compute allocation under current accounting. No GPU training is included.

Use the first reviews to measure diagram yield, review time, corrections/ambiguity,
artwork independence and rendering cost. A disagreement rate requires a separately
chosen comparison review; it cannot be inferred from one reviewer. A complete
learning tranche requires
an evidence-based follow-up allocation and separately budgeted training; approval
of these initial limits does not imply either. The approved limits remain unchanged.
