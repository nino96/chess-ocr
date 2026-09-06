# Issue #1 local resource ledger

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

This ledger does not authorize #2/#3 training, new source acquisition, model
families, enlarged downloads or any paid resources. Named-laptop measurements
need access to that device; no runtime promotion budget has been silently set
from the GB10 CPU measurements. No model token/quota measurements are available.
