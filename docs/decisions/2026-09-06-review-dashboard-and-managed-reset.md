# Review dashboard and managed reset

Status: active

Owner decision, 2026-09-06.

The loopback dashboard is the primary dataset-review workflow: it presents the
local queue/page thumbnails, retains server-side drafts, submits human reviews and
advances the queue. One human pixel review independent of any model/agent proposal
accepts a matching annotation; a second review is optional for every split,
including qualification. This does not relax the frozen #3 recognition metrics or
the requirement that qualification truth be independently human checked.

The confirmation-required **Start over** action archives managed state under
ignored `work/dataset/archives/<timestamp-id>/`, retains the inbox and approved
budget, and carries cumulative acquisition/review use forward. An archive remains
storage usage; reset is not a budget refund and has no automated restore command.
The **Archives** panel lists dated sizes and supports explicitly confirmed permanent
deletion of one unchanged archive. Deletion reclaims storage without changing
active data, inbox files or cumulative usage; unsafe paths and links are rejected.
These are local workflow controls, not a real collection, qualification result or
browser-test claim.
