# Dashboard enhancements before larger collection

Status: active

Owner direction, 2026-09-06; all work stays in issue #2. The dashboard gains the
focused queues and assisted-review framework described below. Successful
synthetic workflow tests still do not establish large-collection throughput.
The bullets below are delivery work, not approval for more
acquisition/compute/reviews:

- **Annotation assistance:** the provider framework, bounded jobs, editable
  proposal UI and metric capture are in scope. The 2026-09-07 four-page
  smoke check found false/missed boards in both current localizers; this is
  recorded as negative diagnostic evidence, not a winner-selection gate. Reserve
  the measured human promotion comparison for the issue #3 candidate versus
  unchanged FENShot.
- **Geometry-triggered label re-read:** deferred until a reviewed label adapter
  rectifies arbitrary four-corner grids. FENShot's unchanged axis-aligned tile
  preprocessor must not be silently applied to a human-edited perspective grid;
  the required stale-result, budget and edit-preservation behavior is recorded in
  [assisted dataset review](../assisted-review.md).
- **Focused review queues:** proposal/deferred/ambiguous/pending/accepted filters,
  deferral and optional five-minute progress. Source/condition coverage-gap
  prioritization must not silently exclude negatives, difficult pages or misses.
- **Concurrent acquisition and review:** remove the whole-job writer-lock conflict
  so rendering does not block saving reviews. Retain transactional revision guards,
  request cancellation/recovery, bounded concurrency and budget accounting. Test
  simultaneous saves, worker stop/resume, reset exclusion and stale results.
- **Append-only page selection:** allow additional explicitly selected pages of an
  admitted PDF without changing source identity, prior annotations or frozen split
  membership. Preview incremental costs, enforce remaining limits, and make retry
  idempotent; test restart/recovery and duplicate selection.
- **Acquisition control in the app:** clearly distinguish local PDF admission,
  source discovery/rights review, and execution of an admitted source queue.
  Add a UI for reviewing and admitting prepared public-source manifests with their
  pinned hashes and rights evidence, plus queued/running/failed counts, progress,
  remaining budget, actionable stop reasons and explicit repaired-job retry.
  This does not authorize an unattended crawler, new rights decisions or spending.
- **Measured scale gate:** under a recorded bounded test allocation, measure queue
  responsiveness, draft-save latency, memory and recovery with representative
  larger queues. Agree acceptable limits before the run and report distributions,
  dataset/code hashes and untested conditions; use original synthetic fixtures
  for tooling load tests and authorized real reviews for human-throughput evidence.

Use the initial manual reviews to estimate annotation effort. The owner approved
the larger bounded [kickoff allocation](../budget.md#issue-2-synthetic-first-kickoff--2026-09-07);
reset/archive deletion cannot replenish it. Deliver these usability/scale checks
before claiming readiness for hundreds of boards. This roadmap does not complete
issue #2's real dataset outcome.
