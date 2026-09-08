# Automatic annotation proposals after feasibility

Status: active

Owner direction, 2026-09-06; implementation decision recorded 2026-09-07; owned
by issue #2. Automatic board proposals and piece-label prefilling are delivered
as the bounded, provider-separated framework in
[assisted dataset review](../assisted-review.md).

Owner supersession, 2026-09-07: do not spend human review effort selecting a
classical-versus-FENShot winner. Keep unchanged FENShot as the current baseline
and the classical localizer as a cheap diagnostic/fallback. Use only a small
nonqualification smoke screen to validate that path. The consequential paired
human promotion comparison is the new issue #3 model versus FENShot on identical
real development pages; retain the same time, miss, false-board, geometry and
piece-correction measures.

- Use a small bounded screen to smoke-test the classical multi-board grid detector
  against the existing FENShot path. Preserve full-page negatives, missed boards,
  small/multiple boards and difficult geometry; do not select only successes or
  treat this smoke evidence as a provider promotion decision.
- Choose localization separately from square-label proposals. The unchanged
  chess-trained FENShot classifier is an initial label-prefilling candidate,
  not a mandated detector or teacher. The unadapted YOLOX/ImageNet checkpoints
  are not assumed to provide useful chess annotations.
- The default pair is FENShot localization plus FENShot labels; the classical
  grid provider is a switchable localizer. Providers are immutable validated
  manifests, not arbitrary code plugins. A valid result prefills only an
  untouched draft; switching providers preserves edits and uses an explicit,
  undoable per-board replacement.
- Evaluate the new issue #3 proposal providers against FENShot by total human
  annotation time, missed and false board proposals, grid corrections and piece
  corrections on identical real development pages. Record a measured
  advance/defer/reject decision rather than assuming the new model wins.
  Qualification inputs remain untouched.
- Implement bounded resumable proposal jobs, source/model/preprocessing identity,
  editable candidate outlines and labels, and preservation of human corrections
  across retries, geometry edits and late results. Provide executable tests and
  documented start/status/stop/review commands before larger-scale use.
- Predictions remain proposals: confidence cannot replace independent human
  verification or admit labels automatically. One human pixel review, independent
  of the proposal/model, accepts a matching annotation; a second review is
  optional, never mandatory. Keep all input-specific evidence
  and private-derived metadata local; commit reviewed public recipes/provenance.
  Freeze the comparison limits within an approved
  allocation; this decision does not increase the current collection/GPU budget.

Handoff gate: after the issue #3 adapter is available, report its measured
review-time benefit and failure coverage against FENShot, or the specific reason
it does not advance. The classical smoke is an implementation check, not a
blocker or winner-selection gate for larger annotation.

Issue #3 training proceeds independently from merge commit
`977d3ab40187203d43a2c485fd1a3adc89e3e174`. Its future trained localizer and
labeler integrate through the fixed manifest and adapter contracts after their
exact preprocessing/output contract is merged; issue #2 does not edit the
concurrent training worktree.
