# V2 evaluation work — PRs #10 and #11, 2026-09-08

Record of the work merged by PRs #10 and #11 on host gx10-b210, merge commit
`ee97d0a`, 2026-09-08. Nothing in this file is current. It is a changelog kept as
evidence of what those two pull requests changed and what they had demonstrated
at that date.

The durable half of the original guide — the vocabulary, the recognition
pipeline, the two diagnostic modes, the reference-first rule, the two export
formats, the dashboard integration and the subsystem-to-file map — now lives in
[architecture](../architecture.md). The operating commands are in
[local candidate testing](../local-candidate.md) and
[assisted dataset review](../assisted-review.md). Read those; do not read this
file for how the system works today.

## The short version

The repository already shipped FENShot as its baseline. A v2 candidate had been
trained on synthetic pages, but synthetic accuracy alone cannot show that it
works on real books or screenshots.

The work did two things:

1. PR #10 preserved and corrected the audit of the completed v2 training run.
2. PR #11 made that frozen candidate testable beside unchanged FENShot in the
   browser and dataset dashboard.

No new model was trained. No private screenshot or model binary was committed.
Qualification data was not opened.

## What PR #10 changed

PR #10 did not change the learned weights. It corrected how the existing
synthetic run is described and evaluated:

- preserved the completed v2 run and selected step 9,000;
- clamped detector average precision to the valid range from zero to one;
- treated partial and unsupported pages as pages with no valid complete board;
- reported false detections on those pages separately;
- added post-hoc evaluation that cannot train or alter checkpoints;
- reconciled GPU accounting without counting inherited attempts twice;
- recalculated the synthetic calibration result.

The corrected synthetic acceptance threshold is `1.0`: no threshold with useful
recall also met the synthetic false-positive constraint. This is an honest
warning, not evidence that the model is useless. PR #11 therefore used `0.01`
only to produce possible detector regions for grid refinement.

## What PR #11 added

PR #11 turned the frozen local ONNX files into a bounded evaluation candidate:
the shared candidate runtime, the nine-line grid refiner, perspective
rectification, the paired browser diagnostic with its two export schemas, and
the schema-3 dataset provider manifests. The shape of that pipeline is described
in [architecture](../architecture.md).

## Schema-3 browser run results

The schema-3 local bundle and fixed ONNX providers share detector
letterboxing/decode, deterministic nine-line refinement, perspective
rectification and classifier tiling. A reference-first browser diagnostic
compares v2 with unchanged FENShot in automatic and manual-grid modes and emits
separate sensitive and aggregate schemas. The retained ONNX files passed Node
WASM plus Chromium, Firefox and WebKit production runs with no external requests.
This is runtime integration evidence only: no private six-image diagnostic, real
development truth, promotion comparison, named-laptop memory gate or
qualification was run.

## What had actually been demonstrated at this date

- strict contracts, bounds, cancellation, cleanup, privacy filtering, and stale
  result handling have executable tests;
- the retained v2 ONNX files execute through Node WASM;
- the actual files execute in Chromium, Firefox, and WebKit in automatic and
  manual-grid paths with no unintended external requests;
- the normal browser and offline-reload suites still pass in all three engines.

These checks demonstrate integration, not real-world accuracy. The timings were
measured on the GB10 development host, not the named laptop or physical mobile
devices.

## What remained unfinished at this date

The candidate could not be promoted. The next independently reviewable stages
recorded at the time were:

1. review and merge PR #11 only with owner approval;
2. resolve the 11 cross-split duplicate blockers and freeze a source/artwork-
   held-out real development membership;
3. review six development pages reference-first without model proposals;
4. use proposals for the remaining bounded tranche, while fully inspecting each
   page and all 64 squares;
5. lock truth, score frozen v2 and unchanged FENShot on identical pages, and
   apply the recorded promotion gates;
6. separately run six private screenshots through both diagnostic modes;
7. access qualification only after development decisions and the candidate are
   frozen.

No automatic retraining, threshold sweep, new seed, model-family change, model
publication, or qualification access was authorized by that work.
