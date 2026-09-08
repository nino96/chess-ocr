# Issue #3 synthetic bootstrap run log — 2026-09-07

Record of the issue #3 training attempts on host gx10-b210 (NVIDIA GB10), from
the first preflight attempt through the completed corrective v2 detector run and
its evaluation correction, 2026-09-07. Nothing in this file is current.

The durable half of the original document — the hypothesis and boundaries, the
GPU-second accounting, the controller commands, the gates and stop conditions,
and what may be delivered after the bootstrap — is in
[the training runbook](../training-runbook.md). The decision that authorized this
work is
[the joint issue #3 bootstrap decision](../decisions/2026-09-07-joint-issue-3-bootstrap.md).

The machine-readable sources of truth these attempts froze are
[`synthetic-bootstrap-v1.json`](../../recipes/synthetic-bootstrap-v1.json) and
[`synthetic-bootstrap-v2.json`](../../recipes/synthetic-bootstrap-v2.json). Both
are immutable; the v1 recipe remains evidence for retained v1 runs only.

## Corrective v2 triage

The completed v1 run is retained as diagnostic evidence, not repaired in place.
Its detector and ONNX export are `legacy-bgr-div255-v1`, synthetic-only and
uncalibrated. Native/browser agreement proved agreement on that self-defined
tensor; it did not prove compatibility with the pinned checkpoint's official
RGB/ImageNet-normalized transfer preprocessing.

Two training causes are confirmed and require a new immutable v2 recipe:

1. The detector fine-tune consumed BGR pixels divided by 255 instead of the
   checkpoint-compatible RGB tensor normalized with ImageNet mean/std.
2. EMA used a fixed `0.9998` decay from the first update instead of the pinned
   YOLOX ramp `0.9998 × (1 - exp(-updates / 2000))`, making the early average
   insufficiently responsive.

The v1 lifecycle also performed checkpoint selection, final development
evaluation and calibration inside the detector process immediately before ONNX
export. When export failed, those completed results were not durably marked and
the progress file still said `running`. V2 must persist `trained`, `calibrating`,
`exporting`, `export_failed` and `complete` states, and provide an export-only
retry that cannot repeat optimization, final development evaluation or
calibration.

The corrected machine-readable source of truth is
`synthetic-bootstrap-v2.json`. The v1
recipe remains immutable evidence for retained runs and must not be reused for
the corrected detector.

The first corrected run keeps the 9,000-update detector schedule, seed, batch
size, BN policy and learning rates. It starts from the original COCO checkpoint,
reuses the completed classifier checkpoint, and does not continue the legacy
detector. No seed sweep, backbone-LR change or model-family addition is authorized.
The incremental reservation is at most 600 preflight plus 5,400 detector
GPU-seconds, with the 24,000 CPU-second ceiling retained.

## Completed corrective v2 run and evaluation correction

Run `ec8d5a3fb66d74e5ec9dfafd36173988e8c22bb15b2d5b57cc51dbeb9e83c1ce`
completed the frozen 9,000-update schedule and selected step 9,000. The preflight
measured `2.384185791015625e-7` maximum input difference against the pinned
upstream preprocessing for both non-square pages and the colored sentinel.
Exact stochastic recovery was zero-difference for live and EMA weights in both
stages. Live and EMA development recall at IoU 0.5 were 1.0 at steps 1,500,
2,000 and 9,000, so the prospective transition stop did not fire. The selected
ONNX/native raw-output maximum difference was `0.0002722740173339844`.

The run used 13.6674 preflight and 3,211.7127 detector GPU-seconds. The
cross-run deduplicated ledger is 12,496.457073617727 GPU-seconds in 12 unique
attempt segments. No classifier optimization, new seed, schedule, backbone or
model family was added.

The original terminal report is retained unchanged even though it serialized
synthetic AP as `1.0000000000000007` and omitted partial pages from development
and calibration. Those are reporting/evaluation faults, not reasons to rewrite a
frozen run. Current code clamps every AP component and aggregate to `[0,1]`,
admits synthetic partial/unsupported pages as no-valid-board DEV/CAL cases, and
reports their false detections separately from ordinary negative pages. The
post-hoc `training audit` command loads the selected checkpoint and ONNX, verifies
the protected run-file checksums before and after, and writes only ignored audit
evidence. It performs no optimization and does not alter progress, curves,
checkpoints, calibration history or exports.

Corrected audit `7763cf7ecd9f80898df65a6e15d1255bcaded8d4a5689b4b83e94f22126f2eb4`
evaluated all 1,192 development pages (including 28 partials) and all 708
calibration pages (including 12 partials). Development AP50:95 is
`0.9825175869014556`, with 1.0 recall at IoU 0.5, zero false detections on 90
ordinary negatives, and 93 detections on partial pages at the 0.01 reporting
threshold. Calibration AP50:95 is `0.9888820647714669`, with zero false
detections on 52 ordinary negatives and 41 on partial pages. Under the frozen
maximum 0.05 false positives per no-valid-board page, no threshold with nonzero
recall survives: the corrected threshold is `1.0` with recall 0.0. The audit
report SHA-256 is `0d1fe1a8f23651c26923e6ad7461421984727299b1d2acb59102b6362d934f80`.

The one bounded diagnosis inspected an original synthetic partial case. It
contains only five visible grid rows, while retaining enough board texture for
YOLOX to localize the board-shaped region. This is a required downstream
refinement/rejection case, not authority to relax calibration, relabel the page,
retrain, sweep thresholds or change models. A bundle using the corrected
threshold therefore abstains automatically; any later low detector threshold
must be explicitly identified as a proposal threshold before nine-line grid
refinement, never as calibrated board acceptance.

## Cumulative GPU-second totals

The corrected run completed all 9,000 updates and selected step 9,000, using
3,225.3801 GPU-seconds including preflight. The selected ONNX passed native
raw-output parity. Its original synthetic report remains immutable; an
evaluation-only audit clamps AP to `[0,1]` and includes partial/unsupported
pages as separately reported no-valid-board cases. These saturated synthetic
results do not change the required shared grid refinement, reference-first real
development comparison, named-laptop budget or qualification gates.

## Preflight attempt record

Attempt 1 (`bce9ea1878e7e5ef67eabd6f3ba0175e6da41968ee4deb240bb29080fe107f85`)
stopped before optimization after 5.696 charged GPU-seconds. The container resolved
Pillow 12.3.0 instead of the required 11.1.0, and its capability-dropped root user
could not write to the host-owned output directory. The bounded diagnosis changed
no data, model or schedule: it adds the exact local Pillow/NumPy wheels to a
training-specific hash-frozen overlay, runs as the invoking UID/GID, uses `/output`
instead of the system `/run`, and imports the first attempt's charge into the
repaired frozen run. The failed run and log remain retained.

Attempt 2 (`387645f0c500f8c61ef7a853a49e44cc6844f28eba1c557a9b7103140909440b`)
also stopped before optimization after 3.578 charged GPU-seconds, bringing the
cumulative charge to 9.274 seconds. The gate incorrectly compared absolute
416-pixel letterboxed YOLOX targets with normalized coordinates in the original
non-square page frame. The bounded diagnosis keeps the data, models, seed and
schedule unchanged. It converts the independent normalized reference into the
continuous YOLOX letterbox frame, adds non-square and resize-rounding regression
tests, broadens parity coverage, and moves all input-contract checks ahead of GPU
allocation. A third run is authorized only if that CPU container validation and
the repository gates pass; it inherits both failed attempts and their full charge.

Attempt 3 (`8b1bd10d3df35c30c26ac4a362d052cf0c9d23c22daba45a8b783c060aa9817e`)
passed the new CPU barrier, then failed GPU preflight after 7.750 charged
GPU-seconds during the detector tiny-set fit. The retained CUDA log showed a
device-side BCE assertion after the first optimizer update; no scheduled model
training ran. The bounded diagnosis reproduced finite first-step CPU loss but
non-finite gradients after the next update, even with the then-assumed 0..1 input scale.
The configured 0.005 detector head rate required the explicit norm-10 gradient
clip now bound in the recipe. The repair applies that clip to preflight, timed
projection and both detector stages, and extends the CPU barrier to three
mixed-page updates. It changes no data, seed, model family, schedule length or
GPU allocation; a replacement run must inherit all three attempts and their
17.023 charged GPU-seconds.

The first detector-only continuation (`89985eb34922497d9e093f0eed865ec923678bb769178fffe51463dd28b14d48`)
stopped before allocating a GPU because its classifier export destination did
not yet exist. The second continuation
(`038dc1f749f3fc5e60f5a97099cae329f4fa40f48776be29eeca9cd2e33b41cf`)
created and verified the classifier ONNX artifact, but incorrectly repeated the
entire development and calibration evaluation. That consumed 30,248.9 CPU-seconds
over 2,785 wall-seconds, beyond the frozen 14,400 CPU-second ceiling. Its GPU
preflight then correctly stopped at the first resource guard after 4.335 charged
GPU-seconds; no detector optimization ran. The repair freezes and verifies the
source run's completed progress, curves, selected checkpoint and development
evidence, exports that checkpoint directly, and limits continuation validation to
native-to-ONNX parity. It also retains and verifies the export log and model hashes.
The replacement run inherits the second continuation's 4.335 GPU-second charge,
keeps the same data, selected checkpoint, seed, models and complete detector
schedule, and must pass the strengthened all-stage timing and recovery preflight
before detector optimization.

The replacement completed all 9,000 detector updates and selected step 9,000,
then failed its final export because the trained YOLOX raw output differed between
native PyTorch and ONNX Runtime by 0.0002151, above an inconsistent 0.0001 export
cutoff. A bounded CPU diagnosis retained the graph and checkpoint and compared
four deterministic tensors plus 16 synthetic development pages. Development-page
raw maximum absolute drift was 0.0004187; decoded detection counts were identical
at score thresholds 0.001, 0.01, 0.1 and 0.3, maximum decoded box drift was
0.00235 pixels, and maximum score drift was 0.0000408. The detector export gate
therefore uses the existing reviewed YOLOX native/browser raw-output ceiling of
0.001 while the classifier keeps 0.0001. This changes no model, data, checkpoint
selection or metric and does not by itself establish browser WASM parity.

That failure also exposed the lifecycle fault described above: the selected
weights and final evaluation existed, but the process had not persisted a trained
state and calibration before entering export. V2 export failure must retain those
artifacts and return `export_failed`; `pnpm run training -- export --run-root PATH
--segment detector` must retry export only.

The retained classifier and detector ONNX exports can now be loaded explicitly
into both local review surfaces through a hash-bound ignored candidate manifest;
the operator commands and limitations are in [local candidate testing](../local-candidate.md).
That v1 bundle remains legacy diagnostic evidence and is not promoted into the
schema-3 paired runtime. The v2 bundle binds the shared refiner, tensor contracts,
separate proposal/calibrated thresholds and limits. FENShot remains the browser
default. Dataset proposals remain drafts, and neither UI treats synthetic-only
confidence as calibration or qualification.
