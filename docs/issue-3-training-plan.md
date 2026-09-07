# Issue #3 synthetic bootstrap decision

Owner decision, 2026-09-07. This is the first bounded experiment in issue #3,
not recognition delivery or qualification. The completed issue #2 synthetic seed
permits training both starting browser candidates in one schedule: the 13-class
MobileNetV3 square classifier and one-class YOLOX-Nano inner-grid detector.
This supersedes the earlier classifier-only wording for this bootstrap without
adding a seed, model family, sweep, or GPU time.

The machine-readable source of truth is
[`synthetic-bootstrap-v1.json`](../recipes/synthetic-bootstrap-v1.json). It binds
the completed 12,000-page/15,923-board corpus, starting checkpoint identities,
single seed, split policy, preprocessing, trainable stages, update counts,
selection rules and resource ceilings. Operational paths and run state stay in
ignored `work/training/`.

## Corrective v2 triage — 2026-09-07

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

The first corrected run keeps the 9,000-update detector schedule, seed, batch
size, BN policy and learning rates. It starts from the original COCO checkpoint,
reuses the completed classifier checkpoint, and does not continue the legacy
detector. No seed sweep, backbone-LR change or model-family addition is authorized.
The incremental reservation is at most 600 preflight plus 5,400 detector
GPU-seconds, with the 24,000 CPU-second ceiling retained.

## Boundaries and hypothesis

Hypothesis: task heads followed by lower-rate adaptation of the pinned native
checkpoints can learn the audited synthetic piece designs and page geometry well
enough to reduce proposal corrections on separately reviewed real pages. Training
both models now tests the available positive, negative, multiple-board, small-board,
affine and projective cases. It does not establish real localization because the
seed lacks source diversity and several target detector conditions.

All synthetic pages retain TRAIN purpose. A deterministic 85/10/5 internal
train/development/calibration partition keeps each page, repeated position parent
and shared degradation seed connected. It is used for optimization, checkpoint
selection and diagnostic calibration only. It is never qualification.
The frozen seed contains 11,751 white-bottom and 4,172 black-bottom boards; every
internal partition must contain both. Metrics report the two orientations
separately. Recognition remains image-relative and orientation remains unknown
until the user selects it unless a later orientation model is independently
validated.

The GPU ceiling remains eight hours total: twenty minutes for mandatory native
parity/backward/tiny-fit/throughput/recovery gates, 100 minutes for the classifier,
340 minutes for the detector, and twenty minutes for at most one bounded diagnosis.
Future runs reserve 24,000 CPU-seconds (six hours forty minutes) and 16 GiB of
output storage while preserving at least 30% filesystem free space. The CPU ceiling
retains the measured 8,806 CPU-seconds used by the completed classifier path and
adds provisional detector and contingency capacity; it does not enlarge the
unchanged eight-hour GPU ceiling. Each initialized run keeps the exact recipe value
it froze, so this revision does not alter an active or historical run. Failed and
interrupted attempts remain charged.

## Gates and stop conditions

Before substantive optimization, verify corpus and checkpoint hashes, actual
image/label/tensor ordering, finite gradients, expected parameter updates, tiny
known-label fit, negative-page loss, native preprocessing and actual stochastic
checkpoint recovery. Use the existing digest-pinned NVIDIA container with cuDNN
disabled; its default cuDNN path previously failed native parity. If backward,
resume, deterministic execution or the complete schedule's measured projection
fails, stop before the run and report the specific blocker.

The container runs as the invoking host UID/GID, writes only through `/output`,
and resolves NumPy 2.2.4, Pillow 11.1.0, ONNX 1.17.0, ONNX Runtime 1.20.1,
OpenCV headless 4.11.0.86, safetensors 0.5.3 and timm 1.0.15 from a hash-frozen ignored overlay built from
the existing local wheelhouse. A repaired attempt imports every prior attempt and
charge; a new run directory cannot reset the cumulative reservation when
`--prior-run` is supplied. A reviewed detector-only continuation may instead omit
`--prior-run`, provide the hash-verified completed classifier checkpoint, and
consume a separately frozen reservation without redoing classifier optimization.
The continuation validates that the checkpoint is the complete development-selected
candidate, reuses its retained development evidence, and performs bounded native-to-ONNX
parity. It does not repeat the full development/calibration evaluation.

Before `start` can allocate a GPU, the exact frozen, network-disabled container
runs a CPU-only validation segment. The v1 segment verified dependency pins,
output writes, classifier tensors/labels, BGR/padding geometry and a self-defined
divide-by-255 detector tensor. That detector check was internally consistent but
not checkpoint-compatible. V2 must instead compare an OpenCV BGR source against
the pinned upstream helper using non-square geometry, colored channel sentinels,
actual ImageNet RGB mean/std and a maximum tensor difference of `1e-6`. Failure
leaves a retained validation log and consumes no GPU time; timeout cleanup stops
the named validation container before returning control.

The v2 graph boundary is raw letterboxed RGB float32 `0..255`, pad `114`; its graph
wrapper divides by 255, subtracts ImageNet RGB mean and divides by ImageNet RGB
standard deviation. Training feeds the equivalent normalized RGB tensor directly.
Native export and browser preprocessing must agree on that identifier.

The detector recipe clips its head/full-model gradient norm at 10.0. This is a
mechanics-stability guard justified by the bounded diagnosis: the configured
0.005 head learning rate produced non-finite gradients after one un-clipped
synthetic update, while the clipped path remained finite and reduced loss across
five CPU updates. No seed, model family, corpus or schedule length changed.

The schedule is complete only after 10,000 classifier and 9,000 detector updates
plus all frozen development/calibration evaluations. Do not shorten it to call a
mechanics pilot successful. One failed comparison permits one bounded causal
diagnosis, not a new seed, resolution, model family or sweep.

Report full curves and effective class/design/effect exposure. For the classifier,
report exact boards, square confusion, occupied/class/color errors, NLL, confidence
coverage and confident errors. For the detector, report multi-board recall, negative
false positives, AP, IoU and normalized box error by size/layout/effect. Compare
the selected candidates with unchanged FENShot on identical inputs.

## Delivery after the bootstrap

A candidate may prefill issue #2 reviews only as a visible proposal. The next
promotion decision requires separately reviewed real development data, the bounded
classical detector comparison, shared inner-grid refinement and paired end-to-end
evidence. Fresh human-checked qualification remains untouched until candidate
freeze. FENShot stays the shipped default until all issue #3 quality, offline WASM,
runtime and browser gates pass.

Checkpoints, ONNX files and raw evidence remain ignored. Model publication and a
public artifact mechanism require separate rights review and owner approval.

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
the operator commands and limitations are in [local candidate testing](local-candidate.md).
That v1 bundle is legacy diagnostic input only and must be regenerated with the
legacy preprocessing identifier under manifest schema 2 before corrected code
will load it. FENShot remains the browser default. Dataset proposals remain drafts,
and neither UI treats synthetic-only confidence as calibration or qualification.
