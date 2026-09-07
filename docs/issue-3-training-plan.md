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
The run also reserves four CPU hours and 16 GiB of output storage while preserving
at least 30% filesystem free space. Failed and interrupted attempts remain charged.

## Gates and stop conditions

Before substantive optimization, verify corpus and checkpoint hashes, actual
image/label/tensor ordering, finite gradients, expected parameter updates, tiny
known-label fit, negative-page loss, native preprocessing and actual stochastic
checkpoint recovery. Use the existing digest-pinned NVIDIA container with cuDNN
disabled; its default cuDNN path previously failed native parity. If backward,
resume, deterministic execution or the complete schedule's measured projection
fails, stop before the run and report the specific blocker.

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
