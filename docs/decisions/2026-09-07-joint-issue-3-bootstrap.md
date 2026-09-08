# Joint issue #3 synthetic bootstrap and corrective detector v2

Status: active

Owner decision, 2026-09-07.

The completed synthetic seed may train both pinned browser candidates in one
frozen schedule: MobileNetV3 for 13 image-relative square classes and YOLOX-Nano
for one-class inner-grid localization. This supersedes the earlier classifier-only
bootstrap wording, but does not add another model family, seed, sweep or GPU time.
The exact recipe, gates and limitations are in
[the training runbook](../training-runbook.md).

Synthetic development/calibration remains TRAIN-purpose diagnostic evidence. A
successful run may improve annotation proposals but cannot establish real-page
promotion, touch qualification, replace the classical-detector comparison or
complete issue #3. Keep FENShot as the shipped default until the existing paired
real-development, qualification, WASM and browser gates pass.

## Corrective detector bootstrap v2

The retained v1 detector is `legacy-bgr-div255-v1`, synthetic-only and
uncalibrated. Its browser/native parity established agreement on a self-defined
tensor, not compatibility with the pinned checkpoint's official preprocessing.
The two confirmed training causes are incompatible BGR/divide-by-255 transfer
preprocessing and fixed `0.9998` EMA decay. A third fault let ONNX export failure
erase the durable lifecycle distinction between completed training/evaluation,
calibration and export.

Issue #1 owns corrected upstream preprocessing parity. Issue #2 owns a
source-held-out independently human-reviewed real development tranche; proposals
are not truth and qualification remains untouched. Issue #3 owns immutable recipe
v2, raw RGB `0..255` graph input with in-graph ImageNet normalization, equivalent
normalized RGB training, ramped/resumable EMA, durable lifecycle/retry, live-versus-
EMA stage gates, detector-only rerun, calibration, inner-grid refinement and the
unchanged FENShot/end-to-end/browser comparisons.

The first corrected detector run starts from the original COCO checkpoint and
reuses the completed classifier. Keep its existing 9,000 updates, seed, batch size,
BN policy and learning rates; do not add a seed sweep, backbone-LR change or model
family without corrected live-model evidence. Reserve at most 600 preflight plus
5,400 detector GPU-seconds and retain the 24,000 CPU-second ceiling. This remains
a synthetic bootstrap, not promotion or issue completion.
