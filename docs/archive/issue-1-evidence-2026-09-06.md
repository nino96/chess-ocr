# Issue #1 implementation evidence — 2026-09-06

Record collected on host gx10-b210 (Linux ARM64, NVIDIA GB10) in a working tree
based on commit `ab60d74`, 2026-09-06, with a later 2026-09-08 paired-browser
section appended below at the time it was run. Nothing in this file is current.

The gates that remain open, and how to resume, are in
[issue #1 outstanding gates](../issue-1-evidence.md). Native checkpoints, export
commands and the parity contract are in [native runtime](../native-runtime.md).

Evidence was collected in a working tree based on `ab60d74`. The implementation
PR isolates issue #1 changes onto `main`; the earlier documentation commit remains
on its original branch. Runtime source/model identities are recorded below.
Reference host: gx10-b210, Linux ARM64, Node 24.19.0/npm 11.17.0, Python 3.12.3.
NVIDIA GB10 driver 580.159.03 works outside sandbox. Browser inference uses CPU.
No downloaded image, dataset, model binary, private diagnostic, or generated run
is tracked. All inputs below are original procedural raster patterns.

## Implemented issue #1 baseline

The repository contains the npm FENShot control, versioned raster/geometry/
probability contract, cancellable WASM worker and small editable browser demo.
Exact native starting checkpoints were exported with their unchanged COCO/ImageNet
heads; source preprocessing and actual browser WASM parity were checked on original
synthetic inputs. No dataset training, real-diagram qualification or recognition
superiority is claimed. Required laptop runtime budgets and unavailable physical
OS/device gates remain explicit. The selected pinned GB10 native inference probe
requires cuDNN disabled to pass CPU/GPU parity; training validation remains a
later gate.

## Delivered path

PNG/JPEG -> explicit automatic FENShot or manual grid -> 64 editable image-relative
squares -> edited JSON/placement. Unknown orientation remains explicit. A worker
performs WASM inference, with verified assets, timeout, cancellation and recovery.
A versioned service worker verifies and caches the build for offline reload.

FENShot model SHA-256 is
`883f6a8e639e6d6b6399b3fda0508ad772e3c6f9cefa2e678a13f27b9fa6248d`.
Full runtime identities: [assets.lock.json](../../assets.lock.json).
Schema `chess-ocr/1`; preprocessing `fenshot-0.1.4/rgba-gray-bilinear-256/1`.

## Native/browser parity

`npm run test:parity` passed actual ORT Web 1.29.0 WASM in Chromium 153.0.8010.12,
Firefox 155.0 and WebKit 26.6. Inputs reproduce the project-defined native tensors
exactly (max absolute tensor difference **0**), including MobileNet's antialiased
bicubic 416->256 resize/224 center crop/ImageNet normalization and the legacy
YOLOX exact-416 RGB->BGR conversion. This proved browser/native agreement on that
self-defined tensor, not compatibility with the pinned YOLOX checkpoint's official
RGB/ImageNet-normalized preprocessing.

The corrective v2 preflight subsequently exercised raw OpenCV BGR inputs,
non-square 612×792 and 792×612 letterboxing, an asymmetric colored-channel
sentinel, pad value 114, and the pinned helper's ImageNet mean/std. The candidate
RGB tensor agreed within `2.384185791015625e-7` (required maximum `1e-6`), target
geometry agreed exactly, and the selected trained ONNX raw output agreed with its
native checkpoint within `0.0002722740173339844` (required maximum `0.001`). This
closes the corrective preprocessing-parity task, not the named-laptop or real-page
recognition gates.

| Model                                      | Browser/native maximum absolute output difference | Elements |
| ------------------------------------------ | ------------------------------------------------: | -------: |
| MobileNetV3 Small, unchanged ImageNet head |                                      0.0000228882 |     1000 |
| YOLOX-Nano, unchanged COCO head            |                                      0.0000616908 |   301665 |

Prospective thresholds in the harness: input max absolute <=1e-6, output <=1e-3.
Native checkpoints, source/notice hashes and CPU export evidence are in
[native-runtime.md](../native-runtime.md). Browser raw parity report is ignored
`work/evidence/parity-2026-09-06T09-47-47-823Z.json`, bound to input manifest
`245323e464764af882cb1fba619094a6b0cf18fcd2ecf23acecce603414828c6`.
YOLOX raw grid decode/class-aware NMS has executable tests; these COCO detections
are not passed off as chess-board localization in the demo.

## Runtime distributions — synthetic host probe

`npm run eval`: 3 cold worker/session starts and 10 warm runs on a 256px original
checker pattern. This measures the dev-served runtime harness, including worker
startup in cold samples. Browser/HTTP/WASM compiler caches are not cleared between
cold samples. It is not a fresh-download laptop distribution.

| Browser  | Cold worker min/median/max ms | Warm min/median/max ms | Four explicit 1024px windows, ms |
| -------- | ----------------------------- | ---------------------- | -------------------------------: |
| Chromium | 335.5 / 382 / 522.1           | 20.3 / 21 / 28.3       |                            113.3 |
| Firefox  | 398 / 454 / 476               | 24 / 25 / 28           |                              121 |
| WebKit   | 313 / 372 / 391               | 20 / 21 / 22           |                              102 |

Four-window work uses the 1600x1200 tiling plan with sequential synthetic crops;
it measures extra classifier work, not page-localization recall. Tiling is an
explicit utility/probe, not a hidden demo fan-out. Cancellation settled within
0–1 ms at available clock precision. Raw samples: `work/evidence/benchmark-2026-09-06T09-47-53-970Z.json`.
The earlier asset-copy implementation's raw `work/evidence/benchmark.json` is
also retained (including its 2332.3 ms first Chromium cold sample); it is not
rewritten as the new implementation's evidence. The Chromium `performance.memory` sample is coarse main-thread JS heap only;
it does **not** measure peak process/worker/WASM memory. Firefox/WebKit expose no
comparable value here. Peak memory and the required named-laptop budgets remain
unmeasured; do not use GB10 CPU timings as laptop acceptance.

The build now imports npm assets directly using the reviewed chess-reader
approach. Model 1,289,483 bytes; runtime WASM 13,961,845; bootstrap 24,218.
Total runtime+model 15,275,546 bytes, plus application bundles. No arbitrary model
size cutoff or trained-model latency/accuracy promotion gate is inferred.

## Verification and CI scope

Initial validation commands (before the pnpm migration): `npm ci`, `npm run setup`, `npm run check`, `npm test`,
`npm run build`, `npm run test:smoke`, `npm run test:browser`,
`npm run test:parity`, `npm run eval`, and the native checks documented separately.
Fresh `npm ci`/asset verification/type/lint/payload/doc-link/build checks passed;
15 Node unit tests and 18 browser integration tests passed. Native tooling has
4 passing tests and a clean `pip check`. The broad browser checks are manual integration gates under the owner's revised
CI direction; ordinary CI uses one affected-path Chromium smoke.

WebKit's simulated-offline switch caused an internal navigation error. The
replacement test shuts down the origin HTTP server, then reloads and executes
WASM from verified cache in real WebKit. This tests absence of the origin service
without weakening the offline requirement. Physical iPad remains deferred.


Delegated work: Luna medium gathered artifact provenance; Terra medium implemented
bounded native export/environment tooling. Lead owns contract, demo, integration,
reuse review and final validation. No token/quota measurement is available.


## V2 paired-browser integration — 2026-09-08

The retained v2 detector and classifier were loaded from their hash-bound local
schema-3 bundle and exercised—not mocked—through the production offline build.
The verifier ran automatic full-page and manual four-corner modes sequentially
against unchanged FENShot and observed zero external requests:

| Engine   | Automatic paired run | Manual-grid paired run |
| -------- | -------------------: | ---------------------: |
| Chromium |          1,078.75 ms |              295.25 ms |
| Firefox  |          1,553.54 ms |              433.30 ms |
| WebKit   |          1,221.28 ms |              337.03 ms |

Command:

```sh
pnpm run candidate:verify:browser work/candidates/synthetic-bootstrap-v2-detector-v3.json \
  work/training/synthetic-bootstrap-v2-detector/classifier/selected.onnx \
  work/training/synthetic-bootstrap-v2-detector/detector/selected.onnx
```

These are synthetic in-memory smoke timings from this GB10 environment, not the
named-laptop gate and not real-page accuracy evidence. Private screenshot and
physical-device checks remain unrun.


## Raw final report SHA-256

Reports include source-code hashes. Local-only evidence, not reproducible from
this repository:

- `work/evidence/parity-2026-09-06T09-47-47-823Z.json`: `34f4e410d9bb171fe759b74d82c37cf066e55d20ed13e3061f624dc07c845ffd`
- `work/evidence/benchmark-2026-09-06T09-47-53-970Z.json`: `8f42227d56d71bcbbf7a0d76b1b762374df21a952b9456e2093da70b53bcbd75`

## pnpm migration — owner correction

The current toolchain uses pnpm 11.11.0, matching chess-reader, and
`pnpm-lock.yaml` replaces the npm lock. Import preserved every package/version
identity; model/runtime byte checks still pass. Dependency lifecycle scripts
remain disabled via `pnpm-workspace.yaml`. README, scripts, CI caches/install
steps and the evaluation hash inventory now use pnpm. AGENTS.md makes the
package-manager choice explicit.

Validation: clean `pnpm install --frozen-lockfile`, `pnpm run setup`,
`pnpm run check`, `pnpm test` (15 passed), `pnpm run build`, and
`pnpm run test:smoke` (1 Chromium test passed). The production offline-build hash
is unchanged: `3e57c5aeb2bbd74195b8e7270d881683f7aceeda8d60076ce546430bb8ba2a40`.
No unchanged native/model parity run or full browser matrix was repeated.
Earlier npm command names above remain historical evidence.

The pnpm lock SHA-256 is
`13b72f0407f4e893b945f2570fdbe85cddb5d268a964263b44bfe87378157cc9`.
The earlier npm lock is retained only in ignored
`work/evidence/npm-lock-before-pnpm.json` for audit.
