# Issue #1 implementation evidence

Evidence was collected in a working tree based on `ab60d74`. The implementation
PR isolates issue #1 changes onto `main`; the earlier documentation commit remains
on its original branch. Runtime source/model identities are recorded below.
Reference host: gx10-b210, Linux ARM64, Node 24.19.0/npm 11.17.0, Python 3.12.3.
NVIDIA GB10 driver 580.159.03 works outside sandbox. Browser inference uses CPU.
No downloaded image, dataset, model binary, private diagnostic, or generated run
is tracked. All inputs below are original procedural raster patterns.

## Delivered path

PNG/JPEG -> explicit automatic FENShot or manual grid -> 64 editable image-relative
squares -> edited JSON/placement. Unknown orientation remains explicit. A worker
performs WASM inference, with verified assets, timeout, cancellation and recovery.
A versioned service worker verifies and caches the build for offline reload.

FENShot model SHA-256 is
`883f6a8e639e6d6b6399b3fda0508ad772e3c6f9cefa2e678a13f27b9fa6248d`.
Full runtime identities: [assets.lock.json](../assets.lock.json).
Schema `chess-ocr/1`; preprocessing `fenshot-0.1.4/rgba-gray-bilinear-256/1`.

## Native/browser parity

`npm run test:parity` passed actual ORT Web 1.29.0 WASM in Chromium 153.0.8010.12,
Firefox 155.0 and WebKit 26.6. Inputs reproduce the native preprocessing exactly
(max absolute tensor difference **0**), including MobileNet's antialiased bicubic
416->256 resize/224 center crop/ImageNet normalization and YOLOX's exact416
RGB->BGR conversion. This single shape does not validate arbitrary page resizing.

| Model | Browser/native maximum absolute output difference | Elements |
| --- | ---: | ---: |
| MobileNetV3 Small, unchanged ImageNet head | 0.0000228882 | 1000 |
| YOLOX-Nano, unchanged COCO head | 0.0000616908 | 301665 |

Prospective thresholds in the harness: input max absolute <=1e-6, output <=1e-3.
Native checkpoints, source/notice hashes and CPU export evidence are in
[native-runtime.md](native-runtime.md). Browser raw parity report is ignored
`work/evidence/parity-2026-09-06T09-47-47-823Z.json`, bound to input manifest
`245323e464764af882cb1fba619094a6b0cf18fcd2ecf23acecce603414828c6`.
YOLOX raw grid decode/class-aware NMS has executable tests; these COCO detections
are not passed off as chess-board localization in the demo.

## Runtime distributions — synthetic host probe

`npm run eval`: 3 cold worker/session starts and 10 warm runs on a 256px original
checker pattern. This measures the dev-served runtime harness, including worker
startup in cold samples. Browser/HTTP/WASM compiler caches are not cleared between
cold samples. It is not a fresh-download laptop distribution.

| Browser | Cold worker min/median/max ms | Warm min/median/max ms | Four explicit 1024px windows, ms |
| --- | --- | --- | ---: |
| Chromium | 335.5 / 382 / 522.1 | 20.3 / 21 / 28.3 | 113.3 |
| Firefox | 398 / 454 / 476 | 24 / 25 / 28 | 121 |
| WebKit | 313 / 372 / 391 | 20 / 21 / 22 | 102 |

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

## Outstanding acceptance gates

* No named laptop, physical macOS/Windows host or iPad is available in this session.
  OS/device setup and laptop peak memory/runtime budgets are not claimed complete.
* The selected GB10 container passes these two native forward probes only with
  cuDNN disabled (MobileNet max abs 1.48e-5, YOLOX 5.82e-5). The default cuDNN
  path failed and remains recorded. Backward/optimizer/recovery validation is #3.
* CPU native lock is Linux ARM64-specific; it must not be copied to other platforms.
* Original repository source is licensed under MIT; third-party packages and
  model artifacts retain their separate licenses and notices.
  package is private. Model release/publication is a separate decision.
* No real-data accuracy, qualified localization, skew support or superiority over
  FENShot is claimed. Dataset and substantive training stay in #2/#3.

Delegated work: Luna medium gathered artifact provenance; Terra medium implemented
bounded native export/environment tooling. Lead owns contract, demo, integration,
reuse review and final validation. No token/quota measurement is available.


## Resuming

Use [the resource ledger](budget.md), preserved local raw reports and their
`codeHashes` to reuse unchanged evidence. Next owner-dependent acceptance action:
run the runtime harness on the chosen laptop and set its prospective budgets;
schedule physical OS/device gates only when available. Package publication needs
the owner's source-license decision. No open background acquisition/training job
is part of this handoff.

Raw final report SHA-256 (reports include source-code hashes):

* `work/evidence/parity-2026-09-06T09-47-47-823Z.json`: `34f4e410d9bb171fe759b74d82c37cf066e55d20ed13e3061f624dc07c845ffd`
* `work/evidence/benchmark-2026-09-06T09-47-53-970Z.json`: `8f42227d56d71bcbbf7a0d76b1b762374df21a952b9456e2093da70b53bcbd75`

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
