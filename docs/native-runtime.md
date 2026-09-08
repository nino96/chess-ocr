# Native starting-model export and browser parity

This is bounded issue #1 evidence, not a chess-quality result. Native assets,
checkpoints, tensors, outputs and ONNX files remain ignored under `cache/native`,
`work/native` and `artifacts/native`; nothing here authorizes redistribution.

## Contract for browser parity

Corrective note (2026-09-07): the original YOLOX vector below proved native and
browser agreement on a project-defined BGR `0..255` tensor. It did not establish
that this tensor matched the pinned checkpoint's official transfer preprocessing.
It is retained as `legacy-bgr-div255-v1` evidence, not as a checkpoint-compatible
training contract.

Run `python/native_runtime.py prepare-inputs`. It creates an original synthetic,
font-free `416x416` raw RGB raster plus exact little-endian float32 input tensors
and `work/native/parity-input-manifest.json`. The manifest is the browser test
vector contract: hash the source bytes, reproduce preprocessing, load the tensor
binary as float32 little-endian, and compare native outputs after ONNX Runtime Web
WASM is executed. Browser decoding may differ from Pillow by a few bicubic pixels;
that is a failed parity check to diagnose, not an excuse to silently accept a new
preprocessing version.

| legacy probe          | graph input                              | source preprocessing                                                                              | graph output                                                                                      |
| --------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| YOLOX-Nano COCO       | `[1,3,416,416]`, `float32` BGR, `0..255` | RGB source; aspect-fit to 416, pad BGR channels with 114; no mean/std                             | raw `[1,3549,85]`: `tx,ty,tw,th,obj,80 class scores`; browser applies grid/stride decode then NMS |
| MobileNetV3 Small 100 | `[1,3,224,224]`, `float32` RGB           | bicubic resize shorter side to 256, center crop 224, then `(x/255-mean)/std` with ImageNet values | `[1,1000]` ImageNet logits                                                                        |

The corrected YOLOX v2 contract is separate: the native/browser graph accepts raw
letterboxed RGB float32 `0..255` with pad `114`, and an in-graph wrapper divides by
255 then applies ImageNet RGB mean/std normalization. Training feeds the equivalent
normalized RGB tensor directly. V2 parity must compare an OpenCV BGR source through
the pinned upstream helper with non-square geometry and channel sentinels at
maximum tensor difference `1e-6`.

For YOLOX, `decode_in_inference` is deliberately **false**. The rows are ordered
by strides 8, 16 and 32 (52² + 26² + 13² = 3549). For each row, generate grid
coordinates and stride in that order, decode `xy=(tx,ty)+grid`, `wh=exp(tw,th)`,
then multiply all four by stride. Objectness and class columns already have sigmoid
applied. This has to be reproduced before NMS; it is not a graph output of `cxcywh`.

Neither output is a chess prediction. YOLOX uses its unchanged 80 COCO head and
MobileNet uses its unchanged 1000 ImageNet head. A later task may replace heads
only with an approved training recipe.

The Linux x86_64 CI-only native-tooling check stays small and does not install the
ARM64 runtime stack:

```sh
python -m pip install --require-hashes -r python/requirements-native-test-linux-x86_64.txt
python -m unittest python/test_native_runtime.py
```

## Rights review before local admission

The code repository license does not automatically license weights. Review the
linked artifact before downloading and place the factual result in
`work/native/status.md` and `work/native/download-manifest.json`. The tracked
[native-notices.json](../python/native-notices.json) preserves the public notice
URLs, review date and SHA-256 identities of the reviewed LICENSE/model card:

- YOLOX-Nano is requested only from the official
  [0.1.1rc0 GitHub release](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth).
  Local admission uses the official release's association with the YOLOX project,
  source commit `e1052df71842031413f6030723c3607b839c80ce`, and that project's
  [Apache-2.0 LICENSE](https://github.com/Megvii-BaseDetection/YOLOX/blob/e1052df71842031413f6030723c3607b839c80ce/LICENSE).
  The asset has no separate GitHub release license field. This is therefore a
  reviewed basis for local issue-#1 evaluation/export only; retain the Apache
  notice and obtain a separate artifact review before redistribution or release.
- MobileNet is requested only from the pinned commit of the official
  [timm Hugging Face repository](https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k).
  Its model card declares Apache-2.0 at full revision
  `1824797e7887cbec1990e4adbd6675960a36c589`. Record the model-file URL and
  SHA-256 before local admission. It is a safetensors file; no pickle loader is
  used.

The downloader accepts only the two fixed identities and expected SHA-256 values,
uses HTTPS and temporary files, and enforces the issue's **4 GiB total download**
ceiling across `cache/native` model files. YOLOX is read only with
`torch.load(..., weights_only=True)`. A failed rights review is a stop condition:
keep the model out of cache and report the blocked artifact rather than trying
another mirror.

## Environment and commands

`python/requirements-native-cpu-linux-aarch64.txt` is the generated exact-wheel,
hash-bound CPU Linux ARM64 lock, from its `.in` input. It is platform-specific
and records only binary wheels. The tested command uses `--require-hashes`; no
unpinned `pip` or `pip-tools` upgrade is needed to consume it. GB10 needs a
separate, driver-compatible NVIDIA wheel or container digest;
`python/requirements-native-gb10-linux-aarch64.in` intentionally does not select
a generic CUDA wheel. That prevents an ARM64 CUDA selection from being copied to
CPU, macOS or Windows.

After an approved artifact-specific rights review:

```sh
python3.12 -m venv work/native/venv
. work/native/venv/bin/activate
python -m pip download --require-hashes --dest work/native/wheelhouse -r python/requirements-native-cpu-linux-aarch64.txt
python -m pip install --no-index --find-links work/native/wheelhouse --require-hashes -r python/requirements-native-cpu-linux-aarch64.txt
python python/native_runtime.py prepare-inputs
python python/native_runtime.py download --mobilenet
python python/native_runtime.py export --model mobilenet --weights cache/native/mobilenetv3_small_100.lamb_in1k.safetensors
```

For YOLOX, clone the official repository to ignored `cache/native/yolox` at a
recorded immutable commit, then use the exact release asset and explicit source:

```sh
git clone --no-checkout https://github.com/Megvii-BaseDetection/YOLOX.git cache/native/yolox
git -C cache/native/yolox checkout --detach e1052df71842031413f6030723c3607b839c80ce
python python/native_runtime.py download --yolox
python python/native_runtime.py export --model yolox --weights cache/native/yolox_nano.pth --yolox-source cache/native/yolox
```

Each export produces ignored `.onnx`, output float32 binary and a manifest bound
to model, input-manifest and output hashes. Native ONNX Runtime and browser WASM
execution remain separate checks; this document does not claim the browser result.

## GPU execution constraint

The pinned NVIDIA container's **default cuDNN path does not agree with the CPU
references**. `python/native_gpu_reference.py` therefore defaults to
`torch.backends.cudnn.enabled=False` and exits nonzero for nonfinite outputs or
a maximum absolute disagreement above `1e-3`. `--enable-cudnn` reproduces the
failed path and is not a supported configuration.

Within that configuration the container is a validated GB10 **native inference**
environment for these two models. It is not evidence of training throughput,
backward-pass correctness, optimizer recovery or GPU export parity; those gates
belong to issue #3 and are in [the training runbook](training-runbook.md).

The measured export identities, container digest, overlay build and the observed
cuDNN failure are in
[the 2026-09-06 native ARM64 run record](archive/native-arm64-run-2026-09-06.md).

Native tooling tests (no GPU/model runs):

```sh
work/native/venv/bin/python -m unittest python/test_native_runtime.py
```

The small `python/requirements-native-test-linux-x86_64.txt` CI lock admits only
NumPy/Pillow Linux x86_64 CPython 3.12 wheels. Their hashes were checked from ARM64;
execution on an x86_64 CI host is configured, not claimed locally observed.
