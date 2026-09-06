# Native starting-model export and browser parity

This is bounded issue #1 evidence, not a chess-quality result. Native assets,
checkpoints, tensors, outputs and ONNX files remain ignored under `cache/native`,
`work/native` and `artifacts/native`; nothing here authorizes redistribution.

## Contract for browser parity

Run `python/native_runtime.py prepare-inputs`. It creates an original synthetic,
font-free `416x416` raw RGB raster plus exact little-endian float32 input tensors
and `work/native/parity-input-manifest.json`. The manifest is the browser test
vector contract: hash the source bytes, reproduce preprocessing, load the tensor
binary as float32 little-endian, and compare native outputs after ONNX Runtime Web
WASM is executed. Browser decoding may differ from Pillow by a few bicubic pixels;
that is a failed parity check to diagnose, not an excuse to silently accept a new
preprocessing version.

| model | graph input | source preprocessing | graph output |
| --- | --- | --- | --- |
| YOLOX-Nano COCO | `[1,3,416,416]`, `float32` BGR, `0..255` | RGB source; aspect-fit to 416, pad BGR channels with 114; no mean/std | raw `[1,3549,85]`: `tx,ty,tw,th,obj,80 class scores`; browser applies grid/stride decode then NMS |
| MobileNetV3 Small 100 | `[1,3,224,224]`, `float32` RGB | bicubic resize shorter side to 256, center crop 224, then `(x/255-mean)/std` with ImageNet values | `[1,1000]` ImageNet logits |

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

* YOLOX-Nano is requested only from the official
  [0.1.1rc0 GitHub release](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth).
  Local admission uses the official release's association with the YOLOX project,
  source commit `e1052df71842031413f6030723c3607b839c80ce`, and that project's
  [Apache-2.0 LICENSE](https://github.com/Megvii-BaseDetection/YOLOX/blob/e1052df71842031413f6030723c3607b839c80ce/LICENSE).
  The asset has no separate GitHub release license field. This is therefore a
  reviewed basis for local issue-#1 evaluation/export only; retain the Apache
  notice and obtain a separate artifact review before redistribution or release.
* MobileNet is requested only from the pinned commit of the official
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

## Observed Linux ARM64 run (2026-09-06)

Python 3.12.3, Torch `2.6.0+cpu`, timm `1.0.15`, ONNX `1.17.0`, and ONNX
Runtime `1.20.1` produced the following ignored assets. CPU ORT agreed with the
native reference forward to the stated maximum absolute difference.

| model | source identity | checkpoint SHA-256 | ONNX SHA-256 | output SHA-256 | native/ORT max abs |
| --- | --- | --- | --- | --- | --- |
| YOLOX-Nano | release `0.1.1rc0`, source `e1052df71842031413f6030723c3607b839c80ce` | `cd28f55fbbc1829f99d9ac9b38a16d259a22889739c8728ea877610201feff7b` | `b1b8da1585106dc116bf591b83c6197e0f3c8711f88181e55e601aec031c8e7c` | `f831ac8f73d2deaa5535f2bb06ce40fc426a235fc801be596bc577a1ee20322a` | `7.319450378417969e-05` |
| MobileNetV3 Small 100 | HF `1824797e7887cbec1990e4adbd6675960a36c589` | `46d2c063b18125884c48937afa4c49e18128869e52e8db96df48bf0a4d7ff697` | `ffa1e75320f1ad7829d04940e32a3fb707b9cf228f2da7215ba253e4ec71e9a0` | `ed9ded4abd9d8be4ffeb4f270d6740328611fde8ff89efe942e780eb14694c40` | `3.0517578125e-05` |

`onnx.checker.check_model` and `pip check` passed. The host reported NVIDIA GB10,
driver `580.159.03`, CUDA 13.0. The separate existing NVIDIA image
`nvcr.io/nvidia/vllm@sha256:95c498a475142c20c989c65e5d223348c09fed83ba17ddf44f117610c0bd3268`
ran a CUDA tensor square/sum successfully (Torch `2.13.0a0+9186a08b2c.nv26.07`,
CUDA 13.3); it is evidence of host CUDA operation, not a project CUDA lock.

## GB10 pinned-container inference check

The already-present image
`nvcr.io/nvidia/vllm@sha256:95c498a475142c20c989c65e5d223348c09fed83ba17ddf44f117610c0bd3268`
ran both existing models on NVIDIA GB10 (capability `[12,1]`) with its Torch
`2.13.0a0+9186a08b2c.nv26.07`, CUDA 13.3, and a read-only local model/source/input
set. The ignored `work/native/gpu-overlay` contains only the explicitly versioned
local wheels needed by `python/native_gpu_reference.py`; no pull or network input
is needed after the wheelhouse is prepared.

Create that overlay once, using only wheels already resolved by the CPU lock:

```sh
docker run --rm \
  -v "$PWD/work/native/wheelhouse:/wheelhouse:ro" -v "$PWD/work/native/gpu-overlay:/overlay" \
  nvcr.io/nvidia/vllm@sha256:95c498a475142c20c989c65e5d223348c09fed83ba17ddf44f117610c0bd3268 \
  python -m pip install --no-index --find-links /wheelhouse --target /overlay --no-deps \
  timm==1.0.15 safetensors==0.5.3 loguru==0.7.3 opencv-python-headless==4.11.0.86 \
  thop==0.1.1.post2209072238 tabulate==0.9.0 pyyaml==6.0.3
```

```sh
docker run --rm --gpus all \
  -v "$PWD/cache/native:/models:ro" -v "$PWD/cache/native/yolox:/yolox:ro" \
  -v "$PWD/work/native:/inputs:ro" -v "$PWD/artifacts/native:/references:ro" \
  -v "$PWD/work/native/gpu-overlay:/overlay:ro" -v "$PWD/python:/runner:ro" \
  nvcr.io/nvidia/vllm@sha256:95c498a475142c20c989c65e5d223348c09fed83ba17ddf44f117610c0bd3268 \
  /bin/bash -lc 'PYTHONPATH=/overlay:/yolox python /runner/native_gpu_reference.py --model both --weights /models --inputs /inputs --references /references --yolox-source /yolox'
```

The initial default-cuDNN outputs were finite but did **not** agree with the CPU references: maximum
absolute difference was `104.27913665771484` for MobileNet and
`4.224334239959717` for YOLOX. One bounded CPU-only diagnosis in that same image
gave `1.239776611328125e-05` and `8.654594421386719e-05`, respectively. The
observed incompatibility is therefore on the selected GPU path, not source or
preprocessing identity. A lead-owned follow-up changed only `torch.backends.cudnn.enabled=False`:
MobileNet then agreed within `1.4781951904296875e-05`, YOLOX within
`5.817413330078125e-05`. The runner now defaults to that explicit configuration
and exits nonzero for nonfinite outputs or maximum absolute disagreement >1e-3.
`--enable-cudnn` reproduces the failed path and must not be used as a supported
configuration. The command above therefore selects the tested non-cuDNN path.

This is a pinned, validated GB10 **native inference** environment for these two
models, not evidence of training throughput, backward-pass correctness, optimizer
recovery or GPU export parity. Those training gates remain in #3. No host driver
changes or alternative image pulls were made. The raw lead diagnosis is ignored
`work/native/cudnn-disabled-result.txt`; the default-path failures remain recorded.


Native tooling tests (no GPU/model runs):

```sh
work/native/venv/bin/python -m unittest python/test_native_runtime.py
```

The small `python/requirements-native-test-linux-x86_64.txt` CI lock admits only
NumPy/Pillow Linux x86_64 CPython 3.12 wheels. Their hashes were checked from ARM64;
execution on an x86_64 CI host is configured, not claimed locally observed.
