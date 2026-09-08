# Native ARM64 export and GB10 container run — 2026-09-06

Record produced on host gx10-b210 (Linux ARM64, NVIDIA GB10, driver 580.159.03),
Python 3.12.3, at the issue #1 evidence commit `ab60d74`, 2026-09-06. Nothing in
this file is current: it is the observed output of one export and one pinned
container inference check. The contract, rights review and export commands that
remain in force are in [native runtime](../native-runtime.md).

## Observed Linux ARM64 run (2026-09-06)

Python 3.12.3, Torch `2.6.0+cpu`, timm `1.0.15`, ONNX `1.17.0`, and ONNX
Runtime `1.20.1` produced the following ignored assets. CPU ORT agreed with the
native reference forward to the stated maximum absolute difference.

| model                 | source identity                                                       | checkpoint SHA-256                                                 | ONNX SHA-256                                                       | output SHA-256                                                     | native/ORT max abs      |
| --------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------ | ------------------------------------------------------------------ | ----------------------- |
| YOLOX-Nano            | release `0.1.1rc0`, source `e1052df71842031413f6030723c3607b839c80ce` | `cd28f55fbbc1829f99d9ac9b38a16d259a22889739c8728ea877610201feff7b` | `b1b8da1585106dc116bf591b83c6197e0f3c8711f88181e55e601aec031c8e7c` | `f831ac8f73d2deaa5535f2bb06ce40fc426a235fc801be596bc577a1ee20322a` | `7.319450378417969e-05` |
| MobileNetV3 Small 100 | HF `1824797e7887cbec1990e4adbd6675960a36c589`                         | `46d2c063b18125884c48937afa4c49e18128869e52e8db96df48bf0a4d7ff697` | `ffa1e75320f1ad7829d04940e32a3fb707b9cf228f2da7215ba253e4ec71e9a0` | `ed9ded4abd9d8be4ffeb4f270d6740328611fde8ff89efe942e780eb14694c40` | `3.0517578125e-05`      |

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
  onnx==1.17.0 onnxruntime==1.20.1 timm==1.0.15 safetensors==0.5.3 loguru==0.7.3 opencv-python-headless==4.11.0.86 \
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
