# Chess OCR

Offline-first recognition of **printed 2D chess diagrams**, with a browser demo
and an explicitly opt-in GB10/server backend. A standalone recognizer for apps
such as chess-reader, not another ebook reader or chess engine.

## Status

This repository currently contains the bootstrap rules and implementation plan,
not a working recognizer. There is no application package, dependency lock,
training environment, browser command or server endpoint yet. Issue #1 delivers
the first runnable baseline; setup below prepares its prerequisites without
pretending unimplemented commands exist.

Read [AGENTS.md](AGENTS.md) before contributing and [PLAN.md](PLAN.md) for the
model, dataset, evaluation and privacy decisions.

## Four delivery issues

| Issue                                              | Outcome                                                                                       |
| -------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| [#1](https://github.com/nino96/chess-ocr/issues/1) | Runnable offline browser baseline, shared contract, pinned environment and actual WASM checks |
| [#2](https://github.com/nino96/chess-ocr/issues/2) | Source-diverse real training data, page/grid labels, review tools and qualification pool      |
| [#3](https://github.com/nino96/chess-ocr/issues/3) | Improved end-to-end offline recognition, qualified and delivered through the demo/library     |
| [#4](https://github.com/nino96/chess-ocr/issues/4) | Explicit opt-in GB10/cloud backend with the same contract and security/privacy controls       |

Start #1. Source research for #2 can proceed independently; #3 uses the accepted
first real-data tranche rather than waiting for the final corpus. #4 can proceed
when measured offline limits justify it; it does not require #3 to succeed.
Do not create another issue for every source, failed seed or pipeline check.

## Intended recognition path

Page/selection -> board detection -> inner-grid refinement -> square recognition
-> editable position with uncertainty.

Starting hypotheses: COCO-pretrained YOLOX-Nano for localization and native
ImageNet-pretrained `timm/mobilenetv3_small_100.lamb_in1k` for 13-class square
recognition, exported to ONNX Runtime Web WASM CPU. Keep unchanged FENShot as a
baseline. A larger RF-DETR Small recognizer is the optional server candidate.
None is claimed to meet chess accuracy, GB10 compatibility or browser latency
requirements before measurement. No arbitrary 2 MB weight limit; usable offline
latency/memory and preservation of working cases decide.

ChessQueries (ViT encoder with 64 square queries and a DETR-style decoder) is
also an owner-added GB10 evaluation candidate in [issue #4](https://github.com/nino96/chess-ocr/issues/4).
That issue includes a proposed isolated Python/CUDA inference recipe, pinned
source/weight identities and bounded screening gates. Setup and printed-diagram
quality are not yet validated; its inclusion does not select it for delivery.

## Environment preparation

Baseline prerequisites: Git, **Node.js 24 LTS**, **Python 3.12**, and optionally
Poppler for PDF-page extraction. Browser development and CPU checks do not need
a GPU. Issue #1 must pin exact package-manager/dependency/platform versions and
replace this prerequisite-only section with tested install/run commands.

Useful official references: [Git installation](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git),
[Node downloads](https://nodejs.org/en/download),
[Python downloads](https://www.python.org/downloads/),
[Poppler](https://poppler.freedesktop.org/).
Python 3.12 is in its security-maintenance phase: distribution/package-manager
availability and installer patch versions differ. Do not infer a supported
training stack from the interpreter version alone.

### Linux (Ubuntu 24.04 example)

Install Git, Python/venv and the optional PDF tools using your distribution:

```sh
sudo apt-get update
sudo apt-get install git python3.12 python3.12-venv poppler-utils
```

Install Node 24 LTS using its official distribution or an already trusted
version manager. On GB10 choose an ARM64 build, not an x86_64 binary. Other Linux
distributions use their corresponding package manager; the Ubuntu commands are
not universal.

```sh
git clone https://github.com/nino96/chess-ocr.git
cd chess-ocr
git --version
node --version
python3.12 --version
python3.12 -m venv .venv
. .venv/bin/activate
python --version
python -m pip --version
pdftoppm -v
pdfinfo -v
```

### macOS (Intel or Apple Silicon)

Install Git and Node 24 LTS using the official installers or your trusted
package manager. If Homebrew is already installed:

```sh
brew install git node@24 python@3.12 poppler
export PATH="$(brew --prefix node@24)/bin:$PATH"
git clone https://github.com/nino96/chess-ocr.git
cd chess-ocr
git --version
node --version
python3.12 --version
python3.12 -m venv .venv
. .venv/bin/activate
python --version
python -m pip --version
pdftoppm -v
pdfinfo -v
```

Use architecture-native binaries. CPU development is the baseline; MPS training
is optional and must be explicitly tested for numerical/recovery behavior.
The temporary PATH change above applies to the current terminal only.

### Windows (PowerShell)

Install Git, Node 24 LTS and Python 3.12 from their official installers. With
WinGet, Git/Python installation can alternatively use:

```powershell
winget install --id Git.Git --exact
winget install --id Python.Python.3.12 --exact
```

Restart PowerShell after installing tools. Install/select Node 24 LTS explicitly
rather than assuming a moving LTS package always selects that major version.

```powershell
git clone https://github.com/nino96/chess-ocr.git
Set-Location chess-ocr
git --version
node --version
py -3.12 --version
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip --version
```

Using the venv executable directly avoids changing PowerShell execution policy.
Once implemented, Node/browser development can run natively. For PDF extraction,
use WSL2 with Ubuntu and the Linux instructions, or an independently reviewed
native Poppler installation and verify `pdftoppm -v` / `pdfinfo -v`. There is no
claimed tested native Windows Poppler bundle in this bootstrap. Do not download
an arbitrary binary zip just to make setup appear complete.

Create a separate clone/venv inside WSL's Linux filesystem if using WSL. Never
share a Windows venv or node_modules with Linux. Pin one reference renderer for
dataset reproduction; platform render differences must not silently replace
hash-bound reviewed images.

### Optional NVIDIA GB10 training/server

Use Linux ARM64 with a compatible NVIDIA driver and a deliberately selected
PyTorch/CUDA stack. First record:

```sh
uname -m
nvidia-smi
```

Issue #1 must establish a tested pinned wheel or container environment and a
real CUDA operation, device-capability and export/parity check. Do not assume
that a generic CUDA wheel, x86_64 lock or `cuda.is_available()` alone proves
GB10 support. A CPU environment remains useful without GPU training.
See [NVIDIA Blackwell compatibility guidance](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/).

### What to run after prerequisites

For this bootstrap: inspect the four issues and choose #1. **Do not run invented
`pnpm install`, training or server commands:** no manifests/locks/scripts exist
yet. #1 must add the real cross-platform CPU setup, optional GPU setup, demo,
checks and test commands and validate them on the named available platforms.
The OS instructions above are prerequisite recipes, not completed OS test runs.

## Local assets and privacy

Keep acquired originals under ignored `data/` or `cache/`; derived pages/crops,
tensors, runs and checkpoints under ignored `work/` or `artifacts/`. Do not put
downloads, model binaries or secrets in Git/LFS. Public URLs, rights decisions,
hashes, recipes, safe factual labels and aggregate metrics can be versioned
after review. Exceptions for tiny original synthetic test fixtures require
their own source/hash/expected-result manifest.

Offline mode must not contact a server after readiness. Server mode is optional,
explicitly configured and consented; no silent fallback or image/FEN retention.
No private diagnostic material is authorized for training or uploading by this
bootstrap. Source-code licensing must be selected with the owner before package
publication; third-party code, data and model licenses remain separate.
