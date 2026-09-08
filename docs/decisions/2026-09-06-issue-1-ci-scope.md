# Issue #1 CI scope

Status: active

Owner direction, 2026-09-06.

This repository's main ongoing work is OCR training and evaluation. Routine CI
runs source/type/contract/unit/payload checks and a production build. Browser
input changes receive one Chromium WASM smoke test; native-only changes do not
trigger browser tests. Firefox/WebKit, offline-origin shutdown, corrupt assets,
and accessibility/touch checks are retained as manual browser integration gates
for relevant runtime/demo changes, not an every-training-change CI matrix.
Initial multi-browser evidence is retained; it is not physical iPad evidence.
