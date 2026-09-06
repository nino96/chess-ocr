# Browser artifact admission — 2026-09-06

Local evaluation only; no model release/publication is approved. Source-code
license selection remains with the owner; npm package is explicitly private.

## FENShot control

Exact package `@scoriiu/fenshot@0.1.4` is locked by registry tarball integrity in
[pnpm-lock.yaml](../pnpm-lock.yaml). Upstream tag `v0.1.4` maps to `5e68f7a04e1261328572caf74a2d4a44a342a6c7`
(as preserved by the chess-reader provenance review). The source distribution itself,
not a moving Git branch, is the frozen control. Exact model/runtime byte SHA-256,
size and local source path are in [assets.lock.json](../assets.lock.json).
`pnpm run setup` rejects any mismatch before Vite bundles the npm assets into ignored `dist/assets`.

The shipped package contains an MIT notice covering the distributed package,
including its shipped `model/chess-tiles-v2.onnx`; no separate model exclusion
is present in that distribution. Its README describes the pretrained chess
classifier and original training artwork. This review admits the publisher's
MIT-distributed artifact for local baseline execution, **not** certification of
all upstream training-image rights, disjointness, or a separately authorized
model release. Unknown inherited training overlap stays unknown. The exact
publisher notice is preserved in [notices/fenshot-MIT.txt](notices/fenshot-MIT.txt).

Sources: [pinned npm distribution](https://registry.npmjs.org/@scoriiu/fenshot/-/fenshot-0.1.4.tgz),
[official repository](https://github.com/scoriiu/fenshot). No weights are committed.

Unchanged `recognizeGray`, grid arbitration, tile extraction and probability
conversion run against unchanged weights in an ORT WASM session. The browser
adapter caps automatic detection at 1600 pixels as the shipped wrapper does,
then maps bounds back to the original decoded image. Manual selection is a
separate, explicitly named fallback using the shipped tile extraction. It does
not establish automatic localization success. FENShot supplies one candidate,
not a multi-board page detector; skew/perspective are unsupported.

The adapter reorders probabilities from bottom-up model tiles and
`1KQRBNPkqrbnp` classes into top-down image rows and the contract's class order.
Its 0.7 confidence floor is inherited and uncalibrated. `resolveOrientation`,
`inferCastling` and `placementToFen` are not used: orientation begins unknown and
unobservable FEN fields are absent. No visible piece is repaired for legality.

## ONNX Runtime Web

Pinned npm `onnxruntime-web@1.29.0`, MIT; publisher notice preserved in
[notices/onnxruntime-MIT.txt](notices/onnxruntime-MIT.txt). Both WASM bootstrap and
binary are verified against checked-in hashes before use. One WASM CPU thread
runs inside a dedicated cancellable worker, with no runtime CDN. Deployment
follows the [official local-asset guidance](https://onnxruntime.ai/docs/tutorials/web/deploy.html).

## Native starting models

See [native-runtime.md](native-runtime.md) for the separate native artifact
reviews and actual export status. COCO and ImageNet heads are not chess heads.
