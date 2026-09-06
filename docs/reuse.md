# Reuse review for issue #1

Reviewed the public chess-reader snapshot
[`c2ece9a422dc573e41a2d9f9a84af7c1240c3ad3`](https://github.com/nino96/chess-reader/tree/c2ece9a422dc573e41a2d9f9a84af7c1240c3ad3)
on 2026-09-06 following the owner's direction to reuse existing work.
No files or production state in chess-reader were changed.

* **Reused/adapted:** `apps/web/src/recognition/assets.ts` npm model/WASM URL
  imports and exact model/runtime identities; `assets.test.ts` installed-byte
  provenance test; third-party notice identities. This avoids duplicate runtime
  copying and reuses the previously reviewed FENShot/ORT pairing. Our manifest
  adds the MJS bootstrap hash, and tests use node:test rather than Vitest.
* **Reviewed for adaptation:** `workerRecognizer.ts`, `workerCore.ts`,
  `pipeline.ts`, `protocol.ts`. These depend on `../study/contracts`, use numeric
  study request IDs, confidence-only placements and white/black orientation.
  This standalone contract requires all 13 probabilities for each image-relative
  square, unknown orientation, original-image selections and explicit geometry.
  The adapter therefore uses the same npm core with a new boundary. Cancellation
  terminates the single bounded worker, including a synchronous WASM run, and
  recreates it on demand; the old abort path posts a cooperative cancel message.
  The old tests informed asset corruption/recovery, stale-result and lifecycle
  coverage rather than importing reader/study dependencies.
* **Referenced, not regenerated:** `experiments/recognition-dataset/handoff/README.md`
  and its existing native FENShot recovery/provenance. Issue #1 does not rebuild
  that recovered native control or replay its training. Public dataset records,
  acquisition jobs and qualification history belong to #2/#3.

No root source-code LICENSE exists in that snapshot. This is limited owner-directed
reuse of their code between their repositories, with provenance retained, not a
new public licensing grant. Both npm dependencies' MIT notices are preserved
separately; code/package publication remains pending owner license selection.
