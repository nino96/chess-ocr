# Synthetic renderer prelaunch validation — 2026-09-07

Record produced on host gx10-b210 (Linux ARM64) at the 2026-09-07 dataset-branch
commit. Nothing in this file is current: it pins the renderer and report bytes of
one passing prelaunch validation and the batch timing measured on that host. The
current fidelity gate, its ceilings and the commands that run it are in
[the deterministic synthetic dataset guide](../synthetic-dataset.md).

This run supersedes the three failing small transformed-board controls recorded
in [the 2026-09-07 test-isolation incident](dataset-incident-2026-09-07.md).

Measured prelaunch validation (2026-09-07): 91 pages, 106 independent controls,
6,784 tile comparisons, 2,496 clean calibration identity checks, and 768 effect
audit squares (1,536 strict class-margin comparisons) passed. Maximum calibration
MAE was 0.4775 under 3; maximum other-control MAE was 17.1213 under 18. Minimum
effect class margin was 7,552 summed byte-error units. These are synthetic
renderer checks, not recognition accuracy on real documents.

Renderer SHA-256: `d16ee83978e8b886c28e9a3b433b13760cd8d08f56c38c109eb3922d1f499827`.
Raw local report SHA-256: `470d45c903845828f1ffbb47f9351b3c6236a5dd5b23b4b94d7a51bbcd7b117c`.
The report and failed predecessors remain under ignored
`work/dataset/bootstrap/fidelity/`; no generated image is published.
One 64-page production-supervisor batch passed in 26.88 seconds on this host.
All 28 JavaScript tests, 59 Python tests, source/privacy/provenance checks and
the offline build passed. Browser inference is unchanged; a new end-to-end
recognition evaluation and training/browser parity remain unrun.
