# Assisted-review bounded diagnostic smoke — 2026-09-07

Record produced on host gx10-b210 against issue #2 commit
`8155a3e0e2bd2672267b097cc72e87a0298cc899`, 2026-09-07. Nothing in this file is
current. The provider contract, job lifecycle and review workspace that remain in
force are in [assisted dataset review](../assisted-review.md).

## Bounded diagnostic result — 2026-09-07

A four-page TRAIN smoke run on `capablanca-1921-20` through
`capablanca-1921-23` was inspected against the source page pixels. It is
negative evidence for both current localization paths, not a provider promotion
comparison:

| Page | Visible board                 | FENShot proposal                        | Classical proposal                           |
| ---- | ----------------------------- | --------------------------------------- | -------------------------------------------- |
| 20   | none                          | false board over the text body          | false text-region grid                       |
| 21   | one board near the upper page | false lower-page grid; missed the board | two false lower-page grids; missed the board |
| 22   | one board near the lower page | false upper-page grid; missed the board | unsupported; missed the board                |
| 23   | none                          | false text-region grid                  | three false text-region grids                |

The labels are consequently not useful because they are extracted from the
wrong crops. The result does not justify more human time choosing between
FENShot and classical. Keep both as diagnostic/baseline evidence only and make
the issue #3 model-versus-FENShot comparison the next meaningful gate. The
source pages and proposal payloads remain local ignored evidence; no image,
crop, label or model artifact is committed.

The provider manifest originally hashed the whole shared TypeScript module.
That binding was corrected to hash only the classical implementation segment;
the startup migration recognizes the old v1 whole-file hash and updates only
the registry metadata. Existing proposal results retain their embedded
historical manifests.

Implementation and synthetic tests establish contracts, isolation, stale-result
handling and editing behavior. Per the 2026-09-07 owner supersession, no
classical-versus-FENShot human winner selection is required: FENShot remains the
baseline and classical remains diagnostic. The pending promotion evidence is the
issue #3 model versus FENShot on identical real development pages after its fixed
adapter is merged. The new model is not assumed to win.
