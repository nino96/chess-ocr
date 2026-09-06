# Dataset test-isolation incident — 2026-09-07

The lead ran `python -m unittest discover -s python -p 'test_dataset*.py'`
instead of the documented canonical-package invocation. Discovery made both
`dataset_pipeline` and `python.dataset_pipeline` importable. The server's
absolute-first import selected the live-root module while tests patched the
other module's root to temporary storage.

The archive-deletion HTTP test consequently reset the live dataset and deleted
the newly created recovery archive. This removed the accepted 12-page/16-board
seed, review history, managed exports and rendered pages. Earlier hash checks had
confirmed that seed was unchanged before this test; those hashes are not backups.
The newly admitted historical queue's managed state was also removed. The
original public PDFs and synthetic SVGs retained outside managed payload paths
survive. They cannot reconstruct human decisions.

The lead preserved a post-incident database copy locally. Its integrity check
passes, but accepted labels/reviews are absent and SQLite secure deletion is
enabled. No surviving managed recovery archive or complete label export has been
found. Recovery remains unresolved; do not fabricate review history, describe
proposals as recovered truth, or silently substitute regenerated labels.

## Prevention changes

- Package imports now use package-relative modules; script execution uses the
  matching direct-import modules. The server and reset tests assert module identity.
- Dataset test processes set an explicit test-mode flag. Dataset path resolution
  refuses the live dataset root even if a future test imports the wrong module.
- A regression imports both module names and checks ownership, and directly checks
  that test-mode live-root access is rejected.

The synthetic bulk job was not launched. Its latest independent comparison passes
the 39 design/class calibration controls and 2,496 piece-identity checks, but
three small transformed-board controls still exceed the retained MAE limit.
That fidelity gate remains failed, separately from the lost reviewed-seed blocker.

The owner subsequently required commits before unattended launches. Future runs
must start from reviewed committed code and verified recoverable data snapshots.
No asset/model payload or private annotation is included in this report.
