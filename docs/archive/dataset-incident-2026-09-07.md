# Dataset test-isolation incident — 2026-09-07

Record of an incident on host gx10-b210 at the 2026-09-07 dataset-branch commit.
Nothing in this file is current. The prevention changes it describes are now
standing rules; the live statements of those rules are in
[the dataset pipeline guide](../dataset-pipeline.md) and [AGENTS.md](../../AGENTS.md).

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
found. Recovery was unsuccessful; do not fabricate review history, describe
proposals as recovered truth, or silently substitute regenerated labels.

## Prevention changes

- Package imports now use package-relative modules; script execution uses the
  matching direct-import modules. The server and reset tests assert module identity.
- Dataset test processes set an explicit test-mode flag. Dataset path resolution
  refuses the live dataset root even if a future test imports the wrong module.
- A regression imports both module names and checks ownership, and directly checks
  that test-mode live-root access is rejected.

The synthetic bulk job was not launched. Its independent comparison at the
incident commit passed the 39 design/class calibration controls and 2,496
piece-identity checks, while three small transformed-board controls still
exceeded the retained MAE limit. **Superseded:** that MAE failure was fixed; the
later passing prelaunch validation is
[the 2026-09-07 synthetic prelaunch record](synthetic-prelaunch-validation-2026-09-07.md),
and the current fidelity gate is described in
[the synthetic dataset guide](../synthetic-dataset.md). Those were the gate
results at the incident commit, not final fidelity evidence.

The owner subsequently stated there is no backup, explicitly abandoned the lost
labeled documents, and authorized continuing the dataset goal. Recovery of those
records is no longer a prerequisite. The loss remains permanent in our accounting;
new annotations start as unverified proposals, not restored human truth. Existing
resource charges and surviving evidence must remain intact.

The owner subsequently required commits before unattended launches. Future runs
must start from reviewed committed code. Preserve surviving originals and evidence;
do not imply that a post-incident snapshot contains the lost human decisions.
No asset/model payload or private annotation is included in this report.
