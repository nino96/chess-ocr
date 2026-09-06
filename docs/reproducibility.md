# Public provenance and reproducibility

Owner update, 2026-09-07: reviewed **public-source metadata may be committed**.
This supersedes the earlier blanket restriction on all dataset metadata. Private
input identities, paths, annotations, images, derivatives, results and reviewer
identities remain excluded from Git/issues/PRs. Original public assets and generated
datasets/weights are still ignored; metadata permission is not asset publication.

## What a fresh clone can reproduce today

[Public bootstrap provenance](../provenance/public-bootstrap.json) records the exact
36 acquired piece SVGs, the public Skak manual, its selected pages (2–13), eight
license-evidence files, revisions, SHA-256/byte bounds, attribution and conservative
TRAIN-only artwork groups. It was explicitly projected from public records; no
operational database, personal reviewer identity or private source record was copied.
The strict public-provenance schema rejects extra fields, local file paths as URLs,
credentials, malformed hashes, duplicate IDs and missing evidence references.
Schema validation cannot prove a source is public: manual privacy/rights review is
still required. An HTTPS URL is not a grant to use or publish its contents.

A future agent can reacquire these public inputs into ignored storage, verify exact
hashes and byte ceilings, retain license evidence, and reconstruct the selected PDF
pages with the documented Pillow 11.1.0 / Poppler 24.02.0 pipeline (150 DPI,
maximum side 2400). The public registry is not itself a `dataset add` manifest:
prepare the reviewed local admission/evidence record as documented in
[the pipeline](dataset-pipeline.md), retaining the recorded split and lineage.
Do not replace a missing or changed upstream file with different bytes. Stop and
report the missing original; URL availability is not guaranteed by a hash.

**It cannot yet reproduce the same complete labeled dataset from Git alone.**
The [synthetic generator](synthetic-dataset.md) and public seed recipe are now
implemented, with a required independent fidelity gate before bulk use. The human's accepted page labels,
geometry, original review attestations and operational history remain locally saved
and exported; public-source links cannot reconstruct those decisions. This update
does not publish those annotations or fabricate fresh human reviews. A new session
on the same GX10 retains that local data; a fresh clone on another machine does not.
Private inputs must always be supplied separately by an authorized owner.

## Required as subsequent stages are delivered

- Commit generator/acquisition tooling and dependency locks, exact public asset
  lists, selection and split recipes, seed/configuration and transform provenance.
- Version privacy-reviewed public factual annotations separately when publication
  is authorized; never include reviewer identifiers or mixed private records.
  Where labels stay local, document the resulting reconstruction limitation.
- Record output hashes and renderer/code versions; test rebuilding a bounded
  public subset with the same inputs before claiming exact reproducibility.
- Keep private-derived artifacts, all downloaded originals and generated payloads
  ignored. Never bypass payload protection or use Git LFS to publish them.

## Rights scope of this public seed

Lichess's pinned per-file table identifies the selected sets as CC0, Apache-2.0 and
MIT; the top-level site code license is not used as their artwork license. The
manual and its chess fonts are LPPL; its other font families have the separately
recorded terms. Local acquisition/training was reviewed, not corpus/font/model
publication. The CM font attribution is family-level, not exact pre-subsetting
binary identification. Existing Lichess designs may overlap FENShot training and
are not fresh qualification. Public metadata publication neither changes these
rights decisions nor certifies unseen data, independent artwork or label accuracy.

No acquisition/training job runs when these metadata checks run:

```sh
pnpm run check
pnpm test
```
