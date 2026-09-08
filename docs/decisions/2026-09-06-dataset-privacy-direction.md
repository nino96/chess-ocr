# Issue #2 pipeline and dataset privacy direction

Status: superseded by [public provenance and reconstruction limits](../reproducibility.md)

Owner direction, 2026-09-06.

Dataset implementation can proceed independently of the deferred physical-device
checks. Reviewed public-source URLs, revisions, hashes, rights, selection and
reconstruction recipes belong in Git; private source details and operational or
mixed records remain ignored. Downloaded originals and generated datasets or
weights are never committed.

The [implemented pipeline](../dataset-pipeline.md) provides a local PDF inbox,
explicit local-use ingestion, bounded background rendering/acquisition, status,
stop/resume, hash/revision checks, one-human acceptance independent of model
proposals, and candidate train/dev exports. Complete-page targets preserve
negatives and exclude unsupported cases. Artwork independence remains a reviewed
property, not a count of downloaded files. The first real tranche, verified
coverage/lineage, measured human audit, approved synthetic fidelity and
downstream #3 preprocessing parity were undelivered at the time of this
decision. Pipeline mechanics do not establish
the dataset or the recognition outcome.
