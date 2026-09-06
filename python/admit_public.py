"""Admit one already-reviewed public PDF from a tracked registry, without fetching."""
import argparse
import json
from pathlib import Path

if __package__:
    from . import dataset_pipeline as d
else:
    import dataset_pipeline as d


def admit(registry, source_id, original, evidence):
    registry = d.read_json(registry)
    d.require(registry.get("schema") == "chess-ocr-public-provenance/1" and registry.get("public") is True, "reviewed public registry required")
    records = {r["id"]:r for r in registry["records"]}
    r = records[source_id]
    d.require(r["kind"] == "pdf" and len(r["evidence_ids"]) == 1, "single reviewed PDF/evidence pair required")
    e = records[r["evidence_ids"][0]]
    for path, entry in ((Path(original),r),(Path(evidence),e)):
        d.require(not path.is_symlink() and path.stat().st_size == entry["bytes"]
                  and path.stat().st_size <= entry["max_bytes"] and d.digest(path) == entry["sha256"], "public artifact integrity mismatch")
    body = {"schema": d.SCHEMA, "id": r["id"], "sha256": r["sha256"], "split": r["split"],
            "format": "pdf", "max_bytes": r["max_bytes"], "pages": r["pages"], "revision": r["revision"],
            "attribution": r["attribution"], "edition": r["revision"],
            "selection_reason": "Fixed PDF pages 20-43 before inference; retain negatives and difficulties",
            "pretrained_overlap": "unknown; public access does not establish training-inventory disjointness",
            "private": False, "real": True, "lineage_reviewed": True,
            "lineage": {"document": [r["lineage"][0]], "edition": [r["lineage"][0]],
                        "artwork": [r["lineage"][-1]], "parent": [r["lineage"][0]]},
            "conditions": r.get("conditions", ["historical-print", "hatching", "scan", "text-and-diagrams"]),
            "url": r["url"], "rights": {"reviewer": "agent-public-artifact-review",
                "evidence_url": e["url"], "evidence_sha256": e["sha256"], "review_date": "2026-09-07",
                "license": r["license"], "exclusions": "No font extraction or payload/model publication; page-pixel local use only",
                "acquisition": "approved", "training": "approved" if r["split"] == "train" else "unknown",
                "evaluation": "approved", "redistribution": "unknown", "model_publication": "unknown"}}
    with d.writer(),d.connect() as db:
        existing = db.execute("SELECT body FROM sources WHERE id=?", (source_id,)).fetchone()
        if existing:
            d.require(json.loads(existing[0]) == body, "existing source differs; immutable admission")
            return {"state":"already-admitted", "id":source_id}
        d.require_free_space(r["bytes"])
        destination = d.local_path(f"originals/{source_id}.pdf")
        if destination.exists():
            d.require(d.digest(destination) == r["sha256"], "stored original differs")
        else:
            d.atomic(destination, Path(original).read_bytes())
    manifest = d.local_path(f"bootstrap/{source_id}-admission.json")
    d.write_json(manifest, {**body,"evidence_file":str(Path(evidence).resolve())})
    return d.add_source(manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("registry", "source_id", "original", "evidence"):
        parser.add_argument(arg)
    args = parser.parse_args()
    print(json.dumps(admit(args.registry,args.source_id,args.original,args.evidence)))
