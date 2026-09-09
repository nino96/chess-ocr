"""Proposal registry/job tests with original temporary synthetic pages."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import ImageDraw

os.environ["CHESS_OCR_TESTING"] = "1"

from python import dataset_pipeline as p
from python import dataset_proposals as proposals


class ProposalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root_patch = patch.object(p, "ROOT", Path(self.temp.name) / "work")
        self.root_patch.start()
        p.initialize()
        p.set_budget(argparse.Namespace(sources=4, pages=20, download_bytes=2**20,
                     storage_bytes=2**28, cpu_seconds=7200, review_limit=20))
        self.train = self.sample("train", "train")
        self.development = self.sample("development", "dev")
        self.qualification = self.sample("qualification", "qualification")
        proposals.initialize()

    def tearDown(self):
        self.root_patch.stop()
        self.temp.cleanup()

    def sample(self, source_id, split):
        image = p.pillow().new("RGB", (160,160), "white")
        original = p.local_path(f"originals/{source_id}.png")
        original.parent.mkdir(parents=True, exist_ok=True)
        image.save(original)
        evidence = p.local_path(f"rights/{source_id}.evidence")
        p.atomic(evidence, b"original synthetic evidence")
        source = {"schema": p.SCHEMA, "id": source_id, "sha256": p.digest(original),
                  "split": split, "format": "png", "max_bytes": 100000, "pages": [1],
                  "revision": "1", "attribution": "Original test", "edition": "1",
                  "selection_reason": "test", "pretrained_overlap": "unknown", "private": False,
                  "real": True, "lineage_reviewed": True,
                  "lineage": {key: [source_id] for key in ("document","edition","artwork","parent")},
                  "conditions": ["procedural-test"], "url": "https://example.invalid/test.png",
                  "rights": {"reviewer": "test", "evidence_url": "https://example.invalid/rights",
                             "license": "original-test", "exclusions": "none", "review_date": "2026-09-07",
                             "evidence_sha256": p.digest(evidence), "acquisition": "approved",
                             "training": "approved", "evaluation": "approved", "redistribution": "denied",
                             "model_publication": "unknown"}}
        page = p.local_path(f"pages/{source_id}-1.png")
        page.parent.mkdir(parents=True, exist_ok=True)
        image.save(page)
        with p.connect() as db:
            db.execute("INSERT INTO sources VALUES (?,?,?)", (source_id,p.canonical(source),p.identity(source)))
            db.execute("INSERT INTO samples(id,source,page,image,sha,width,height,phash) VALUES (?,?,?,?,?,?,?,?)",
                       (f"{source_id}-1",source_id,1,f"pages/{source_id}-1.png",p.digest(page),160,160,"0"*16))
        return f"{source_id}-1"

    def test_registry_is_capability_separated_and_hash_bound(self):
        registry = proposals.providers()
        self.assertEqual(registry["schema"], "chess-ocr-provider-registry/1")
        self.assertEqual({x["capability"] for x in registry["providers"]}, {"localization","labels"})
        self.assertEqual(registry["defaults"], {"localization": "fenshot-localizer-v1",
                                                "labels": "fenshot-labeler-v1",
                                                "board_reread": "fenshot-rectified-labeler-v1"})
        for provider in registry["providers"]:
            self.assertEqual(len(provider["manifest_sha256"]), 64)
        with p.connect() as db:
            manifests = {row["id"]: json.loads(row["body"]) for row in db.execute(
                "SELECT id,body FROM provider_manifests")}
        self.assertIsNone(manifests["fenshot-localizer-v1"]["artifact"])
        self.assertIsNotNone(manifests["fenshot-labeler-v1"]["artifact"])
        self.assertEqual(manifests["fenshot-rectified-labeler-v1"]["preprocessing"],
                         "four-corner-rgba768-bilinear/fenshot-gray32-v1")
        self.assertNotEqual(manifests["fenshot-localizer-v1"]["model"]["sha256"],
                            manifests["fenshot-labeler-v1"]["model"]["sha256"])
        future = {**manifests["fenshot-labeler-v1"], "id": "future-localizer",
                  "capability": "localization", "runtime": "chess-ocr-onnx-localizer-v1",
                  "preprocessing": "yolox-rgb-imagenet-v2"}
        with self.assertRaisesRegex(p.Invalid, "configuration required"):
            proposals.validate_manifest(future, allow_builtin=True)
        future["configuration"] = {
            "kind": "chess-ocr-onnx-localizer/1", "input": "images",
            "output": "predictions", "inputShape": [1,3,416,416],
            "proposalScoreThreshold": .01, "calibratedAcceptanceThreshold": 1,
            "nmsIou": .65, "refinement": {
                "id": "nine-line-grid-refiner-v1",
                "implementationSha256": p.digest(p.REPO / "src/grid.ts"),
                "regionExpansion": .12, "outputSize": 768, "classifierTileSize": 96,
                "maxCandidates": 4,
            },
        }
        proposals.validate_manifest(future, allow_builtin=True)
        with p.connect() as db:
            db.execute("INSERT INTO provider_manifests VALUES (?,?,?,?,?,0)",
                       (future["id"], future["capability"], future["runtime"],
                        p.canonical(future), p.identity(future)))
        self.assertIn("future-localizer", {item["id"] for item in proposals.providers()["providers"]})

    def test_missing_square_evidence_must_remain_uncertain(self):
        with p.connect() as db:
            provider_bodies = [json.loads(row[0]) for row in db.execute(
                "SELECT body FROM provider_manifests WHERE id IN "
                "('fenshot-localizer-v1','fenshot-labeler-v1') ORDER BY capability")]
        result = {"schema": proposals.PROPOSAL_SCHEMA, "runId": "a"*64,
                  "sampleId": self.train, "revision": 0,
                  "imageSha256": p.digest(p.local_path(f"pages/{self.train}.png")),
                  "status": "ok", "warnings": [], "providers": provider_bodies,
                  "timings": {"totalMs": 1}, "boards": [{"id": "board",
                      "corners": [[0,0],[160,0],[160,160],[0,160]],
                      "labels": ["."]*64, "orientation": "unknown",
                      "probabilities": [None]*64, "uncertain": [False]*64}]}
        with self.assertRaisesRegex(p.Invalid, "missing proposal evidence"):
            proposals.validate_result(result, 160, 160)

    def test_train_scope_freezes_inputs_and_excludes_qualification(self):
        with patch.object(proposals, "_launch", side_effect=lambda run: {"state": "starting", "run_id": run}):
            result = proposals.create_run("fenshot-localizer-v1", "fenshot-labeler-v1", "train-pending", 10)
        with p.connect() as db:
            body = json.loads(db.execute("SELECT body FROM proposal_runs WHERE id=?", (result["run_id"],)).fetchone()[0])
        self.assertEqual([sample["id"] for sample in body["samples"]], [self.train])
        self.assertNotIn(self.qualification, p.canonical(body))
        with p.connect() as db:
            db.execute("UPDATE proposal_runs SET state='complete'")
        with self.assertRaisesRegex(p.Invalid, "exclude qualification"):
            proposals.create_run("fenshot-localizer-v1", "fenshot-labeler-v1", "qualification", 1)

    def test_development_scope_is_frozen_before_inference_and_excludes_qualification(self):
        with patch.object(proposals, "_launch", side_effect=lambda run: {"state": "starting", "run_id": run}):
            result = proposals.create_run("fenshot-localizer-v1", "fenshot-labeler-v1", "dev-pending", 10)
        with p.connect() as db:
            body = json.loads(db.execute("SELECT body FROM proposal_runs WHERE id=?", (result["run_id"],)).fetchone()[0])
        self.assertEqual([sample["id"] for sample in body["samples"]], [self.development])
        self.assertNotIn(self.qualification, p.canonical(body))

    def test_board_reread_binds_draft_geometry_provider_and_forbids_qualification(self):
        with p.connect() as db:
            db.execute("""CREATE TABLE web_drafts(sample TEXT PRIMARY KEY, revision INTEGER NOT NULL,
                       image_sha TEXT NOT NULL, version INTEGER NOT NULL, body TEXT NOT NULL)""")
            image_sha = db.execute("SELECT sha FROM samples WHERE id=?", (self.train,)).fetchone()[0]
            board = {"corners": [[0,0],[160,0],[160,160],[0,160]],
                     "labels": ["."]*64, "orientation": "unknown"}
            draft = {"boards": [board], "kind": "boards", "reviewer": "", "human": False,
                     "complete": False, "elapsed_seconds": 0, "proposal_base": [],
                     "proposal_run": None, "touched": {"boards": [], "corners": [],
                     "squares": [], "corrections": []}}
            db.execute("INSERT INTO web_drafts VALUES (?,?,?,?,?)",
                       (self.train,0,image_sha,1,p.canonical(draft)))
        request = {"sample_id": self.train, "revision": 0, "image_sha256": image_sha,
                   "draft_version": 1, "board_index": 0, "corners": board["corners"],
                   "labeler": "fenshot-rectified-labeler-v1"}
        with patch.object(proposals, "_launch_board_reread",
                          side_effect=lambda value: {"schema": proposals.REREAD_SCHEMA,
                                                     "request_id": value, "state": "starting"}):
            started = proposals.create_board_reread(request)
        with p.connect() as db:
            row = db.execute("SELECT body FROM board_rereads WHERE id=?", (started["request_id"],)).fetchone()
            body = json.loads(row[0])
            provider = body["labeler"]["manifest"]
            result = {"schema": proposals.REREAD_SCHEMA, "requestId": started["request_id"],
                      "sampleId": self.train, "revision": 0, "imageSha256": image_sha,
                      "draftVersion": 1, "boardIndex": 0, "corners": board["corners"],
                      "labels": ["P"] + ["."]*63, "probabilities": [None]*64,
                      "uncertain": [True]*64, "provider": provider, "warnings": [],
                      "timings": {"totalMs": 1}}
            proposals.validate_reread_result(result, body, 160, 160)
            db.execute("UPDATE board_rereads SET state='complete',result=? WHERE id=?",
                       (p.canonical(result), started["request_id"]))
        applied = proposals.apply_board_reread({"request_id": started["request_id"],
                  "sample_id": self.train, "revision": 0, "image_sha256": image_sha,
                  "draft_version": 1, "board_index": 0, "corners": board["corners"]})
        self.assertEqual(applied["result"]["labels"][0], "P")
        with p.connect() as db:
            db.execute("UPDATE web_drafts SET version=2 WHERE sample=?", (self.train,))
        with self.assertRaisesRegex(p.Invalid, "stale for this draft"):
            proposals.apply_board_reread({"request_id": started["request_id"],
                      "sample_id": self.train, "revision": 0, "image_sha256": image_sha,
                      "draft_version": 2, "board_index": 0, "corners": board["corners"]})
        with self.assertRaisesRegex(p.Invalid, "draft changed"):
            proposals.apply_board_reread({"request_id": started["request_id"],
                      "sample_id": self.train, "revision": 0, "image_sha256": image_sha,
                      "draft_version": 1, "board_index": 0, "corners": board["corners"]})

        with p.connect() as db:
            qualification_sha = db.execute("SELECT sha FROM samples WHERE id=?", (self.qualification,)).fetchone()[0]
            db.execute("INSERT INTO web_drafts VALUES (?,?,?,?,?)",
                       (self.qualification,0,qualification_sha,1,p.canonical(draft)))
        with self.assertRaisesRegex(p.Invalid, "TRAIN and DEV"):
            proposals.create_board_reread({**request, "sample_id": self.qualification,
                                           "image_sha256": qualification_sha})

    def test_board_reread_executes_the_pinned_runner_on_exact_geometry(self):
        with p.connect() as db:
            db.execute("""CREATE TABLE web_drafts(sample TEXT PRIMARY KEY, revision INTEGER NOT NULL,
                       image_sha TEXT NOT NULL, version INTEGER NOT NULL, body TEXT NOT NULL)""")
            image_sha = db.execute("SELECT sha FROM samples WHERE id=?", (self.train,)).fetchone()[0]
            board = {"corners": [[0,0],[160,0],[160,160],[0,160]],
                     "labels": ["."]*64, "orientation": "unknown"}
            draft = {"boards": [board], "kind": "boards", "reviewer": "", "human": False,
                     "complete": False, "elapsed_seconds": 0, "proposal_base": [],
                     "proposal_run": None, "touched": {"boards": [], "corners": [],
                     "squares": [], "corrections": []}}
            db.execute("INSERT INTO web_drafts VALUES (?,?,?,?,?)",
                       (self.train,0,image_sha,1,p.canonical(draft)))
        request = {"sample_id": self.train, "revision": 0, "image_sha256": image_sha,
                   "draft_version": 1, "board_index": 0, "corners": board["corners"],
                   "labeler": "fenshot-rectified-labeler-v1"}
        with patch.object(proposals, "_launch_board_reread",
                          side_effect=lambda value: {"schema": proposals.REREAD_SCHEMA,
                                                     "request_id": value, "state": "starting"}):
            started = proposals.create_board_reread(request)
        result = proposals.run_board_reread(started["request_id"])
        self.assertEqual(result["state"], "complete", result)
        self.assertEqual(len(result["result"]["labels"]), 64)
        self.assertEqual(result["result"]["corners"], board["corners"])

    def test_stale_heartbeat_run_is_explicitly_resumable(self):
        with patch.object(proposals, "_launch", side_effect=lambda run: {"state": "starting", "run_id": run}):
            result = proposals.create_run("fenshot-localizer-v1", "fenshot-labeler-v1", "train-pending", 1)
        with p.connect() as db:
            db.execute("UPDATE proposal_runs SET state='running',heartbeat=0 WHERE id=?", (result["run_id"],))
        self.assertEqual(proposals.status()["runs"][0]["state"], "heartbeat-stale-resume-required")
        with patch.object(proposals, "_launch", side_effect=lambda run: {"state": "running", "run_id": run}):
            resumed = proposals.resume(result["run_id"])
        self.assertEqual(resumed["state"], "running")

    def test_only_current_hash_bound_results_are_exposed(self):
        with p.connect() as db:
            provider_bodies = [json.loads(row[0]) for row in db.execute(
                "SELECT body FROM provider_manifests WHERE id IN ('fenshot-localizer-v1','fenshot-labeler-v1') ORDER BY capability")]
        result = {"schema": proposals.PROPOSAL_SCHEMA, "runId": "a"*64, "sampleId": self.train,
                  "revision": 0, "imageSha256": p.digest(p.local_path(f"pages/{self.train}.png")),
                  "status": "unsupported", "boards": [], "warnings": [], "providers": provider_bodies,
                  "timings": {"totalMs": 1}}
        with p.connect() as db:
            db.execute("INSERT INTO proposal_runs VALUES (?,?,?,?,?,NULL)",
                       ("a"*64,"complete",p.canonical({}),0,0))
            db.execute("INSERT INTO proposal_results VALUES (?,?,?,?,?,?)",
                       ("a"*64,self.train,0,result["imageSha256"],p.canonical(result),0))
        self.assertEqual(len(proposals.sample_results(self.train)["results"]), 1)
        with p.connect() as db:
            db.execute("UPDATE samples SET revision=1 WHERE id=?", (self.train,))
        self.assertEqual(proposals.sample_results(self.train)["results"], [])

    def test_deferral_is_bounded_stale_safe_and_never_accepts(self):
        payload = {"sample_id": self.train, "revision": 0,
                   "image_sha256": p.digest(p.local_path(f"pages/{self.train}.png")),
                   "reason": "extensive-repair", "elapsed_seconds": 12, "proposal_run": None}
        self.assertEqual(proposals.defer(payload)["state"], "deferred")
        with p.connect() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM review_deferrals").fetchone()[0], 1)
            self.assertEqual(db.execute("SELECT accepted FROM samples WHERE id=?", (self.train,)).fetchone()[0], 0)
        with self.assertRaisesRegex(p.Invalid, "invalid deferral"):
            proposals.defer({**payload, "reason": "free form"})
        with self.assertRaisesRegex(p.Invalid, "stale deferral"):
            proposals.defer({**payload, "revision": 1})

    def test_exact_fenshot_wasm_runner_emits_hash_bound_result(self):
        page = p.local_path(f"pages/{self.train}.png")
        image = p.pillow().new("RGB", (192, 192), "white")
        draw = ImageDraw.Draw(image)
        for row in range(8):
            for column in range(8):
                value = 32 if (row + column) % 2 else 224
                draw.rectangle((32 + column * 16, 32 + row * 16,
                                47 + column * 16, 47 + row * 16),
                               fill=(value, value, value))
        image.save(page)
        with p.connect() as db:
            db.execute("UPDATE samples SET sha=?,width=192,height=192 WHERE id=?",
                       (p.digest(page), self.train))
        with patch.object(proposals, "_launch", side_effect=lambda run: {"state": "starting", "run_id": run}):
            run = proposals.create_run("fenshot-localizer-v1", "fenshot-labeler-v1", "train-pending", 1)
        proposals._run_child(run["run_id"], self.train)
        result = p.read_json(p.local_path(f"proposals/staging/{run['run_id']}/{self.train}/result.json"))
        self.assertEqual(result["schema"], proposals.PROPOSAL_SCHEMA)
        self.assertEqual(result["sampleId"], self.train)
        self.assertEqual(result["imageSha256"], p.digest(p.local_path(f"pages/{self.train}.png")))
        self.assertEqual(result["status"], "ok")
        self.assertTrue(all(len(probabilities) == 13
                            for probabilities in result["boards"][0]["probabilities"]))


if __name__ == "__main__":
    unittest.main()
