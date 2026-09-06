"""Offline tests using procedurally generated original rasters/PDFs in temp storage."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from python import dataset_pipeline as p


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root_patch = patch.object(p, "ROOT", Path(self.temp.name) / "work")
        self.root_patch.start()
        p.initialize()
        p.set_budget(argparse.Namespace(sources=12, pages=2000, download_bytes=2*1024**3,
                     storage_bytes=8*1024**3, cpu_seconds=14400, review_limit=20))

    def tearDown(self):
        p.signal.alarm(0)
        self.root_patch.stop()
        self.temp.cleanup()

    def image(self, name="original.png", color="white"):
        image = p.pillow().new("RGB", (160, 160), color)
        path = p.local_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        return path

    def source(self, sid="test", split="train", group="design"):
        original = self.image(f"originals/{sid}.png")
        evidence = p.local_path(f"rights/{sid}.evidence")
        p.atomic(evidence, b"original test evidence")
        body = {"schema": p.SCHEMA, "id": sid, "sha256": p.digest(original), "split": split,
                "format": "png", "max_bytes": 100000, "pages": [1], "revision": "1",
                "attribution": "Original procedural test", "edition": "1", "selection_reason": "test",
                "pretrained_overlap": "unknown", "private": False, "real": True,
                "lineage_reviewed": True, "lineage": {k: [group] for k in ("document", "edition", "artwork", "parent")},
                "conditions": ["procedural-test"], "url": "https://example.invalid/test.png",
                "rights": {"reviewer": "test", "evidence_url": "https://example.invalid/rights", "license": "original-test",
                           "exclusions": "none", "review_date": "2026-09-06", "evidence_sha256": p.digest(evidence),
                           "acquisition": "approved", "training": "approved", "evaluation": "approved",
                           "redistribution": "denied", "model_publication": "unknown"}}
        with p.connect() as db:
            db.execute("INSERT INTO sources VALUES (?,?,?)", (sid, p.canonical(body), p.identity(body)))
        return body

    def sample(self, sid="test", split="train", group="design"):
        self.source(sid, split, group)
        path = self.image(f"pages/{sid}-1.png")
        if sid == "reserved":
            from PIL import ImageDraw
            image = p.pillow().new("RGB", (160,160), "white")
            draw = ImageDraw.Draw(image)
            for i in range(64):
                x,y = i%8*20,i//8*20
                draw.rectangle((x,y,x+19,y+19), fill=(i*3,255-i*3,i))
            image.save(path)
            original = p.local_path(f"originals/{sid}.png")
            image.save(original)
            with p.connect() as db:
                body = json.loads(db.execute("SELECT body FROM sources WHERE id=?", (sid,)).fetchone()[0])
                body["sha256"] = p.digest(original)
                db.execute("UPDATE sources SET body=?,sha=? WHERE id=?", (p.canonical(body),p.identity(body),sid))
        with p.connect() as db:
            db.execute("INSERT INTO samples(id,source,page,image,sha,width,height,phash) VALUES (?,?,?,?,?,?,?,?)",
                       (sid+"-1", sid, 1, f"pages/{sid}-1.png", p.digest(path), 160, 160, "0"*16))
        return sid+"-1"

    def review_data(self, sample="test-1", reviewer="first", revision=0):
        return {"schema": "chess-ocr-dataset-review/1", "sample_id": sample, "revision": revision,
                "image_sha256": p.digest(p.local_path(f"pages/{sample}.png")), "reviewer": reviewer,
                "human": True, "elapsed_seconds": 45, "kind": "boards", "complete_page": True,
                "boards": [{"corners": [[0,0], [160,0], [160,160], [0,160]],
                            "labels": list("PNBRQKpnbrqk" + "."*52), "orientation": "unknown"}]}

    def submit(self, data):
        return p.submit_review(data)

    def accept(self, sample):
        self.submit(self.review_data(sample))

    def test_single_human_review_accepts_and_retry_is_free(self):
        sample = self.sample()
        data = self.review_data(sample)
        payload = p.review_payload(sample)
        self.assertEqual(payload["sample_id"], sample)
        self.assertTrue(payload["image_data_url"].startswith("data:image/png;base64,"))
        self.assertEqual(self.submit(data)["state"], "accepted")
        data["elapsed_seconds"] = 61
        retry = self.submit(data)
        self.assertEqual(retry, {"state": "accepted", "revision": 1, "idempotent": True})
        with p.connect() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM reviews").fetchone()[0], 1)
            self.assertEqual(db.execute("SELECT accepted FROM samples").fetchone()[0], 1)

    def test_stale_corrupt_and_nonhuman_overwrite_are_rejected(self):
        sample = self.sample()
        data = self.review_data(sample)
        self.submit(data)
        stale = self.review_data(sample, reviewer="late", revision=0)
        stale["boards"][0]["labels"][0] = "."
        with self.assertRaises(p.Invalid):
            self.submit(stale)
        correction = self.review_data(sample, reviewer="editor", revision=1)
        correction["boards"][0]["labels"][0] = "."
        self.assertEqual(self.submit(correction)["state"], "accepted")
        agent_change = self.review_data(sample, reviewer="agent", revision=2)
        agent_change["human"] = False
        agent_change["boards"][0]["labels"][1] = "."
        with self.assertRaisesRegex(p.Invalid, "nonhuman"):
            self.submit(agent_change)
        with p.connect() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM reviews").fetchone()[0], 2)
            self.assertEqual(json.loads(db.execute("SELECT annotation FROM samples").fetchone()[0])["boards"][0]["labels"][0], ".")
        corrupt = self.review_data(sample, reviewer="corrupt", revision=2)
        p.atomic(p.local_path("pages/test-1.png"), b"corrupt")
        with self.assertRaisesRegex(p.Invalid, "stale review/image"):
            self.submit(corrupt)

    def test_agent_review_does_not_qualify(self):
        sample = self.sample(split="qualification")
        data = self.review_data(sample)
        data["human"] = False
        self.assertEqual(self.submit(data)["state"], "needs-human-review")
        with p.connect() as db:
            self.assertEqual(db.execute("SELECT accepted FROM samples").fetchone()[0], 0)
        data.update(revision=1, reviewer="human", human=True)
        self.assertEqual(self.submit(data)["state"], "accepted")

    def test_legacy_single_human_pending_record_promotes_without_budget_charge(self):
        sample = self.sample()
        data = self.review_data(sample)
        content = p.canonical(p.annotation_validate(data, 160, 160))
        with p.connect() as db:
            db.execute("UPDATE samples SET revision=1,annotation=?,accepted=0 WHERE id=?", (content, sample))
            db.execute("INSERT INTO reviews(sample,revision,content,reviewer,human,seconds,decision,at) VALUES (?,?,?,?,?,?,?,?)",
                       (sample, 1, content, "legacy-human", 1, 45, "correction", 1))
            before = db.execute("SELECT COUNT(*),SUM(seconds) FROM reviews").fetchone()
        self.assertEqual(p.promote_legacy_single_human_reviews(), {"state": "legacy-single-human-promoted", "promoted": 1})
        self.assertEqual(p.promote_legacy_single_human_reviews()["promoted"], 0)
        with p.connect() as db:
            self.assertEqual(db.execute("SELECT accepted FROM samples WHERE id=?", (sample,)).fetchone()[0], 1)
            self.assertEqual(tuple(db.execute("SELECT COUNT(*),SUM(seconds) FROM reviews").fetchone()), tuple(before))

    def test_geometry_and_label_rejection(self):
        self.sample()
        data = self.review_data()
        for mutate in (lambda d: d["boards"][0]["corners"].reverse(),
                       lambda d: d["boards"][0]["labels"].pop(),
                       lambda d: d["boards"][0]["corners"][0].__setitem__(0, float("nan")),
                       lambda d: d.__setitem__("complete_page", False),
                       lambda d: d.__setitem__("kind", "negative")):
            copy = json.loads(json.dumps(data))
            mutate(copy)
            with self.assertRaises(p.Invalid):
                p.annotation_validate(copy, 160, 160)

    def test_integrity_and_duplicates_block_export(self):
        first = self.sample()
        second = self.sample("other", "dev", "other-art")
        self.accept(first)
        self.accept(second)
        with p.connect() as db:
            db.execute("INSERT OR REPLACE INTO duplicates VALUES (?,?,'exact',NULL)", tuple(sorted((first, second))))
            self.assertIn("unresolved-duplicate", p.validate(db)["errors"])
        with self.assertRaises(p.Invalid):
            p.resolve_duplicate(first, second, "distinct")
        p.resolve_duplicate(first, second, "duplicate")
        with p.connect() as db:
            self.assertIn("cross-split-duplicate", p.validate(db)["errors"])
        p.atomic(p.local_path("pages/test-1.png"), b"corrupt")
        with p.connect() as db:
            self.assertIn("page-integrity", p.validate(db)["errors"])

    def test_excluding_cross_split_duplicate_preserves_remaining_data(self):
        first = self.sample()
        second = self.sample("other", "dev", "other-art")
        self.accept(first)
        self.accept(second)
        p.resolve_duplicate(first,second,"duplicate")
        p.exclude_source("other","duplicate-lineage")
        with p.connect() as db:
            result = p.validate(db)
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["excluded_sources"], 1)
        self.assertEqual(result["coverage"]["train"]["boards"], 1)
        self.assertNotIn("dev",result["coverage"])

    def test_export_tensor_order_and_qualification_isolation(self):
        train = self.sample()
        reserved = self.sample("reserved", "qualification", "reserved-art")
        self.accept(train)
        self.accept(reserved)
        result = p.export_dataset()
        path = p.local_path("exports/" + result["export"])
        records = p.read_json(path / "records.json")["records"]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["sample"], train)
        self.assertEqual(records[0]["labels"], self.review_data()["boards"][0]["labels"])
        import array
        tensor = array.array("f")
        tensor.frombytes((path / "test-1-0.f32").read_bytes())
        self.assertEqual(len(tensor), 64*3*96*96)
        self.assertAlmostEqual(tensor[0], (1-.485)/.229, places=5)
        self.assertAlmostEqual(tensor[96*96], (1-.456)/.224, places=5)
        self.assertTrue((path / "test-1.txt").read_text().startswith("0 0.5 0.5 1 1"))
        with self.assertRaises(p.Invalid):
            p.export_dataset()

    def test_projective_grid_maps_image_relative_squares(self):
        Image = p.pillow()
        image = Image.new("RGB", (160,160))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(image)
        for i in range(64):
            x, y = i%8*20, i//8*20
            draw.rectangle((x,y,x+19,y+19), fill=(i*3, 255-i*3, i))
        grid = p.rectify(image, [[0,0],[160,0],[160,160],[0,160]])
        for i in range(64):
            self.assertEqual(grid.getpixel((i%8*96+48,i//8*96+48)), (i*3,255-i*3,i))

    def test_projective_center_matches_independent_diagonal_intersection(self):
        image = p.pillow().new("RGB", (160,160))
        image.putdata([(x,y,0) for y in range(160) for x in range(160)])
        corners = [[20,20],[140,40],[120,140],[40,120]]
        # Solving the two diagonal line equations gives (78,89.6).
        grid = p.rectify(image,corners,size=80)
        red,green,_ = grid.getpixel((40,40))
        self.assertLessEqual(abs(red-78),2)
        self.assertLessEqual(abs(green-89.6),2)

    def test_source_schema_and_split_leakage(self):
        body = self.source()
        self.assertEqual(p.source_validate(body), body)
        body["id"] = "new"
        body["split"] = "dev"
        body["evidence_file"] = str(p.local_path("rights/test.evidence"))
        path = p.local_path("manifest.json")
        p.write_json(path, body)
        with self.assertRaisesRegex(p.Invalid, "cross-split"):
            p.add_source(path)
        body["url"] = "http://localhost/test"
        with self.assertRaises(p.Invalid):
            p.source_validate(body)

    def test_budget_reservations_survive_failed_attempt(self):
        body = self.source()
        with p.connect() as db:
            db.execute("INSERT INTO jobs(source,stage) VALUES ('test','acquire')")
            job = db.execute("SELECT * FROM jobs").fetchone()
            p.reserve(db, job, body)
        with p.connect() as db:
            self.assertEqual(db.execute("SELECT SUM(bytes) FROM reservations").fetchone()[0], body["max_bytes"])
            budget = p.meta(db, "budget")
            budget["download_bytes"] = body["max_bytes"]
            p.set_meta(db, "budget", budget)
        with p.connect() as db:
            with self.assertRaises(p.Budget):
                p.reserve(db, job, body)

    def test_symlink_and_traversal_rejection(self):
        p.local_path("escape").symlink_to(Path(self.temp.name))
        with self.assertRaises(p.Invalid):
            p.local_path("escape/file")
        with self.assertRaises(p.Invalid):
            p.local_path("../outside")

    def test_local_pdf_ingestion_idempotent_and_bounded(self):
        image = p.pillow().new("RGB", (160,160), "white")
        path = p.local_path("inbox/book.pdf")
        image.save(path, "PDF", save_all=True, append_images=[image, image])
        args = argparse.Namespace(approve_local_use=True, group="owner-group", split="train", reviewer="owner", pages_per_pdf=2)
        result = p.ingest(args)
        self.assertEqual(result["added"], 1)
        self.assertEqual(p.ingest(args)["already_ingested"], 1)
        with p.connect() as db:
            body = json.loads(db.execute("SELECT body FROM sources").fetchone()[0])
            self.assertEqual(body["pages"], [1,3])
            self.assertTrue(body["private"])
            self.assertNotIn("book.pdf", p.canonical(body))
            self.assertEqual(db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0], 2)
        args.split = "dev"
        with self.assertRaises(p.Invalid):
            p.ingest(args)

    def test_child_render_and_finalize_recovery(self):
        self.source()
        with p.connect() as db:
            db.execute("INSERT INTO jobs(source,stage,page) VALUES ('test','render',1)")
            job = db.execute("SELECT * FROM jobs").fetchone()
        # Do not impose process resource limits on the test runner.
        with patch.object(p.resource, "setrlimit"):
            p.child(job["id"])
        with p.connect() as db:
            p.finalize(db, job)
        with patch.object(p.resource, "setrlimit"):
            p.child(job["id"])
        with p.connect() as db:
            p.finalize(db, job)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM samples").fetchone()[0], 1)
            self.assertEqual(db.execute("SELECT state FROM jobs").fetchone()[0], "done")

    def test_download_rejects_private_destinations_without_network(self):
        body = self.source()
        with patch.object(p.socket, "getaddrinfo", return_value=[(2,1,6,"",("127.0.0.1",443))]):
            with self.assertRaisesRegex(p.Invalid, "nonpublic"):
                p.fetch(body, p.local_path("download.part"))

    def test_worker_recovers_interrupted_attempt(self):
        self.source()
        with p.connect() as db:
            db.execute("INSERT INTO jobs(source,stage,page,state,attempts) VALUES ('test','render',1,'running',1)")
        class Process:
            pid = 999999999
            returncode = 0
            def poll(self):
                return self.returncode
        def launch(command, **kwargs):
            self.assertTrue(kwargs["pass_fds"])
            with patch.object(p.resource, "setrlimit"):
                p.child(int(command[-1]))
            return Process()
        with patch.object(p.subprocess, "Popen", side_effect=launch):
            result = p.run()
        self.assertEqual(result["jobs"], {"done": 1})
        self.assertEqual(result["pages"], 1)
        with p.connect() as db:
            self.assertEqual(db.execute("SELECT attempts FROM jobs").fetchone()[0], 2)

    def test_timeout_quarantine_and_explicit_repair_preserve_attempts(self):
        self.source()
        with p.connect() as db:
            db.execute("INSERT INTO jobs(source,stage,page,attempts) VALUES ('test','render',1,2)")
        class Process:
            pid = 999999999
            returncode = None
            def poll(self):
                return self.returncode
            def wait(self):
                self.returncode = -9
        with patch.object(p.subprocess, "Popen", return_value=Process()), \
             patch.object(p.time, "monotonic", side_effect=[0, 100]), \
             patch.object(p.os, "killpg") as kill:
            result = p.run()
        self.assertEqual(result["jobs"], {"quarantined": 1})
        self.assertEqual(result["worker"]["state"], "needs-repair")
        kill.assert_called()
        p.retry_job(1, True)
        with p.connect() as db:
            row = db.execute("SELECT attempts,max_attempts,state FROM jobs").fetchone()
            self.assertEqual(tuple(row), (3,4,"pending"))
            self.assertEqual(db.execute("SELECT COUNT(*) FROM reservations").fetchone()[0], 1)

    def test_stop_kills_attempt_and_keeps_retryable_work(self):
        self.source()
        with p.connect() as db:
            db.execute("INSERT INTO jobs(source,stage,page) VALUES ('test','render',1)")
        class Process:
            pid = 999999999
            returncode = None
            def poll(self):
                p.atomic(p.local_path("stop"), b"stop")
                return self.returncode
            def wait(self):
                self.returncode = -9
        with patch.object(p.subprocess, "Popen", return_value=Process()), patch.object(p.os, "killpg"):
            result = p.run()
        self.assertEqual(result["worker"]["state"], "stopped")
        self.assertEqual(result["jobs"], {"retry": 1})

    def test_download_validation_without_external_network(self):
        body = self.source()
        payload = b"test payload"
        body["sha256"] = hashlib.sha256(payload).hexdigest()
        class Response:
            status = 200
            def __init__(self):
                self.sent = False
            def getheader(self, key, default=None):
                return str(len(payload)) if key == "Content-Length" else default
            def read(self, size):
                if self.sent:
                    return b""
                self.sent = True
                return payload
        class Connection:
            def request(self, *args, **kwargs):
                pass
            def getresponse(self):
                return Response()
            def close(self):
                pass
        with patch.object(p.socket, "getaddrinfo", return_value=[(2,1,6,"",("8.8.8.8",443))]), \
             patch.object(p, "PinnedHTTPS", return_value=Connection()):
            p.fetch(body, p.local_path("download.part"))
            body["sha256"] = "0"*64
            with self.assertRaisesRegex(p.Invalid, "hash mismatch"):
                p.fetch(body, p.local_path("bad.part"))
            body["max_bytes"] = 1
            with self.assertRaisesRegex(p.Invalid, "byte ceiling"):
                p.fetch(body, p.local_path("large.part"))

    def test_failed_export_resumes_without_promoting_partial_files(self):
        sample = self.sample()
        self.accept(sample)
        with patch.object(p, "rectify", side_effect=RuntimeError("simulated process failure")):
            with self.assertRaises(RuntimeError):
                p.export_dataset()
        self.assertEqual(len(list(p.local_path("exports").glob(".partial-*"))), 1)
        result = p.export_dataset()
        self.assertEqual(len(list(p.local_path("exports").glob(".partial-*"))), 0)
        folder = p.local_path("exports/"+result["export"])
        for name, sha in p.read_json(folder/"hashes.json").items():
            self.assertEqual(p.digest(folder/name), sha)

    def test_single_writer_lock(self):
        with p.writer():
            with self.assertRaisesRegex(p.Invalid, "another writer"):
                with p.writer():
                    pass

    def test_real_pdf_worker_process_and_resume(self):
        image = p.pillow().new("RGB", (160,160), "white")
        image.save(p.local_path("inbox/original.pdf"), "PDF", save_all=True, append_images=[image])
        p.ingest(argparse.Namespace(approve_local_use=True, group="original", split="train",
                                   reviewer="test", pages_per_pdf=2))
        real_popen = subprocess.Popen
        def launch(command, **kwargs):
            # Isolate the CLI child in this temporary fixture workspace.
            code = "from python import dataset_pipeline as p; from pathlib import Path; import sys; p.ROOT=Path(sys.argv[1]); p.child(int(sys.argv[2]))"
            return real_popen([p.sys.executable, "-c", code, str(p.ROOT), command[-1]], **kwargs)
        with patch.object(p.subprocess, "Popen", side_effect=launch):
            result = p.run()
            again = p.run()
        self.assertEqual(result["jobs"], {"done": 2})
        self.assertEqual(result["pages"], 2)
        self.assertEqual(result["attempts"], again["attempts"])
        self.assertEqual(result["accepted_pages"], 0)


if __name__ == "__main__":
    unittest.main()
