"""Offline tests of reservation, immutable batch recovery and training ordering."""
import copy
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
import os
from collections import namedtuple
from unittest.mock import patch

os.environ["CHESS_OCR_TESTING"] = "1"

from python import dataset_pipeline as d
from python import synthetic_job as j
from python import synthetic_training as t
from python import admit_public


class SyntheticTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.old_d, self.old_j = d.ROOT, j.ROOT
        d.ROOT = self.root
        j.ROOT = self.root / "synthetic"
        d.initialize()
        self.gate = patch.object(j, "fidelity_gate", return_value="test-gate")
        self.gate.start()
        self.coverage_gate=patch.object(j,"validate_coverage")
        self.coverage_gate.start()
        with d.connect() as db:
            d.set_meta(db, "budget", {"sources": 64, "pages": 2000,
                       "download_bytes": 10000, "storage_bytes": 64*1024**3,
                       "cpu_seconds": 720000, "review_limit": 2000})

    def tearDown(self):
        self.gate.stop()
        self.coverage_gate.stop()
        d.ROOT, j.ROOT = self.old_d, self.old_j
        self.tmp.cleanup()

    def config(self):
        c = j.configuration()
        c.update(pages=2, batch_pages=1, compute_seconds=240)
        return c

    def recipe(self, index):
        return {"index": index, "seed": 1, "width": 80, "height": 80,
                "kind": "boards", "boards": [{"labels": list(d.LABELS)*4+list(d.LABELS[:12]), "set":"chessnut", "position_kind":"teaching", "orientation":"unknown",
                "corners": [[0, 0], [80, 0], [80, 80], [0, 80]]}]}

    def fake_node(self, command, *args, **kwargs):
        if command == "recipes":
            return [self.recipe(i) for i in range(args[1], args[1]+args[2])]
        recipes = json.loads(Path(args[0]).read_text())
        records = []
        for recipe in recipes:
            name = f"page-{recipe['index']:06d}.png"
            path = Path(args[1]) / name
            d.pillow().new("RGB", (80, 80), "white").save(path)
            records.append({"recipe": recipe, "image": name, "image_sha256": d.digest(path)})
        return records

    def test_resume_reuses_batches_and_preserves_reservations(self):
        with patch.object(j, "configuration", return_value=self.config()), \
             patch.object(j, "frozen_identity", side_effect=lambda c: {"configuration": c, "files": {}}), \
             patch.object(j, "node", side_effect=self.fake_node) as node:
            j.initialize()
            j.run()
            count = node.call_count
            j.run()
            self.assertEqual(node.call_count, count)
            self.assertEqual(j.status()["completed_pages"], 2)
            self.assertEqual(j.status()["reserved_attempt_seconds"], 240)
            with d.connect() as db:
                self.assertEqual(d.cumulative_accounting(db)["reserved_compute_seconds"], 240)
            image = j.ROOT / "images/page-000000.png"
            image.write_bytes(b"corrupt")
            with self.assertRaises(d.Invalid):
                j.run()

    def test_changed_inputs_rejected(self):
        with patch.object(j, "configuration", return_value=self.config()), \
             patch.object(j, "frozen_identity", side_effect=lambda c: {"configuration": c, "files": {"code": "old"}}), \
             patch.object(j, "node", side_effect=self.fake_node):
            j.initialize()
        with patch.object(j, "frozen_identity", return_value={"files": {"code": "new"}}):
            with self.assertRaisesRegex(d.Invalid, "changed code"):
                j.run()

    def test_failed_attempt_charged_and_budget_stops_retry(self):
        c = self.config()
        c["compute_seconds"] = 120
        with patch.object(j, "configuration", return_value=c), \
             patch.object(j, "frozen_identity", side_effect=lambda c: {"configuration": c, "files": {}}), \
             patch.object(j, "node", side_effect=self.fake_node):
            j.initialize()
            with patch.object(j, "node", side_effect=d.Invalid("renderer time ceiling")):
                with self.assertRaises(d.Invalid):
                    j.run()
            self.assertEqual(j.status()["reserved_attempt_seconds"], 120)
            with self.assertRaisesRegex(d.Invalid, "compute reservation exhausted"):
                j.run()

    def test_stop_at_batch_boundary_then_resume(self):
        def stopping(*args, **kwargs):
            result = self.fake_node(*args, **kwargs)
            if args[0] == "render":
                d.atomic(j.ROOT / "stop", b"stop")
            return result
        with patch.object(j, "configuration", return_value=self.config()), \
             patch.object(j, "frozen_identity", side_effect=lambda c: {"configuration": c, "files": {}}), \
             patch.object(j, "node", side_effect=stopping):
            j.initialize()
            j.run()
            self.assertEqual(j.status()["state"], "stopped")
            self.assertEqual(j.status()["completed_pages"], 1)
            (j.ROOT / "stop").unlink()
            j.run()
            self.assertEqual(j.status()["state"], "complete")

    def test_training_tensor_order_and_negative_exclusion(self):
        image = d.pillow().new("RGB", (80, 80))
        for y in range(80):
            for x in range(80):
                image.putpixel((x, y), ((y//10*8+x//10)*4, 80, 160))
        path = self.root / "test.png"
        image.save(path)
        record = {"recipe": self.recipe(0), "image_sha256": d.digest(path)}
        result = t.load_board(path, record, 0)
        self.assertEqual(result["shape"], [64, 3, 96, 96])
        self.assertEqual(result["labels"], [d.LABELS.index(x) for x in self.recipe(0)["boards"][0]["labels"]])
        for i in range(64):
            self.assertAlmostEqual(result["tensor"][i*3*96*96+48*96+48], (i*4/255-.485)/.229, places=5)
        self.assertEqual(t.detector_targets(record), [[.5, .5, 1, 1]])
        record["recipe"]["kind"] = "partial"
        with self.assertRaises(d.Invalid):
            t.detector_targets(record)

    def test_free_space_floor_includes_pending_output(self):
        usage = namedtuple("usage", "total used free")
        with patch.object(d.shutil, "disk_usage", return_value=usage(10000, 6000, 4000)):
            d.require_free_space(1000, concurrent_bytes=0)
            with self.assertRaisesRegex(d.Budget, "30 percent"):
                d.require_free_space(1001, concurrent_bytes=0)

    def test_public_admission_is_hash_bound_and_idempotent(self):
        original, evidence = self.root / "original.pdf", self.root / "evidence.txt"
        d.atomic(original, b"%PDF-original test fixture")
        d.atomic(evidence, b"Original test rights evidence")
        registry = self.root / "registry.json"
        record = {"id":"test-source", "kind":"pdf", "evidence_ids":["test-rights"],
                  "bytes":original.stat().st_size, "max_bytes":10000, "sha256":d.digest(original),
                  "split":"train", "pages":list(range(20,44)), "revision":"original test source",
                  "attribution":"test", "lineage":["test-document","test-artwork"],
                  "license":"original test only", "url":"https://example.org/test.pdf", "conditions":["digital-print"]}
        rights = {"id":"test-rights", "kind":"license-evidence", "bytes":evidence.stat().st_size,
                  "max_bytes":10000, "sha256":d.digest(evidence), "url":"https://example.org/rights"}
        d.write_json(registry, {"schema":"chess-ocr-public-provenance/1", "public":True,"records":[record,rights]})
        self.assertEqual(admit_public.admit(registry,"test-source",original,evidence)["state"],"queued")
        self.assertEqual(admit_public.admit(registry,"test-source",original,evidence)["state"],"already-admitted")
        with d.connect() as db:
            body=json.loads(db.execute("SELECT body FROM sources WHERE id='test-source'").fetchone()[0])
            self.assertEqual(body["conditions"],["digital-print"])
        d.atomic(original,b"changed")
        with self.assertRaises(d.Invalid):
            admit_public.admit(registry,"test-source",original,evidence)

    def test_crash_reservation_is_recovered_from_sqlite(self):
        c = self.config(); c["compute_seconds"] = 360
        with patch.object(j, "configuration", return_value=c), \
             patch.object(j, "frozen_identity", side_effect=lambda c: {"configuration":c,"files":{}}), \
             patch.object(j, "node", side_effect=self.fake_node):
            j.initialize()
            frozen=d.read_json(j.ROOT/"frozen.json")
            with d.connect() as db:
                db.execute("INSERT INTO synthetic_attempts VALUES (?,?,?)",(frozen["run_id"],0,d.canonical({"start":0,"pages":1,"at":1,"state":"running"})))
            j.run()
            self.assertEqual(j.status()["reserved_attempt_seconds"],360)
            self.assertEqual(j.status()["failed_or_interrupted_attempts"],1)
            state=d.read_json(j.ROOT/"state.json");state["reserved_attempt_seconds"]=-1
            d.write_json(j.ROOT/"state.json",state)
            with self.assertRaisesRegex(d.Invalid,"accounting mismatch"):
                j.run()

    def test_renderer_timeout_terminates_process_group(self):
        real_popen = subprocess.Popen
        processes = []
        def sleeper(command, **kwargs):
            proc=real_popen(["timeout","--signal=KILL","1",sys.executable,"-c","import time; time.sleep(60)"],**kwargs)
            processes.append(proc)
            return proc
        with patch.object(j.subprocess,"Popen",side_effect=sleeper):
            with self.assertRaises(d.Invalid):
                j.node("unused",timeout=1)
        self.assertIsNotNone(processes[0].poll())

    def test_failed_database_reservation_remains_resumable(self):
        import sqlite3
        with patch.object(j, "configuration", return_value=self.config()), \
             patch.object(j, "frozen_identity", side_effect=lambda c: {"configuration":c,"files":{}}), \
             patch.object(j, "node", side_effect=self.fake_node):
            j.initialize()
            with d.connect() as db:
                db.execute("CREATE TRIGGER fail_attempt BEFORE INSERT ON synthetic_attempts BEGIN SELECT RAISE(ABORT, 'injected disk failure'); END")
            with self.assertRaises(sqlite3.IntegrityError):
                j.run()
            self.assertEqual(j.status()["reserved_attempt_seconds"], 0)
            self.assertEqual(j.status()["attempt_count"], 0)
            with d.connect() as db:
                db.execute("DROP TRIGGER fail_attempt")
            j.run()
            self.assertEqual(j.status()["state"], "complete")

    def test_incomplete_coverage_rejects_before_reservation(self):
        self.coverage_gate.stop()
        with patch.object(j,"configuration",return_value=self.config()), \
             patch.object(j,"frozen_identity",side_effect=lambda c:{"configuration":c,"files":{}}), \
             patch.object(j,"node",side_effect=self.fake_node):
            with self.assertRaisesRegex(d.Invalid,"missing class/background/effect"):
                j.initialize()
            with d.connect() as db:
                self.assertEqual(d.cumulative_accounting(db)["reserved_compute_seconds"],0)
            self.assertFalse((j.ROOT/"frozen.json").exists())

    def test_live_dataset_is_refused_even_with_duplicate_import(self):
        with patch.object(d,"ROOT", d.REPO/"work/dataset"):
            with self.assertRaisesRegex(d.Invalid,"test process refused"):
                d.local_path("state.sqlite3")
        code = "import sys; sys.path.insert(0,'python'); import dataset_pipeline; from python import dataset_pipeline as p, dataset_server as s, dataset_reset as r; assert s.p is p and r.p is p and s.reset is r"
        subprocess.run([sys.executable,"-c",code],cwd=d.REPO,check=True,timeout=10)


if __name__ == "__main__":
    unittest.main()
