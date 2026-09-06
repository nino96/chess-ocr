"""Reset tests use only procedural local files in a temporary dataset root."""
import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from python import dataset_pipeline as p
from python import dataset_reset as reset


class DatasetResetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.patch = patch.object(p, "ROOT", Path(self.temp.name) / "dataset")
        self.patch.start()
        self.assertIs(reset.p, p)
        p.initialize()
        p.set_budget(argparse.Namespace(sources=3, pages=10, download_bytes=1000,
                     storage_bytes=20 * 1024 * 1024, cpu_seconds=1000, review_limit=3))
        p.atomic(p.local_path("inbox/keep.pdf"), b"%PDF- procedural inbox")
        image = p.pillow().new("RGB", (32, 32), "white")
        original = p.local_path("originals/source.png")
        original.parent.mkdir(parents=True, exist_ok=True)
        image.save(original)
        page = p.local_path("pages/source-1.png")
        page.parent.mkdir(parents=True, exist_ok=True)
        image.save(page)
        with p.connect() as db:
            db.execute("INSERT INTO sources VALUES (?,?,?)", ("source", "{}", "test"))
            db.execute("INSERT INTO jobs(source,stage,state) VALUES ('source','render','done')")
            db.execute("INSERT INTO samples(id,source,page,image,sha,width,height,phash) VALUES (?,?,?,?,?,?,?,?)",
                       ("source-1", "source", 1, "pages/source-1.png", p.digest(page), 32, 32, "0" * 16))
            db.execute("INSERT INTO reviews(sample,revision,content,reviewer,human,seconds,decision,at) VALUES (?,?,?,?,?,?,?,?)",
                       ("source-1", 0, "{}", "human", 1, 12.5, "confirmation", 1))
            db.execute("INSERT INTO reservations(job,bytes,seconds,at) VALUES (?,?,?,?)", (1, 33, 44, 1))
            db.execute("CREATE TABLE web_drafts(sample TEXT PRIMARY KEY, revision INTEGER, image_sha TEXT, version INTEGER, body TEXT)")
            db.execute("INSERT INTO web_drafts VALUES ('source-1',0,'x',1,'{}')")

    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()

    def counts(self):
        with p.connect() as db:
            return {name: db.execute("SELECT COUNT(*) FROM " + name).fetchone()[0]
                    for name in ("sources", "jobs", "samples", "reviews", "reservations", "web_drafts")}

    def test_confirmation_lock_and_archive_reset(self):
        with self.assertRaisesRegex(p.Invalid, "START OVER"):
            reset.reset_dataset("start over")
        with p.writer():
            with self.assertRaisesRegex(p.Invalid, "writer is active"):
                reset.reset_dataset("START OVER")
        result = reset.reset_dataset("START OVER")
        self.assertEqual(result["state"], "reset")
        self.assertTrue(result["archive"].startswith("archives/"))
        self.assertTrue(p.local_path("inbox/keep.pdf").exists())
        archive = p.local_path(result["archive"])
        self.assertTrue((archive / "state.sqlite3").is_file())
        self.assertTrue((archive / "originals/source.png").is_file())
        self.assertEqual(self.counts(), {name: 0 for name in self.counts()})
        status = p.status()
        self.assertEqual((status["reserved_download_bytes"], status["reserved_compute_seconds"], status["attempts"]), (33, 44, 1))
        self.assertEqual((status["review_decisions"], status["review_seconds"]), (1, 12.5))
        self.assertEqual(status["budget"]["review_limit"], 3)
        again = reset.reset_dataset("START OVER")
        self.assertNotEqual(again["archive"], result["archive"])
        self.assertEqual(p.status()["review_decisions"], 1)

    def test_interrupted_payload_move_recovers_before_status(self):
        with self.assertRaisesRegex(RuntimeError, "injected"):
            reset.reset_dataset("START OVER", _interrupt_at="payload-moved")
        self.assertTrue(p.local_path("reset.pending.json").exists())
        # Any normal pipeline read finishes the durable operation first.
        status = p.status()
        self.assertFalse(p.local_path("reset.pending.json").exists())
        self.assertEqual(status["sources"], 0)
        self.assertEqual(status["review_decisions"], 1)
        self.assertTrue(p.local_path("inbox/keep.pdf").exists())

    def test_prepared_phase_recovery(self):
        with self.assertRaisesRegex(RuntimeError, "injected"):
            reset.reset_dataset("START OVER", _interrupt_at="prepared")
        self.assertEqual(p.status()["sources"], 0)
        self.assertFalse(p.local_path("reset.pending.json").exists())

    def test_backed_up_phase_recovery(self):
        with self.assertRaisesRegex(RuntimeError, "injected"):
            reset.reset_dataset("START OVER", _interrupt_at="backed-up")
        self.assertEqual(p.status()["sources"], 0)
        self.assertFalse(p.local_path("reset.pending.json").exists())

    def test_partial_archive_is_not_accepted_during_recovery(self):
        p.local_path("archives/partial").mkdir(parents=True)
        with p.connect() as db:
            carryover = p.cumulative_accounting(db)
        p.write_json(p.local_path("reset.pending.json"), {"schema": "chess-ocr-dataset-reset/1",
                     "phase": "prepared", "archive": "archives/partial",
                     "carryover": carryover})
        with self.assertRaisesRegex(p.Invalid, "incomplete reset archive"):
            p.status()

    def test_review_ceiling_uses_lifetime_carryover_after_reset(self):
        reset.reset_dataset("START OVER")
        with p.connect() as db:
            budget = p.meta(db, "budget")
            budget["review_limit"] = 1
            p.set_meta(db, "budget", budget)
            db.execute("INSERT INTO sources VALUES (?,?,?)", ("new", "{}", "new"))
        image = p.pillow().new("RGB", (32, 32), "white")
        page = p.local_path("pages/new-1.png")
        page.parent.mkdir(parents=True)
        image.save(page)
        with p.connect() as db:
            db.execute("INSERT INTO samples(id,source,page,image,sha,width,height,phash) VALUES (?,?,?,?,?,?,?,?)",
                       ("new-1", "new", 1, "pages/new-1.png", p.digest(page), 32, 32, "0" * 16))
        review = {"schema": "chess-ocr-dataset-review/1", "sample_id": "new-1", "revision": 0,
                  "image_sha256": p.digest(page), "reviewer": "new-human", "human": True,
                  "elapsed_seconds": 2, "kind": "negative", "boards": [], "complete_page": True}
        with self.assertRaisesRegex(p.Invalid, "review batch limit"):
            p.submit_review(review)

    def test_compute_budget_gates_do_not_refund_after_reset(self):
        reset.reset_dataset("START OVER")
        with p.connect() as db:
            budget = p.meta(db, "budget")
            budget["cpu_seconds"] = 44
            p.set_meta(db, "budget", budget)
        args = argparse.Namespace(approve_local_use=True, group="same", split="train", reviewer="owner", pages_per_pdf=1)
        with self.assertRaisesRegex(p.Invalid, "inspection compute budget"):
            p.ingest(args)
        with p.connect() as db:
            with self.assertRaisesRegex(p.Budget, "reservation"):
                p.reserve(db, {"id": 7, "stage": "render"}, {"max_bytes": 1})
            budget = p.meta(db, "budget")
            budget["cpu_seconds"] = 1000
            budget["download_bytes"] = 33
            p.set_meta(db, "budget", budget)
            with self.assertRaisesRegex(p.Budget, "reservation"):
                p.reserve(db, {"id": 8, "stage": "acquire"}, {"max_bytes": 1})
            budget["cpu_seconds"] = 44
            p.set_meta(db, "budget", budget)
        page = p.local_path("pages/export-1.png")
        page.parent.mkdir(parents=True)
        p.pillow().new("RGB", (32, 32), "white").save(page)
        annotation = p.canonical({"kind": "negative", "boards": [], "complete_page": True})
        with p.connect() as db:
            db.execute("INSERT INTO sources VALUES (?,?,?)", ("export", '{"split":"train"}', "export"))
            db.execute("INSERT INTO samples(id,source,page,image,sha,width,height,phash,annotation,accepted) VALUES (?,?,?,?,?,?,?,?,?,1)",
                       ("export-1", "export", 1, "pages/export-1.png", p.digest(page), 32, 32, "0" * 16, annotation))
        with patch.object(p, "validate", return_value={"errors": []}):
            with self.assertRaisesRegex(p.Invalid, "export compute reservation"):
                p.build_export()

    def test_archive_listing_delete_confirmation_and_retained_active_state(self):
        result = reset.reset_dataset("START OVER")
        archive = reset.list_archives()["archives"][0]
        self.assertEqual(archive["id"], Path(result["archive"]).name)
        self.assertGreater(archive["bytes"], 0)
        p.atomic(p.local_path("pages/active.txt"), b"new active data")
        before = p.status()
        with self.assertRaisesRegex(p.Invalid, "DELETE"):
            reset.delete_archive(archive["id"], archive["version"], "")
        with p.writer():
            with self.assertRaisesRegex(p.Invalid, "writer is active"):
                reset.delete_archive(archive["id"], archive["version"], "DELETE")
        deleted = reset.delete_archive(archive["id"], archive["version"], "DELETE")
        self.assertEqual(deleted["state"], "deleted")
        self.assertEqual(reset.list_archives(), {"archives": []})
        self.assertTrue(p.local_path("inbox/keep.pdf").exists())
        self.assertEqual(p.local_path("pages/active.txt").read_bytes(), b"new active data")
        after = p.status()
        for key in ("budget", "review_decisions", "review_seconds", "reserved_compute_seconds", "reserved_download_bytes", "attempts"):
            self.assertEqual(after[key], before[key])
        self.assertLess(after["storage_bytes"], before["storage_bytes"])
        self.assertEqual(reset.delete_archive(archive["id"], archive["version"], "DELETE")["state"], "already-deleted")

    def test_archive_delete_rejects_traversal_links_and_changed_selection(self):
        reset.reset_dataset("START OVER")
        archive = reset.list_archives()["archives"][0]
        for name in ("..", "../inbox", "/", "inbox", "archives/" + archive["id"]):
            with self.assertRaises(p.Invalid):
                reset.delete_archive(name, archive["version"], "DELETE")
        folder = p.local_path("archives/" + archive["id"])
        p.atomic(folder / "extra.txt", b"changed since listing")
        with self.assertRaisesRegex(p.Invalid, "archive changed"):
            reset.delete_archive(archive["id"], archive["version"], "DELETE")
        self.assertTrue((folder / "state.sqlite3").exists())
        link = folder / "link"
        link.symlink_to(p.local_path("inbox"), target_is_directory=True)
        self.assertIsNotNone(reset.list_archives()["archives"][0]["error"])
        with self.assertRaisesRegex(p.Invalid, "link"):
            reset.delete_archive(archive["id"], archive["version"], "DELETE")
        self.assertTrue(p.local_path("inbox/keep.pdf").exists())
        link.unlink()
        root_link = p.local_path("archives/20260906T000000Z-aaaaaaaaaaaa")
        root_link.symlink_to(p.local_path("inbox"), target_is_directory=True)
        with self.assertRaises(p.Invalid):
            reset.delete_archive(root_link.name, archive["version"], "DELETE")

    def test_archive_deletion_can_resume_after_partial_failure(self):
        reset.reset_dataset("START OVER")
        archive = reset.list_archives()["archives"][0]
        original_unlink = reset.os.unlink
        calls = 0

        def fail_once(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("simulated interruption")
            return original_unlink(*args, **kwargs)

        with patch.object(reset.os, "unlink", side_effect=fail_once):
            with self.assertRaisesRegex(p.Invalid, "Deletion stopped"):
                reset.delete_archive(archive["id"], archive["version"], "DELETE")
        remaining = reset.list_archives()["archives"][0]
        self.assertNotEqual(remaining["version"], archive["version"])
        reset.delete_archive(remaining["id"], remaining["version"], "DELETE")
        self.assertEqual(reset.list_archives(), {"archives": []})

    def test_archive_delete_time_limit_and_pending_reset_refusal(self):
        reset.reset_dataset("START OVER")
        archive = reset.list_archives()["archives"][0]
        inventory = {"version": archive["version"], "bytes": archive["bytes"]}
        with patch.object(reset, "_archive_inventory", return_value=inventory), patch.object(reset.time, "monotonic", side_effect=[0, 0, 100]):
            with self.assertRaisesRegex(p.Invalid, "deletion time limit"):
                reset.delete_archive(archive["id"], archive["version"], "DELETE")
        self.assertTrue(p.local_path("archives/" + archive["id"] + "/state.sqlite3").exists())
        with self.assertRaises(RuntimeError):
            reset.reset_dataset("START OVER", _interrupt_at="prepared")
        with self.assertRaisesRegex(p.Invalid, "reset recovery"):
            reset.delete_archive(archive["id"], archive["version"], "DELETE")
        self.assertTrue(p.local_path("reset.pending.json").exists())


if __name__ == "__main__":
    unittest.main()
