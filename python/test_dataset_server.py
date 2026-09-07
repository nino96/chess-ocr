"""Loopback-only HTTP tests with original synthetic local fixtures."""
import argparse
import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest
import os
from unittest.mock import patch

os.environ["CHESS_OCR_TESTING"] = "1"

from python import dataset_pipeline as p
from python import dataset_server as s


class DatasetServerTests(unittest.TestCase):
    def setUp(self):
        self.assertIs(s.p, p)
        self.assertIs(s.reset.p, p)
        self.temp = tempfile.TemporaryDirectory()
        self.root_patch = patch.object(p, "ROOT", Path(self.temp.name) / "work")
        self.root_patch.start()
        p.initialize()
        p.set_budget(argparse.Namespace(sources=2, pages=20, download_bytes=2**20,
                     storage_bytes=2**28, cpu_seconds=600, review_limit=20))
        self.sample = self.make_sample()
        s.initialize()
        self.server = s.Server(("127.0.0.1", 0))
        self.port = self.server.server_port
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.root_patch.stop()
        self.temp.cleanup()

    def make_sample(self):
        image = p.pillow().new("RGB", (160, 160), "white")
        original = p.local_path("originals/test.png")
        original.parent.mkdir(parents=True, exist_ok=True)
        image.save(original)
        evidence = p.local_path("rights/test.evidence")
        p.atomic(evidence, b"original synthetic evidence")
        source = {
            "schema": p.SCHEMA, "id": "test", "sha256": p.digest(original),
            "split": "train", "format": "png", "max_bytes": 100000,
            "pages": [1], "revision": "1", "attribution": "Original test",
            "edition": "1", "selection_reason": "test", "pretrained_overlap": "unknown",
            "private": False, "real": True, "lineage_reviewed": True,
            "lineage": {key: ["test-design"] for key in ("document", "edition", "artwork", "parent")},
            "conditions": ["procedural-test"], "url": "https://example.invalid/test.png",
            "rights": {"reviewer": "test", "evidence_url": "https://example.invalid/rights",
                       "license": "original-test", "exclusions": "none", "review_date": "2026-09-06",
                       "evidence_sha256": p.digest(evidence), "acquisition": "approved",
                       "training": "approved", "evaluation": "approved", "redistribution": "denied",
                       "model_publication": "unknown"},
        }
        page = p.local_path("pages/test-1.png")
        page.parent.mkdir(parents=True, exist_ok=True)
        image.save(page)
        with p.connect() as db:
            db.execute("INSERT INTO sources VALUES (?,?,?)", ("test", p.canonical(source), p.identity(source)))
            db.execute("INSERT INTO samples(id,source,page,image,sha,width,height,phash) VALUES (?,?,?,?,?,?,?,?)",
                       ("test-1", "test", 1, "pages/test-1.png", p.digest(page), 160, 160, "0" * 16))
        return "test-1"

    def make_second_sample(self, source_id, split):
        image = p.pillow().new("RGB", (160, 160), "white")
        original = p.local_path(f"originals/{source_id}.png")
        image.save(original)
        evidence = p.local_path(f"rights/{source_id}.evidence")
        p.atomic(evidence, b"second original synthetic evidence")
        with p.connect() as db:
            source = json.loads(db.execute("SELECT body FROM sources WHERE id='test'").fetchone()[0])
            source.update(id=source_id, split=split, sha256=p.digest(original))
            source["lineage"] = {key: [f"{source_id}-design"] for key in source["lineage"]}
            source["rights"]["evidence_sha256"] = p.digest(evidence)
            page = p.local_path(f"pages/{source_id}-1.png")
            image.save(page)
            db.execute("INSERT INTO sources VALUES (?,?,?)", (source_id, p.canonical(source), p.identity(source)))
            db.execute("INSERT INTO samples(id,source,page,image,sha,width,height,phash) VALUES (?,?,?,?,?,?,?,?)",
                       (f"{source_id}-1", source_id, 1, f"pages/{source_id}-1.png", p.digest(page), 160, 160, "0" * 16))
        return f"{source_id}-1"

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=2)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def handshake(self):
        status, headers, _ = self.request("GET", "/")
        self.assertEqual(status, 200)
        return headers["Set-Cookie"].split(";", 1)[0]

    def write_headers(self, cookie):
        return {"Cookie": cookie, "Origin": f"http://127.0.0.1:{self.port}",
                "X-Dataset-Request": "1", "Content-Type": "application/json"}

    def review_data(self, revision=0, reviewer="human"):
        return {"schema": "chess-ocr-dataset-review/1", "sample_id": self.sample,
                "revision": revision, "image_sha256": p.digest(p.local_path("pages/test-1.png")),
                "reviewer": reviewer, "human": True, "elapsed_seconds": 10,
                "kind": "boards", "complete_page": True,
                "boards": [{"corners": [[0, 0], [160, 0], [160, 160], [0, 160]],
                            "labels": list("PNBRQKpnbrqk" + "." * 52), "orientation": "unknown"}]}

    def post(self, path, value, headers):
        return self.request("POST", path, p.canonical(value).encode(), headers)

    def test_handshake_and_request_guards(self):
        status, _, _ = self.request("GET", "/api/queue")
        self.assertEqual(status, 400)
        status, _, _ = self.request("GET", "/", headers={"Sec-Fetch-Site": "cross-site"})
        self.assertEqual(status, 400)
        status, _, _ = self.request("GET", "/", headers={"Host": "example.invalid:8766"})
        self.assertEqual(status, 400)
        cookie = self.handshake()
        status, _, body = self.request("GET", "/api/queue", headers={"Cookie": cookie})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["pages"][0]["id"], self.sample)
        status, _, _ = self.post("/api/action", {"action": "validate"}, {"Cookie": cookie, "Content-Type": "application/json"})
        self.assertEqual(status, 409)
        status, _, _ = self.post("/api/action", {"action": "validate"},
                                 {"Cookie": cookie, "Origin": f"http://127.0.0.1:{self.port}", "Content-Type": "application/json"})
        self.assertEqual(status, 409)
        headers = self.write_headers(cookie)
        headers["Sec-Fetch-Site"] = "cross-site"
        status, _, _ = self.post("/api/action", {"action": "validate"}, headers)
        self.assertEqual(status, 409)

    def test_queue_exposes_only_cross_split_duplicate_candidates(self):
        same_split = self.make_second_sample("same", "train")
        cross_split = self.make_second_sample("cross", "dev")
        with p.connect() as db:
            db.execute("INSERT INTO duplicates VALUES (?,?,'perceptual',NULL)", tuple(sorted((self.sample, same_split))))
            db.execute("INSERT INTO duplicates VALUES (?,?,'perceptual',NULL)", tuple(sorted((self.sample, cross_split))))
        cookie = self.handshake()
        status, _, body = self.request("GET", "/api/queue", headers={"Cookie": cookie})
        self.assertEqual(status, 200)
        duplicates = json.loads(body)["duplicates"]
        self.assertEqual([(pair["a"], pair["b"]) for pair in duplicates], [tuple(sorted((self.sample, cross_split)))])
        with p.connect() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM duplicates").fetchone()[0], 2)

    def test_size_and_path_integrity_rejections(self):
        cookie = self.handshake()
        headers = self.write_headers(cookie)
        headers["Content-Length"] = str(p.MAX_CONFIG + 1)
        status, _, _ = self.request("POST", "/api/action", b"{}", headers)
        self.assertEqual(status, 409)
        status, _, _ = self.request("GET", "/review/../state.sqlite3", headers={"Cookie": cookie})
        self.assertEqual(status, 400)
        page = p.local_path("pages/test-1.png")
        page.unlink()
        page.symlink_to(p.local_path("originals/test.png"))
        status, _, _ = self.request("GET", "/thumbnail/test-1", headers={"Cookie": cookie})
        self.assertEqual(status, 400)
        page.unlink()
        p.atomic(page, b"corrupt")
        status, _, _ = self.request("GET", "/thumbnail/test-1", headers={"Cookie": cookie})
        self.assertEqual(status, 400)

    def test_drafts_conflict_and_do_not_accept(self):
        cookie = self.handshake()
        state = {"boards": [], "kind": "negative", "reviewer": "human", "human": True,
                 "complete": True, "elapsed_seconds": 0}
        envelope = {"sample_id": self.sample, "revision": 0,
                    "image_sha256": p.digest(p.local_path("pages/test-1.png")), "version": 0, "state": state}
        status, _, body = self.post("/api/draft", envelope, self.write_headers(cookie))
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {"state": "saved", "version": 1})
        conflict = dict(envelope, state={**state, "reviewer": "other"})
        status, _, _ = self.post("/api/draft", conflict, self.write_headers(cookie))
        self.assertEqual(status, 409)
        status, _, _ = self.request("GET", f"/review/{self.sample}", headers={"Cookie": cookie})
        self.assertEqual(status, 200)
        with p.connect() as db:
            self.assertEqual(db.execute("SELECT accepted FROM samples WHERE id=?", (self.sample,)).fetchone()[0], 0)
        invalid = dict(envelope, state={"boards": []})
        status, _, _ = self.post("/api/draft", invalid, self.write_headers(cookie))
        self.assertEqual(status, 409)

    def test_provider_routes_extended_draft_deferral_and_metrics(self):
        cookie = self.handshake()
        headers = self.write_headers(cookie)
        status, _, body = self.request("GET", "/api/providers", headers={"Cookie": cookie})
        self.assertEqual(status, 200)
        registry = json.loads(body)
        self.assertEqual({x["capability"] for x in registry["providers"]}, {"localization", "labels"})
        with patch.object(s.proposals, "create_run",
                          return_value={"state": "running", "run_id": "a" * 64}) as start:
            status, _, body = self.post("/api/proposals/start", {
                "localizer": "classical-grid-v1", "labeler": "fenshot-labeler-v1",
                "scope": "train-pending", "max_pages": 20}, headers)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["state"], "running")
        start.assert_called_once_with("classical-grid-v1", "fenshot-labeler-v1",
                                      "train-pending", 20)
        status, _, body = self.request("GET", f"/api/proposals/{self.sample}", headers={"Cookie": cookie})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["results"], [])

        board = self.review_data()["boards"][0]
        state = {"boards": [board], "kind": "boards", "reviewer": "human", "human": True,
                 "complete": True, "elapsed_seconds": 12, "proposal_base": [board],
                 "proposal_run": None,
                 "touched": {"boards": [], "corners": [], "squares": ["0:0"], "corrections": ["0:0"]}}
        envelope = {"sample_id": self.sample, "revision": 0,
                    "image_sha256": p.digest(p.local_path("pages/test-1.png")), "version": 0, "state": state}
        self.assertEqual(self.post("/api/draft", envelope, headers)[0], 200)
        review = dict(self.review_data(), draft_version=1)
        status, _, body = self.post("/api/review", review, headers)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["assistance_metrics"], "recorded")
        with p.connect() as db:
            metric = json.loads(db.execute("SELECT body FROM review_metrics").fetchone()[0])
        self.assertEqual(metric["missed_boards"], 0)
        self.assertEqual(metric["false_boards"], 0)
        self.assertEqual(metric["matched_boards"][0]["piece_corrections"], 0)

        second = self.make_second_sample("defer", "train")
        payload = {"sample_id": second, "revision": 0,
                   "image_sha256": p.digest(p.local_path(f"pages/{second}.png")),
                   "reason": "ambiguous", "elapsed_seconds": 5, "proposal_run": None}
        self.assertEqual(self.post("/api/defer", payload, headers)[0], 200)
        status, _, body = self.request("GET", "/api/queue", headers={"Cookie": cookie})
        deferred = next(x for x in json.loads(body)["pages"] if x["id"] == second)
        self.assertEqual(deferred["deferred_reason"], "ambiguous")

    def test_direct_human_acceptance_stale_submission_and_action_allowlist(self):
        cookie = self.handshake()
        review = self.review_data()
        review["draft_version"] = 0
        status, _, body = self.post("/api/review", review, self.write_headers(cookie))
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["state"], "accepted")
        # A legacy pending human decision can be confirmed without a new charge,
        # even if startup migration was deferred while a rendering job held the lock.
        with p.connect() as db:
            db.execute("UPDATE samples SET accepted=0 WHERE id=?", (self.sample,))
        status, _, body = self.post("/api/review", review, self.write_headers(cookie))
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["state"], "accepted")
        self.assertEqual(p.status()["review_decisions"], 1)

        stale = self.review_data(reviewer="late")
        stale["draft_version"] = 0
        stale["boards"][0]["labels"][0] = "."
        status, _, _ = self.post("/api/review", stale, self.write_headers(cookie))
        self.assertEqual(status, 409)
        status, _, _ = self.post("/api/action", {"action": "unknown"}, self.write_headers(cookie))
        self.assertEqual(status, 409)
        status, _, _ = self.post("/api/action", {"action": "validate", "extra": True}, self.write_headers(cookie))
        self.assertEqual(status, 409)

    def test_archive_delete_http_guards_and_shape(self):
        cookie = self.handshake()
        status, _, _ = self.request("GET", "/api/archives")
        self.assertEqual(status, 400)
        self.assertEqual(self.post("/api/reset", {"confirmation": "START OVER"}, self.write_headers(cookie))[0], 200)
        status, _, body = self.request("GET", "/api/archives", headers={"Cookie": cookie})
        self.assertEqual(status, 200)
        archive = json.loads(body)["archives"][0]
        payload = {"id": archive["id"], "version": archive["version"], "confirmation": "DELETE"}
        self.assertEqual(self.post("/api/archive-delete", payload, {"Cookie": cookie, "Content-Type": "application/json"})[0], 409)
        self.assertEqual(self.post("/api/archive-delete", {**payload, "extra": True}, self.write_headers(cookie))[0], 409)
        self.assertEqual(self.post("/api/archive-delete", {**payload, "id": "../inbox"}, self.write_headers(cookie))[0], 409)
        self.assertEqual(self.post("/api/archive-delete", payload, self.write_headers(cookie))[0], 200)
        self.assertEqual(s.reset.list_archives(), {"archives": []})

    def test_reset_is_confirmed_and_preserves_inbox_and_usage(self):
        cookie = self.handshake()
        p.atomic(p.local_path("inbox/keep.pdf"), b"%PDF- original synthetic placeholder")
        review = dict(self.review_data(), draft_version=0)
        self.assertEqual(self.post("/api/review", review, self.write_headers(cookie))[0], 200)
        self.assertEqual(self.post("/api/reset", {"confirmation": "wrong"}, self.write_headers(cookie))[0], 409)
        self.assertEqual(p.status()["pages"], 1)
        status, _, body = self.post("/api/reset", {"confirmation": "START OVER"}, self.write_headers(cookie))
        self.assertEqual(status, 200)
        result = json.loads(body)
        self.assertTrue(p.local_path(result["archive"]).is_dir())
        self.assertTrue(p.local_path("inbox/keep.pdf").exists())
        self.assertEqual(p.status()["pages"], 0)
        self.assertEqual(p.status()["review_decisions"], 1)


if __name__ == "__main__":
    unittest.main()
