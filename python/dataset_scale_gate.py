"""Isolated 500-page loopback queue/draft scale gate using original fixtures."""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import resource
import tempfile
import threading
import time

os.environ["CHESS_OCR_TESTING"] = "1"

if __package__:
    from . import dataset_pipeline as p
    from . import dataset_server as server
else:
    import dataset_pipeline as p
    import dataset_server as server


PAGES = 500


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def request(port, method, path, *, cookie=None, value=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    headers = {"Cookie": cookie} if cookie else {}
    body = None
    if value is not None:
        body = p.canonical(value).encode()
        headers.update({"Content-Type": "application/json", "X-Dataset-Request": "1",
                        "Origin": f"http://127.0.0.1:{port}"})
    started = time.perf_counter()
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        payload = response.read()
        return response.status, dict(response.getheaders()), payload, time.perf_counter() - started
    finally:
        connection.close()


def start_server():
    server.initialize()
    app = server.Server(("127.0.0.1", 0))
    thread = threading.Thread(target=app.serve_forever, daemon=True)
    thread.start()
    status, headers, _, _ = request(app.server_port, "GET", "/")
    p.require(status == 200, "scale-gate handshake failed")
    cookie = headers["Set-Cookie"].split(";", 1)[0]
    return app, thread, cookie


def stop_server(app, thread):
    app.shutdown()
    app.server_close()
    thread.join(timeout=2)
    p.require(not thread.is_alive(), "scale-gate server did not stop")


def fixture():
    p.initialize()
    p.set_budget(argparse.Namespace(sources=2, pages=PAGES, download_bytes=2**20,
                 storage_bytes=2**30, cpu_seconds=3600, review_limit=PAGES + 20))
    image = p.pillow().new("RGB", (32, 32), "white")
    page = p.local_path("fixture.png")
    image.save(page)
    payload = page.read_bytes()
    image_sha = hashlib.sha256(payload).hexdigest()
    source = {"schema": p.SCHEMA, "id": "scale", "sha256": image_sha,
              "split": "train", "format": "pdf", "max_bytes": len(payload),
              "pages": list(range(1, PAGES + 1)), "revision": "scale-gate",
              "attribution": "Original synthetic scale fixture", "edition": "1",
              "selection_reason": "fixed 500-page queue scale gate",
              "pretrained_overlap": "not-applicable", "private": False, "real": True,
              "lineage_reviewed": True,
              "lineage": {key: ["scale-gate"] for key in ("document", "edition", "artwork", "parent")},
              "conditions": ["synthetic-scale-fixture"], "url": "https://example.invalid/scale.pdf",
              "rights": {"reviewer": "test", "evidence_url": "https://example.invalid/original",
                         "license": "original-test", "exclusions": "none",
                         "review_date": "2026-09-10", "evidence_sha256": image_sha,
                         "acquisition": "approved", "training": "approved",
                         "evaluation": "approved", "redistribution": "denied",
                         "model_publication": "unknown"}}
    page_root = p.local_path("pages")
    page_root.mkdir(parents=True, exist_ok=True)
    with p.connect() as db:
        db.execute("INSERT INTO sources VALUES (?,?,?)",
                   (source["id"], p.canonical(source), p.identity(source)))
        for index in range(1, PAGES + 1):
            path = page_root / f"scale-{index}.png"
            p.atomic(path, payload)
            db.execute("""INSERT INTO samples(id,source,page,image,sha,width,height,phash)
                       VALUES (?,?,?,?,?,?,?,?)""",
                       (f"scale-{index}", "scale", index, f"pages/scale-{index}.png",
                        image_sha, 32, 32, "0" * 16))


def run():
    original_root = p.ROOT
    with tempfile.TemporaryDirectory(prefix="chess-ocr-scale-") as directory:
        p.ROOT = Path(directory) / "work"
        try:
            fixture()
            app, thread, cookie = start_server()
            queue_times = []
            save_times = []
            try:
                for _ in range(25):
                    status, _, body, elapsed = request(app.server_port, "GET", "/api/queue",
                                                       cookie=cookie)
                    p.require(status == 200 and len(json.loads(body)["pages"]) == PAGES,
                              "scale-gate queue response changed")
                    queue_times.append(elapsed)
                state = {"boards": [], "kind": "negative", "reviewer": "scale-human",
                         "human": True, "complete": True, "elapsed_seconds": 0}
                for index in range(1, PAGES + 1):
                    sample = f"scale-{index}"
                    status, _, _, elapsed = request(app.server_port, "POST", "/api/draft",
                        cookie=cookie, value={"sample_id": sample, "revision": 0,
                        "image_sha256": hashlib.sha256(p.local_path(f"pages/{sample}.png").read_bytes()).hexdigest(),
                        "version": 0, "state": state})
                    p.require(status == 200, "scale-gate draft save failed")
                    save_times.append(elapsed)
                stale = {"sample_id": "scale-1", "revision": 0,
                         "image_sha256": hashlib.sha256(p.local_path("pages/scale-1.png").read_bytes()).hexdigest(),
                         "version": 0, "state": {**state, "reviewer": "stale-tab"}}
                p.require(request(app.server_port, "POST", "/api/draft", cookie=cookie,
                                  value=stale)[0] == 409, "stale draft was not rejected")
            finally:
                stop_server(app, thread)

            app, thread, cookie = start_server()
            try:
                status, _, body, resumed_time = request(app.server_port, "GET", "/api/queue",
                                                        cookie=cookie)
                queue = json.loads(body)
                p.require(status == 200 and sum(page["draft"] for page in queue["pages"]) == PAGES,
                          "scale-gate restart lost drafts")
            finally:
                stop_server(app, thread)
            with p.connect() as db:
                saved = db.execute("SELECT COUNT(*) FROM web_drafts WHERE version=1").fetchone()[0]
            rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            result = {"schema": "chess-ocr-dataset-scale-gate/1", "pages": PAGES,
                      "drafts": saved, "stale_rejected": True,
                      "queue_p95_ms": round(percentile(queue_times, .95) * 1000, 3),
                      "save_p95_ms": round(percentile(save_times, .95) * 1000, 3),
                      "restart_queue_ms": round(resumed_time * 1000, 3),
                      "rss_mib": round(rss_bytes / 2**20, 3), "stop_resume": True}
            p.require(saved == PAGES and result["queue_p95_ms"] < 1000
                      and result["save_p95_ms"] < 500 and rss_bytes < 512 * 2**20,
                      "dataset scale gate failed")
            return result
        finally:
            p.ROOT = original_root


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
