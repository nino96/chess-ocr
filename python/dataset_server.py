"""Private loopback dataset review application; no general-purpose file serving."""
from __future__ import annotations

import argparse
import base64
import io
import json
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import secrets
import socket
import urllib.parse

if __package__:
    from . import dataset_pipeline as p
    from . import dataset_reset as reset
else:
    import dataset_pipeline as p
    import dataset_reset as reset

UI = Path(__file__).parent / "dataset_app.html"
SCRIPT = p.REPO / "work/dataset-ui/dataset-app.js"


def initialize():
    with p.connect() as db:
        db.execute("""CREATE TABLE IF NOT EXISTS web_drafts(
            sample TEXT PRIMARY KEY REFERENCES samples(id), revision INTEGER NOT NULL,
            image_sha TEXT NOT NULL, version INTEGER NOT NULL, body TEXT NOT NULL)""")
    try:
        p.promote_legacy_single_human_reviews()
    except p.Invalid:
        # Rendering owns the writer lock; the app must still open to show status/stop.
        pass


def sample_row(db, sample_id):
    p.token(sample_id)
    row = db.execute("SELECT * FROM samples WHERE id=? AND source NOT IN (SELECT source FROM exclusions)",
                     (sample_id,)).fetchone()
    p.require(row is not None, "page unavailable")
    return row


def draft_state(value):
    p.require(isinstance(value, dict) and set(value) == {"boards", "kind", "reviewer", "human", "complete", "elapsed_seconds"}, "invalid draft")
    p.require(value["kind"] in {"boards", "negative", "partial", "unsupported"}, "invalid page kind")
    p.require(isinstance(value["reviewer"], str) and len(value["reviewer"]) <= 80, "invalid reviewer")
    p.require(type(value["human"]) is bool and type(value["complete"]) is bool, "invalid declarations")
    elapsed = value["elapsed_seconds"]
    p.require(type(elapsed) in (int, float) and 0 <= elapsed <= 14400, "invalid review time")
    p.require(isinstance(value["boards"], list) and len(value["boards"]) <= 64, "invalid draft boards")
    for b in value["boards"]:
        p.require(isinstance(b, dict) and set(b) == {"corners", "labels", "orientation"}, "invalid draft board")
        p.require(b["orientation"] in {"unknown", "white-bottom", "black-bottom"}, "invalid orientation")
        p.require(isinstance(b["labels"], list) and len(b["labels"]) == 64
                  and all(isinstance(x, str) and len(x) == 1 and x in p.LABELS for x in b["labels"]), "invalid labels")
        # In-progress geometry can be nonconvex. Submission performs full validation.
        p.require(isinstance(b["corners"], list) and len(b["corners"]) == 4, "invalid corners")
        for point in b["corners"]:
            p.require(isinstance(point, list) and len(point) == 2
                      and all(type(x) in (int, float) and -p.MAX_EDGE <= x <= p.MAX_EDGE for x in point), "invalid coordinates")
    return value


def save_draft(data):
    p.require(isinstance(data, dict) and set(data) == {"sample_id", "revision", "image_sha256", "version", "state"}, "invalid draft envelope")
    body = p.canonical(draft_state(data["state"]))
    p.require(type(data["version"]) is int and data["version"] >= 0, "invalid draft version")
    with p.writer(), p.connect() as db:
        row = sample_row(db, data["sample_id"])
        p.require(type(data["revision"]) is int and data["revision"] == row["revision"]
                  and data["image_sha256"] == row["sha"], "page changed; reload before saving")
        prior = db.execute("SELECT * FROM web_drafts WHERE sample=?", (row["id"],)).fetchone()
        version = prior["version"] if prior else 0
        if (prior and prior["body"] == body and prior["revision"] == row["revision"]
                and data["version"] in {version, version - 1}):
            return {"version": version, "state": "saved"}
        p.require(data["version"] == version, "draft changed in another tab; reload before saving")
        p.require(p.size_on_disk() + len(body.encode()) + 65536 <= p.meta(db, "budget")["storage_bytes"], "draft storage ceiling")
        db.execute("INSERT OR REPLACE INTO web_drafts VALUES (?,?,?,?,?)",
                   (row["id"], row["revision"], row["sha"], version + 1, body))
    return {"version": version + 1, "state": "saved"}


def queue(candidate=None):
    with p.connect() as db:
        sources = []
        for number, row in enumerate(db.execute("SELECT id,body FROM sources WHERE id NOT IN (SELECT source FROM exclusions) ORDER BY id"), 1):
            body = json.loads(row["body"])
            sources.append({"id": row["id"], "label": f"Document {number}", "split": body["split"], "selected_pages": len(body["pages"])})
        pages = [dict(r) for r in db.execute("""SELECT s.id,s.source,s.page,s.revision,s.accepted,
                CASE WHEN d.sample IS NULL THEN 0 ELSE 1 END AS draft
                FROM samples s LEFT JOIN web_drafts d ON s.id=d.sample AND s.revision=d.revision
                WHERE s.source NOT IN (SELECT source FROM exclusions) ORDER BY s.source,s.page""")]
        # Same-split candidates are retained and reported by pipeline status/audits.
        # The dashboard queue is reserved for cross-split leakage investigation.
        duplicates = [dict(r) for r in db.execute("""SELECT d.* FROM duplicates d
            JOIN samples a ON a.id=d.a JOIN sources sa ON a.source=sa.id
            JOIN samples b ON b.id=d.b JOIN sources sb ON b.source=sb.id
            WHERE d.decision IS NULL AND json_extract(sa.body, '$.split') != json_extract(sb.body, '$.split')
            AND a.source NOT IN (SELECT source FROM exclusions)
            AND b.source NOT IN (SELECT source FROM exclusions)""")]
    return {"schema": "chess-ocr-dataset-app/1", "sources": sources, "pages": pages,
            "duplicates": duplicates, "status": p.status(),
            "candidate": candidate.public_identity if candidate else None}


def candidate_proposal(data, candidate):
    p.require(candidate is not None, "no local candidate was configured")
    p.require(isinstance(data, dict) and set(data) == {"sample_id", "revision", "image_sha256"},
              "invalid candidate request")
    with p.connect() as db:
        row = sample_row(db, data["sample_id"])
    p.require(type(data["revision"]) is int and data["revision"] == row["revision"]
              and data["image_sha256"] == row["sha"], "page changed; reload before proposing")
    path = p.local_path(row["image"])
    p.require(p.digest(path) == row["sha"], "corrupt page")
    image = p.load_image(path).convert("RGB")
    result = candidate.recognize(image)
    with p.connect() as db:
        current = sample_row(db, data["sample_id"])
    p.require(current["revision"] == row["revision"] and current["sha"] == row["sha"],
              "page changed while proposing; proposal discarded")
    p.require(isinstance(result, dict) and set(result) == {"boards", "model", "warning"}
              and isinstance(result["boards"], list) and len(result["boards"]) <= 64
              and isinstance(result["warning"], str) and len(result["warning"]) <= 300
              and result["model"] == candidate.public_identity, "invalid candidate result")
    if result["boards"]:
        p.annotation_validate({"kind": "boards", "complete_page": True,
                               "boards": result["boards"]}, row["width"], row["height"])
    return {"schema": "chess-ocr-dataset-candidate/1", **result}


def review_html(sample_id):
    with p.connect() as db:
        row = sample_row(db, sample_id)
        draft = db.execute("SELECT * FROM web_drafts WHERE sample=?", (sample_id,)).fetchone()
    payload = p.review_payload(sample_id)
    payload["connected"] = True
    payload["draft_version"] = draft["version"] if draft else 0
    payload["draft"] = (json.loads(draft["body"]) if draft and draft["revision"] == row["revision"]
                        and draft["image_sha"] == row["sha"] else None)
    return (Path(__file__).parent / "dataset_review.html").read_text().replace(
        "__PAYLOAD_BASE64__", base64.b64encode(p.canonical(payload).encode()).decode()).encode()


class Server(HTTPServer):
    # A single handler bounds concurrent decoding/writes; socket timeouts bound slow clients.
    request_queue_size = 8

    def __init__(self, address, candidate=None):
        self.session = secrets.token_urlsafe(32)
        self.candidate = candidate
        super().__init__(address, Handler)

    def get_request(self):
        sock, address = super().get_request()
        sock.settimeout(10)
        return sock, address


class Handler(BaseHTTPRequestHandler):
    server_version = "DatasetReview"

    def log_message(self, *_args):
        pass

    def reply(self, status, body, content_type="application/json", cookie=False):
        if not isinstance(body, bytes):
            body = p.canonical(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "SAMEORIGIN")
        self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-src 'self'; frame-ancestors 'self'; base-uri 'none'; form-action 'none'")
        if cookie:
            self.send_header("Set-Cookie", f"dataset_session={self.server.session}; HttpOnly; SameSite=Strict; Path=/")
        self.end_headers()
        self.wfile.write(body)

    def guard(self, write=False):
        host = self.headers.get("Host", "")
        parsed = urllib.parse.urlsplit("http://" + host)
        p.require(parsed.hostname in {"127.0.0.1", "localhost"} and parsed.path == ""
                  and parsed.username is None and parsed.password is None and not parsed.query and not parsed.fragment
                  and parsed.port is not None and 0 < parsed.port <= 65535, "loopback host required")
        p.require(self.headers.get("Sec-Fetch-Site") not in {"cross-site"}, "cross-site request rejected")
        if write:
            p.require(self.headers.get("Origin") == "http://" + host
                      and self.headers.get("X-Dataset-Request") == "1", "same-origin request required")
        if self.path != "/" or write:
            cookie = SimpleCookie()
            cookie.load(self.headers.get("Cookie", ""))
            p.require("dataset_session" in cookie and secrets.compare_digest(cookie["dataset_session"].value, self.server.session), "open the app to start a session")

    def do_GET(self):
        try:
            self.guard()
            route = urllib.parse.urlsplit(self.path)
            if self.path == "/":
                self.reply(200, UI.read_bytes(), "text/html; charset=utf-8", cookie=True)
            elif self.path == "/app.js":
                self.reply(200, SCRIPT.read_bytes(), "text/javascript; charset=utf-8")
            elif self.path == "/api/queue":
                self.reply(200, queue(self.server.candidate))
            elif self.path == "/api/archives":
                self.reply(200, reset.list_archives())
            elif route.path.startswith("/review/"):
                query = urllib.parse.parse_qs(route.query, strict_parsing=True, max_num_fields=1)
                p.require(not query or set(query) == {"session"}, "invalid editor session")
                if query:
                    p.token(query["session"][0])
                self.reply(200, review_html(route.path.removeprefix("/review/")), "text/html; charset=utf-8")
            elif self.path.startswith("/thumbnail/"):
                with p.connect() as db:
                    row = sample_row(db, self.path.removeprefix("/thumbnail/"))
                path = p.local_path(row["image"])
                p.require(p.digest(path) == row["sha"], "corrupt page")
                img = p.load_image(path)
                img.thumbnail((160, 200))
                output = io.BytesIO()
                img.save(output, format="PNG")
                self.reply(200, output.getvalue(), "image/png")
            else:
                self.reply(404, {"error": "not found"})
        except (p.Invalid, ValueError, KeyError, TypeError):
            self.reply(400, {"error": "Page unavailable or request rejected. Reload the app; check dataset status if this persists."})
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            pass
        except Exception:
            self.reply(500, {"error": "Local operation failed. Check dataset status."})

    def do_POST(self):
        try:
            self.guard(write=True)
            p.require(self.headers.get("Content-Type") == "application/json"
                      and not self.headers.get("Transfer-Encoding"), "JSON body required")
            length = int(self.headers.get("Content-Length", "0"))
            p.require(0 < length <= p.MAX_CONFIG, "request byte ceiling")
            raw = self.rfile.read(length)
            p.require(len(raw) == length, "incomplete request")
            data = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(p.Invalid("invalid number")))
            p.require(isinstance(data, dict), "object required")
            if self.path == "/api/draft":
                result = save_draft(data)
            elif self.path == "/api/review":
                version = data.pop("draft_version", None)
                def check_current(db):
                    sample_row(db, data["sample_id"])
                    draft = db.execute("SELECT version FROM web_drafts WHERE sample=?", (data["sample_id"],)).fetchone()
                    p.require(type(version) is int and version == (draft[0] if draft else 0), "draft changed in another tab; reload before submitting")
                p.require(data.get("human") is True, "confirm human review before submitting")
                result = p.submit_review(data, check_current=check_current)
                # Keep the old revision's draft as a retry receipt. It is not restored
                # over the accepted revision; its monotonic version prevents tab races.
            elif self.path == "/api/candidate":
                result = candidate_proposal(data, self.server.candidate)
            elif self.path == "/api/action":
                p.require(set(data) == {"action"}, "invalid action")
                action = data["action"]
                if action == "start":
                    result = p.start()
                elif action == "stop":
                    p.atomic(p.local_path("stop"), b"stop\n")
                    result = {"state": "stop-requested"}
                elif action == "validate":
                    with p.connect() as db:
                        result = p.validate(db)
                elif action == "export":
                    result = p.start(export=True)
                else:
                    raise p.Invalid("unknown action")
            elif self.path == "/api/ingest":
                p.require(set(data) == {"group", "split", "reviewer", "pages_per_pdf", "approve_local_use"}, "invalid ingestion request")
                p.token(data["group"])
                p.require(data["split"] in {"train", "dev"} and data["approve_local_use"] is True
                          and isinstance(data["reviewer"], str) and 0 < len(data["reviewer"].strip()) <= 80
                          and type(data["pages_per_pdf"]) is int and 1 <= data["pages_per_pdf"] <= 2000, "complete ingestion declarations")
                result = p.ingest(argparse.Namespace(**data))
            elif self.path == "/api/duplicate":
                p.require(set(data) == {"a", "b", "decision"}, "invalid duplicate decision")
                result = p.resolve_duplicate(data["a"], data["b"], data["decision"])
            elif self.path == "/api/reset":
                p.require(set(data) == {"confirmation"}, "invalid reset request")
                result = reset.reset_dataset(data["confirmation"])
            elif self.path == "/api/archive-delete":
                p.require(set(data) == {"id", "version", "confirmation"}, "invalid archive deletion request")
                result = reset.delete_archive(data["id"], data["version"], data["confirmation"])
            else:
                self.reply(404, {"error": "not found"})
                return
            self.reply(200, result)
        except p.Invalid as error:
            self.reply(409, {"error": str(error)})
        except (ValueError, KeyError, TypeError):
            self.reply(400, {"error": "Invalid request schema."})
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            pass
        except Exception:
            self.reply(500, {"error": "Local operation failed. Inspect status; saved edits remain on disk."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--candidate-classifier", type=Path)
    parser.add_argument("--candidate-detector", type=Path)
    parser.add_argument("--candidate-overlay-root", type=Path, default=Path("work/training-overlay"))
    args = parser.parse_args()
    p.require(1024 <= args.port <= 65535, "invalid port")
    candidate_arguments = (args.candidate_manifest, args.candidate_classifier, args.candidate_detector)
    p.require(all(candidate_arguments) or not any(candidate_arguments),
              "candidate manifest, classifier and detector must be supplied together")
    candidate = None
    if all(candidate_arguments):
        if __package__:
            from .local_candidate import LocalCandidate
        else:
            from local_candidate import LocalCandidate
        candidate = LocalCandidate(args.candidate_manifest, args.candidate_classifier,
                                   args.candidate_detector, args.candidate_overlay_root, p.REPO)
    initialize()
    server = Server(("127.0.0.1", args.port), candidate)
    print(f"Dataset review: http://127.0.0.1:{server.server_port} (forward this port in VS Code)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
