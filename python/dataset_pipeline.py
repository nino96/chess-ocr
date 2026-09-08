"""Local-only, resumable dataset acquisition and reviewed export (schema v1)."""
from __future__ import annotations

import argparse
import array
import base64
import contextlib
import datetime as dt
import fcntl
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import shutil
import socket
import sqlite3
import ssl
import subprocess
import sys
import time
import urllib.parse
import http.client
import uuid

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "work" / "dataset"
SCHEMA = "chess-ocr-dataset/1"
LABELS = ".PNBRQKpnbrqk"
SPLITS = {"train", "dev", "qualification", "regression"}
MAX_CONFIG = 2 * 1024 * 1024
MAX_PIXELS = 24_000_000
MAX_EDGE = 8192


class Invalid(ValueError):
    pass


class Budget(Invalid):
    pass


def require(value, message):
    if not value:
        raise Invalid(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read_json(path):
    path = Path(path)
    require(not path.is_symlink() and path.stat().st_size <= MAX_CONFIG, "unsafe configuration")
    return json.loads(path.read_text(), parse_constant=lambda _: (_ for _ in ()).throw(Invalid("nonfinite JSON")))


def local_path(relative):
    if os.environ.get("CHESS_OCR_TESTING") == "1":
        require(ROOT.resolve() != (REPO / "work/dataset").resolve(),
                "test process refused access to live dataset")
    path = ROOT / relative
    require(".." not in Path(relative).parts, "path traversal")
    require(path.is_relative_to(ROOT), "path escaped workspace")
    for parent in [path, *path.parents]:
        if parent == REPO:
            break
        require(not parent.is_symlink(), "symlink in workspace")
    require(path.resolve().is_relative_to(ROOT.resolve()), "path escaped workspace")
    return path


def atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temp.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def write_json(path, value):
    atomic(path, (canonical(value) + "\n").encode())


def connect():
    # A reset may have been interrupted after its durable marker was written.
    # Finish it before any caller can observe or mutate a mixed generation.
    if (ROOT / "reset.pending.json").exists():
        if __package__:
            from .dataset_reset import recover_reset
        else:
            from dataset_reset import recover_reset
        recover_reset()
    local_path("state.sqlite3")
    require((ROOT / "state.sqlite3").exists(), "run init first")
    db = sqlite3.connect(ROOT / "state.sqlite3", timeout=10)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    return db


@contextlib.contextmanager
def _writer_lock():
    ROOT.mkdir(parents=True, exist_ok=True)
    with local_path("writer.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Invalid("another writer is active; status and stop remain available") from None
        yield lock


@contextlib.contextmanager
def writer():
    # Recovery must finish before a normal writer obtains the lock; this avoids
    # a connection observing the generation between payload move and DB clear.
    if (ROOT / "reset.pending.json").exists():
        if __package__:
            from .dataset_reset import recover_reset
        else:
            from dataset_reset import recover_reset
        recover_reset()
    with _writer_lock() as lock:
        yield lock


def initialize():
    with writer():
        local_path("inbox").mkdir(exist_ok=True)
        db = sqlite3.connect(local_path("state.sqlite3"))
        db.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS sources(id TEXT PRIMARY KEY, body TEXT NOT NULL, sha TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS exclusions(source TEXT PRIMARY KEY REFERENCES sources(id), reason TEXT NOT NULL, at REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS jobs(id INTEGER PRIMARY KEY, source TEXT NOT NULL REFERENCES sources(id),
          stage TEXT NOT NULL, page INTEGER NOT NULL DEFAULT 0, state TEXT NOT NULL DEFAULT 'pending',
          attempts INTEGER NOT NULL DEFAULT 0, max_attempts INTEGER NOT NULL DEFAULT 3,
          error TEXT, next_at REAL NOT NULL DEFAULT 0,
          UNIQUE(source,stage,page));
        CREATE TABLE IF NOT EXISTS samples(id TEXT PRIMARY KEY, source TEXT NOT NULL REFERENCES sources(id),
          page INTEGER NOT NULL, image TEXT NOT NULL, sha TEXT NOT NULL, width INTEGER NOT NULL,
          height INTEGER NOT NULL, phash TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 0,
          annotation TEXT, accepted INTEGER NOT NULL DEFAULT 0, UNIQUE(source,page));
        CREATE TABLE IF NOT EXISTS reviews(id INTEGER PRIMARY KEY, sample TEXT NOT NULL REFERENCES samples(id),
          revision INTEGER NOT NULL, content TEXT NOT NULL, reviewer TEXT NOT NULL, human INTEGER NOT NULL,
          seconds REAL NOT NULL, decision TEXT NOT NULL, at REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS duplicates(a TEXT NOT NULL, b TEXT NOT NULL, reason TEXT NOT NULL,
          decision TEXT, PRIMARY KEY(a,b));
        CREATE TABLE IF NOT EXISTS board_signatures(sample TEXT NOT NULL REFERENCES samples(id),
          number INTEGER NOT NULL, revision INTEGER NOT NULL, sha TEXT NOT NULL, phash TEXT NOT NULL,
          PRIMARY KEY(sample,number));
        CREATE TABLE IF NOT EXISTS reservations(id INTEGER PRIMARY KEY, job INTEGER NOT NULL,
          bytes INTEGER NOT NULL, seconds INTEGER NOT NULL, at REAL NOT NULL);
        """)
        if "max_attempts" not in {r[1] for r in db.execute("PRAGMA table_info(jobs)")}:
            db.execute("ALTER TABLE jobs ADD COLUMN max_attempts INTEGER NOT NULL DEFAULT 3")
        defaults = {"budget": {"sources": 0, "pages": 0, "download_bytes": 0,
                    "storage_bytes": 0, "cpu_seconds": 0, "review_limit": 20},
                    "worker": {"state": "idle"}, "schema": SCHEMA,
                    "carryover": {"reserved_download_bytes": 0, "reserved_compute_seconds": 0,
                                  "attempts": 0, "review_decisions": 0, "review_seconds": 0}}
        with db:
            for key, value in defaults.items():
                db.execute("INSERT OR IGNORE INTO meta VALUES (?,?)", (key, canonical(value)))
            _promote_legacy_single_human_reviews(db)
        db.close()
    return {"state": "initialized", "next_action": "set an authorized local budget, then add reviewed source manifests"}


def ingest(args):
    """Explicitly authorize only PDFs placed in the dedicated local inbox."""
    require(args.approve_local_use, "explicit local-use authorization required")
    group = token(args.group)
    require(args.split in {"train", "dev"}, "unverified private lineage cannot enter qualification")
    require(0 < args.pages_per_pdf <= 2000, "bounded page selection required")
    require(args.reviewer.strip() and len(args.reviewer) <= 80, "rights reviewer required")
    with writer(), connect() as db:
        budget = meta(db, "budget")
        inbox = local_path("inbox")
        candidates = sorted(inbox.glob("*.pdf")) + sorted(inbox.glob("*.PDF"))
        require(candidates, "inbox contains no PDFs")
        require(len(candidates) <= budget["sources"], "inbox source ceiling")
        added, reused = 0, 0
        for path in candidates:
            require(not path.is_symlink() and path.is_file() and 0 < path.stat().st_size <= 512*1024**2, "unsafe inbox PDF")
            with path.open("rb") as stream:
                require(stream.read(5) == b"%PDF-", "invalid PDF signature")
            sha = digest(path)
            source_id = "local-" + sha[:20]
            existing = db.execute("SELECT body FROM sources WHERE id=?", (source_id,)).fetchone()
            if existing:
                body = json.loads(existing[0])
                require(body["split"] == args.split and body["lineage"]["artwork"] == [group], "existing source split/group is immutable")
                reused += 1
                continue
            for row in db.execute("SELECT body FROM sources"):
                other = json.loads(row[0])
                require(not (group in other["lineage"]["artwork"] and other["split"] != args.split), "shared artwork crosses splits")
            require(db.execute("SELECT COUNT(*) FROM sources").fetchone()[0] < budget["sources"], "source budget exhausted")
            # pdfinfo is bounded and receives only the user's dedicated inbox inputs.
            def limits():
                resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
                resource.setrlimit(resource.RLIMIT_AS, (512*1024**2, 512*1024**2))
                resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_CONFIG, MAX_CONFIG))
            require(cumulative_accounting(db)["reserved_compute_seconds"] + 15 <= budget["cpu_seconds"], "inspection compute budget exhausted")
            with db:
                db.execute("INSERT INTO reservations(job,bytes,seconds,at) VALUES (0,0,15,?)", (time.time(),))
            info_path = local_path("staging/pdfinfo.txt")
            info_path.parent.mkdir(parents=True, exist_ok=True)
            with info_path.open("wb") as output:
                subprocess.run(["pdfinfo", str(path)], stdout=output, stderr=subprocess.DEVNULL,
                               timeout=15, check=True, preexec_fn=limits)
            require(info_path.stat().st_size <= MAX_CONFIG, "PDF metadata ceiling")
            match = re.search(rb"^Pages:\s+(\d+)\s*$", info_path.read_bytes(), re.MULTILINE)
            info_path.unlink()
            require(match is not None, "PDF page count unavailable")
            count = int(match[1])
            require(0 < count <= 100000, "PDF page count ceiling")
            # Uniform fixed page sampling avoids selecting only detector successes.
            n = min(count, args.pages_per_pdf)
            pages = sorted({1 + round(i * (count - 1) / max(1, n - 1)) for i in range(n)})
            current_pages = sum(len(json.loads(r[0])["pages"]) for r in db.execute("SELECT body FROM sources"))
            require(current_pages + len(pages) <= budget["pages"], "page budget exhausted")
            require(size_on_disk() + path.stat().st_size + MAX_CONFIG < budget["storage_bytes"], "local copy storage ceiling")
            evidence = {"schema": SCHEMA, "action": "owner-approved-local-use", "reviewer": args.reviewer.strip(),
                        "date": dt.date.today().isoformat(), "training": args.split == "train", "evaluation": args.split == "dev",
                        "sha256": sha, "publication": "not-authorized"}
            evidence_bytes = canonical(evidence).encode()
            body = {"schema": SCHEMA, "id": source_id, "sha256": sha, "url": "local-inbox",
                    "revision": sha, "edition": "owner-supplied-unverified", "attribution": "local owner-supplied PDF",
                    "format": "pdf", "private": True, "real": True, "max_bytes": path.stat().st_size,
                    "split": args.split, "pages": pages, "total_pages": count,
                    "selection_reason": "uniform page sample independent of recognition outcomes",
                    "pretrained_overlap": "unknown", "lineage_reviewed": False,
                    "lineage": {"document": [source_id], "edition": [group], "artwork": [group], "parent": [source_id]},
                    "conditions": ["owner-supplied", "appearance-unreviewed"],
                    "rights": {"reviewer": args.reviewer.strip(), "review_date": dt.date.today().isoformat(),
                               "license": "owner authorization for local use only", "evidence_url": "local-owner-declaration",
                               "evidence_sha256": hashlib.sha256(evidence_bytes).hexdigest(), "exclusions": "all publication",
                               "acquisition": "approved", "training": "approved" if args.split == "train" else "unknown",
                               "evaluation": "approved" if args.split == "dev" else "unknown",
                               "redistribution": "denied", "model_publication": "unknown"}}
            original = local_path(f"originals/{source_id}.pdf")
            atomic(original, path.read_bytes())
            require(digest(original) == sha, "inbox changed while copying")
            atomic(local_path(f"rights/{source_id}.evidence"), evidence_bytes)
            with db:
                db.execute("INSERT INTO sources VALUES (?,?,?)", (source_id, canonical(body), identity(body)))
                for page in pages:
                    db.execute("INSERT INTO jobs(source,stage,page) VALUES (?,'render',?)", (source_id, page))
            added += 1
    return {"state": "queued", "added": added, "already_ingested": reused,
            "next_action": "start; local use only, labels and artwork remain unverified"}


def meta(db, key):
    return json.loads(db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()[0])


def set_meta(db, key, value):
    db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, canonical(value)))


def set_budget(args):
    budget = {key: getattr(args, key) for key in ("sources", "pages", "download_bytes", "storage_bytes", "cpu_seconds", "review_limit")}
    require(all(type(v) is int and v > 0 for v in budget.values()), "positive budget ceilings required")
    with writer(), connect() as db:
        set_meta(db, "budget", budget)
    return {"budget": budget}


def token(value):
    require(isinstance(value, str) and re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", value), "invalid identifier")
    return value


def sha256(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "invalid sha256")
    return value


def source_validate(body):
    require(body.get("schema") == SCHEMA, "invalid source schema")
    token(body["id"])
    require(len(body["id"]) <= 64, "source identifier must leave room for page suffix")
    sha256(body["sha256"])
    require(body["split"] in SPLITS, "invalid split")
    require(body["format"] in {"pdf", "png", "jpeg"}, "unsupported source format")
    require(type(body["max_bytes"]) is int and 0 < body["max_bytes"] <= 512 * 1024 * 1024, "invalid byte ceiling")
    pages = body["pages"]
    require(isinstance(pages, list) and 0 < len(pages) <= 2000 and len(set(pages)) == len(pages)
            and all(type(p) is int and 1 <= p <= 100000 for p in pages), "explicit unique page selection required")
    require(body["format"] == "pdf" or pages == [1], "image sources have one page")
    for field in ("revision", "attribution", "edition", "selection_reason", "pretrained_overlap"):
        require(isinstance(body.get(field), str) and 0 < len(body[field]) <= 4000, "missing source provenance")
    require(body.get("private") is False, "private inputs require a separately approved ingestion path")
    require(body.get("real") is True, "this acquisition lane admits real documents only")
    groups = body["lineage"]
    require(isinstance(groups, dict) and set(groups) == {"document", "edition", "artwork", "parent"}, "complete lineage required")
    for values in groups.values():
        require(isinstance(values, list) and values, "unresolved lineage cannot enter a split")
        for value in values:
            token(value)
    require(body.get("lineage_reviewed") is True, "lineage review required")
    require(isinstance(body.get("conditions"), list) and body["conditions"], "coverage tags required")
    for condition in body["conditions"]:
        token(condition)
    rights = body["rights"]
    for field in ("reviewer", "evidence_url", "license", "exclusions"):
        require(isinstance(rights.get(field), str) and rights[field], "missing rights evidence")
    dt.date.fromisoformat(rights["review_date"])
    sha256(rights["evidence_sha256"])
    require(rights.get("acquisition") == "approved", "acquisition rights unapproved")
    require(rights.get("training" if body["split"] == "train" else "evaluation") == "approved", "split use unapproved")
    for field in ("redistribution", "model_publication"):
        require(rights.get(field) in {"approved", "denied", "unknown"}, "separate downstream rights required")
    parsed = urllib.parse.urlsplit(body["url"])
    require(parsed.scheme == "https" and parsed.hostname and not parsed.username and not parsed.password
            and parsed.port in (None, 443) and not parsed.fragment, "plain HTTPS source required")
    return body


def lineage(body):
    return {(kind, value) for kind, values in body["lineage"].items() for value in values}


def add_source(path):
    body = source_validate(read_json(path))
    with writer(), connect() as db:
        budget = meta(db, "budget")
        sources = [json.loads(row[0]) for row in db.execute("SELECT body FROM sources")]
        require(not any(s["id"] == body["id"] for s in sources), "source identity is immutable")
        require(len(sources) < budget["sources"], "source budget exhausted")
        require(sum(len(s["pages"]) for s in sources) + len(body["pages"]) <= budget["pages"], "page budget exhausted")
        for other in sources:
            require(not ((lineage(other) & lineage(body) or other["sha256"] == body["sha256"])
                         and other["split"] != body["split"]), "cross-split lineage or original duplicate")
        evidence = Path(body.pop("evidence_file"))
        require(not evidence.is_symlink() and evidence.stat().st_size <= MAX_CONFIG, "unsafe rights evidence")
        require(digest(evidence) == body["rights"]["evidence_sha256"], "rights evidence hash mismatch")
        dest = local_path(f"rights/{body['id']}.evidence")
        atomic(dest, evidence.read_bytes())
        db.execute("INSERT INTO sources VALUES (?,?,?)", (body["id"], canonical(body), identity(body)))
        db.execute("INSERT INTO jobs(source,stage) VALUES (?, 'acquire')", (body["id"],))
    return {"state": "queued", "sources_added": 1}


def size_on_disk():
    total = 0
    for path in ROOT.rglob("*"):
        require(not path.is_symlink(), "symlink in workspace")
        if path.is_file():
            total += path.stat().st_size
    return total


def require_free_space(additional_bytes, concurrent_bytes=1024**3):
    """Owner floor plus headroom for the other bounded dataset worker."""
    usage = shutil.disk_usage(ROOT)
    if usage.free - additional_bytes - concurrent_bytes < math.ceil(usage.total * .30):
        raise Budget("30 percent filesystem free-space floor would be crossed")


def outstanding_synthetic_storage(db):
    row = db.execute("SELECT value FROM meta WHERE key='synthetic_storage_reservation'").fetchone()
    if not row:
        return 0
    reserved = json.loads(row[0])["bytes"]
    root = local_path("synthetic")
    present = sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) if root.exists() else 0
    return max(0, reserved - present)


def reserve(db, job, source):
    budget = meta(db, "budget")
    used = cumulative_accounting(db)
    byte_count = source["max_bytes"] if job["stage"] == "acquire" else 0
    seconds = 120 if job["stage"] == "acquire" else 90
    # Reserve full worst-case storage, including temporary images, before starting.
    storage = source["max_bytes"] if job["stage"] == "acquire" else MAX_PIXELS * 12
    require_free_space(storage)
    if used["reserved_download_bytes"] + byte_count > budget["download_bytes"] or used["reserved_compute_seconds"] + seconds > budget["cpu_seconds"]:
        raise Budget("attempt reservation exceeds download/compute ceiling")
    if size_on_disk() + storage + outstanding_synthetic_storage(db) > budget["storage_bytes"]:
        raise Budget("attempt reservation exceeds storage ceiling")
    db.execute("INSERT INTO reservations(job,bytes,seconds,at) VALUES (?,?,?,?)", (job["id"], byte_count, seconds, time.time()))
    return seconds


def pillow():
    import PIL
    require(PIL.__version__ == "11.1.0", "requires locked Pillow 11.1.0")
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    return Image


def load_image(path):
    Image = pillow()
    require(not Path(path).is_symlink(), "symlink image")
    with Image.open(path) as image:
        require(image.format in {"PNG", "JPEG", "PPM"}, "unsupported decoded image")
        require(0 < image.width <= MAX_EDGE and 0 < image.height <= MAX_EDGE
                and image.width * image.height <= MAX_PIXELS, "decoded raster limit")
        require(getattr(image, "n_frames", 1) == 1, "animated input unsupported")
        # Preserve image-relative encoded pixel coordinates; orientation is not inferred.
        image.load()
        return image.convert("RGB")


def phash(image):
    Image = pillow()
    values = list(image.convert("L").resize((9, 8), Image.Resampling.BILINEAR).getdata())
    bits = [values[y * 9 + x] > values[y * 9 + x + 1] for y in range(8) for x in range(8)]
    return f"{sum(int(bit) << i for i, bit in enumerate(bits)):016x}"


class PinnedHTTPS(http.client.HTTPSConnection):
    def __init__(self, host, address):
        super().__init__(host, timeout=15, context=ssl.create_default_context())
        self.address = address

    def connect(self):
        sock = socket.create_connection((self.address, 443), self.timeout)
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


def fetch(source, destination):
    parsed = urllib.parse.urlsplit(source["url"])
    addresses = sorted({info[4][0] for info in socket.getaddrinfo(parsed.hostname, 443, type=socket.SOCK_STREAM)})
    require(addresses and all(ipaddress.ip_address(address).is_global for address in addresses), "nonpublic destination rejected")
    connection = PinnedHTTPS(parsed.hostname, addresses[0])
    try:
        connection.request("GET", urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query, "")),
                           headers={"User-Agent": "chess-ocr-local-dataset/1", "Accept-Encoding": "identity"})
        response = connection.getresponse()
        if response.status in (408, 429) or response.status >= 500:
            raise OSError("transient HTTP failure")
        require(response.status == 200, "HTTP rejected (redirects require reviewed exact URL)")
        require(response.getheader("Content-Encoding", "identity") == "identity", "encoded response rejected")
        length = response.getheader("Content-Length")
        require(length is None or 0 < int(length) <= source["max_bytes"], "response byte ceiling")
        count = 0
        with destination.open("xb") as stream:
            while chunk := response.read(65536):
                count += len(chunk)
                require(count <= source["max_bytes"], "response byte ceiling")
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        require(count > 0 and (length is None or count == int(length)), "incomplete response")
        require(digest(destination) == source["sha256"], "source hash mismatch")
    finally:
        connection.close()


def child(job_id):
    # One isolated process group per attempt. The parent enforces wall time/stop.
    # Also enforce wall time if the supervisor itself is killed mid-download.
    signal.alarm(120)
    resource.setrlimit(resource.RLIMIT_CPU, (85, 85))
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (512 * 1024**2, 512 * 1024**2))
    with connect() as db:
        job = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        source = json.loads(db.execute("SELECT body FROM sources WHERE id=?", (job["source"],)).fetchone()[0])
    staging = local_path(f"staging/{job_id}")
    staging.mkdir(parents=True, exist_ok=True)
    for old in staging.iterdir():
        require(old.is_file() and not old.is_symlink(), "unsafe staging")
        old.unlink()
    original = local_path(f"originals/{source['id']}.{source['format']}")
    if job["stage"] == "acquire":
        if original.exists():
            require(digest(original) == source["sha256"], "stored source corrupt")
        else:
            candidate = staging / "download.part"
            fetch(source, candidate)
            if source["format"] == "pdf":
                with candidate.open("rb") as stream:
                    require(stream.read(5) == b"%PDF-", "invalid PDF signature")
            else:
                load_image(candidate)
            original.parent.mkdir(parents=True, exist_ok=True)
            os.replace(candidate, original)
        return
    require(digest(original) == source["sha256"], "stored source corrupt")
    if source["format"] == "pdf":
        version = subprocess.run(["pdftoppm", "-v"], capture_output=True, timeout=5, check=True)
        require(b"pdftoppm version 24.02.0" in version.stderr, "requires reviewed Poppler 24.02.0 renderer")
        # Fixed recipe; one explicitly selected page, bounded resolution, no archive extraction.
        subprocess.run(["pdftoppm", "-f", str(job["page"]), "-l", str(job["page"]),
                        "-singlefile", "-scale-to", "2400", "-r", "150", "-png",
                        str(original), str(staging / "page")], check=True, timeout=70,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        image = load_image(staging / "page.png")
    else:
        image = load_image(original)
    image.save(staging / "result.png")
    write_json(staging / "result.json", {"width": image.width, "height": image.height,
               "sha": digest(staging / "result.png"), "phash": phash(image),
               "recipe": {"pipeline_sha256": digest(__file__), "source_sha256": source["sha256"],
                          "pillow": "11.1.0", "poppler": "24.02.0" if source["format"] == "pdf" else None,
                          "pdf_scale_to": 2400, "pdf_dpi": 150, "orientation": "encoded-image-relative"}})


def finalize(db, job):
    if job["stage"] == "acquire":
        body = json.loads(db.execute("SELECT body FROM sources WHERE id=?", (job["source"],)).fetchone()[0])
        for page in body["pages"]:
            db.execute("INSERT OR IGNORE INTO jobs(source,stage,page) VALUES (?, 'render', ?)", (job["source"], page))
    else:
        result = read_json(local_path(f"staging/{job['id']}/result.json"))
        sample = f"{job['source']}-{job['page']}"
        image = f"pages/{sample}.png"
        dest = local_path(image)
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(local_path(f"staging/{job['id']}/result.png"), dest)
        db.execute("INSERT OR IGNORE INTO samples(id,source,page,image,sha,width,height,phash) VALUES (?,?,?,?,?,?,?,?)",
                   (sample, job["source"], job["page"], image, result["sha"], result["width"], result["height"], result["phash"]))
        set_meta(db, "render:" + sample, result["recipe"])
        for other in db.execute("SELECT id,sha,phash FROM samples WHERE id<>?", (sample,)):
            distance = (int(other["phash"], 16) ^ int(result["phash"], 16)).bit_count()
            if other["sha"] == result["sha"] or distance <= 4:
                a, b = sorted((sample, other["id"]))
                db.execute("INSERT OR IGNORE INTO duplicates VALUES (?,?,?,NULL)",
                           (a, b, "exact" if other["sha"] == result["sha"] else "perceptual"))
    db.execute("UPDATE jobs SET state='done',error=NULL WHERE id=?", (job["id"],))


def run(clear_stop=True):
    with writer() as lock, connect() as db:
        if clear_stop:
            local_path("stop").unlink(missing_ok=True)
        # Full attempt reservations survive crashes; never refund uncertain work.
        with db:
            db.execute("UPDATE jobs SET state=CASE WHEN attempts>=max_attempts THEN 'quarantined' ELSE 'retry' END,error='interrupted' WHERE state='running'")
            set_meta(db, "worker", {"state": "running", "pid": os.getpid(), "heartbeat": time.time()})
        state = "idle"
        try:
            while not local_path("stop").exists():
                job = db.execute("SELECT * FROM jobs WHERE state IN ('pending','retry') AND next_at<=? ORDER BY id LIMIT 1", (time.time(),)).fetchone()
                if job is None:
                    state = ("waiting-retry" if db.execute("SELECT 1 FROM jobs WHERE state='retry'").fetchone()
                             else "needs-repair" if db.execute("SELECT 1 FROM jobs WHERE state='quarantined'").fetchone()
                             else "needs-review" if db.execute("SELECT 1 FROM samples").fetchone()
                             else "awaiting-sources")
                    break
                source = json.loads(db.execute("SELECT body FROM sources WHERE id=?", (job["source"],)).fetchone()[0])
                try:
                    with db:
                        ceiling = reserve(db, job, source)
                        db.execute("UPDATE jobs SET state='running',attempts=attempts+1 WHERE id=?", (job["id"],))
                except Budget:
                    state = "budget-blocked"
                    break
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "_child", str(job["id"])],
                                           start_new_session=True, pass_fds=(lock.fileno(),),
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                started = time.monotonic()
                reason = None
                while process.poll() is None:
                    if local_path("stop").exists() or time.monotonic() - started > ceiling:
                        reason = "stopped" if local_path("stop").exists() else "timeout"
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                        break
                    with db:
                        set_meta(db, "worker", {"state": "running", "pid": os.getpid(), "heartbeat": time.time(), "job": job["id"]})
                    time.sleep(0.5)
                # A failed decoder must not leave grandchildren consuming resources.
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                with db:
                    if process.returncode == 0:
                        finalize(db, job)
                    else:
                        permanent = process.returncode == 2
                        attempts = job["attempts"] + 1
                        db.execute("UPDATE jobs SET state=?,error=?,next_at=? WHERE id=?",
                                   ("quarantined" if permanent or attempts >= job["max_attempts"] else "retry",
                                    reason or ("validation-failed" if permanent else "transient-failure"),
                                    time.time() + min(60, 2**attempts), job["id"]))
                if reason == "stopped":
                    state = "stopped"
                    break
                # Waits belong to this bounded local worker, never to an AI turn.
                while db.execute("SELECT 1 FROM jobs WHERE state='retry' AND next_at>?", (time.time(),)).fetchone() and not db.execute("SELECT 1 FROM jobs WHERE state='pending' OR (state='retry' AND next_at<=?)", (time.time(),)).fetchone():
                    if local_path("stop").exists():
                        break
                    time.sleep(0.5)
            else:
                state = "stopped"
        except Exception:
            state = "interrupted-resume-required"
            raise
        finally:
            with db:
                set_meta(db, "worker", {"state": state, "heartbeat": time.time()})
    return status()


def start(export=False):
    # Startup handshake avoids reporting a job started if lock acquisition failed.
    with writer(), connect() as db:
        local_path("stop").unlink(missing_ok=True)
        set_meta(db, "worker", {"state": "starting", "heartbeat": time.time()})
    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "_export" if export else "_run"],
                               start_new_session=True, stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(40):
        with connect() as db:
            worker = meta(db, "worker")
        if worker["state"] != "starting":
            return {"state": worker["state"], "pid": process.pid}
        if process.poll() is not None:
            raise Invalid("worker startup failed")
        time.sleep(0.05)
    raise Invalid("worker startup not confirmed; inspect status")


def retry_job(job_id, after_repair):
    require(after_repair, "explicit repair acknowledgement required")
    with writer(), connect() as db:
        job = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        require(job is not None and job["state"] == "quarantined", "only quarantined attempts need manual repair")
        # Explicitly authorizes one further attempt; prior reservations never disappear.
        db.execute("UPDATE jobs SET state='pending',max_attempts=attempts+1,next_at=0 WHERE id=?", (job_id,))
    return {"state": "queued-one-repair-attempt", "job": job_id}


def status():
    with connect() as db:
        worker = meta(db, "worker")
        if worker["state"] in {"running", "starting", "exporting"} and time.time() - worker["heartbeat"] > 100:
            worker["state"] = "heartbeat-stale-check-process-or-resume"
        counts = dict(db.execute("SELECT state,COUNT(*) FROM jobs GROUP BY state"))
        samples = db.execute("SELECT COUNT(*),COALESCE(SUM(accepted),0) FROM samples WHERE source NOT IN (SELECT source FROM exclusions)").fetchone()
        accounting = cumulative_accounting(db)
        duplicate_counts = db.execute("""SELECT
              SUM(CASE WHEN d.decision IS NULL AND json_extract(sa.body, '$.split') = json_extract(sb.body, '$.split') THEN 1 ELSE 0 END),
              SUM(CASE WHEN d.decision IS NULL AND json_extract(sa.body, '$.split') != json_extract(sb.body, '$.split') THEN 1 ELSE 0 END)
            FROM duplicates d
            JOIN samples a ON d.a=a.id JOIN sources sa ON a.source=sa.id
            JOIN samples b ON d.b=b.id JOIN sources sb ON b.source=sb.id
            WHERE a.source NOT IN (SELECT source FROM exclusions)
              AND b.source NOT IN (SELECT source FROM exclusions)""").fetchone()
        same_split_candidates, cross_split_blockers = (int(value or 0) for value in duplicate_counts)
        sources = db.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
        return {"schema": SCHEMA, "worker": worker, "jobs": counts, "pages": samples[0],
                "sources": sources,
                "excluded_sources": db.execute("SELECT COUNT(*) FROM exclusions").fetchone()[0],
                "accepted_pages": samples[1], "review_queue": samples[0] - samples[1],
                "unresolved_duplicate_pairs": cross_split_blockers,
                "same_split_duplicate_candidates": same_split_candidates, "review_decisions": accounting["review_decisions"],
                "review_seconds": accounting["review_seconds"], "budget": meta(db, "budget"),
                "reserved_download_bytes": accounting["reserved_download_bytes"], "reserved_compute_seconds": accounting["reserved_compute_seconds"],
                "attempts": accounting["attempts"], "storage_bytes": size_on_disk(),
                "next_action": ("set authorized budget; ingest inbox or add reviewed public manifests" if not sources
                                else "inspect queue and repair quarantined jobs" if counts.get("quarantined")
                                else "resolve cross-split duplicate queue" if cross_split_blockers
                                else "review pages; validate before export"),
                "recognition_qualified": False}


def cumulative_accounting(db):
    """Lifetime charges, including archived generations after a reset."""
    carry = meta(db, "carryover") if db.execute("SELECT 1 FROM meta WHERE key='carryover'").fetchone() else {}
    reserved = db.execute("SELECT COALESCE(SUM(bytes),0),COALESCE(SUM(seconds),0),COUNT(*) FROM reservations").fetchone()
    reviews = db.execute("SELECT COUNT(*),COALESCE(SUM(seconds),0) FROM reviews").fetchone()
    return {"reserved_download_bytes": int(carry.get("reserved_download_bytes", 0)) + reserved[0],
            "reserved_compute_seconds": int(carry.get("reserved_compute_seconds", 0)) + reserved[1],
            "attempts": int(carry.get("attempts", 0)) + reserved[2],
            "review_decisions": int(carry.get("review_decisions", 0)) + reviews[0],
            "review_seconds": float(carry.get("review_seconds", 0)) + reviews[1]}


def annotation_validate(data, width, height):
    require(data.get("kind") in {"boards", "negative", "partial", "unsupported"}, "invalid page kind")
    require(data.get("complete_page") is True, "all page diagrams must be inspected")
    boards = data["boards"]
    require(isinstance(boards, list) and len(boards) <= 64, "invalid board list")
    require((data["kind"] == "boards" and len(boards) > 0) or (data["kind"] != "boards" and not boards), "kind/board mismatch")
    for board in boards:
        labels = board["labels"]
        require(isinstance(labels, list) and len(labels) == 64 and all(type(x) is str and x in LABELS and len(x) == 1 for x in labels), "64 image-relative labels required")
        require(board["orientation"] in {"unknown", "white-bottom", "black-bottom"}, "invalid orientation")
        corners = board["corners"]
        require(isinstance(corners, list) and len(corners) == 4, "four corners required")
        for point in corners:
            require(isinstance(point, list) and len(point) == 2 and all(type(v) in (int, float) and math.isfinite(v) for v in point), "invalid point")
            require(0 <= point[0] <= width and 0 <= point[1] <= height, "corner outside image")
        for i in range(4):
            a, b, c = corners[i], corners[(i+1) % 4], corners[(i+2) % 4]
            require((b[0]-a[0])*(c[1]-b[1])-(b[1]-a[1])*(c[0]-b[0]) > 1, "corners must be convex clockwise TL TR BR BL")
        require(corners[0][1] + corners[1][1] < corners[2][1] + corners[3][1]
                and corners[0][0] + corners[3][0] < corners[1][0] + corners[2][0], "image-relative corner ordering required")
        require(min(math.dist(corners[i], corners[(i+1) % 4]) for i in range(4)) >= 16, "grid too small")
        require(set(board) == {"labels", "orientation", "corners"}, "unknown board fields")
    return {"kind": data["kind"], "boards": boards, "complete_page": True}


def _promote_legacy_single_human_reviews(db):
    """Promote pre-single-review records without writing a new review decision."""
    rows = db.execute("""
        SELECT s.id FROM samples s
        WHERE s.accepted=0 AND s.annotation IS NOT NULL
          AND EXISTS (
            SELECT 1 FROM reviews r
            WHERE r.sample=s.id AND r.revision=s.revision
              AND r.content=s.annotation AND r.human=1
          )
    """).fetchall()
    db.executemany("UPDATE samples SET accepted=1 WHERE id=?", ((row[0],) for row in rows))
    return len(rows)


def promote_legacy_single_human_reviews():
    """Idempotently apply the one-human policy to old pending review records."""
    with writer(), connect() as db:
        promoted = _promote_legacy_single_human_reviews(db)
    return {"state": "legacy-single-human-promoted", "promoted": promoted}


def review_payload(sample_id):
    token(sample_id)
    with connect() as db:
        sample = db.execute("SELECT * FROM samples WHERE id=?", (sample_id,)).fetchone()
        require(sample is not None, "unknown sample")
        require(digest(local_path(sample["image"])) == sample["sha"], "stale/corrupt page")
        annotation = json.loads(sample["annotation"]) if sample["annotation"] else {"kind": "boards", "boards": []}
        return {"schema": "chess-ocr-dataset-review/1", "sample_id": sample_id,
                "revision": sample["revision"], "image_sha256": sample["sha"],
                "width": sample["width"], "height": sample["height"], **annotation,
                "image_data_url": "data:image/png;base64," + base64.b64encode(local_path(sample["image"]).read_bytes()).decode()}


def review(sample_id):
    payload = review_payload(sample_id)
    with writer(), connect() as db:
        template = (Path(__file__).parent / "dataset_review.html").read_text()
        html = template.replace("__PAYLOAD_BASE64__", base64.b64encode(canonical(payload).encode()).decode())
        path = local_path(f"review/{sample_id}.html")
        require(size_on_disk() + len(html.encode()) <= meta(db, "budget")["storage_bytes"], "review storage ceiling")
        atomic(path, html.encode())
    return {"review_file": str(path), "state": "proposal; import one human review"}


def import_review(path):
    return submit_review(read_json(path))


def submit_review(data, *, check_current=None):
    require(isinstance(data, dict), "invalid review payload")
    require(data.get("schema") == "chess-ocr-dataset-review/1", "invalid review schema")
    token(data["sample_id"])
    require(isinstance(data.get("reviewer"), str) and 0 < len(data["reviewer"].strip()) <= 80, "reviewer identity required")
    require(type(data.get("human")) is bool, "reviewer type required")
    require(type(data.get("elapsed_seconds")) in (int, float) and math.isfinite(data["elapsed_seconds"])
            and 0 < data["elapsed_seconds"] <= 14400, "bounded review time required")
    with writer(), connect() as db:
        sample = db.execute("SELECT * FROM samples WHERE id=?", (data["sample_id"],)).fetchone()
        require(sample is not None, "unknown sample")
        if check_current is not None:
            check_current(db)
        require(data.get("image_sha256") == sample["sha"] and digest(local_path(sample["image"])) == sample["sha"], "stale review/image")
        requested_revision = data.get("revision")
        require(type(requested_revision) is int and requested_revision in {sample["revision"], sample["revision"] - 1},
                "stale review/image")
        annotation = annotation_validate(data, sample["width"], sample["height"])
        content = canonical(annotation)
        reviewer = data["reviewer"].strip()
        # Retried identical submissions are no-ops, including when the first
        # submission advanced the revision before its response was delivered.
        prior = db.execute("SELECT 1 FROM reviews WHERE sample=? AND revision=? AND content=? AND reviewer=? AND human=?",
                           (sample["id"], sample["revision"], sample["annotation"], reviewer,
                            int(data["human"]))).fetchone()
        if prior is not None and content == sample["annotation"]:
            accepted = bool(sample["accepted"] or data["human"])
            if accepted and not sample["accepted"]:
                db.execute("UPDATE samples SET accepted=1 WHERE id=?", (sample["id"],))
            return {"state": "accepted" if accepted else "needs-human-review",
                    "revision": sample["revision"], "idempotent": True}
        require(requested_revision == sample["revision"], "stale review/image")
        require(cumulative_accounting(db)["review_decisions"] < meta(db, "budget")["review_limit"], "review batch limit; inspect first-batch cost before extending")
        changed = content != sample["annotation"]
        require(not (changed and sample["accepted"] and not data["human"]),
                "nonhuman review cannot overwrite accepted human annotation")
        revision = sample["revision"] + int(changed)
        existing = db.execute("SELECT reviewer,human FROM reviews WHERE sample=? AND revision=? AND content=?",
                              (sample["id"], revision, content)).fetchall() if not changed else []
        require(reviewer not in {r["reviewer"] for r in existing}, "same reviewer cannot confirm own annotation")
        # One matching human review is sufficient; agent-only decisions never accept.
        accepted = bool(data["human"]) or any(r["human"] for r in existing)
        db.execute("INSERT INTO reviews(sample,revision,content,reviewer,human,seconds,decision,at) VALUES (?,?,?,?,?,?,?,?)",
                   (sample["id"], revision, content, reviewer, int(data["human"]), data["elapsed_seconds"],
                    "correction" if changed else "confirmation", time.time()))
        db.execute("UPDATE samples SET revision=?,annotation=?,accepted=? WHERE id=?",
                   (revision, content, int(accepted), sample["id"]))
        if changed:
            db.execute("DELETE FROM board_signatures WHERE sample=?", (sample["id"],))
            db.execute("DELETE FROM duplicates WHERE (a=? OR b=?) AND reason LIKE 'board-%'", (sample["id"], sample["id"]))
            image = load_image(local_path(sample["image"]))
            for number, board in enumerate(annotation["boards"]):
                grid = rectify(image, board["corners"], size=128)
                grid_sha = hashlib.sha256(grid.tobytes()).hexdigest()
                grid_phash = phash(grid)
                for other in db.execute("SELECT * FROM board_signatures WHERE sample<>?", (sample["id"],)):
                    distance = (int(grid_phash, 16) ^ int(other["phash"], 16)).bit_count()
                    if grid_sha == other["sha"] or distance <= 4:
                        a, b = sorted((sample["id"], other["sample"]))
                        # Quarantine repeated artwork/placements even when page layouts differ.
                        prior = db.execute("SELECT reason FROM duplicates WHERE a=? AND b=?", (a,b)).fetchone()
                        if prior is None or prior["reason"] != "exact":
                            db.execute("INSERT OR REPLACE INTO duplicates VALUES (?,?,?,NULL)",
                                       (a, b, "board-exact" if grid_sha == other["sha"] else "board-perceptual"))
                db.execute("INSERT INTO board_signatures VALUES (?,?,?,?,?)",
                           (sample["id"], number, revision, grid_sha, grid_phash))
    return {"state": "accepted" if accepted else "needs-human-review", "revision": revision}


def resolve_duplicate(a, b, decision):
    a, b = sorted((token(a), token(b)))
    require(decision in {"distinct", "duplicate"}, "invalid duplicate decision")
    with writer(), connect() as db:
        pair = db.execute("""SELECT d.* FROM duplicates d
            JOIN samples a ON a.id=d.a JOIN sources sa ON a.source=sa.id
            JOIN samples b ON b.id=d.b JOIN sources sb ON b.source=sb.id
            WHERE d.a=? AND d.b=? AND d.decision IS NULL
            AND json_extract(sa.body, '$.split') != json_extract(sb.body, '$.split')
            AND a.source NOT IN (SELECT source FROM exclusions)
            AND b.source NOT IN (SELECT source FROM exclusions)""", (a, b)).fetchone()
        require(pair is not None, "unknown, inactive or already resolved cross-split pair")
        require(not (pair["reason"] in {"exact", "board-exact"} and decision == "distinct"), "exact bytes cannot be declared distinct")
        db.execute("UPDATE duplicates SET decision=? WHERE a=? AND b=?", (decision, a, b))
    return {"state": "recorded"}


def exclude_source(source_id, reason):
    token(source_id)
    token(reason)
    with writer(), connect() as db:
        sources = {r["id"]: json.loads(r["body"]) for r in db.execute("SELECT * FROM sources")}
        require(source_id in sources, "unknown source")
        selected, edges = {source_id}, lineage(sources[source_id])
        while True:
            related = {sid for sid,s in sources.items() if edges & lineage(s)}
            if related <= selected:
                break
            selected.update(related)
            for sid in selected:
                edges.update(lineage(sources[sid]))
        for sid in selected:
            db.execute("INSERT OR IGNORE INTO exclusions VALUES (?,?,?)", (sid, reason, time.time()))
            db.execute("UPDATE jobs SET state='excluded' WHERE source=? AND state<>'done'", (sid,))
    return {"state": "excluded-related-source-component", "sources": len(selected), "history_preserved": True}


def duplicate_exclusions(db, excluded_sources):
    sample_sources = dict(db.execute("SELECT id,source FROM samples"))
    return {r["b"] for r in db.execute("SELECT * FROM duplicates WHERE decision='duplicate'")
            if sample_sources[r["a"]] not in excluded_sources and sample_sources[r["b"]] not in excluded_sources}


def same_split_duplicate_audit(db, excluded_sources, included_splits=None):
    """Record unresolved same-split candidates without changing annotations or membership."""
    pairs = []
    for pair in db.execute("""SELECT d.a,d.b,d.reason,json_extract(sa.body, '$.split') AS split
                            FROM duplicates d
                            JOIN samples a ON d.a=a.id JOIN sources sa ON a.source=sa.id
                            JOIN samples b ON d.b=b.id JOIN sources sb ON b.source=sb.id
                            WHERE d.decision IS NULL AND json_extract(sa.body, '$.split')=json_extract(sb.body, '$.split')
                              AND a.source NOT IN (SELECT source FROM exclusions)
                              AND b.source NOT IN (SELECT source FROM exclusions)
                            ORDER BY d.a,d.b"""):
        if included_splits is None or pair["split"] in included_splits:
            pairs.append({"pair": [pair["a"], pair["b"]], "reason": pair["reason"], "split": pair["split"]})
    return {"schema": SCHEMA, "policy": "retained-unresolved-same-split-candidates",
            "limitation": "Candidates may repeat pages or boards; they are retained without a distinct or duplicate determination and can increase within-split multiplicity.",
            "pairs": pairs}


def validate(db):
    errors = []
    coverage = {}
    excluded_sources = {r[0] for r in db.execute("SELECT source FROM exclusions")}
    sources = {r["id"]: json.loads(r["body"]) for r in db.execute("SELECT * FROM sources")}
    for row in db.execute("SELECT * FROM sources"):
        if identity(json.loads(row["body"])) != row["sha"]:
            errors.append("source-record-integrity")
    source_list = [s for sid,s in sources.items() if sid not in excluded_sources]
    for i, source in enumerate(source_list):
        for other in source_list[:i]:
            if (lineage(source) & lineage(other) or source["sha256"] == other["sha256"]) and source["split"] != other["split"]:
                errors.append("cross-split-lineage")
    for sid, source in sources.items():
        if sid in excluded_sources:
            continue
        original = local_path(f"originals/{sid}.{source['format']}")
        if not original.exists() or digest(original) != source["sha256"]:
            errors.append("source-integrity")
        evidence = local_path(f"rights/{sid}.evidence")
        if not evidence.exists() or digest(evidence) != source["rights"]["evidence_sha256"]:
            errors.append("rights-integrity")
    samples = {r["id"]: r for r in db.execute("SELECT * FROM samples")}
    for pair in db.execute("SELECT * FROM duplicates"):
        if samples[pair["a"]]["source"] in excluded_sources or samples[pair["b"]]["source"] in excluded_sources:
            continue
        same_split = sources[samples[pair["a"]]["source"]]["split"] == sources[samples[pair["b"]]["source"]]["split"]
        if pair["decision"] is None and not same_split:
            errors.append("unresolved-duplicate")
        elif pair["decision"] == "duplicate" and not same_split:
            errors.append("cross-split-duplicate")
    excluded = duplicate_exclusions(db, excluded_sources)
    for sample in samples.values():
        if sample["source"] in excluded_sources:
            continue
        if not local_path(sample["image"]).exists() or digest(local_path(sample["image"])) != sample["sha"]:
            errors.append("page-integrity")
        if not sample["accepted"] or sample["id"] in excluded:
            continue
        reviewers = db.execute("SELECT DISTINCT reviewer FROM reviews WHERE sample=? AND revision=? AND content=? AND human=1",
                               (sample["id"], sample["revision"], sample["annotation"])).fetchall()
        if len(reviewers) < 1:
            errors.append("review-integrity")
        source = sources[sample["source"]]
        ann = annotation_validate(json.loads(sample["annotation"]), sample["width"], sample["height"])
        entry = coverage.setdefault(source["split"], {"pages": 0, "boards": 0, "groups": set(), "sources": set(), "conditions": {}, "classes": dict.fromkeys(LABELS, 0), "kinds": {}, "class_square_parity": {}})
        entry["pages"] += 1
        entry["boards"] += len(ann["boards"])
        entry["groups"].update(source["lineage"]["artwork"])
        entry["sources"].add(source["id"])
        entry["kinds"][ann["kind"]] = entry["kinds"].get(ann["kind"], 0) + 1
        for condition in source["conditions"]:
            entry["conditions"][condition] = entry["conditions"].get(condition, 0) + 1
        for board in ann["boards"]:
            for i, label in enumerate(board["labels"]):
                entry["classes"][label] += 1
                key = label + str((i // 8 + i % 8) % 2)
                entry["class_square_parity"][key] = entry["class_square_parity"].get(key, 0) + 1
    for entry in coverage.values():
        entry["artwork_groups"] = len(entry.pop("groups"))
        members = [sources[s] for s in entry.pop("sources")]
        components = []
        for source in members:
            edges = lineage(source)
            merged = [component for component in components if component & edges]
            components = [component for component in components if not component & edges]
            components.append(edges.union(*merged))
        entry["declared_independent_components"] = len(components)
        entry["independence_verified"] = all(s["lineage_reviewed"] for s in members)
        entry["missing_class_parity"] = [label + str(parity) for label in LABELS for parity in range(2) if not entry["class_square_parity"].get(label + str(parity))]
    jobs = dict(db.execute("SELECT state,COUNT(*) FROM jobs WHERE source NOT IN (SELECT source FROM exclusions) GROUP BY state"))
    if any(state != "done" and count for state, count in jobs.items()):
        errors.append("acquisition-incomplete")
    return {"schema": SCHEMA, "errors": sorted(set(errors)), "coverage": coverage,
            "unreviewed_pages": sum(not r["accepted"] for r in samples.values()),
            "excluded_duplicates": len(excluded), "recognition_qualified": False,
            "excluded_sources": len(excluded_sources),
            "same_split_duplicate_audit": same_split_duplicate_audit(db, excluded_sources),
            "unverified_lineage_sources": sum(not s["lineage_reviewed"] for s in sources.values()),
            "dataset_delivery_ready": False,
            "qualification_exported": False}


def solve(matrix, values):
    rows = [list(map(float, row)) + [float(value)] for row, value in zip(matrix, values)]
    for i in range(len(rows)):
        pivot = max(range(i, len(rows)), key=lambda j: abs(rows[j][i]))
        rows[i], rows[pivot] = rows[pivot], rows[i]
        require(abs(rows[i][i]) > 1e-10, "singular grid")
        scale = rows[i][i]
        rows[i] = [v / scale for v in rows[i]]
        for j in range(len(rows)):
            if j != i:
                scale = rows[j][i]
                rows[j] = [v - scale * p for v, p in zip(rows[j], rows[i])]
    return [row[-1] for row in rows]


def rectify(image, corners, size=768):
    Image = pillow()
    matrix, values = [], []
    for (u, v), (x, y) in zip([(0, 0), (size, 0), (size, size), (0, size)], corners):
        matrix.extend([[u, v, 1, 0, 0, 0, -x*u, -x*v], [0, 0, 0, u, v, 1, -y*u, -y*v]])
        values.extend([x, y])
    coeff = solve(matrix, values)
    return image.transform((size, size), Image.Transform.PERSPECTIVE, coeff, Image.Resampling.BICUBIC)


def export_dataset(clear_stop=True):
    try:
        result = build_export(clear_stop=clear_stop)
    except Exception:
        with connect() as db:
            current = meta(db, "worker")
            if current.get("pid") == os.getpid() and current["state"] == "exporting":
                set_meta(db, "worker", {"state": "export-interrupted", "heartbeat": time.time()})
        raise
    with connect() as db:
        set_meta(db, "worker", {"state": "exported", "heartbeat": time.time(), "export": result["export"]})
    return result


def build_export(clear_stop=True):
    Image = pillow()
    with writer(), connect() as db:
        if clear_stop:
            local_path("stop").unlink(missing_ok=True)
        with db:
            set_meta(db, "worker", {"state": "exporting", "pid": os.getpid(), "heartbeat": time.time()})
        report = validate(db)
        require(not report["errors"], "validation blockers; run validate")
        selected = db.execute("SELECT s.*,o.body FROM samples s JOIN sources o ON s.source=o.id WHERE s.accepted=1 ORDER BY s.id").fetchall()
        excluded_sources = {r[0] for r in db.execute("SELECT source FROM exclusions")}
        excluded = duplicate_exclusions(db, excluded_sources)
        selected = [r for r in selected if json.loads(r["body"])["split"] in {"train", "dev"} and r["id"] not in excluded and r["source"] not in excluded_sources]
        require(selected, "no accepted train/dev pages")
        duplicate_audit = same_split_duplicate_audit(db, excluded_sources, {"train", "dev"})
        recipe = {"schema": SCHEMA, "pipeline_sha256": digest(__file__), "pillow": "11.1.0", "grid": 768,
                  "tile": 96, "tensor": "64x3x96x96 float32 little-endian ImageNet normalized",
                  "order": "image-relative row-major", "classes": list(LABELS),
                  "production_parity": "pending issue #3; candidate preprocessing only",
                  "same_split_duplicate_audit": duplicate_audit,
                  "sources": {r["source"]: identity(json.loads(r["body"])) for r in selected},
                  "samples": [[r["id"], r["sha"], r["revision"], identity(json.loads(r["annotation"]))] for r in selected]}
        export_id = identity(recipe)
        output = local_path(f"exports/{export_id}")
        require(not output.exists(), "immutable export already exists; reuse it")
        estimate = sum(len(json.loads(r["annotation"])["boards"]) * (64*3*96*96*4 + 8*1024**2) + local_path(r["image"]).stat().st_size for r in selected)
        require_free_space(estimate)
        require(size_on_disk() + estimate + outstanding_synthetic_storage(db) <= meta(db, "budget")["storage_bytes"], "export storage reservation exceeds budget")
        seconds = 10 * sum(len(json.loads(r["annotation"])["boards"]) for r in selected) + 5*len(selected)
        require(cumulative_accounting(db)["reserved_compute_seconds"] + seconds <= meta(db, "budget")["cpu_seconds"], "export compute reservation exceeds budget")
        with db:
            db.execute("INSERT INTO reservations(job,bytes,seconds,at) VALUES (0,0,?,?)", (seconds,time.time()))
        deadline = time.monotonic() + seconds
        staging = local_path(f"exports/.partial-{export_id}")
        staging.mkdir(parents=True, exist_ok=True)
        records, pages = [], []
        for sample in selected:
            require(time.monotonic() < deadline, "export time ceiling; partial output retained for resume")
            require(not local_path("stop").exists(), "export stopped; partial output retained")
            with db:
                set_meta(db, "worker", {"state": "exporting", "pid": os.getpid(), "heartbeat": time.time(),
                                       "completed_pages": len(pages), "total_pages": len(selected)})
            body = json.loads(sample["body"])
            ann = json.loads(sample["annotation"])
            image = load_image(local_path(sample["image"]))
            boxes = []
            for number, board in enumerate(ann["boards"]):
                require(time.monotonic() < deadline and not local_path("stop").exists(), "export interrupted; partial output retained")
                name = f"{sample['id']}-{number}"
                grid = rectify(image, board["corners"])
                grid.save(staging / f"{name}.png")
                tensor = array.array("f")
                for index in range(64):
                    x, y = index % 8 * 96, index // 8 * 96
                    tile = grid.crop((x, y, x+96, y+96))
                    raw = tile.tobytes()
                    for channel, (mean, std) in enumerate(zip((.485, .456, .406), (.229, .224, .225))):
                        tensor.extend((raw[i] / 255 - mean) / std for i in range(channel, len(raw), 3))
                if sys.byteorder != "little":
                    tensor.byteswap()
                with (staging / f"{name}.f32").open("wb") as stream:
                    tensor.tofile(stream)
                xs, ys = zip(*board["corners"])
                box = [(min(xs)+max(xs))/2/image.width, (min(ys)+max(ys))/2/image.height,
                       (max(xs)-min(xs))/image.width, (max(ys)-min(ys))/image.height]
                boxes.append("0 " + " ".join(format(v, ".9g") for v in box))
                records.append({"sample": sample["id"], "split": body["split"], "board": number,
                                "grid": f"{name}.png", "tensor": f"{name}.f32", **board})
            # Unsupported/partial pages must never become false negative detector labels.
            if ann["kind"] in {"boards", "negative"}:
                atomic(staging / f"{sample['id']}.png", local_path(sample["image"]).read_bytes())
                atomic(staging / f"{sample['id']}.txt", ("\n".join(boxes) + ("\n" if boxes else "")).encode())
            pages.append({"sample": sample["id"], "source": sample["source"], "split": body["split"],
                          "kind": ann["kind"], "image_sha256": sample["sha"], "revision": sample["revision"],
                          "detector_eligible": ann["kind"] in {"boards", "negative"}})
        write_json(staging / "records.json", {"schema": SCHEMA, "records": records, "pages": pages})
        write_json(staging / "recipe.json", recipe)
        write_json(staging / "coverage.json", report)
        write_json(staging / "same-split-duplicate-audit.json", duplicate_audit)
        write_json(staging / "hashes.json", {p.name: digest(p) for p in sorted(staging.iterdir()) if p.name != "hashes.json"})
        os.replace(staging, output)
    return {"state": "exported", "export": export_id, "boards": len(records), "qualification_exported": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("init", "start", "run", "_run", "stop", "status", "validate", "export", "export-start", "_export", "queue"):
        sub.add_parser(command)
    budget = sub.add_parser("budget")
    for key in ("sources", "pages", "download_bytes", "storage_bytes", "cpu_seconds", "review_limit"):
        budget.add_argument("--" + key.replace("_", "-"), type=int, required=True)
    for command in ("add", "import-review"):
        sub.add_parser(command).add_argument("file")
    sub.add_parser("review").add_argument("sample")
    duplicate = sub.add_parser("duplicate")
    duplicate.add_argument("a")
    duplicate.add_argument("b")
    duplicate.add_argument("decision", choices=("distinct", "duplicate"))
    retry = sub.add_parser("retry")
    retry.add_argument("job", type=int)
    retry.add_argument("--after-repair", action="store_true")
    exclude = sub.add_parser("exclude-source")
    exclude.add_argument("source")
    exclude.add_argument("--reason", required=True)
    ingest_parser = sub.add_parser("ingest", help="queue owner-authorized PDFs from ignored work/dataset/inbox")
    ingest_parser.add_argument("--approve-local-use", action="store_true")
    ingest_parser.add_argument("--group", required=True, help="conservative shared edition/artwork group")
    ingest_parser.add_argument("--split", choices=("train", "dev"), default="train")
    ingest_parser.add_argument("--reviewer", required=True)
    ingest_parser.add_argument("--pages-per-pdf", type=int, default=40)
    sub.add_parser("_child").add_argument("job", type=int)
    args = parser.parse_args()
    try:
        if args.command == "_child":
            child(args.job)
            return
        if args.command == "init":
            result = initialize()
        elif args.command == "budget":
            result = set_budget(args)
        elif args.command == "add":
            result = add_source(args.file)
        elif args.command == "ingest":
            result = ingest(args)
        elif args.command == "start":
            result = start()
        elif args.command == "export-start":
            result = start(export=True)
        elif args.command in {"run", "_run"}:
            result = run(clear_stop=args.command == "run")
        elif args.command == "stop":
            atomic(local_path("stop"), b"stop\n")
            result = {"state": "stop-requested"}
        elif args.command == "status":
            result = status()
        elif args.command == "review":
            result = review(args.sample)
        elif args.command == "import-review":
            result = import_review(args.file)
        elif args.command == "duplicate":
            result = resolve_duplicate(args.a, args.b, args.decision)
        elif args.command == "retry":
            result = retry_job(args.job, args.after_repair)
        elif args.command == "exclude-source":
            result = exclude_source(args.source, args.reason)
        elif args.command in {"export", "_export"}:
            result = export_dataset(clear_stop=args.command != "_export")
        elif args.command == "queue":
            with connect() as db:
                result = {"samples": [dict(r) for r in db.execute("SELECT id,revision,accepted FROM samples ORDER BY id")],
                          "sources": [dict(r) for r in db.execute("SELECT id FROM sources ORDER BY id")],
                          "excluded_sources": [dict(r) for r in db.execute("SELECT * FROM exclusions")],
                          "duplicates": [dict(r) for r in db.execute("SELECT * FROM duplicates")],
                          "blocked_jobs": [dict(r) for r in db.execute("SELECT id,stage,state,error FROM jobs WHERE state='quarantined'")]}
        else:
            with connect() as db:
                result = validate(db)
        print(json.dumps(result, indent=2))
        if args.command == "validate" and result["errors"]:
            sys.exit(1)
    except Invalid as error:
        print(json.dumps({"state": "validation-failed", "reason": str(error)}), file=sys.stderr)
        sys.exit(2)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        # Never echo URLs, filenames, image contents, positions, or exception bodies.
        print('{"state":"validation-failed","action":"check local input schema, hashes, limits and queue"}', file=sys.stderr)
        sys.exit(2)
    except Exception:
        print('{"state":"operation-failed","action":"inspect local status; recover within remaining budget"}', file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
