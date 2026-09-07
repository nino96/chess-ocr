"""Bounded local proposal providers and resumable dataset proposal jobs.

Proposal bytes are machine evidence only.  This module never changes a sample's
annotation or accepted state; the normal human review transaction remains the
only path to accepted truth.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import uuid

if __package__:
    from . import dataset_pipeline as p
else:
    import dataset_pipeline as p


SCHEMA = "chess-ocr-provider-manifest/1"
PROPOSAL_SCHEMA = "chess-ocr-dataset-proposal/1"
MAX_PAGES = 100
ATTEMPT_SECONDS = 45
RUN_SECONDS = 2 * 60 * 60
STOP = "proposal-stop"
SUPPORTED_RUNTIMES = {
    "fenshot-localizer-v1": "localization",
    "fenshot-labeler-v1": "labels",
    "classical-grid-v1": "localization",
    "chess-ocr-onnx-localizer-v1": "localization",
    "chess-ocr-onnx-labeler-v1": "labels",
}
RUNNABLE_RUNTIMES = {
    "fenshot-localizer-v1", "fenshot-labeler-v1", "classical-grid-v1"
}


def _hex(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _provider_tables(db):
    db.executescript("""
        CREATE TABLE IF NOT EXISTS provider_manifests(
          id TEXT PRIMARY KEY, capability TEXT NOT NULL, runtime TEXT NOT NULL,
          body TEXT NOT NULL, sha TEXT NOT NULL, builtin INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS proposal_runs(
          id TEXT PRIMARY KEY, state TEXT NOT NULL, body TEXT NOT NULL,
          created REAL NOT NULL, heartbeat REAL NOT NULL, error TEXT);
        CREATE TABLE IF NOT EXISTS proposal_attempts(
          id INTEGER PRIMARY KEY, run TEXT NOT NULL, sample TEXT NOT NULL,
          ordinal INTEGER NOT NULL, state TEXT NOT NULL, reserved_seconds INTEGER NOT NULL,
          started REAL NOT NULL, elapsed REAL, error TEXT, UNIQUE(run,sample,ordinal));
        CREATE TABLE IF NOT EXISTS proposal_results(
          run TEXT NOT NULL, sample TEXT NOT NULL, sample_revision INTEGER NOT NULL,
          image_sha TEXT NOT NULL, body TEXT NOT NULL, created REAL NOT NULL,
          PRIMARY KEY(run,sample));
        CREATE TABLE IF NOT EXISTS review_deferrals(
          id INTEGER PRIMARY KEY, sample TEXT NOT NULL, revision INTEGER NOT NULL,
          image_sha TEXT NOT NULL, reason TEXT NOT NULL, seconds REAL NOT NULL,
          proposal_run TEXT, created REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS review_metrics(
          review_id INTEGER PRIMARY KEY, body TEXT NOT NULL);
    """)


def _builtin_manifests():
    model = p.REPO / "node_modules/@scoriiu/fenshot/model/chess-tiles-v2.onnx"
    fenshot_dist = p.REPO / "node_modules/@scoriiu/fenshot/dist"
    p.require(model.is_file() and not model.is_symlink(), "run pnpm install/setup before proposals")
    fenshot_localizer_sha = p.identity({
        name: p.digest(fenshot_dist / name) for name in ("detect.js", "tiles.js")
    })
    model_sha = p.digest(model)
    limits = {"max_pixels": min(p.MAX_PIXELS, 16_000_000), "max_dimension": p.MAX_EDGE,
              "timeout_ms": 30000, "max_candidates": 16}
    identity = {"name": "@scoriiu/fenshot", "version": "0.1.4", "sha256": model_sha}
    artifact = {"path": str(model.relative_to(p.REPO)), "sha256": model_sha}
    classical_path = p.REPO / "src/proposals/index.ts"
    classical_sha = p.digest(classical_path) if classical_path.is_file() else "0" * 64
    return [
        {"schema": SCHEMA, "id": "fenshot-localizer-v1", "capability": "localization",
         "runtime": "fenshot-localizer-v1",
         "model": {"name": "@scoriiu/fenshot grid detector", "version": "0.1.4",
                   "sha256": fenshot_localizer_sha},
         "preprocessing": "fenshot-0.1.4/rgba-gray-grid/1",
         "artifact": None, "limits": limits},
        {"schema": SCHEMA, "id": "fenshot-labeler-v1", "capability": "labels",
         "runtime": "fenshot-labeler-v1", "model": identity,
         "preprocessing": "fenshot-0.1.4/rgba-gray-bilinear-256/1",
         "artifact": artifact, "limits": limits},
        {"schema": SCHEMA, "id": "classical-grid-v1", "capability": "localization",
         "runtime": "classical-grid-v1",
         "model": {"name": "chess-ocr classical grid", "version": "1", "sha256": classical_sha},
         "preprocessing": "chess-ocr/classical-grid-gray/1", "artifact": None,
         "limits": limits},
    ]


def validate_manifest(value, *, allow_builtin=False):
    p.require(isinstance(value, dict) and set(value) == {
        "schema", "id", "capability", "runtime", "model", "preprocessing", "artifact", "limits"
    }, "invalid provider manifest")
    p.require(value["schema"] == SCHEMA, "invalid provider schema")
    p.token(value["id"])
    runtime = value["runtime"]
    p.require(runtime in SUPPORTED_RUNTIMES and SUPPORTED_RUNTIMES[runtime] == value["capability"],
              "unsupported provider runtime/capability")
    p.require(isinstance(value["preprocessing"], str) and 0 < len(value["preprocessing"]) <= 160,
              "invalid provider preprocessing")
    model = value["model"]
    p.require(isinstance(model, dict) and set(model) == {"name", "version", "sha256"}
              and all(isinstance(model[k], str) and 0 < len(model[k]) <= 160 for k in ("name", "version"))
              and _hex(model["sha256"]), "invalid provider model identity")
    limits = value["limits"]
    p.require(isinstance(limits, dict) and set(limits) == {
        "max_pixels", "max_dimension", "timeout_ms", "max_candidates"
    }, "invalid provider limits")
    p.require(type(limits["max_pixels"]) is int and 0 < limits["max_pixels"] <= p.MAX_PIXELS
              and type(limits["max_dimension"]) is int and 0 < limits["max_dimension"] <= p.MAX_EDGE
              and type(limits["timeout_ms"]) is int and 100 <= limits["timeout_ms"] <= 30000
              and type(limits["max_candidates"]) is int and 1 <= limits["max_candidates"] <= 16,
              "provider limits exceed dataset bounds")
    artifact = value["artifact"]
    if artifact is None:
        p.require(runtime in {"classical-grid-v1", "fenshot-localizer-v1"},
                  "model provider artifact required")
    else:
        p.require(isinstance(artifact, dict) and set(artifact) == {"path", "sha256"}
                  and isinstance(artifact["path"], str) and _hex(artifact["sha256"]),
                  "invalid provider artifact")
        path = (p.REPO / artifact["path"]).resolve()
        allowed = [p.REPO / "work", p.REPO / "artifacts"]
        if allow_builtin:
            allowed.append(p.REPO / "node_modules")
        p.require(any(path.is_relative_to(root.resolve()) for root in allowed), "provider artifact outside approved local roots")
        p.require(path.is_file() and not path.is_symlink() and p.digest(path) == artifact["sha256"],
                  "provider artifact missing or changed")
        p.require(artifact["sha256"] == model["sha256"], "artifact/model hash mismatch")
    return value


def initialize():
    with p.writer(), p.connect() as db:
        _provider_tables(db)
        for manifest in _builtin_manifests():
            validate_manifest(manifest, allow_builtin=True)
            body = p.canonical(manifest)
            sha = p.identity(manifest)
            prior = db.execute("SELECT sha FROM provider_manifests WHERE id=?", (manifest["id"],)).fetchone()
            p.require(prior is None or prior["sha"] == sha,
                      "built-in provider identity changed; version its provider id")
            db.execute("INSERT OR IGNORE INTO provider_manifests VALUES (?,?,?,?,?,1)",
                       (manifest["id"], manifest["capability"], manifest["runtime"], body, sha))


def register_manifest(path):
    candidate = Path(path).resolve()
    p.require(any(candidate.is_relative_to((p.REPO / root).resolve()) for root in ("work", "artifacts")),
              "provider manifest must be under ignored work/ or artifacts/")
    p.require(candidate.is_file() and not candidate.is_symlink(), "unsafe provider manifest")
    manifest = validate_manifest(p.read_json(candidate))
    with p.writer(), p.connect() as db:
        _provider_tables(db)
        prior = db.execute("SELECT sha FROM provider_manifests WHERE id=?", (manifest["id"],)).fetchone()
        sha = p.identity(manifest)
        p.require(prior is None or prior["sha"] == sha, "provider id already has another immutable manifest")
        db.execute("INSERT OR IGNORE INTO provider_manifests VALUES (?,?,?,?,?,0)",
                   (manifest["id"], manifest["capability"], manifest["runtime"], p.canonical(manifest), sha))
    return {"state": "registered", "id": manifest["id"], "sha256": sha}


def providers():
    with p.connect() as db:
        _provider_tables(db)
        values = []
        placeholders = ",".join("?" for _ in RUNNABLE_RUNTIMES)
        for row in db.execute(
                f"SELECT * FROM provider_manifests WHERE runtime IN ({placeholders}) ORDER BY capability,id",
                tuple(sorted(RUNNABLE_RUNTIMES))):
            body = json.loads(row["body"])
            values.append({"id": row["id"], "capability": row["capability"],
                           "runtime": row["runtime"], "manifest_sha256": row["sha"],
                           "model": body["model"], "preprocessing": body["preprocessing"],
                           "limits": body["limits"]})
    return {"schema": "chess-ocr-provider-registry/1", "providers": values,
            "defaults": {"localization": "fenshot-localizer-v1", "labels": "fenshot-labeler-v1"}}


def _selected_samples(db, scope, maximum):
    where = {
        "train-pending": "json_extract(o.body,'$.split')='train' AND s.accepted=0",
        "train-all": "json_extract(o.body,'$.split')='train'",
        "accepted-train": "json_extract(o.body,'$.split')='train' AND s.accepted=1",
    }.get(scope)
    p.require(where is not None, "proposal scope must exclude qualification")
    rows = db.execute(f"""SELECT s.id,s.revision,s.sha FROM samples s JOIN sources o ON o.id=s.source
        WHERE s.source NOT IN (SELECT source FROM exclusions) AND {where} ORDER BY s.source,s.page LIMIT ?""",
                      (maximum,)).fetchall()
    p.require(rows, "no eligible TRAIN pages for proposal run")
    return [{"id": r["id"], "revision": r["revision"], "image_sha256": r["sha"]} for r in rows]


def create_run(localizer, labeler, scope="train-pending", max_pages=MAX_PAGES):
    p.require(type(max_pages) is int and 1 <= max_pages <= MAX_PAGES, "proposal page bound")
    with p.writer(), p.connect() as db:
        _provider_tables(db)
        active = db.execute("SELECT 1 FROM proposal_runs WHERE state IN ('starting','running')").fetchone()
        p.require(active is None, "another proposal run is active")
        loc = db.execute("SELECT * FROM provider_manifests WHERE id=? AND capability='localization'", (localizer,)).fetchone()
        lab = db.execute("SELECT * FROM provider_manifests WHERE id=? AND capability='labels'", (labeler,)).fetchone()
        p.require(loc is not None and lab is not None
                  and loc["runtime"] in RUNNABLE_RUNTIMES
                  and lab["runtime"] in RUNNABLE_RUNTIMES,
                  "select runnable registered localization and label providers")
        samples = _selected_samples(db, scope, max_pages)
        body = {"schema": "chess-ocr-proposal-run/1", "localizer": {"id": loc["id"], "sha256": loc["sha"]},
                "labeler": {"id": lab["id"], "sha256": lab["sha"]}, "scope": scope,
                "samples": samples, "limits": {"pages": len(samples), "attempt_seconds": ATTEMPT_SECONDS,
                                                   "run_seconds": RUN_SECONDS}, "nonce": uuid.uuid4().hex}
        run_id = p.identity(body)
        now = time.time()
        db.execute("INSERT INTO proposal_runs VALUES (?,?,?,?,?,NULL)",
                   (run_id, "starting", p.canonical(body), now, now))
        p.local_path(STOP).unlink(missing_ok=True)
    return _launch(run_id)


def _launch(run_id):
    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "run", run_id],
                               cwd=p.REPO, start_new_session=True, stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        with p.connect() as db:
            row = db.execute("SELECT state FROM proposal_runs WHERE id=?", (run_id,)).fetchone()
        if row and row["state"] != "starting":
            return {"state": row["state"], "run_id": run_id, "pid": process.pid}
        if process.poll() is not None:
            raise p.Invalid("proposal worker startup failed; run in foreground for diagnosis")
        time.sleep(.05)
    raise p.Invalid("proposal worker startup not confirmed; inspect proposal status")


def resume(run_id, after_repair=False):
    p.require(isinstance(run_id, str) and _hex(run_id), "invalid proposal run")
    with p.writer(), p.connect() as db:
        _provider_tables(db)
        row = db.execute("SELECT state,heartbeat FROM proposal_runs WHERE id=?", (run_id,)).fetchone()
        stale = (row is not None and row["state"] in {"starting", "running"}
                 and row["heartbeat"] <= time.time() - 90)
        p.require(row is not None and (row["state"] in {"stopped", "interrupted"}
                  or stale or (after_repair and row["state"] == "needs-repair")),
                  "run is not resumable")
        db.execute("UPDATE proposal_runs SET state='starting',error=NULL,heartbeat=? WHERE id=?", (time.time(),run_id))
        p.local_path(STOP).unlink(missing_ok=True)
    return _launch(run_id)


def _attempt(db, run_id, sample):
    accounting = p.cumulative_accounting(db)
    budget = p.meta(db, "budget")
    p.require(accounting["reserved_compute_seconds"] + ATTEMPT_SECONDS <= budget["cpu_seconds"],
              "proposal compute reservation exceeds dataset budget")
    ordinal = db.execute("SELECT COUNT(*) FROM proposal_attempts WHERE run=? AND sample=?",
                         (run_id,sample)).fetchone()[0]
    db.execute("INSERT INTO proposal_attempts(run,sample,ordinal,state,reserved_seconds,started) VALUES (?,?,?,?,?,?)",
               (run_id,sample,ordinal,"running",ATTEMPT_SECONDS,time.time()))
    db.execute("INSERT INTO reservations(job,bytes,seconds,at) VALUES (0,0,?,?)", (ATTEMPT_SECONDS,time.time()))
    return ordinal


def validate_result(value, width, height):
    p.require(isinstance(value,dict) and set(value) == {"schema","runId","sampleId","revision",
              "imageSha256","status","boards","warnings","providers","timings"}
              and value["schema"] == PROPOSAL_SCHEMA and value["status"] in {"ok","unsupported"},
              "invalid proposal result")
    p.require(isinstance(value["warnings"],list) and len(value["warnings"]) <= 60
              and all(isinstance(x,str) and len(x) <= 300 for x in value["warnings"]), "invalid proposal warnings")
    p.require(isinstance(value["providers"],list) and len(value["providers"]) == 2, "invalid proposal providers")
    for manifest in value["providers"]:
        validate_manifest(manifest, allow_builtin=True)
    timing = value["timings"]
    p.require(isinstance(timing,dict) and set(timing) == {"totalMs"}
              and type(timing["totalMs"]) in (int,float) and 0 <= timing["totalMs"] <= 120000,
              "invalid proposal timing")
    boards = value["boards"]
    p.require(isinstance(boards,list) and len(boards) <= 16
              and ((value["status"] == "ok") == bool(boards)), "invalid proposal board status")
    for board in boards:
        p.require(isinstance(board,dict) and set(board) == {"id","corners","labels","orientation",
                  "probabilities","uncertain"} and isinstance(board["id"],str) and 0 < len(board["id"]) <= 100,
                  "invalid proposal board")
        p.annotation_validate({"kind":"boards","complete_page":True,"boards":[
            {"corners":board["corners"],"labels":board["labels"],"orientation":board["orientation"]}
        ]},width,height)
        p.require(isinstance(board["uncertain"],list) and len(board["uncertain"]) == 64
                  and all(type(x) is bool for x in board["uncertain"]), "invalid proposal uncertainty")
        p.require(isinstance(board["probabilities"],list) and len(board["probabilities"]) == 64,
                  "invalid proposal probabilities")
        for probabilities in board["probabilities"]:
            p.require(probabilities is None or (isinstance(probabilities,list) and len(probabilities) == 13
                      and all(type(x) in (int,float) and 0 <= x <= 1 for x in probabilities)
                      and abs(sum(probabilities)-1) <= 1e-4), "invalid proposal probabilities")
        p.require(all(probabilities is not None or board["uncertain"][index]
                      for index, probabilities in enumerate(board["probabilities"])),
                  "missing proposal evidence must remain uncertain")
    return value


def _run_child(run_id, sample_id):
    staging = p.local_path(f"proposals/staging/{run_id}/{sample_id}")
    staging.mkdir(parents=True, exist_ok=True)
    for path in staging.iterdir():
        p.require(path.is_file() and not path.is_symlink(), "unsafe proposal staging")
        path.unlink()
    with p.connect() as db:
        run = db.execute("SELECT body FROM proposal_runs WHERE id=?", (run_id,)).fetchone()
        sample = db.execute("SELECT * FROM samples WHERE id=?", (sample_id,)).fetchone()
        p.require(run is not None and sample is not None, "proposal input disappeared")
        body = json.loads(run["body"])
        loc = json.loads(db.execute("SELECT body FROM provider_manifests WHERE id=?", (body["localizer"]["id"],)).fetchone()[0])
        lab = json.loads(db.execute("SELECT body FROM provider_manifests WHERE id=?", (body["labeler"]["id"],)).fetchone()[0])
    page = p.local_path(sample["image"])
    p.require(p.digest(page) == sample["sha"], "proposal source image changed")
    image = p.load_image(page).convert("RGBA")
    raw = staging / "image.rgba"
    p.atomic(raw, image.tobytes())
    request = {"schema": "chess-ocr-proposal-request/1", "run_id": run_id, "sample_id": sample_id,
               "revision": sample["revision"], "image_sha256": sample["sha"],
               "width": image.width, "height": image.height, "rgba_path": str(raw),
               "config_sha256": p.identity({"localizer": loc, "labeler": lab}),
               "localizer": loc, "labeler": lab}
    request_path, output = staging / "request.json", staging / "result.json"
    p.write_json(request_path, request)
    completed = subprocess.run(["node", "--experimental-strip-types", "scripts/proposal-runner.ts",
                                str(request_path), str(output)], cwd=p.REPO, timeout=35,
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    p.require(completed.returncode == 0 and output.is_file(), "proposal provider failed")
    value = validate_result(p.read_json(output), image.width, image.height)
    p.require(value.get("schema") == PROPOSAL_SCHEMA and value.get("runId") == run_id
              and value.get("sampleId") == sample_id and value.get("revision") == sample["revision"]
              and value.get("imageSha256") == sample["sha"], "invalid proposal result identity")
    raw.unlink(missing_ok=True)


def child(run_id, sample_id):
    signal.alarm(40)
    resource.setrlimit(resource.RLIMIT_CPU, (35,35))
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024**2, 64 * 1024**2))
    _run_child(run_id, sample_id)


def run(run_id):
    p.require(_hex(run_id), "invalid proposal run")
    with p.writer(), p.connect() as db:
        _provider_tables(db)
        row = db.execute("SELECT * FROM proposal_runs WHERE id=?", (run_id,)).fetchone()
        p.require(row is not None and row["state"] == "starting", "proposal run is not starting")
        db.execute("UPDATE proposal_runs SET state='running',heartbeat=? WHERE id=?", (time.time(),run_id))
        body = json.loads(row["body"])
    state, error = "complete", None
    started = time.monotonic()
    try:
        for sample in body["samples"]:
            if p.local_path(STOP).exists():
                state = "stopped"
                break
            p.require(time.monotonic() - started < RUN_SECONDS, "proposal run time ceiling")
            with p.connect() as db:
                if db.execute("SELECT 1 FROM proposal_results WHERE run=? AND sample=?", (run_id,sample["id"])).fetchone():
                    continue
            with p.writer(), p.connect() as db:
                ordinal = _attempt(db, run_id, sample["id"])
                db.execute("UPDATE proposal_runs SET heartbeat=? WHERE id=?", (time.time(),run_id))
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "_child", run_id, sample["id"]],
                                       cwd=p.REPO, start_new_session=True, stdin=subprocess.DEVNULL,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            before = time.monotonic()
            reason = None
            while process.poll() is None:
                if p.local_path(STOP).exists() or time.monotonic() - before > ATTEMPT_SECONDS:
                    reason = "stopped" if p.local_path(STOP).exists() else "timeout"
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    break
                time.sleep(.1)
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            elapsed = time.monotonic() - before
            with p.writer(), p.connect() as db:
                if process.returncode == 0:
                    result_path = p.local_path(f"proposals/staging/{run_id}/{sample['id']}/result.json")
                    current_dimensions = db.execute("SELECT width,height FROM samples WHERE id=?", (sample["id"],)).fetchone()
                    current = db.execute("SELECT revision,sha FROM samples WHERE id=?", (sample["id"],)).fetchone()
                    if current is None or current_dimensions is None or current["revision"] != sample["revision"] or current["sha"] != sample["image_sha256"]:
                        db.execute("UPDATE proposal_attempts SET state='stale',elapsed=? WHERE run=? AND sample=? AND ordinal=?",
                                   (elapsed,run_id,sample["id"],ordinal))
                    else:
                        value = validate_result(p.read_json(result_path), current_dimensions["width"], current_dimensions["height"])
                        db.execute("INSERT OR REPLACE INTO proposal_results VALUES (?,?,?,?,?,?)",
                                   (run_id,sample["id"],sample["revision"],sample["image_sha256"],p.canonical(value),time.time()))
                        db.execute("UPDATE proposal_attempts SET state='done',elapsed=? WHERE run=? AND sample=? AND ordinal=?",
                                   (elapsed,run_id,sample["id"],ordinal))
                else:
                    failure = reason or "provider-failed"
                    db.execute("UPDATE proposal_attempts SET state='failed',elapsed=?,error=? WHERE run=? AND sample=? AND ordinal=?",
                               (elapsed,failure,run_id,sample["id"],ordinal))
                    state, error = ("stopped", None) if reason == "stopped" else ("needs-repair", failure)
            if process.returncode != 0:
                break
    except Exception as exc:
        state, error = "interrupted", str(exc)
        raise
    finally:
        with p.writer(), p.connect() as db:
            if db.execute("SELECT 1 FROM proposal_runs WHERE id=?", (run_id,)).fetchone():
                db.execute("UPDATE proposal_runs SET state=?,heartbeat=?,error=? WHERE id=?",
                           (state,time.time(),error,run_id))
    return status()


def stop():
    p.atomic(p.local_path(STOP), b"stop\n")
    return {"state": "stop-requested", "boundary": "current bounded page"}


def status():
    with p.connect() as db:
        _provider_tables(db)
        rows = []
        for run in db.execute("SELECT * FROM proposal_runs ORDER BY created DESC LIMIT 20"):
            body = json.loads(run["body"])
            counts = dict(db.execute("SELECT state,COUNT(*) FROM proposal_attempts WHERE run=? GROUP BY state", (run["id"],)))
            completed = db.execute("SELECT COUNT(*) FROM proposal_results WHERE run=?", (run["id"],)).fetchone()[0]
            state = run["state"]
            if state in {"starting","running"} and time.time() - run["heartbeat"] > 90:
                state = "heartbeat-stale-resume-required"
            rows.append({"id": run["id"], "state": state, "localizer": body["localizer"],
                         "labeler": body["labeler"], "scope": body["scope"], "pages": len(body["samples"]),
                         "completed": completed, "attempts": counts, "error": run["error"]})
    return {"schema": "chess-ocr-proposal-status/1", "runs": rows,
            "active": next((r["id"] for r in rows if r["state"] in {"starting","running"}), None)}


def sample_results(sample_id):
    p.token(sample_id)
    with p.connect() as db:
        _provider_tables(db)
        sample = db.execute("SELECT revision,sha,width,height FROM samples WHERE id=?", (sample_id,)).fetchone()
        p.require(sample is not None, "unknown sample")
        results = [validate_result(json.loads(row["body"]), sample["width"], sample["height"]) for row in db.execute(
            "SELECT body FROM proposal_results WHERE sample=? AND sample_revision=? AND image_sha=? ORDER BY created DESC",
            (sample_id,sample["revision"],sample["sha"]))]
    return {"schema": "chess-ocr-sample-proposals/1", "sample_id": sample_id, "results": results}


def defer(data):
    reasons = {"ambiguous", "geometry", "labels", "image-quality", "unreadable-geometry",
               "ambiguous-pieces", "extensive-repair", "false-proposal", "missing-board"}
    p.require(isinstance(data, dict) and set(data) == {
        "sample_id", "revision", "image_sha256", "reason", "elapsed_seconds", "proposal_run"
    }, "invalid deferral")
    p.token(data["sample_id"])
    p.require(data["reason"] in reasons and type(data["elapsed_seconds"]) in (int,float)
              and 0 <= data["elapsed_seconds"] <= 14400
              and (data["proposal_run"] is None or _hex(data["proposal_run"])), "invalid deferral")
    with p.writer(), p.connect() as db:
        _provider_tables(db)
        sample = db.execute("SELECT revision,sha FROM samples WHERE id=?", (data["sample_id"],)).fetchone()
        p.require(sample is not None and sample["revision"] == data["revision"]
                  and sample["sha"] == data["image_sha256"], "stale deferral")
        db.execute("INSERT INTO review_deferrals(sample,revision,image_sha,reason,seconds,proposal_run,created) VALUES (?,?,?,?,?,?,?)",
                   (data["sample_id"],data["revision"],data["image_sha256"],data["reason"],data["elapsed_seconds"],data["proposal_run"],time.time()))
    return {"state": "deferred", "reason": data["reason"]}


def _polygon_area(points):
    return abs(sum(points[i][0] * points[(i + 1) % len(points)][1]
                   - points[(i + 1) % len(points)][0] * points[i][1]
                   for i in range(len(points))) / 2) if len(points) >= 3 else 0


def _intersection(subject, clip):
    """Sutherland-Hodgman intersection for clockwise image-coordinate quads."""
    output = [list(point) for point in subject]
    for index, edge_start in enumerate(clip):
        edge_end = clip[(index + 1) % len(clip)]
        prior = output
        output = []
        if not prior:
            break

        def inside(point):
            return ((edge_end[0] - edge_start[0]) * (point[1] - edge_start[1])
                    - (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0])) >= 0

        def crossing(a, b):
            ex, ey = edge_end[0] - edge_start[0], edge_end[1] - edge_start[1]
            dx, dy = b[0] - a[0], b[1] - a[1]
            denominator = dx * ey - dy * ex
            if abs(denominator) < 1e-12:
                return list(b)
            t = ((edge_start[0] - a[0]) * ey - (edge_start[1] - a[1]) * ex) / denominator
            return [a[0] + t * dx, a[1] + t * dy]

        previous = prior[-1]
        for current in prior:
            if inside(current):
                if not inside(previous):
                    output.append(crossing(previous,current))
                output.append(list(current))
            elif inside(previous):
                output.append(crossing(previous,current))
            previous = current
    return output


def _iou(a, b):
    overlap = _polygon_area(_intersection(a,b))
    union = _polygon_area(a) + _polygon_area(b) - overlap
    return overlap / union if union > 0 else 0


def review_metrics(final_annotation, draft, elapsed_seconds):
    proposed = draft.get("proposal_base", []) if isinstance(draft, dict) else []
    final = final_annotation.get("boards", [])
    pairs = sorted((( _iou(a["corners"], b["corners"]), ai, bi)
                    for ai,a in enumerate(proposed) for bi,b in enumerate(final)), reverse=True)
    used_a, used_b, matched = set(), set(), []
    for overlap, ai, bi in pairs:
        if overlap < .5 or ai in used_a or bi in used_b:
            continue
        used_a.add(ai); used_b.add(bi)
        a, b = proposed[ai], final[bi]
        distances = [((a["corners"][i][0]-b["corners"][i][0])**2
                      + (a["corners"][i][1]-b["corners"][i][1])**2)**.5 for i in range(4)]
        edge = sum((((b["corners"][(i+1)%4][0]-b["corners"][i][0])**2
                     + (b["corners"][(i+1)%4][1]-b["corners"][i][1])**2)**.5) for i in range(4)) / 4
        matched.append({"proposal_board": ai, "review_board": bi, "iou": overlap,
                        "mean_corner_pixels": sum(distances)/4,
                        "mean_corner_squares": (sum(distances)/4)/(edge/8) if edge else None,
                        "piece_corrections": sum(x != y for x,y in zip(a["labels"],b["labels"]))})
    touched = draft.get("touched", {}) if isinstance(draft, dict) else {}
    return {"schema": "chess-ocr-review-metrics/1", "proposal_run": draft.get("proposal_run") if isinstance(draft,dict) else None,
            "active_seconds": elapsed_seconds, "proposal_boards": len(proposed), "review_boards": len(final),
            "matched_boards": matched, "false_boards": len(proposed)-len(used_a),
            "missed_boards": len(final)-len(used_b),
            "touched": touched if isinstance(touched,dict) else {}}


def record_review_metrics(sample_id, reviewer, final_annotation, elapsed_seconds):
    """Attach derived assistance cost to the immutable human review decision."""
    with p.writer(), p.connect() as db:
        _provider_tables(db)
        review = db.execute("SELECT id FROM reviews WHERE sample=? AND reviewer=? ORDER BY id DESC LIMIT 1",
                            (sample_id,reviewer)).fetchone()
        p.require(review is not None, "review decision missing for metrics")
        draft = db.execute("SELECT body FROM web_drafts WHERE sample=?", (sample_id,)).fetchone()
        body = json.loads(draft["body"]) if draft else {}
        metric = review_metrics(final_annotation,body,elapsed_seconds)
        db.execute("INSERT OR IGNORE INTO review_metrics VALUES (?,?)", (review["id"],p.canonical(metric)))
    return metric


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    sub.add_parser("providers")
    register = sub.add_parser("register")
    register.add_argument("manifest")
    start = sub.add_parser("start")
    start.add_argument("--localizer", default="fenshot-localizer-v1")
    start.add_argument("--labeler", default="fenshot-labeler-v1")
    start.add_argument("--scope", choices=("train-pending","train-all","accepted-train"), default="train-pending")
    start.add_argument("--max-pages", type=int, default=MAX_PAGES)
    resume_parser = sub.add_parser("resume")
    resume_parser.add_argument("run_id")
    resume_parser.add_argument("--after-repair", action="store_true")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("run_id")
    child_parser = sub.add_parser("_child")
    child_parser.add_argument("run_id")
    child_parser.add_argument("sample_id")
    sub.add_parser("status")
    sub.add_parser("stop")
    args = parser.parse_args()
    if args.command == "_child":
        child(args.run_id,args.sample_id)
        return
    if args.command not in {"init", "run"}:
        initialize()
    result = {"init": initialize, "providers": providers, "register": lambda: register_manifest(args.manifest),
              "start": lambda: create_run(args.localizer,args.labeler,args.scope,args.max_pages),
              "resume": lambda: resume(args.run_id,args.after_repair), "run": lambda: run(args.run_id),
              "status": status, "stop": stop}[args.command]()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
