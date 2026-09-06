"""Bounded, hash-frozen synthetic jobs; payloads and operational state stay local."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import signal
import resource
from collections import Counter
import subprocess
import sys
import time

if __package__:
    from . import dataset_pipeline as d
else:
    import dataset_pipeline as d

ROOT = d.REPO / "work/dataset/synthetic"
CONFIG = d.REPO / "recipes/synthetic-seed-v1.json"
CODE = ["scripts/synthetic-render.mjs", "python/synthetic_job.py",
        "scripts/synthetic-degradation.mjs", "scripts/synthetic-perspective.mjs",
        "python/synthetic_training.py", "python/dataset_pipeline.py", "pnpm-lock.yaml",
        "python/requirements-dataset-linux-cp312.txt", "provenance/public-bootstrap.json"]


def fidelity_gate():
    report = d.read_json(d.REPO / "work/dataset/bootstrap/fidelity/report.json")
    d.require(report["state"] == "passed" and report["controls"] >= 90
              and report.get("calibration_controls") == 39 and report.get("rebuild_pages") == 3
              and report.get("independent_identity_squares") == 2496,
              "complete independent renderer fidelity required")
    audit=report.get("effect_audit",{})
    d.require(audit.get("state")=="passed" and audit.get("controls")==12 and audit.get("squares")==768 and audit.get("minimum_class_margin",0)>0,"effect label preservation required")
    for key, path in [("renderer_sha256", "scripts/synthetic-render.mjs"),
                      ("control_sha256", "scripts/synthetic-fidelity.mjs"),
                      ("effects_sha256", "scripts/synthetic-degradation.mjs"),
                      ("perspective_sha256", "scripts/synthetic-perspective.mjs"),
                      ("effect_reference_sha256", "python/synthetic_effect_reference.py"),
                      ("checker_sha256", "python/synthetic_fidelity.py"),
                      ("training_sha256", "python/synthetic_training.py"),
                      ("provenance_sha256", "provenance/public-bootstrap.json")]:
        d.require(report[key] == d.digest(d.REPO / path), "stale fidelity evidence")
    d.require(report["runtime"] == runtime_identity(), "fidelity runtime changed")
    return d.identity(report)


def safe_root():
    return d.local_path("synthetic")


@contextlib.contextmanager
def lock():
    safe_root().mkdir(parents=True, exist_ok=True)
    with (ROOT / "job.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise d.Invalid("synthetic worker already active") from None
        yield


def configuration(path=CONFIG):
    c = d.read_json(path)
    d.require(c.get("schema") == "chess-ocr-synthetic-recipe/1", "recipe schema")
    for key, maximum in [("pages", 20000), ("max_boards", 20000), ("batch_pages", 64),
                         ("attempt_seconds", 120), ("compute_seconds", 28800),
                         ("storage_bytes", 16 * 1024**3)]:
        d.require(type(c.get(key)) is int and 0 < c[key] <= maximum, "invalid recipe bound")
    d.require(type(c.get("seed")) is int and 0 <= c["seed"] < 2**32, "seed bound")
    d.require(c.get("split") == "train" and c.get("label_order") == d.LABELS,
              "training-only image-relative schema required")
    d.require(c.get("sets") == ["chessnut", "fantasy", "rhosgfx"], "asset allowlist")
    return c


def frozen_identity(c):
    return {"configuration": c, "files": {p: d.digest(d.REPO / p) for p in CODE},
            "node": subprocess.check_output(["node", "--version"], text=True).strip(),
            "runtime": runtime_identity()}


def runtime_identity():
    browser = subprocess.check_output(["node", "--input-type=module", "-e",
              "import {chromium} from '@playwright/test'; process.stdout.write(chromium.executablePath())"],
              cwd=d.REPO, text=True, timeout=10).strip()
    fonts = sorted(set(subprocess.check_output(["fc-list", "--format", "%{file}\\n"],
                                               text=True, timeout=10).splitlines()))
    return {"chromium_sha256": d.digest(browser),
            "font_hashes": sorted(d.digest(path) for path in fonts),
            "font_selection": subprocess.check_output(["fc-match", "-s", "system-ui,sans-serif", "--format", "%{file}:%{family}:%{style}\\n"], text=True, timeout=10),
            "fontconfig_hashes": {str(p): d.digest(p) for p in sorted(Path('/etc/fonts').rglob('*')) if p.is_file()},
            "locale": {k: os.environ.get(k, "") for k in ("LANG", "LC_ALL", "LC_CTYPE")}}


def node(*args, timeout=120):
    # Kill the entire renderer process group on a ceiling, including Chromium.
    def limits():
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
        resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout))
        resource.setrlimit(resource.RLIMIT_FSIZE, (16*1024**2, 16*1024**2))
    proc = subprocess.Popen(["timeout", "--signal=KILL", str(timeout), "node", "--max-old-space-size=256", "scripts/synthetic-render.mjs", *map(str, args)],
                            cwd=d.REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            start_new_session=True, preexec_fn=limits)
    try:
        deadline = time.monotonic() + timeout
        while True:
            try:
                out, _ = proc.communicate(timeout=min(.5, max(.01, deadline-time.monotonic())))
                break
            except subprocess.TimeoutExpired:
                if time.monotonic() >= deadline:
                    raise
                rss = 0
                for path in Path("/proc").glob("[0-9]*/stat"):
                    try:
                        fields = path.read_text().split(") ", 1)[1].split()
                        if int(fields[2]) == proc.pid:
                            rss += int(fields[21]) * os.sysconf("SC_PAGE_SIZE")
                    except (OSError, ValueError, IndexError):
                        continue
                if rss > 4*1024**3:
                    raise subprocess.TimeoutExpired("memory ceiling", timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        raise d.Invalid("renderer time/memory ceiling; failed attempt retained") from None
    d.require(proc.returncode == 0, "renderer failed; input/renderer inspection required")
    d.require(len(out) <= 32 * 1024**2, "renderer metadata limit")
    return json.loads(out)


def initialize():
    with lock(), d.writer(), d.connect() as db:
        d.require(not (ROOT / "frozen.json").exists(), "already initialized; resume existing job")
        c = configuration()
        gate = fidelity_gate()
        frozen = frozen_identity(c)
        recipes = []
        for start in range(0, c["pages"], 64):
            recipes.extend(node("recipes", c["seed"], start, min(64, c["pages"]-start)))
        boards = sum(len(r["boards"]) for r in recipes)
        equivalents = boards + sum(len(r.get("unsupported_boards", [])) for r in recipes)
        d.require(0 < equivalents <= c["max_boards"], "board-equivalent ceiling")
        cov=coverage(recipes)
        validate_coverage(cov)
        budget, used = d.meta(db, "budget"), d.cumulative_accounting(db)
        d.require(used["reserved_compute_seconds"] + c["compute_seconds"] <= budget["cpu_seconds"], "global compute ceiling")
        d.require(d.size_on_disk() + c["storage_bytes"] <= budget["storage_bytes"], "global storage ceiling")
        d.require_free_space(c["storage_bytes"])
        # Reserve once, in the existing ledger. Failed attempts draw down this
        # reservation, never refund it. The transaction precedes any bulk rendering.
        with db:
            db.execute("CREATE TABLE IF NOT EXISTS synthetic_attempts(run TEXT NOT NULL, ordinal INTEGER NOT NULL, body TEXT NOT NULL, PRIMARY KEY(run,ordinal))")
            db.execute("INSERT INTO reservations(job,bytes,seconds,at) VALUES (0,0,?,?)",
                       (c["compute_seconds"], time.time()))
            d.set_meta(db, "synthetic_storage_reservation", {"bytes": c["storage_bytes"]})
        frozen.update({"recipes_sha256": d.identity(recipes), "boards": boards, "board_equivalents": equivalents, "fidelity_sha256": gate})
        frozen["run_id"] = d.identity(frozen)
        d.write_json(ROOT / "recipes.json", recipes)
        d.write_json(ROOT / "coverage.json", cov)
        d.write_json(ROOT / "frozen.json", frozen)
        d.write_json(ROOT / "state.json", {"state": "ready", "completed_pages": 0,
                     "reserved_attempt_seconds": 0, "attempts": [], "pid": None})
        return {"state": "ready", "pages": len(recipes), "boards": boards,
                "identity": d.identity(frozen)}


def coverage(recipes):
    counts = {name: Counter() for name in ("page_kinds", "sets", "position_kinds", "classes", "class_background", "set_class_background", "set_effect_class_background", "effects", "perspective", "joint_conditions", "position_repetition")}
    for r in recipes:
        counts["page_kinds"][r["kind"]] += 1
        condition=r.get("condition",{})
        effect=condition.get("degradation",{}).get("variant","unknown")
        counts["effects"][effect]+=1
        counts["perspective"]["projective" if condition.get("perspective") else "affine-only"]+=1
        for b in r["boards"]:
            counts["sets"][b["set"]] += 1
            counts["position_kinds"][b["position_kind"]] += 1
            counts["position_repetition"][d.identity(b["labels"])] += 1
            summary_condition={**condition,"degradation":effect}
            counts["joint_conditions"][d.canonical({"set":b["set"],"position":b["position_kind"],"layout":r.get("layout"),"condition":summary_condition,"orientation":b["orientation"]})] += 1
            for i, label in enumerate(b["labels"]):
                parity=(i//8+i%8)%2
                counts["classes"][label] += 1
                counts["class_background"][f"{label}:{parity}"] += 1
                counts["set_class_background"][f"{b['set']}:{label}:{parity}"] += 1
                counts["set_effect_class_background"][f"{b['set']}:{effect}:{label}:{parity}"] += 1
    missing=[f"{s}:{p}:{bg}" for s in ("chessnut","fantasy","rhosgfx") for p in d.LABELS for bg in (0,1) if not counts["set_class_background"][f"{s}:{p}:{bg}"]]
    missing_effects=[f"{s}:{e}:{p}:{bg}" for s in ("chessnut","fantasy","rhosgfx") for e in ("blank","paper","faded","soft") for p in d.LABELS for bg in (0,1) if not counts["set_effect_class_background"][f"{s}:{e}:{p}:{bg}"]]
    repetitions=counts.pop("position_repetition")
    return {**{k:dict(v) for k,v in counts.items()},"missing_set_class_background":missing,
            "missing_set_effect_class_background":missing_effects,
            "unique_image_relative_positions":len(repetitions),"max_position_repetition":max(repetitions.values(),default=0),
            "repeated_position_board_count":sum(n for n in repetitions.values() if n>1),
            "training_boards":sum(counts["sets"].values()),
            "unsupported_board_equivalents":sum(len(r.get("unsupported_boards",[])) for r in recipes),
            "qualification":False,"recognition_improvement":"unmeasured"}


def validate_coverage(cov):
    d.require(not cov["missing_set_class_background"] and not cov["missing_set_effect_class_background"],"missing class/background/effect exposure")
    d.require(all(cov["perspective"].get(k,0)>0 for k in ("projective","affine-only")),"missing perspective exposure")


def process_identity(pid):
    try:
        # starttime protects status from PID reuse; /proc exists on supported Linux.
        fields = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def status():
    safe_root()
    if not (ROOT / "state.json").exists():
        return {"state": "not-initialized"}
    s = d.read_json(ROOT / "state.json")
    live = bool(s.get("pid") and process_identity(s["pid"]) == s.get("process_start"))
    frozen = d.read_json(ROOT / "frozen.json") if (ROOT / "frozen.json").exists() else {}
    return {**{k:v for k,v in s.items() if k != "attempts"}, "process_live": live,
            "total_pages": frozen.get("configuration", {}).get("pages"),
            "training_boards_planned": frozen.get("boards"),
            "attempt_count": len(s.get("attempts", [])),
            "failed_or_interrupted_attempts": sum(a["state"] == "failed-or-interrupted" for a in s.get("attempts", [])),
            "state": "interrupted" if s["state"] == "running" and not live else s["state"]}


def validate_state(s, c):
    attempts = s.get("attempts")
    d.require(isinstance(attempts, list) and len(attempts) <= c["compute_seconds"]//c["attempt_seconds"], "invalid attempt ledger")
    charge = len(attempts)*c["attempt_seconds"]
    d.require(type(s.get("reserved_attempt_seconds")) is int and s["reserved_attempt_seconds"] == charge, "attempt accounting mismatch")
    d.require(type(s.get("completed_pages")) is int and 0 <= s["completed_pages"] <= c["pages"], "invalid completed count")
    for a in attempts:
        d.require(a.get("state") in {"running", "done", "failed-or-interrupted"}, "invalid attempt state")
        d.require(type(a.get("start")) is int and 0 <= a["start"] < c["pages"] and a["start"] % c["batch_pages"] == 0, "invalid attempt range")
        if a["state"] == "running":
            a["state"] = "failed-or-interrupted"
    return charge


def verify_records(records, recipes, image_root):
    d.require(len(records) == len(recipes), "missing batch records")
    for record, recipe in zip(records, recipes):
        d.require(record["recipe"] == recipe, "recipe/output disagreement")
        name = f"page-{recipe['index']:06d}.png"
        d.require(record["image"] == name, "unsafe image name")
        path = image_root / name
        d.require(not path.is_symlink() and path.stat().st_size <= 16*1024**2, "image byte ceiling")
        d.require(d.digest(path) == record["image_sha256"], "output hash mismatch")
        im = d.load_image(path)
        d.require(im.size == (recipe["width"], recipe["height"]), "image geometry mismatch")


def run():
    with lock():
        # Prevent managed reset while allowing independent acquisition/review.
        with d.local_path("reset.lock").open("a") as reset:
            fcntl.flock(reset, fcntl.LOCK_SH | fcntl.LOCK_NB)
            frozen = d.read_json(ROOT / "frozen.json")
            c = frozen["configuration"]
            d.require(frozen["fidelity_sha256"] == fidelity_gate(), "changed fidelity gate")
            expected = frozen_identity(c)
            d.require(all(frozen[k] == v for k, v in expected.items()), "changed code/config/runtime; frozen job rejected")
            recipes_path = ROOT / "recipes.json"
            d.require(not recipes_path.is_symlink() and recipes_path.stat().st_size <= 32*1024**2, "recipes size")
            recipes = json.loads(recipes_path.read_text())
            d.require(d.identity(recipes) == frozen["recipes_sha256"], "changed recipe inputs")
            s = d.read_json(ROOT / "state.json")
            validate_state(s, c)
            with d.connect() as db:
                durable = [json.loads(r[0]) for r in db.execute("SELECT body FROM synthetic_attempts WHERE run=? ORDER BY ordinal", (frozen["run_id"],))]
            d.require(len(s["attempts"]) <= len(durable), "state exceeds durable attempt ledger")
            for i, a in enumerate(durable):
                if i < len(s["attempts"]):
                    d.require(all(s["attempts"][i][k] == a[k] for k in ("start", "pages", "at")), "durable attempt mismatch")
                else:
                    s["attempts"].append({**a, "state": "failed-or-interrupted"})
            s["reserved_attempt_seconds"] = len(durable)*c["attempt_seconds"]
            validate_state(s, c)
            s.update(state="running", pid=os.getpid(), process_start=process_identity(os.getpid()))
            image_root = ROOT / "images"
            image_root.mkdir(exist_ok=True)
            d.write_json(ROOT / "state.json", s)
            try:
                for start in range(0, len(recipes), c["batch_pages"]):
                    batch = recipes[start:start+c["batch_pages"]]
                    manifest = ROOT / f"batch-{start:06d}.json"
                    if manifest.exists():
                        verify_records(d.read_json(manifest), batch, image_root)
                        continue
                    if (ROOT / "stop").exists():
                        s["state"] = "stopped"
                        break
                    d.require(s["reserved_attempt_seconds"] + c["attempt_seconds"] <= c["compute_seconds"], "seed compute reservation exhausted")
                    disk = sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file())
                    d.require(disk + len(batch)*16*1024**2 <= c["storage_bytes"], "seed storage ceiling")
                    d.require_free_space(len(batch)*16*1024**2)
                    d.require(not any(p.is_symlink() for p in ROOT.rglob("*")), "symlink in job")
                    attempt = {"start": start, "pages": len(batch), "at": time.time(), "state": "running"}
                    with d.connect() as db:
                        db.execute("INSERT INTO synthetic_attempts VALUES (?,?,?)", (frozen["run_id"], len(s["attempts"]), d.canonical(attempt)))
                    s["reserved_attempt_seconds"] += c["attempt_seconds"]
                    s["attempts"].append(attempt)
                    d.write_json(ROOT / "state.json", s)
                    d.write_json(ROOT / "active-recipes.json", batch)
                    before = time.monotonic()
                    records = node("render", ROOT / "active-recipes.json", image_root,
                                   timeout=c["attempt_seconds"])
                    verify_records(records, batch, image_root)
                    d.write_json(manifest, records)
                    attempt.update(state="done", elapsed_seconds=time.monotonic()-before)
                    s["completed_pages"] = start + len(batch)
                    s["heartbeat"] = time.time()
                    d.write_json(ROOT / "state.json", s)
                else:
                    s["state"] = "complete"
                    s["completed_pages"] = len(recipes)
                    s["boards"] = frozen["boards"]
            except Exception as error:
                s.update(state="failed", error=str(error))
                if s["attempts"] and s["attempts"][-1]["state"] == "running":
                    s["attempts"][-1]["state"] = "failed-or-interrupted"
                raise
            finally:
                s["heartbeat"] = time.time()
                d.write_json(ROOT / "state.json", s)
    return status()


def start():
    with lock():
        d.require((ROOT / "frozen.json").exists(), "initialize first")
        (ROOT / "stop").unlink(missing_ok=True)
    proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "run"],
                            cwd=d.REPO, start_new_session=True, stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        s = status()
        if s.get("pid") == proc.pid and s.get("process_live"):
            return {"state": "running", "pid": proc.pid}
        if proc.poll() is not None:
            return {"state": "startup-failed", "exit_code": proc.returncode,
                    "next_action": "run in foreground to inspect frozen input or gate error"}
        time.sleep(.05)
    return {"state": "startup-unconfirmed", "pid": proc.pid, "next_action": "inspect status; do not restart"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["init", "start", "run", "status", "stop"])
    command = parser.parse_args().command
    if command == "stop":
        safe_root().mkdir(parents=True, exist_ok=True)
        d.atomic(ROOT / "stop", b"stop\n")
        result = {"state": "stop-requested", "boundary": "current bounded batch"}
    else:
        try:
            result = {"init": initialize, "start": start, "run": run, "status": status}[command]()
        except Exception as error:
            if command == "run" and (ROOT / "state.json").exists():
                # Do not overwrite the active worker if this was a duplicate launch.
                s = status()
                if not s.get("process_live") or s.get("pid") == os.getpid():
                    original = d.read_json(ROOT / "state.json")
                    original.update(state="failed", error=str(error), heartbeat=time.time())
                    d.write_json(ROOT / "state.json", original)
            raise
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
