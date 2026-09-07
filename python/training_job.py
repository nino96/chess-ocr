#!/usr/bin/env python3
"""Hash-frozen, bounded controller for the issue #3 synthetic bootstrap."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
from typing import Any


REPO = Path(__file__).resolve().parents[1]
DEFAULT_RECIPE = REPO / "recipes" / "synthetic-bootstrap-v1.json"
DEFAULT_RUN = REPO / "work" / "training" / "synthetic-bootstrap-v1"
HASHED_CODE = (
    "python/training.py",
    "python/training_job.py",
    "scripts/training.mjs",
    "recipes/synthetic-bootstrap-v1.json",
)
SPLITS = ("train", "development", "calibration")


class Invalid(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Invalid(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def identity(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def read_json(path: Path) -> Any:
    require(path.is_file() and not path.is_symlink(), f"missing or unsafe file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def safe_directory(path: Path, *, create: bool = False) -> Path:
    path = path.resolve()
    if create:
        path.mkdir(parents=True, exist_ok=True)
    require(path.is_dir() and not path.is_symlink(), f"unsafe directory: {path}")
    require(path != Path("/") and len(path.parts) >= 3, "directory is too broad")
    return path


def configuration(path: Path = DEFAULT_RECIPE) -> dict[str, Any]:
    value = read_json(path)
    require(value.get("schema") == "chess-ocr-training-recipe/1", "training recipe schema")
    require(value.get("run") == "synthetic-bootstrap-v1", "unexpected run name")
    require(type(value.get("seed")) is int and 0 <= value["seed"] < 2**32, "seed")
    dataset = value.get("dataset", {})
    require(dataset.get("label_order") == ".PNBRQKpnbrqk", "label order")
    ratios = dataset.get("split", {})
    require(ratios.get("algorithm") == "connected-page-parent-effect-greedy-v1", "split algorithm")
    require(abs(sum(float(ratios.get(k, 0)) for k in SPLITS) - 1) < 1e-9, "split ratios")
    resources = value.get("resources", {})
    require(resources.get("gpu_seconds") == 28800, "GPU ceiling")
    require(sum(resources.get(k, 0) for k in (
        "preflight_gpu_seconds", "classifier_gpu_seconds", "detector_gpu_seconds",
        "diagnosis_gpu_seconds")) == resources["gpu_seconds"], "GPU allocation")
    require(0 < resources.get("storage_bytes", 0) <= 16 * 1024**3, "storage ceiling")
    require(resources.get("minimum_free_space_ratio") == 0.3, "free-space floor")
    require(value.get("environment", {}).get("cudnn_enabled") is False, "cuDNN must remain disabled")
    require(value["environment"].get("precision") == "float32", "training precision")
    require(sum(s["updates"] for s in value["classifier"]["stages"]) == 10000, "classifier schedule")
    require(sum(s["updates"] for s in value["detector"]["stages"]) == 9000, "detector schedule")
    require(0 < value["detector"].get("gradient_clip_norm", 0) <= 100, "detector gradient clip")
    return value


class UnionFind:
    def __init__(self, count: int):
        self.parent = list(range(count))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def freeze_split(recipes: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Keep pages connected by repeated positions and four-page effect controls."""
    require(len(recipes) == config["dataset"]["expected_pages"], "page count")
    groups = UnionFind(len(recipes))
    seen: dict[tuple[str, str], int] = {}
    for ordinal, page in enumerate(recipes):
        require(page.get("index") == ordinal, "recipe ordering")
        degradation = page.get("condition", {}).get("degradation", {})
        keys = [("effect", str(degradation.get("seed")))]
        keys.extend(("parent", str(board.get("parent"))) for board in page.get("boards", []))
        for key in keys:
            if key in seen:
                groups.union(ordinal, seen[key])
            else:
                seen[key] = ordinal
    components: dict[int, list[int]] = {}
    for ordinal in range(len(recipes)):
        components.setdefault(groups.find(ordinal), []).append(ordinal)
    seed = str(config["seed"])
    ordered = sorted(
        components.values(),
        key=lambda items: (-len(items), hashlib.sha256(f"{seed}:{items[0]}".encode()).hexdigest()),
    )
    ratios = config["dataset"]["split"]
    targets = {name: len(recipes) * float(ratios[name]) for name in SPLITS}
    assigned: dict[str, list[int]] = {name: [] for name in SPLITS}
    counts = {name: 0 for name in SPLITS}
    for component in ordered:
        # Pick the most under-filled target; the stable tuple makes the split reproducible.
        split = min(SPLITS, key=lambda name: (counts[name] / targets[name], SPLITS.index(name)))
        assigned[split].extend(component)
        counts[split] += len(component)
    page_split = {}
    summaries = {}
    for name in SPLITS:
        pages = sorted(assigned[name])
        for ordinal in pages:
            page_split[str(ordinal)] = name
        subset = [recipes[i] for i in pages]
        boards = [board for page in subset for board in page.get("boards", [])]
        summaries[name] = {
            "pages": len(pages),
            "boards": len(boards),
            "positive_pages": sum(page.get("kind") == "boards" for page in subset),
            "negative_pages": sum(page.get("kind") == "negative" for page in subset),
            "partial_pages": sum(page.get("kind") == "partial" for page in subset),
            "sets": sorted({board["set"] for board in boards}),
            "effects": sorted({page["condition"]["degradation"]["variant"] for page in subset}),
            "layouts": sorted({page.get("layout") for page in subset}),
            "position_kinds": sorted({board["position_kind"] for board in boards}),
            "orientations": {orientation: sum(board.get("orientation") == orientation for board in boards)
                             for orientation in ("white-bottom", "black-bottom")},
        }
        require(summaries[name]["boards"] > 0, f"{name} has no classifier boards")
        require(summaries[name]["negative_pages"] > 0, f"{name} has no detector negatives")
        for key in ("sets", "effects", "layouts", "position_kinds"):
            require(summaries[name][key], f"{name} missing {key}")
        require(all(summaries[name]["orientations"].values()), f"{name} missing an orientation")
    require(len(page_split) == len(recipes), "split membership")
    return {
        "schema": "chess-ocr-synthetic-internal-split/1",
        "algorithm": ratios["algorithm"],
        "seed": config["seed"],
        "dataset_run_id": config["dataset"]["run_id"],
        "page_split": page_split,
        "summary": summaries,
    }


def verify_dataset(root: Path, config: dict[str, Any], *, verify_images: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = safe_directory(root)
    expected = config["dataset"]
    for name, key in (("frozen.json", "frozen_sha256"), ("recipes.json", "recipes_sha256"),
                      ("coverage.json", "coverage_sha256")):
        require(sha256(root / name) == expected[key], f"dataset {name} hash changed")
    frozen = read_json(root / "frozen.json")
    state = read_json(root / "state.json")
    coverage = read_json(root / "coverage.json")
    require(frozen.get("run_id") == expected["run_id"], "dataset run identity")
    require(state.get("state") == "complete" and state.get("completed_pages") == expected["expected_pages"], "dataset incomplete")
    require(coverage.get("training_boards") == expected["expected_training_boards"], "dataset board count")
    recipes = read_json(root / "recipes.json")
    require(sum(len(page.get("boards", [])) for page in recipes) == expected["expected_training_boards"], "recipe board count")
    if verify_images:
        seen = 0
        for start in range(0, len(recipes), 64):
            records = read_json(root / f"batch-{start:06d}.json")
            require(len(records) == min(64, len(recipes) - start), "synthetic batch length")
            for offset, record in enumerate(records):
                page = start + offset
                image = root / "images" / f"page-{page:06d}.png"
                require(record.get("recipe") == recipes[page], "synthetic batch recipe disagreement")
                require(record.get("image") == image.name and sha256(image) == record.get("image_sha256"), "synthetic image changed")
                seen += 1
        require(seen == expected["expected_pages"], "synthetic image count")
    return frozen, recipes


def verify_native(root: Path, config: dict[str, Any]) -> None:
    root = safe_directory(root)
    classifier = root / "cache/native/mobilenetv3_small_100.lamb_in1k.safetensors"
    detector = root / "cache/native/yolox_nano.pth"
    require(sha256(classifier) == config["starting_models"]["classifier"]["sha256"], "classifier checkpoint hash")
    require(sha256(detector) == config["starting_models"]["detector"]["sha256"], "detector checkpoint hash")
    source = root / "cache/native/yolox"
    commit = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], check=True,
                            text=True, capture_output=True).stdout.strip()
    require(commit == config["starting_models"]["detector"]["revision"], "YOLOX source revision")
    status = subprocess.run(["git", "-C", str(source), "status", "--porcelain"], check=True,
                            text=True, capture_output=True).stdout
    require(not status, "YOLOX source checkout is dirty")


def process_start(pid: int) -> str | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return None


@contextlib.contextmanager
def lock(root: Path):
    with (root / "job.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Invalid("training worker already active") from None
        yield


def code_identity() -> dict[str, str]:
    return {path: sha256(REPO / path) for path in HASHED_CODE}


def git_identity() -> dict[str, str]:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
                          text=True, capture_output=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, check=True,
                            text=True, capture_output=True).stdout
    require(not status, "commit reviewed changes before initializing training")
    return {"head": head}


def docker_identity(image: str) -> str:
    result = subprocess.run(["docker", "image", "inspect", image, "--format", "{{.Id}}"],
                            text=True, capture_output=True)
    require(result.returncode == 0 and result.stdout.strip(), "pinned training image is unavailable")
    return result.stdout.strip()


def directory_identity(root: Path) -> dict[str, str]:
    root = safe_directory(root)
    result = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), f"symlink in dependency overlay: {path}")
        if path.is_file():
            require(path.stat().st_size <= 128 * 1024**2, f"unexpectedly large overlay file: {path}")
            result[str(path.relative_to(root))] = sha256(path)
    require(result, "empty dependency overlay")
    return result


def initialize(args: argparse.Namespace) -> dict[str, Any]:
    config = configuration(args.recipe)
    run = safe_directory(args.run_root, create=True)
    require(not (run / "frozen.json").exists(), "training run already initialized")
    dataset = safe_directory(args.dataset_root)
    native = safe_directory(args.native_root)
    _, recipes = verify_dataset(dataset, config, verify_images=True)
    verify_native(native, config)
    overlay = safe_directory(args.overlay_root)
    classifier_checkpoint = None
    if args.detector_only:
        require(args.classifier_checkpoint is not None, "detector-only mode requires --classifier-checkpoint")
        classifier_checkpoint = args.classifier_checkpoint.resolve()
        require(classifier_checkpoint.is_file() and not classifier_checkpoint.is_symlink(),
                "classifier checkpoint is missing or unsafe")
    prior_attempts = []
    prior_charge = 0.0
    if args.prior_run:
        prior = read_json(safe_directory(args.prior_run) / "state.json")
        require(prior.get("run_id") and prior.get("state") in {"failed", "budget-blocked", "interrupted", "stopped"}, "invalid prior training attempt")
        prior_attempts = [{**attempt, "prior_run_id": attempt.get("prior_run_id", prior["run_id"])}
                          for attempt in prior.get("attempts", [])]
        prior_charge = float(prior.get("gpu_seconds_charged", 0))
        require(prior_charge < config["resources"]["gpu_seconds"], "prior attempt exhausted GPU budget")
    split = freeze_split(recipes, config)
    usage = shutil.disk_usage(run)
    resources = config["resources"]
    require(usage.free - resources["storage_bytes"] >= usage.total * resources["minimum_free_space_ratio"], "training storage would cross free-space floor")
    frozen = {
        "schema": "chess-ocr-training-frozen/1",
        "recipe": config,
        "recipe_sha256": sha256(args.recipe),
        "dataset_root": str(dataset),
        "native_root": str(native),
        "repository_root": str(REPO),
        "code": code_identity(),
        "git": git_identity(),
        "container_id": docker_identity(config["environment"]["image"]),
        "dependency_overlay": directory_identity(overlay),
        "dependency_overlay_root": str(overlay),
        "container_user": {"uid": os.getuid(), "gid": os.getgid()},
        "prior_gpu_seconds_charged": prior_charge,
        "mode": "detector-only" if args.detector_only else "full",
        "classifier_checkpoint": ({"path": str(classifier_checkpoint), "sha256": sha256(classifier_checkpoint)}
                                   if classifier_checkpoint else None),
        "split_sha256": identity(split),
        "created_at": time.time(),
    }
    frozen["run_id"] = identity(frozen)
    write_json(run / "split.json", split)
    write_json(run / "frozen.json", frozen)
    write_json(run / "state.json", {
        "schema": "chess-ocr-training-state/1", "state": "ready", "stage": "preflight",
        "pid": None, "process_start": None, "attempts": prior_attempts, "gpu_seconds_charged": prior_charge,
        "run_id": frozen["run_id"],
    })
    return {"state": "ready", "run_id": frozen["run_id"], "split": split["summary"]}


def verify_frozen(run: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    frozen = read_json(run / "frozen.json")
    config = configuration(DEFAULT_RECIPE)
    require(frozen.get("recipe") == config and frozen.get("recipe_sha256") == sha256(DEFAULT_RECIPE), "training recipe changed")
    require(frozen.get("code") == code_identity(), "training code changed")
    require(frozen.get("git") == git_identity(), "training commit changed")
    require(frozen.get("container_id") == docker_identity(config["environment"]["image"]), "training image changed")
    require(frozen.get("dependency_overlay") == directory_identity(Path(frozen["dependency_overlay_root"])), "training dependency overlay changed")
    require(frozen.get("container_user") == {"uid": os.getuid(), "gid": os.getgid()}, "container user changed")
    if frozen.get("mode") == "detector-only":
        checkpoint = frozen.get("classifier_checkpoint") or {}
        path = Path(checkpoint.get("path", ""))
        require(path.is_file() and not path.is_symlink() and sha256(path) == checkpoint.get("sha256"),
                "classifier checkpoint changed or missing")
    verify_dataset(Path(frozen["dataset_root"]), config, verify_images=True)
    verify_native(Path(frozen["native_root"]), config)
    split = read_json(run / "split.json")
    require(identity(split) == frozen["split_sha256"], "split changed")
    return frozen, config


def container_name(frozen: dict[str, Any], segment: str) -> str:
    return f"chess-ocr-{frozen['run_id'][:12]}-{segment}"


def container_command(run: Path, frozen: dict[str, Any], segment: str) -> list[str]:
    config = frozen["recipe"]
    overlay = Path(frozen["dependency_overlay_root"])
    require(overlay.is_dir() and not overlay.is_symlink(), "verified GPU dependency overlay missing")
    command = ["docker", "run", "--rm", "--name", container_name(frozen, segment)]
    if segment not in {"validate", "classifier-export"}:
        command.extend(("--gpus", "all"))
    command.extend([
        "--network", "none", "--read-only",
        "--user", f"{frozen['container_user']['uid']}:{frozen['container_user']['gid']}", "--entrypoint", "python",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--tmpfs", "/tmp:rw,noexec,nosuid,size=2g",
        "-v", f"{frozen['repository_root']}:/repo:ro",
        "-v", f"{frozen['dataset_root']}:/dataset:ro",
        "-v", f"{frozen['native_root']}:/native:ro",
        "-v", f"{run}:/output:rw",
        "-v", f"{overlay}:/overlay:ro",
        "-e", "PYTHONPATH=/overlay:/native/cache/native/yolox:/repo/python",
        config["environment"]["image"], "/repo/python/training.py", segment,
        "--recipe", "/repo/recipes/synthetic-bootstrap-v1.json",
        "--dataset", "/dataset", "--native", "/native", "--run", "/output",
        "--split", "/output/split.json",
    ])
    if frozen.get("mode") == "detector-only":
        checkpoint = Path(frozen["classifier_checkpoint"]["path"])
        command.extend(("--classifier-checkpoint", "/classifier-checkpoint.pt"))
        # Insert the read-only checkpoint mount before the environment and image arguments.
        mount_at = command.index("-e")
        command[mount_at:mount_at] = ["-v", f"{checkpoint}:/classifier-checkpoint.pt:ro"]
    return command


def validate_before_gpu(run: Path, frozen: dict[str, Any]) -> dict[str, Any]:
    marker = run / "validation.complete.json"
    log_path = run / "validation.log"
    if marker.exists():
        result = read_json(marker)
        require(result.get("run_id") == frozen["run_id"] and result.get("log_sha256") == sha256(log_path),
                "invalid CPU validation marker")
        return result
    started = time.monotonic()
    try:
        with log_path.open("ab") as log:
            result = subprocess.run(container_command(run, frozen, "validate"), stdout=log,
                                    stderr=subprocess.STDOUT, timeout=300)
    except subprocess.TimeoutExpired:
        stop_container(frozen, "validate")
        raise Invalid("CPU prelaunch validation timed out; container stopped (no GPU charged)") from None
    require(result.returncode == 0, "CPU prelaunch validation failed; inspect validation.log (no GPU charged)")
    completed = {"run_id": frozen["run_id"], "segment": "validate", "state": "passed",
                 "elapsed_seconds": time.monotonic() - started, "finished_at": time.time(),
                 "log_sha256": sha256(log_path)}
    write_json(marker, completed)
    return completed


def directory_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file() and not path.is_symlink())


def remaining_seconds(state: dict[str, Any], segment: str, resources: dict[str, Any]) -> int:
    key = f"{segment}_gpu_seconds"
    used_segment = sum(float(attempt.get("elapsed_seconds", 0)) for attempt in state["attempts"]
                       if attempt.get("segment") == segment)
    used_total = float(state.get("gpu_seconds_charged", 0))
    return max(0, int(min(resources[key] - used_segment, resources["gpu_seconds"] - used_total)))


def stop_container(frozen: dict[str, Any], segment: str) -> None:
    try:
        subprocess.run(["docker", "stop", "--time", "30", container_name(frozen, segment)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=45)
    except subprocess.SubprocessError:
        subprocess.run(["docker", "kill", container_name(frozen, segment)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)


def materialize_classifier_checkpoint(run: Path, frozen: dict[str, Any]) -> None:
    if frozen.get("mode") != "detector-only":
        return
    marker = run / "classifier.complete.json"
    if marker.exists():
        require(read_json(marker).get("run_id") == frozen["run_id"], "invalid classifier export marker")
        return
    log_path = run / "classifier-export.log"
    try:
        with log_path.open("ab") as log:
            result = subprocess.run(container_command(run, frozen, "classifier-export"), stdout=log,
                                    stderr=subprocess.STDOUT, timeout=3600)
    except subprocess.TimeoutExpired:
        stop_container(frozen, "classifier-export")
        raise Invalid("classifier checkpoint export timed out") from None
    require(result.returncode == 0, "classifier checkpoint export failed; inspect its retained log")
    write_json(marker, {"run_id": frozen["run_id"], "segment": "classifier-export",
                        "finished_at": time.time(), "log_sha256": sha256(log_path)})


def run_worker(run: Path) -> None:
    run = safe_directory(run)
    with lock(run):
        frozen, config = verify_frozen(run)
        state = read_json(run / "state.json")
        resources = config["resources"]
        segments = (
            ("preflight", resources["preflight_gpu_seconds"]),
            ("classifier", resources["classifier_gpu_seconds"]),
            ("detector", resources["detector_gpu_seconds"]),
        )
        state.update(state="running", pid=os.getpid(), process_start=process_start(os.getpid()))
        write_json(run / "state.json", state)
        try:
            materialize_classifier_checkpoint(run, frozen)
            for segment, _ in segments:
                marker = run / f"{segment}.complete.json"
                if marker.exists():
                    require(read_json(marker).get("run_id") == frozen["run_id"], f"invalid {segment} marker")
                    continue
                if (run / "stop").exists():
                    state.update(state="stopped", stage=segment)
                    break
                state["stage"] = segment
                ceiling = remaining_seconds(state, segment, resources)
                require(ceiling > 0, f"{segment} and/or total GPU reservation exhausted")
                require(directory_size(run) <= resources["storage_bytes"], "training output storage ceiling exceeded")
                usage = shutil.disk_usage(run)
                require(usage.free >= usage.total * resources["minimum_free_space_ratio"], "filesystem free-space floor crossed")
                attempt = {"segment": segment, "started_at": time.time(), "state": "running"}
                state["attempts"].append(attempt)
                write_json(run / "state.json", state)
                started = time.monotonic()
                interrupted = False
                child = None
                previous_handlers = {}
                def request_interrupt(signum, _frame):
                    nonlocal interrupted
                    interrupted = True
                    if child is not None:
                        stop_container(frozen, segment)
                try:
                    for signum in (signal.SIGTERM, signal.SIGINT):
                        previous_handlers[signum] = signal.signal(signum, request_interrupt)
                    with (run / f"{segment}.log").open("ab") as log:
                        child = subprocess.Popen(container_command(run, frozen, segment), stdout=log,
                                                 stderr=subprocess.STDOUT, start_new_session=True)
                        try:
                            returncode = child.wait(timeout=ceiling)
                        except subprocess.TimeoutExpired:
                            stop_container(frozen, segment)
                            returncode = child.wait(timeout=45)
                            raise
                    elapsed = min(ceiling, time.monotonic() - started)
                except subprocess.TimeoutExpired:
                    elapsed = ceiling
                    state["gpu_seconds_charged"] += elapsed
                    attempt.update(state="budget-blocked", elapsed_seconds=elapsed,
                                   finished_at=time.time(), error="segment time ceiling exceeded")
                    state.update(state="budget-blocked", error=f"{segment} exceeded its frozen time ceiling")
                    write_json(run / "state.json", state)
                    return
                finally:
                    for signum, handler in previous_handlers.items():
                        signal.signal(signum, handler)
                if interrupted:
                    state["gpu_seconds_charged"] += elapsed
                    attempt.update(state="interrupted", elapsed_seconds=elapsed, finished_at=time.time())
                    state.update(state="interrupted", error="supervisor signal forwarded to training container")
                    write_json(run / "state.json", state)
                    return
                state["gpu_seconds_charged"] += elapsed
                if returncode == 75:
                    attempt.update(state="stopped", elapsed_seconds=elapsed, returncode=returncode, finished_at=time.time())
                    state.update(state="stopped")
                    write_json(run / "state.json", state)
                    return
                attempt.update(state="complete" if returncode == 0 else "failed", elapsed_seconds=elapsed,
                               returncode=returncode, finished_at=time.time())
                require(state["gpu_seconds_charged"] <= resources["gpu_seconds"], "GPU budget exhausted")
                write_json(run / "state.json", state)
                require(returncode == 0, f"{segment} failed; inspect its retained log")
                write_json(marker, {"run_id": frozen["run_id"], "segment": segment,
                                    "finished_at": time.time(), "log_sha256": sha256(run / f"{segment}.log")})
            else:
                state.update(state="complete", stage="complete")
        except Exception as error:
            state.update(state="failed", error=str(error))
        finally:
            state.update(pid=None, process_start=None, finished_at=time.time())
            write_json(run / "state.json", state)


def start(run: Path) -> dict[str, Any]:
    run = safe_directory(run)
    frozen, _ = verify_frozen(run)
    state = read_json(run / "state.json")
    live = state.get("pid") and process_start(state["pid"]) == state.get("process_start")
    require(not live, "training worker already active")
    require(state.get("state") not in {"complete", "budget-blocked"}, f"training is {state.get('state')}")
    try:
        validation = validate_before_gpu(run, frozen)
    except (Invalid, OSError, subprocess.SubprocessError) as error:
        state.update(state="validation-failed", error=str(error), finished_at=time.time())
        write_json(run / "state.json", state)
        raise
    state["cpu_validation"] = validation
    state.pop("error", None)
    write_json(run / "state.json", state)
    (run / "stop").unlink(missing_ok=True)
    log = (run / "supervisor.log").open("ab")
    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "run", "--run-root", str(run)],
                               cwd=REPO, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        time.sleep(0.1)
        current = read_json(run / "state.json")
        if current.get("pid") == process.pid and current.get("state") == "running":
            return {"state": "running", "pid": process.pid, "stage": current["stage"]}
        if process.poll() is not None:
            raise Invalid("training supervisor failed to start; inspect supervisor.log")
    process.send_signal(signal.SIGTERM)
    raise Invalid("training supervisor did not confirm startup")


def stop(run: Path) -> dict[str, Any]:
    run = safe_directory(run)
    state = read_json(run / "state.json")
    write_json(run / "stop", {"requested_at": time.time(), "run_id": state.get("run_id")})
    return {"state": state.get("state"), "stop": "requested at next checkpoint boundary"}


def status(run: Path) -> dict[str, Any]:
    if not run.exists():
        return {"state": "not-initialized"}
    run = safe_directory(run)
    state = read_json(run / "state.json")
    live = bool(state.get("pid") and process_start(state["pid"]) == state.get("process_start"))
    frozen = read_json(run / "frozen.json") if (run / "frozen.json").exists() else {}
    resources = frozen.get("recipe", {}).get("resources", {})
    attempts = state.get("attempts", [])
    consumed_by_segment = {segment: sum(float(attempt.get("elapsed_seconds", 0)) for attempt in attempts
                                        if attempt.get("segment") == segment)
                           for segment in ("preflight", "classifier", "detector")}
    capacity = float(resources.get("gpu_seconds", 0))
    reported = {key: value for key, value in state.items() if key != "attempts"}
    reported.update({"process_live": live,
                     "current_attempt": attempts[-1] if attempts else None,
                     "budget": {"unit": "GPU-seconds (one second of allocated active GPU-container wall time)",
                                "capacity_seconds": capacity,
                                "consumed_seconds": float(state.get("gpu_seconds_charged", 0)),
                                "remaining_seconds": max(0.0, capacity - float(state.get("gpu_seconds_charged", 0))),
                                "by_segment": {segment: {"capacity_seconds": float(resources.get(f"{segment}_gpu_seconds", 0)),
                                                           "consumed_seconds": consumed_by_segment[segment],
                                                           "remaining_seconds": max(0.0, float(resources.get(f"{segment}_gpu_seconds", 0)) - consumed_by_segment[segment])}
                                                for segment in consumed_by_segment}}})
    if state.get("state") == "running" and not live:
        reported["state"] = "interrupted"
    for model in ("classifier", "detector"):
        progress = run / model / "progress.json"
        if progress.exists():
            reported[model] = read_json(progress)
    return reported


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    sub = value.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    init.add_argument("--dataset-root", type=Path, required=True)
    init.add_argument("--native-root", type=Path, required=True)
    init.add_argument("--overlay-root", type=Path, required=True)
    init.add_argument("--prior-run", type=Path)
    init.add_argument("--detector-only", action="store_true")
    init.add_argument("--classifier-checkpoint", type=Path)
    init.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    for name in ("start", "stop", "run"):
        command = sub.add_parser(name)
        command.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    status_command = sub.add_parser("status")
    status_command.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    status_command.add_argument("--history", action="store_true")
    return value


def main() -> None:
    args = parser().parse_args()
    try:
        if args.command == "init":
            result = initialize(args)
        elif args.command == "start":
            result = start(args.run_root)
        elif args.command == "stop":
            result = stop(args.run_root)
        elif args.command == "status":
            result = status(args.run_root)
            if args.history:
                result["attempts"] = read_json(args.run_root / "state.json").get("attempts", [])
        else:
            run_worker(args.run_root)
            result = status(args.run_root)
        print(json.dumps(result, indent=2, sort_keys=True))
    except (Invalid, OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        print(json.dumps({"state": "error", "error": str(error)}, indent=2), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
