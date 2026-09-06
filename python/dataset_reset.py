"""Crash-recoverable, local-only reset for a dataset generation.

The inbox is deliberately outside the reset payload: it is owner-controlled
input, not managed dataset state.  Old managed material is moved, never
deleted, to a counted archive below the dataset root.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import os
import hashlib
import re
import stat
import time
from pathlib import Path
import sqlite3
import uuid

try:  # Supports both `python python/dataset_pipeline.py` and package tests.
    import dataset_pipeline as p
except ModuleNotFoundError:
    from python import dataset_pipeline as p

CONFIRMATION = "START OVER"
MARKER = "reset.pending.json"
PAYLOAD_DIRS = ("originals", "rights", "pages", "staging", "review", "exports")
ARCHIVE_ID = re.compile(r"[0-9]{8}T[0-9]{6}Z-[0-9a-f]{12}")
DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _entries(directory, limit, deadline):
    result = []
    with os.scandir(directory) as entries:
        for entry in entries:
            p.require(len(result) < limit and time.monotonic() < deadline, "archive inspection limit; use local operator cleanup")
            result.append(entry)
    return sorted(result, key=lambda x: x.name)


def _archive_inventory(fd, deadline):
    """Bounded metadata-only inventory; never follow links or read source contents."""
    digest = hashlib.sha256()
    total = count = 0
    device = os.fstat(fd).st_dev

    def walk(directory, prefix, depth):
        nonlocal total, count
        p.require(depth <= 32, "archive nesting limit")
        info = os.fstat(directory)
        p.require(info.st_dev == device, "archive contains another filesystem")
        digest.update(p.canonical([prefix, info.st_dev, info.st_ino, info.st_ctime_ns]).encode())
        for entry in _entries(directory, 250000 - count, deadline):
            count += 1
            p.require(count <= 250000 and time.monotonic() < deadline, "archive inspection limit; use local operator cleanup")
            info = entry.stat(follow_symlinks=False)
            p.require(stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode), "archive contains a link or unsupported file")
            relative = prefix + "/" + entry.name
            digest.update(p.canonical([relative, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]).encode())
            if stat.S_ISDIR(info.st_mode):
                child = os.open(entry.name, DIRECTORY_FLAGS, dir_fd=directory)
                try:
                    walk(child, relative, depth + 1)
                finally:
                    os.close(child)
            else:
                total += info.st_size

    walk(fd, "", 0)
    return {"bytes": total, "version": digest.hexdigest()}


def list_archives():
    root = p.local_path("archives")
    if not root.exists():
        return {"archives": []}
    deadline = time.monotonic() + 10
    records = []
    for entry in reversed(_entries(root, 10000, deadline)):
        if not ARCHIVE_ID.fullmatch(entry.name):
            continue
        p.require(len(records) < 1000 and time.monotonic() < deadline, "archive listing limit")
        record = {"id": entry.name, "bytes": None, "version": None, "error": None}
        try:
            fd = os.open(p.local_path("archives/" + entry.name), DIRECTORY_FLAGS)
            try:
                record.update(_archive_inventory(fd, deadline))
            finally:
                os.close(fd)
        except (OSError, p.Invalid):
            record["error"] = "Cannot safely inspect this archive; local operator cleanup is required."
        records.append(record)
    return {"archives": records}


def delete_archive(archive_id, version, confirmation):
    try:
        return _delete_archive(archive_id, version, confirmation)
    except OSError:
        raise p.Invalid("Deletion stopped. Refresh archives to check remaining files before retrying.") from None


def _delete_archive(archive_id, version, confirmation):
    """Permanently delete one confirmed archive using directory-relative operations."""
    p.require(isinstance(archive_id, str) and ARCHIVE_ID.fullmatch(archive_id), "invalid archive identifier")
    p.require(isinstance(version, str) and re.fullmatch(r"[0-9a-f]{64}", version), "invalid archive version")
    p.require(confirmation == "DELETE", "type DELETE exactly to delete this archive")
    # Unlike writer(), this does not start pending reset recovery as a side effect.
    with p._writer_lock(), _reset_guard(blocking=False):
        p.require(not _marker_path().exists(), "finish dataset reset recovery before deleting archives")
        root = p.local_path("archives")
        target = p.local_path("archives/" + archive_id)
        if not target.exists():
            return {"state": "already-deleted", "bytes": 0}
        parent = os.open(root, DIRECTORY_FLAGS)
        try:
            fd = os.open(archive_id, DIRECTORY_FLAGS, dir_fd=parent)
            try:
                inventory = _archive_inventory(fd, time.monotonic() + 10)
                p.require(inventory["version"] == version, "archive changed; refresh the archive list before deleting")
                device = os.fstat(fd).st_dev
                deadline = time.monotonic() + 15

                def remove_contents(directory, depth=0):
                    p.require(depth <= 32 and os.fstat(directory).st_dev == device, "unsafe archive contents")
                    with os.scandir(directory) as entries:
                        for entry in entries:
                            p.require(time.monotonic() < deadline, "deletion time limit; refresh and delete the remaining archive files")
                            if entry.is_dir(follow_symlinks=False):
                                child = os.open(entry.name, DIRECTORY_FLAGS, dir_fd=directory)
                                try:
                                    remove_contents(child, depth + 1)
                                finally:
                                    os.close(child)
                                os.rmdir(entry.name, dir_fd=directory)
                            else:
                                os.unlink(entry.name, dir_fd=directory)

                remove_contents(fd)
                p.require(os.stat(archive_id, dir_fd=parent, follow_symlinks=False).st_ino == os.fstat(fd).st_ino,
                          "archive location changed; refresh before retrying")
                os.rmdir(archive_id, dir_fd=parent)
                os.fsync(parent)
            finally:
                os.close(fd)
        finally:
            os.close(parent)
    return {"state": "deleted", "bytes": inventory["bytes"]}


@contextlib.contextmanager
def _reset_guard(blocking):
    p.ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = p.local_path("reset.lock")
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            raise p.Invalid("dataset reset is already in progress; wait for recovery") from None
        yield


def _marker_path():
    return p.local_path(MARKER)


def _archive_path(marker):
    value = marker.get("archive")
    p.require(isinstance(value, str) and value.startswith("archives/"), "unsafe reset marker")
    path = p.local_path(value)
    p.require(path.parent == p.local_path("archives"), "unsafe reset archive")
    return path


def _read_marker():
    marker = p.read_json(_marker_path())
    p.require(isinstance(marker, dict) and marker.get("schema") == "chess-ocr-dataset-reset/1"
              and marker.get("phase") in {"prepared", "backed-up", "payload-moved", "cleared"}, "unsafe reset marker")
    p.require(isinstance(marker.get("carryover"), dict), "unsafe reset marker")
    _archive_path(marker)
    return marker


def _write_marker(marker):
    p.write_json(_marker_path(), marker)


def _backup_database(archive):
    active = p.local_path("state.sqlite3")
    p.require(active.exists() and not active.is_symlink(), "run init first")
    target = archive / "state.sqlite3"
    if target.exists():
        p.require(archive.is_dir() and not archive.is_symlink(), "unsafe reset archive")
        check = sqlite3.connect("file:" + str(target) + "?mode=ro", uri=True)
        try:
            p.require(check.execute("PRAGMA integrity_check").fetchone()[0] == "ok", "incomplete reset archive")
            p.require(check.execute("SELECT value FROM meta WHERE key='schema'").fetchone() is not None,
                      "incomplete reset archive")
        finally:
            check.close()
        return
    p.require(not archive.exists(), "incomplete reset archive; recovery requires operator inspection")
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.parent / (".reset-backup-" + uuid.uuid4().hex)
    temporary.mkdir()
    target = temporary / "state.sqlite3"
    source = sqlite3.connect(active)
    destination = sqlite3.connect(target)
    try:
        source.backup(destination)
        destination.commit()
    finally:
        destination.close()
        source.close()
    with target.open("rb") as stream:
        os.fsync(stream.fileno())
    directory_fd = os.open(temporary, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    os.replace(temporary, archive)
    parent_fd = os.open(archive.parent, os.O_RDONLY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _move_payload(archive):
    for name in PAYLOAD_DIRS:
        source = p.local_path(name)
        target = archive / name
        if source.exists():
            p.require(not source.is_symlink() and not target.exists(), "reset payload collision; recovery requires operator inspection")
            os.replace(source, target)


def _clear_active(marker):
    active = p.local_path("state.sqlite3")
    db = sqlite3.connect(active)
    try:
        db.execute("PRAGMA foreign_keys=ON")
        names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        with db:
            for name in ("web_drafts", "board_signatures", "duplicates", "reviews", "samples", "exclusions", "jobs", "reservations", "sources"):
                if name in names:
                    db.execute("DELETE FROM " + name)
            db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", ("carryover", p.canonical(marker["carryover"])))
            db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", ("worker", p.canonical({"state": "idle"})))
    finally:
        db.close()
    p.local_path("stop").unlink(missing_ok=True)


def _complete(marker, interrupt_at=None):
    archive = _archive_path(marker)
    if marker["phase"] == "prepared":
        if interrupt_at == "prepared":
            raise RuntimeError("injected reset interruption")
        _backup_database(archive)
        marker["phase"] = "backed-up"
        _write_marker(marker)
    if marker["phase"] == "backed-up":
        if interrupt_at == "backed-up":
            raise RuntimeError("injected reset interruption")
        _move_payload(archive)
        marker["phase"] = "payload-moved"
        _write_marker(marker)
        if interrupt_at == "payload-moved":
            raise RuntimeError("injected reset interruption")
    if marker["phase"] == "payload-moved":
        _clear_active(marker)
        marker["phase"] = "cleared"
        _write_marker(marker)
    if marker["phase"] == "cleared":
        _marker_path().unlink()


def recover_reset():
    """Complete a previously interrupted reset before normal pipeline access."""
    if not _marker_path().exists():
        return False
    # Do not run recovery concurrently with an ordinary writer.  Use the raw
    # writer lock because writer() itself invokes this function on a marker.
    with p._writer_lock(), _reset_guard(blocking=False):
        if _marker_path().exists():
            _complete(_read_marker())
            return True
    return False


def reset_preview():
    """Generic, non-private description suitable for a local UI confirmation."""
    return {"confirmation": CONFIRMATION, "preserves": ["inbox", "budget ceilings", "lifetime charges"],
            "archives": "managed dataset files are moved under archives/; they still count toward storage",
            "clears": ["sources", "jobs", "pages", "reviews", "drafts", "exports"]}


def reset_dataset(confirmation, *, _interrupt_at=None):
    """Archive the active generation and empty it after an exact owner confirmation."""
    p.require(confirmation == CONFIRMATION, "type START OVER exactly to reset the dataset")
    # writer is the worker's exclusive lock.  Taking it first is the active-worker
    # refusal and prevents reset from racing ordinary writes.
    with p.writer(), _reset_guard(blocking=False):
        p.require(not _marker_path().exists(), "dataset reset recovery is required before starting another reset")
        active = p.local_path("state.sqlite3")
        p.require(active.exists(), "run init first")
        # The only transient duplicate is SQLite's consistent backup.  Enforce
        # its bounded overhead before writing any marker or moving payload.
        with p.connect() as db:
            budget = p.meta(db, "budget")
            p.require(p.size_on_disk() + active.stat().st_size <= budget["storage_bytes"],
                      "reset archive backup exceeds storage ceiling")
            carryover = p.cumulative_accounting(db)
        archive_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:12]
        marker = {"schema": "chess-ocr-dataset-reset/1", "phase": "prepared",
                  "archive": "archives/" + archive_id, "carryover": carryover}
        _write_marker(marker)
        _complete(marker, _interrupt_at)
    return {"state": "reset", "archive": marker["archive"], "status": p.status()}
