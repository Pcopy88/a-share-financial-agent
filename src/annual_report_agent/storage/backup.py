"""Consistent stopped-service backup; restore verifies all hashes into a new directory."""

import hashlib
import json
import shutil
import sqlite3
import tempfile
import zipfile
from contextlib import closing
from pathlib import Path, PurePosixPath

from annual_report_agent.storage.process_lock import RuntimeLock


def backup(runtime_dir, destination):
    source, destination = Path(runtime_dir).resolve(), Path(destination).resolve()
    if not source.is_dir():
        raise ValueError("runtime directory missing")
    if destination.is_relative_to(source):
        raise ValueError("archive must be outside runtime directory")
    if destination.exists():
        raise ValueError("archive already exists; choose a new filename")
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = {}
    temporary = destination.with_suffix(destination.suffix + ".part")
    with RuntimeLock(source), zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_symlink():
                raise ValueError("backup does not follow symlinks")
            if not path.is_file() or path.name in {".env", ".runtime.lock"}:
                continue
            relative = path.relative_to(source).as_posix()
            manifest[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            archive.write(path, "payload/" + relative)
        archive.writestr("manifest.json", json.dumps({"schema": 1, "files": manifest,
                                                    "source_runtime_dir": str(source)}))
    temporary.replace(destination)
    return destination


def restore(archive_path, destination):
    destination = Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("restore destination must be empty")
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or "manifest.json" not in names:
            raise ValueError("archive manifest missing or duplicate entries")
        if sum(info.file_size for info in archive.infolist()) > 30 * 1024**3:
            raise ValueError("backup exceeds 30 GiB restore limit")
        manifest = json.loads(archive.read("manifest.json"))
        files = manifest.get("files", {})
        if manifest.get("schema") != 1 or set(names) != {"manifest.json", *("payload/" + key for key in files)}:
            raise ValueError("archive content differs from manifest")
        for name in files:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
                raise ValueError("unsafe archive path")
            if not (destination / name).resolve().is_relative_to(destination):
                raise ValueError("archive escapes destination")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="restore-", dir=destination.parent) as stage:
            stage = Path(stage)
            for name, expected_hash in files.items():
                path = stage / name
                path.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                with archive.open("payload/" + name) as source, path.open("wb") as target:
                    while block := source.read(1024 * 1024):
                        digest.update(block)
                        target.write(block)
                if digest.hexdigest() != expected_hash:
                    raise ValueError("backup hash mismatch")
            database = stage / "application.sqlite"
            if database.is_file() and manifest.get("source_runtime_dir"):
                with database.open("rb") as stream:
                    is_sqlite = stream.read(16) == b"SQLite format 3\x00"
                if is_sqlite:
                    original = Path(manifest["source_runtime_dir"])
                    with closing(sqlite3.connect(database)) as db, db:
                        for identifier, payload in db.execute(
                                "SELECT id,payload FROM objects WHERE kind='report'").fetchall():
                            report = json.loads(payload)
                            path = Path(report.get("path", ""))
                            if path.is_relative_to(original):
                                report["path"] = str(destination / path.relative_to(original))
                                db.execute("UPDATE objects SET payload=? WHERE kind='report' AND id=?",
                                           (json.dumps(report, ensure_ascii=False), identifier))
            destination.mkdir(parents=True, exist_ok=True)
            for path in stage.iterdir():
                shutil.move(str(path), destination / path.name)
    return destination
