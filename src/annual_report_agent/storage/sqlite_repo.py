"""SQLite operations with parameterized queries and atomic index publication."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


def now() -> str:
    return datetime.now(UTC).isoformat()


def dump(value) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, allow_nan=False, default=str)


class Repository:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS objects (
                    kind TEXT NOT NULL, id TEXT NOT NULL, report_id TEXT,
                    payload TEXT NOT NULL, updated_at TEXT NOT NULL,
                    PRIMARY KEY(kind,id));
                CREATE INDEX IF NOT EXISTS objects_report ON objects(kind,report_id);
                CREATE TABLE IF NOT EXISTS builds (
                    version TEXT PRIMARY KEY, report_id TEXT NOT NULL,
                    profile TEXT NOT NULL, mode TEXT NOT NULL, expected TEXT NOT NULL,
                    encoded TEXT NOT NULL, status TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS active_indexes (
                    report_id TEXT PRIMARY KEY, version TEXT NOT NULL REFERENCES builds(version));
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    profile TEXT, purpose TEXT, text_hash TEXT, vector TEXT NOT NULL,
                    PRIMARY KEY(profile,purpose,text_hash));
                PRAGMA user_version=1;
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def put(self, kind: str, identifier: str, value, report_id: str | None = None):
        with self.connect() as db:
            db.execute("INSERT INTO objects VALUES(?,?,?,?,?) ON CONFLICT(kind,id) DO UPDATE SET "
                       "report_id=excluded.report_id,payload=excluded.payload,updated_at=excluded.updated_at",
                       (kind, identifier, report_id, dump(value), now()))

    def put_many(self, kind: str, records: list[dict], report_id: str | None = None):
        with self.connect() as db:
            db.executemany("INSERT INTO objects VALUES(?,?,?,?,?) ON CONFLICT(kind,id) DO UPDATE SET "
                           "report_id=excluded.report_id,payload=excluded.payload,updated_at=excluded.updated_at",
                           [(kind, r["id"], report_id, dump(r), now()) for r in records])

    def get(self, kind: str, identifier: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT payload FROM objects WHERE kind=? AND id=?",
                             (kind, identifier)).fetchone()
        return json.loads(row[0]) if row else None

    def list(self, kind: str, report_id: str | None = None) -> list[dict]:
        with self.connect() as db:
            if report_id is None:
                rows = db.execute("SELECT payload FROM objects WHERE kind=? ORDER BY id", (kind,)).fetchall()
            else:
                rows = db.execute("SELECT payload FROM objects WHERE kind=? AND report_id=? ORDER BY id",
                                  (kind, report_id)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def create_task(self, resource_type: str, resource_id: str, **metadata) -> dict:
        value = dict(task_id=uuid4().hex, resource_type=resource_type, resource_id=resource_id,
                     status="queued", progress=0.0, error_code=None, message=None,
                     created_at=now(), updated_at=now(), metadata=metadata)
        self.put("task", value["task_id"], value)
        return value

    def get_task(self, task_id: str) -> dict | None:
        return self.get("task", task_id)

    def update_task(self, task_id: str, status: str, progress: float, **changes) -> dict:
        if status not in {"queued", "running", "succeeded", "failed", "interrupted"}:
            raise ValueError("invalid task status")
        if not 0 <= progress <= 1:
            raise ValueError("progress must be in [0,1]")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT payload FROM objects WHERE kind='task' AND id=?", (task_id,)).fetchone()
            if not row:
                raise KeyError(task_id)
            task = json.loads(row[0])
            allowed = {"queued": {"queued", "running", "failed", "interrupted"},
                       "running": {"running", "succeeded", "failed", "interrupted"},
                       "succeeded": {"succeeded"}, "failed": {"failed"},
                       "interrupted": {"interrupted"}}
            if status not in allowed[task["status"]]:
                raise ValueError("invalid task transition; create a new task to retry")
            task.update(changes, status=status, progress=progress, updated_at=now())
            db.execute("UPDATE objects SET payload=?,updated_at=? WHERE kind='task' AND id=?",
                       (dump(task), now(), task_id))
        return task

    def recover_interrupted(self) -> int:
        count = 0
        for task in self.list("task"):
            if task["status"] in {"queued", "running"}:
                self.update_task(task["task_id"], "interrupted", task["progress"],
                                 error_code="PROCESS_RESTARTED", message="服务重启，请重试任务。")
                count += 1
        return count

    def save_build(self, version: str, report_id: str, chunk_ids: list[str],
                   profile: str, mode: str, **payload):
        if not chunk_ids or len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("build needs a nonempty unique chunk list")
        with self.connect() as db:
            existing = db.execute("SELECT * FROM builds WHERE version=?", (version,)).fetchone()
            if existing:
                if (existing["report_id"] != report_id or existing["profile"] != profile
                        or existing["mode"] != mode or json.loads(existing["expected"]) != chunk_ids):
                    raise ValueError("index version collision")
                return
            metadata = dict(index_version=version, report_id=report_id, chunk_ids=chunk_ids,
                            embedding_profile_hash=profile, mode=mode, **payload)
            db.execute("INSERT INTO builds VALUES(?,?,?,?,?,?,?,?)",
                       (version, report_id, profile, mode, dump(chunk_ids), "[]", "building", dump(metadata)))

    def build(self, version: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM builds WHERE version=?", (version,)).fetchone()
        if not row:
            return None
        return json.loads(row["payload"]) | {"build_status": row["status"],
                                             "encoded_ids": json.loads(row["encoded"])}

    def mark_chunk_encoded(self, version: str, chunk_id: str):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT expected,encoded FROM builds WHERE version=?", (version,)).fetchone()
            if not row or chunk_id not in json.loads(row["expected"]):
                raise ValueError("unexpected chunk")
            encoded = set(json.loads(row["encoded"])) | {chunk_id}
            db.execute("UPDATE builds SET encoded=? WHERE version=?", (dump(sorted(encoded)), version))

    def publish_index(self, version: str):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM builds WHERE version=?", (version,)).fetchone()
            if not row or set(json.loads(row["expected"])) != set(json.loads(row["encoded"])):
                raise ValueError("incomplete index cannot be published")
            db.execute("UPDATE builds SET status='ready' WHERE version=?", (version,))
            db.execute("INSERT INTO active_indexes VALUES(?,?) ON CONFLICT(report_id) "
                       "DO UPDATE SET version=excluded.version", (row["report_id"], version))

    def active_index(self, report_id: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT version FROM active_indexes WHERE report_id=?", (report_id,)).fetchone()
        return self.build(row[0]) if row else None

    def cache_embedding(self, profile: str, purpose: str, text: str, vector: list[float]):
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO embedding_cache VALUES(?,?,?,?)",
                       (profile, purpose, text_hash, dump(vector)))

    def cached_embedding(self, profile: str, purpose: str, text: str) -> list[float] | None:
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        with self.connect() as db:
            row = db.execute("SELECT vector FROM embedding_cache WHERE profile=? AND purpose=? AND text_hash=?",
                             (profile, purpose, text_hash)).fetchone()
        return json.loads(row[0]) if row else None
