"""Durable receipt outbox and fail-closed motion deduplication using SQLite."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import threading
import time


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


class JournalConflict(RuntimeError):
    pass


class DurableJournal:
    def __init__(self, path):
        self.path = Path(path)
        if self.path.is_symlink():
            raise ValueError("Journal must not be a symbolic link")
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        os.chmod(self.path, 0o600)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (name TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS counters (name TEXT PRIMARY KEY, value INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS outbox (
                id TEXT PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL,
                digest TEXT NOT NULL, created_at REAL NOT NULL, acknowledged INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS execution (
                run_id TEXT NOT NULL, epoch INTEGER NOT NULL, sequence INTEGER NOT NULL,
                state TEXT NOT NULL, started_at REAL NOT NULL, receipt_id TEXT,
                PRIMARY KEY (run_id, epoch, sequence));
        """)

    @contextmanager
    def _transaction(self):
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise

    def next_sequence(self, namespace):
        with self._transaction():
            row = self._db.execute("SELECT value FROM counters WHERE name=?", (namespace,)).fetchone()
            value = row[0] + 1 if row else 1
            self._db.execute("INSERT OR REPLACE INTO counters VALUES (?, ?)", (namespace, value))
        return value

    def get(self, name, default=None):
        with self._lock:
            row = self._db.execute("SELECT value FROM metadata WHERE name=?", (name,)).fetchone()
        return json.loads(row[0]) if row else default

    def set(self, name, value):
        serialized = encoded(value)
        with self._transaction():
            self._db.execute("INSERT OR REPLACE INTO metadata VALUES (?, ?)", (name, serialized))

    def _enqueue(self, identifier, kind, payload):
        serialized = encoded(payload)
        digest = hashlib.sha256(serialized.encode()).hexdigest()
        existing = self._db.execute("SELECT kind, digest FROM outbox WHERE id=?", (identifier,)).fetchone()
        if existing:
            if existing != (kind, digest):
                raise JournalConflict("An existing receipt identifier cannot refer to different evidence")
            return False
        self._db.execute("INSERT INTO outbox (id, kind, payload, digest, created_at) VALUES (?, ?, ?, ?, ?)",
                         (identifier, kind, serialized, digest, time.time()))
        return True

    def enqueue(self, identifier, kind, payload):
        with self._transaction():
            return self._enqueue(identifier, kind, payload)

    def pending(self, limit=100):
        with self._lock:
            rows = self._db.execute("SELECT id, kind, payload FROM outbox WHERE acknowledged=0 ORDER BY rowid LIMIT ?",
                                    (limit,)).fetchall()
        return [{"id": row[0], "kind": row[1], "payload": json.loads(row[2])} for row in rows]

    def acknowledge(self, identifier):
        with self._transaction():
            cursor = self._db.execute("UPDATE outbox SET acknowledged=1 WHERE id=?", (identifier,))
            if cursor.rowcount != 1:
                raise JournalConflict("Cannot acknowledge an unknown receipt")

    def begin_execution(self, run_id, epoch, sequence):
        """Commit before invoking hardware. An unfinished entry is never replayable."""
        with self._transaction():
            try:
                self._db.execute("INSERT INTO execution VALUES (?, ?, ?, 'begun', ?, NULL)",
                                 (run_id, epoch, sequence, time.time()))
            except sqlite3.IntegrityError:
                raise JournalConflict("This motion sequence has already begun; duplicate execution is forbidden") from None

    def finish_execution(self, run_id, epoch, sequence, receipt_id, payload, *, learning_pending=False):
        """Commit actual evidence and completed motion together before uploading."""
        with self._transaction():
            row = self._db.execute("SELECT state FROM execution WHERE run_id=? AND epoch=? AND sequence=?",
                                  (run_id, epoch, sequence)).fetchone()
            if row != ("begun",):
                raise JournalConflict("Cannot finish a motion without its unique begun entry")
            self._enqueue(receipt_id, "transition_pending" if learning_pending else "transition", payload)
            self._db.execute("UPDATE execution SET state='finished', receipt_id=? WHERE run_id=? AND epoch=? AND sequence=?",
                             (receipt_id, run_id, epoch, sequence))

    def resolve_learning(self, identifier, payload):
        """Only an unsent pending-bootstrap receipt may be enriched before upload."""
        serialized = encoded(payload)
        digest = hashlib.sha256(serialized.encode()).hexdigest()
        with self._transaction():
            row = self._db.execute("SELECT kind, acknowledged FROM outbox WHERE id=?", (identifier,)).fetchone()
            if row != ("transition_pending", 0):
                raise JournalConflict("Only an unsent learning-pending receipt may be resolved")
            self._db.execute("UPDATE outbox SET kind='transition', payload=?, digest=? WHERE id=?",
                             (serialized, digest, identifier))

    def uncertain_executions(self):
        with self._lock:
            rows = self._db.execute("SELECT run_id, epoch, sequence, started_at FROM execution WHERE state='begun'").fetchall()
        return [{"run_id": row[0], "epoch": row[1], "sequence": row[2], "started_at": row[3]} for row in rows]

    def close(self):
        with self._lock:
            self._db.close()
