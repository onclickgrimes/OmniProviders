from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from app import config
from app.domain.models import ModelRuntimeError


class GenerationJobStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS generation_jobs (
                    id TEXT PRIMARY KEY,
                    operation TEXT NOT NULL,
                    model TEXT NOT NULL,
                    status TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    result_json TEXT,
                    error_json TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def create(self, *, operation: str, model: str, request: dict[str, Any]) -> dict[str, Any]:
        job_id = f"video_{uuid4().hex}" if operation == "videos.generate" else f"job_{uuid4().hex}"
        now = time.time()
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO generation_jobs VALUES (?, ?, ?, ?, ?, NULL, NULL, ?, ?)",
                (job_id, operation, model, "queued", json.dumps(request), now, now),
            )
        return self.get(job_id)

    def recover_interrupted(self) -> int:
        """Finish jobs that cannot survive a sidecar process restart."""
        now = time.time()
        error = json.dumps(
            {
                "code": "sidecar_restarted",
                "message": "Generation was interrupted because OmniProviders restarted.",
            }
        )
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE generation_jobs
                SET status='failed', result_json=NULL, error_json=?, updated_at=?
                WHERE status IN ('queued', 'in_progress')
                """,
                (error, now),
            )
        return int(cursor.rowcount or 0)

    def update(
        self,
        job_id: str,
        *,
        status: str,
        result: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._connection() as connection:
            cursor = connection.execute(
                "UPDATE generation_jobs SET status=?, result_json=?, error_json=?, updated_at=? WHERE id=?",
                (
                    status,
                    json.dumps(result) if result is not None else None,
                    json.dumps(error) if error is not None else None,
                    time.time(),
                    job_id,
                ),
            )
            if not cursor.rowcount:
                raise ModelRuntimeError("Generation job not found.", code="job_not_found", status_code=404)
        return self.get(job_id)

    def get(self, job_id: str) -> dict[str, Any]:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM generation_jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise ModelRuntimeError("Generation job not found.", code="job_not_found", status_code=404)
        result = json.loads(row["result_json"]) if row["result_json"] else None
        return {
            "id": row["id"],
            "object": "video",
            "operation": row["operation"],
            "model": row["model"],
            "status": row["status"],
            "created_at": int(row["created_at"]),
            "completed_at": int(row["updated_at"]) if row["status"] in {"completed", "failed", "cancelled"} else None,
            "result": result,
            "error": json.loads(row["error_json"]) if row["error_json"] else None,
        }


_default_job_store: GenerationJobStore | None = None


def default_job_store() -> GenerationJobStore:
    global _default_job_store
    if _default_job_store is None:
        _default_job_store = GenerationJobStore(config.DATABASE_PATH)
    return _default_job_store
