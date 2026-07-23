from __future__ import annotations

import mimetypes
import time
from pathlib import Path
from uuid import uuid4

import httpx

from app import config
from app.domain.models import Artifact, ModelRuntimeError


class ArtifactStore:
    def __init__(self, root: Path, *, ttl_seconds: int = config.ARTIFACT_TTL_SECONDS) -> None:
        self.root = root.resolve()
        self.ttl_seconds = ttl_seconds
        self.root.mkdir(parents=True, exist_ok=True)
        self._artifacts: dict[str, Artifact] = {}

    def save_bytes(
        self,
        data: bytes,
        *,
        mime_type: str = "application/octet-stream",
        filename: str | None = None,
    ) -> Artifact:
        artifact_id = f"artifact_{uuid4().hex}"
        suffix = Path(filename or "").suffix or mimetypes.guess_extension(mime_type) or ".bin"
        safe_name = Path(filename or f"{artifact_id}{suffix}").name
        path = (self.root / f"{artifact_id}{suffix}").resolve()
        if self.root not in path.parents:
            raise ModelRuntimeError("Invalid artifact path.", code="invalid_artifact")
        path.write_bytes(data)
        now = time.time()
        artifact = Artifact(
            id=artifact_id,
            path=str(path),
            mime_type=mime_type,
            filename=safe_name,
            size=len(data),
            created_at=now,
            expires_at=now + self.ttl_seconds,
        )
        self._artifacts[artifact.id] = artifact
        return artifact

    def import_file(
        self,
        source: str | Path,
        *,
        mime_type: str | None = None,
        filename: str | None = None,
    ) -> Artifact:
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise ModelRuntimeError("Generated media file was not found.", code="artifact_not_found")
        resolved_mime = mime_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return self.save_bytes(path.read_bytes(), mime_type=resolved_mime, filename=filename or path.name)

    async def import_url(self, url: str, *, filename: str | None = None) -> Artifact:
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
        mime_type = response.headers.get("content-type", "application/octet-stream").split(";", 1)[0]
        return self.save_bytes(response.content, mime_type=mime_type, filename=filename)

    def get(self, artifact_id: str) -> Artifact:
        artifact = self._artifacts.get(artifact_id)
        if artifact is None:
            matches = [path for path in self.root.glob(f"{artifact_id}.*") if path.is_file()]
            if matches:
                path = max(matches, key=lambda item: item.stat().st_mtime)
                created_at = path.stat().st_mtime
                artifact = Artifact(
                    id=artifact_id,
                    path=str(path),
                    mime_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                    filename=path.name,
                    size=path.stat().st_size,
                    created_at=created_at,
                    expires_at=created_at + self.ttl_seconds,
                )
                self._artifacts[artifact_id] = artifact
        if artifact is None or artifact.expires_at <= time.time() or not Path(artifact.path).is_file():
            raise ModelRuntimeError(
                f"Artifact '{artifact_id}' was not found or has expired.",
                code="artifact_not_found",
                status_code=404,
            )
        return artifact

    def cleanup(self) -> int:
        now = time.time()
        for path in self.root.glob("artifact_*.*"):
            if path.is_file() and path.stat().st_mtime + self.ttl_seconds <= now:
                path.unlink(missing_ok=True)
        expired = [item for item in self._artifacts.values() if item.expires_at <= now]
        for artifact in expired:
            Path(artifact.path).unlink(missing_ok=True)
            self._artifacts.pop(artifact.id, None)
        return len(expired)


_default_artifact_store: ArtifactStore | None = None


def default_artifact_store() -> ArtifactStore:
    global _default_artifact_store
    if _default_artifact_store is None:
        _default_artifact_store = ArtifactStore(config.ARTIFACTS_DIR)
    return _default_artifact_store
