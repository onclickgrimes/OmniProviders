from __future__ import annotations

import base64
import inspect
from pathlib import Path
from typing import Any

from app import config
from app.domain.models import Artifact, ModelRuntimeError
from app.media.artifacts import ArtifactStore
from app.persistence.jobs import GenerationJobStore
from app.runtime.registry import ProviderRegistry


class MediaRuntime:
    def __init__(
        self,
        registry: ProviderRegistry,
        artifacts: ArtifactStore,
        jobs: GenerationJobStore,
    ) -> None:
        self.registry = registry
        self.artifacts = artifacts
        self.jobs = jobs

    async def _store_item(self, item: Any, *, client: Any = None) -> list[Artifact]:
        if isinstance(item, bytes):
            return [self.artifacts.save_bytes(item)]
        if isinstance(item, (str, Path)):
            value = str(item)
            if value.startswith(("http://", "https://")):
                return [await self.artifacts.import_url(value)]
            return [self.artifacts.import_file(value)]
        if isinstance(item, dict):
            if item.get("bytes") is not None:
                raw = item["bytes"]
                data = base64.b64decode(raw) if isinstance(raw, str) else bytes(raw)
                return [
                    self.artifacts.save_bytes(
                        data,
                        mime_type=str(item.get("mime_type") or item.get("mimeType") or "application/octet-stream"),
                        filename=item.get("filename"),
                    )
                ]
            for key in ("path", "file", "url"):
                if item.get(key):
                    return await self._store_item(item[key], client=client)
            return []
        save = getattr(item, "save", None)
        if callable(save):
            saved = save(
                path=str(config.ARTIFACTS_DIR / "provider-output"),
                filename=None,
                verbose=False,
                **({"client": client} if client is not None else {}),
            )
            if inspect.isawaitable(saved):
                saved = await saved
            if isinstance(saved, dict):
                artifacts: list[Artifact] = []
                for value in saved.values():
                    if value:
                        artifacts.extend(await self._store_item(value))
                return artifacts
            return await self._store_item(saved)
        url = getattr(item, "url", None)
        if url:
            return [await self.artifacts.import_url(str(url))]
        return []

    async def _store_result(self, result: dict[str, Any]) -> list[Artifact]:
        candidates = result.get("artifacts") or result.get("media") or result.get("images") or result.get("videos") or []
        if not isinstance(candidates, list):
            candidates = [candidates]
        artifacts: list[Artifact] = []
        for item in candidates:
            artifacts.extend(await self._store_item(item, client=result.get("mediaClient")))
        if not artifacts:
            raise ModelRuntimeError(
                "Provider completed without returning media.",
                code="empty_media_response",
                status_code=502,
            )
        return artifacts

    async def generate_images(self, target: str, payload: dict[str, Any]) -> dict[str, Any]:
        adapter, descriptor = await self.registry.resolve(target, operation="images.generate")
        handler = getattr(adapter, "generate_images", None)
        if not callable(handler):
            raise ModelRuntimeError("Provider has no image adapter.", code="unsupported_operation")
        result = await handler(descriptor.model, payload)
        artifacts = await self._store_result(result)
        return {
            "created": artifacts[0].created_at,
            "model": target,
            "data": [artifact.to_dict() for artifact in artifacts],
            "metadata": {
                key: value
                for key, value in result.items()
                if key not in {"artifacts", "media", "images", "videos", "mediaClient"}
            },
        }

    def create_video_job(self, target: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.jobs.create(operation="videos.generate", model=target, request=payload)

    async def run_video_job(self, job_id: str, target: str, payload: dict[str, Any]) -> None:
        self.jobs.update(job_id, status="in_progress")
        try:
            adapter, descriptor = await self.registry.resolve(target, operation="videos.generate")
            handler = getattr(adapter, "generate_video", None)
            if not callable(handler):
                raise ModelRuntimeError("Provider has no video adapter.", code="unsupported_operation")
            result = await handler(descriptor.model, payload)
            artifacts = await self._store_result(result)
            self.jobs.update(
                job_id,
                status="completed",
                result={
                    "artifacts": [artifact.to_dict() for artifact in artifacts],
                    "metadata": {
                        key: value
                        for key, value in result.items()
                        if key not in {"artifacts", "media", "images", "videos", "mediaClient"}
                    },
                },
            )
        except Exception as exc:
            code = exc.code if isinstance(exc, ModelRuntimeError) else "provider_error"
            self.jobs.update(job_id, status="failed", error={"code": code, "message": str(exc)})

    async def generate_speech(self, target: str, payload: dict[str, Any]) -> Artifact:
        adapter, descriptor = await self.registry.resolve(target, operation="audio.speech")
        handler = getattr(adapter, "generate_speech", None)
        if not callable(handler):
            raise ModelRuntimeError("Provider has no speech adapter.", code="unsupported_operation")
        result = await handler(descriptor.model, payload)
        return (await self._store_result(result))[0]

    async def transcribe(self, target: str, audio: bytes, payload: dict[str, Any]) -> dict[str, Any]:
        adapter, descriptor = await self.registry.resolve(target, operation="audio.transcriptions")
        handler = getattr(adapter, "transcribe", None)
        if not callable(handler):
            raise ModelRuntimeError("Provider has no transcription adapter.", code="unsupported_operation")
        return await handler(descriptor.model, audio, payload)
