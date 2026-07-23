from __future__ import annotations

from typing import Any

from app.domain.models import ModelCapabilities, ModelDescriptor, ModelInvocation, ModelResult, ModelRuntimeError
from app.providers.deepgram.transport import DeepgramError, DeepgramTransport


class DeepgramProviderAdapter:
    provider_id = "deepgram"

    def __init__(self, transport: Any | None = None) -> None:
        self._transport = transport or DeepgramTransport()

    async def list_models(self, *, refresh: bool = False) -> list[ModelDescriptor]:
        del refresh
        if not self._transport.has_auth_config():
            return []
        return [
            ModelDescriptor(
                provider=self.provider_id,
                model=model,
                label=label,
                discovery="static_verified",
                capabilities=ModelCapabilities(
                    input_modalities=frozenset({"audio"}),
                    output_modalities=frozenset({"text"}),
                    operations=frozenset({"audio.transcriptions"}),
                ),
            )
            for model, label in (("nova-3", "Deepgram Nova-3"), ("nova-2", "Deepgram Nova-2"))
        ]

    async def invoke(self, model: str, request: ModelInvocation) -> ModelResult:
        del model, request
        raise ModelRuntimeError("Deepgram only supports audio transcriptions.", code="unsupported_operation")

    async def transcribe(self, model: str, audio: bytes, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return await self._transport.transcribe(
                audio,
                model=model,
                language=payload.get("language"),
                mime_type=payload.get("mime_type"),
            )
        except DeepgramError as exc:
            raise ModelRuntimeError(str(exc), code="provider_error", status_code=502) from exc

    async def status(self, *, validate: bool = False) -> dict[str, Any]:
        del validate
        configured = self._transport.has_auth_config()
        return {
            "success": configured,
            "isLoggedIn": configured,
            **({} if configured else {"env": ["DEEPGRAM_API_KEY", "DEEPGRAM_TOKEN"]}),
        }
