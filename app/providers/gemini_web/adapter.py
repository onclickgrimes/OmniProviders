from __future__ import annotations

from typing import Any

from app.domain.models import (
    ModelCapabilities,
    ModelDescriptor,
    ModelInvocation,
    ModelResult,
    ModelRuntimeError,
)
from app.protocols.openai_input import messages_from_openai_input
from app.providers.gemini_web.transport import GeminiScrapingError, GeminiScrapingService


class GeminiWebProviderAdapter:
    def __init__(self, transport: Any | None = None) -> None:
        self._transport = transport or GeminiScrapingService()

    @property
    def provider_id(self) -> str:
        return "gemini-web"

    async def list_models(self, *, refresh: bool = False) -> list[ModelDescriptor]:
        del refresh
        status = await self._transport.check_login()
        if not status.get("success"):
            return []
        result: list[ModelDescriptor] = []
        for item in status.get("models") or []:
            if not isinstance(item, dict) or not item.get("available", True):
                continue
            model = str(item.get("modelName") or "").strip()
            if not model:
                continue
            result.append(
                ModelDescriptor(
                    provider=self.provider_id,
                    model=model,
                    label=str(item.get("displayName") or model),
                    available=True,
                    discovery="account_live",
                    capabilities=ModelCapabilities(
                        input_modalities=frozenset({"text", "image", "video", "audio", "file"}),
                        output_modalities=frozenset({"text", "image", "video"}),
                        structured_output=False,
                        streaming=False,
                        tool_calling="none",
                        operations=frozenset(
                            {"responses", "chat.completions", "images.generate", "videos.generate"}
                        ),
                    ),
                    metadata={
                        "description": item.get("description"),
                        "accountStatus": status.get("accountStatus"),
                    },
                )
            )
        return result

    async def invoke(self, model: str, request: ModelInvocation) -> ModelResult:
        try:
            text = await self._transport.generate_text_from_messages(
                messages_from_openai_input(request.input),
                model=model,
                response_json=request.response_format == "json",
            )
        except GeminiScrapingError as exc:
            raise ModelRuntimeError(str(exc), code="provider_error", status_code=502) from exc
        return ModelResult(text=text, effective_model=model)

    async def generate_images(self, model: str, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._transport.generate_images({**payload, "model": model})

    async def generate_video(self, model: str, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._transport.generate_video({**payload, "model": model})

    async def status(self, *, validate: bool = False) -> dict[str, Any]:
        return (
            await self._transport.validate_login()
            if validate
            else await self._transport.check_login()
        )

    async def start_login(self) -> dict[str, Any]:
        return self._transport.login_instructions()

    async def close(self) -> None:
        await self._transport.close()
