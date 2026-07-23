from __future__ import annotations

import json
from typing import Any

from app.domain.models import (
    FunctionCall,
    ModelCapabilities,
    ModelDescriptor,
    ModelInvocation,
    ModelResult,
    ModelRuntimeError,
)
from app.protocols.openai_input import messages_from_openai_input
from app.providers.antigravity.transport import (
    AntigravityOAuthError,
    AntigravityOAuthService,
)


class AntigravityProviderAdapter:
    def __init__(self, transport: Any | None = None) -> None:
        self._transport = transport or AntigravityOAuthService()

    @property
    def provider_id(self) -> str:
        return "antigravity"

    async def list_models(self, *, refresh: bool = False) -> list[ModelDescriptor]:
        del refresh
        payload = await self._transport.list_models()
        if not isinstance(payload, dict) or not payload.get("success"):
            return []
        descriptors: list[ModelDescriptor] = []
        for item in payload.get("models") or []:
            if not isinstance(item, dict):
                continue
            model = str(item.get("model") or item.get("id") or "").strip()
            if not model:
                continue
            descriptors.append(
                ModelDescriptor(
                    provider=self.provider_id,
                    model=model,
                    label=str(item.get("label") or model),
                    available=True,
                    discovery="account_live",
                    effective_model=str(item.get("backendModel") or model),
                    capabilities=ModelCapabilities(
                        input_modalities=frozenset({"text", "image", "video", "audio"}),
                        output_modalities=frozenset({"text"}),
                        structured_output=True,
                        streaming=False,
                        tool_calling="native",
                    ),
                    metadata={
                        key: item[key]
                        for key in ("description", "quotaInfo", "thinkingLevel")
                        if item.get(key) is not None
                    },
                )
            )
        return descriptors

    async def invoke(self, model: str, request: ModelInvocation) -> ModelResult:
        messages = messages_from_openai_input(request.input)
        try:
            if request.tools:
                native = await self._transport.generate_native(
                    messages=messages,
                    model=model,
                    tools=list(request.tools),
                    tool_choice=request.tool_choice,
                    temperature=request.temperature,
                )
                calls = tuple(
                    FunctionCall(
                        call_id=str(item.get("call_id") or item.get("id") or ""),
                        name=str(item.get("name") or ""),
                        arguments=(
                            item.get("arguments")
                            if isinstance(item.get("arguments"), str)
                            else json.dumps(item.get("arguments") or {}, ensure_ascii=False)
                        ),
                    )
                    for item in native.get("functionCalls") or []
                    if isinstance(item, dict) and item.get("name")
                )
                return ModelResult(
                    text=str(native.get("text") or ""),
                    effective_model=str(native.get("effectiveModel") or model),
                    function_calls=calls,
                    usage=dict(native.get("usage") or {}),
                )
            text = await self._transport.generate_text_from_messages(
                messages,
                model=model,
                temperature=request.temperature if request.temperature is not None else 0.4,
                response_json=request.response_format == "json",
            )
        except AntigravityOAuthError as exc:
            raise ModelRuntimeError(
                str(exc),
                code="provider_error",
                status_code=502,
            ) from exc
        return ModelResult(text=text, effective_model=model)

    async def status(self, *, validate: bool = False) -> dict[str, Any]:
        return (
            await self._transport.validate_credentials()
            if validate
            else self._transport.check_config()
        )

    async def start_login(self) -> dict[str, Any]:
        return self._transport.start_login()

    async def complete_login(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._transport.complete_login_sync(
            str(payload.get("code") or ""),
            str(payload.get("state") or ""),
        )
