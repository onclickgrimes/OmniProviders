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
from app.providers.kiro.transport import KiroOAuthError, KiroOAuthService


class KiroProviderAdapter:
    def __init__(self, transport: Any | None = None) -> None:
        self._transport = transport or KiroOAuthService()

    @property
    def provider_id(self) -> str:
        return "kiro"

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
                    effective_model=str(item.get("baseModel") or model),
                    capabilities=ModelCapabilities(
                        input_modalities=frozenset({"text", "image"}),
                        output_modalities=frozenset({"text"}),
                        structured_output=True,
                        streaming=False,
                        tool_calling="native",
                    ),
                    metadata={
                        key: item[key]
                        for key in ("effort", "effortSchemaPath", "description")
                        if item.get(key) is not None
                    },
                )
            )
        return descriptors

    async def invoke(self, model: str, request: ModelInvocation) -> ModelResult:
        try:
            if request.tools:
                native = await self._transport.generate_native(
                    messages=messages_from_openai_input(request.input),
                    model=model,
                    tools=list(request.tools),
                    tool_choice=request.tool_choice,
                )
                calls_by_id: dict[str, FunctionCall] = {}
                for event in native.get("events") or []:
                    if not isinstance(event, dict):
                        continue
                    tool_use = event.get("toolUseEvent") or (
                        event if event.get("toolUseId") and event.get("name") else None
                    )
                    if not isinstance(tool_use, dict):
                        continue
                    call_id = str(tool_use.get("toolUseId") or "").strip()
                    name = str(tool_use.get("name") or "").strip()
                    if not call_id or not name:
                        continue
                    raw_input = tool_use.get("input")
                    arguments = (
                        raw_input
                        if isinstance(raw_input, str)
                        else json.dumps(raw_input or {}, ensure_ascii=False)
                    )
                    calls_by_id[call_id] = FunctionCall(
                        call_id=call_id,
                        name=name,
                        arguments=arguments,
                    )
                return ModelResult(
                    text=str(native.get("text") or ""),
                    effective_model=str(native.get("effectiveModel") or model),
                    function_calls=tuple(calls_by_id.values()),
                    usage=dict(native.get("usage") or {}),
                )
            text = await self._transport.generate_text_from_messages(
                messages_from_openai_input(request.input),
                model=model,
                temperature=request.temperature,
                response_json=request.response_format == "json",
            )
        except KiroOAuthError as exc:
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
