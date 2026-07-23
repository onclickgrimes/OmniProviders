from __future__ import annotations

from typing import Any

from app.domain.models import (
    FunctionCall,
    ModelCapabilities,
    ModelDescriptor,
    ModelInvocation,
    ModelResult,
)


class FakeProviderAdapter:
    def __init__(
        self,
        *,
        provider: str,
        models: list[dict[str, Any]],
        response_text: str = "",
        tool_name: str | None = None,
    ) -> None:
        self._provider = provider
        self._models = models
        self._response_text = response_text
        self._tool_name = tool_name
        self.last_request: ModelInvocation | None = None

    @property
    def provider_id(self) -> str:
        return self._provider

    async def list_models(self, *, refresh: bool = False) -> list[ModelDescriptor]:
        del refresh
        return [
            ModelDescriptor(
                provider=self._provider,
                model=str(item["id"]),
                label=str(item.get("label") or item["id"]),
                available=bool(item.get("available", True)),
                capabilities=ModelCapabilities(
                    structured_output=bool(item.get("structured_output", True)),
                    tool_calling="native" if self._tool_name else "none"
                ),
                discovery="account_live",
            )
            for item in self._models
        ]

    async def invoke(self, model: str, request: ModelInvocation) -> ModelResult:
        self.last_request = request
        has_tool_output = isinstance(request.input, list) and any(
            isinstance(item, dict) and item.get("type") == "function_call_output"
            for item in request.input
        )
        if self._tool_name and not has_tool_output:
            return ModelResult(
                effective_model=model,
                function_calls=(
                    FunctionCall(
                        call_id="call_fake",
                        name=self._tool_name,
                        arguments='{"number":2}',
                    ),
                ),
            )
        return ModelResult(text=self._response_text, effective_model=model)
