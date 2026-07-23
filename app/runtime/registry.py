from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Iterable

from app.domain.models import ModelDescriptor, ModelInvocation, ModelResult, ModelRuntimeError
from app.protocols.openai_input import input_modalities
from app.protocols.structured_output import StructuredOutputError, extract_json
from app.providers.base import ProviderAdapter
from app import config


class ProviderRegistry:
    def __init__(self, adapters: Iterable[ProviderAdapter] = ()) -> None:
        self._adapters = {adapter.provider_id: adapter for adapter in adapters}
        self._model_cache: dict[str, tuple[float, list[ModelDescriptor]]] = {}

    def adapters(self) -> tuple[ProviderAdapter, ...]:
        return tuple(self._adapters.values())

    def get_adapter(self, provider: str) -> ProviderAdapter:
        adapter = self._adapters.get(provider)
        if adapter is None:
            raise ModelRuntimeError(
                f"Provider '{provider}' is not configured.",
                code="provider_not_configured",
                status_code=404,
            )
        return adapter

    def invalidate(self, provider: str | None = None) -> None:
        if provider is None:
            self._model_cache.clear()
        else:
            self._model_cache.pop(provider, None)

    async def _models_for(self, adapter: ProviderAdapter, *, refresh: bool) -> list[ModelDescriptor]:
        cached = self._model_cache.get(adapter.provider_id)
        if not refresh and cached and cached[0] > time.monotonic():
            return cached[1]
        try:
            models = await adapter.list_models(refresh=refresh)
        except Exception:
            models = []
        self._model_cache[adapter.provider_id] = (
            time.monotonic() + config.MODEL_CACHE_TTL_SECONDS,
            models,
        )
        return models

    async def list_models(self, *, refresh: bool = False) -> list[ModelDescriptor]:
        results = await asyncio.gather(
            *(self._models_for(adapter, refresh=refresh) for adapter in self._adapters.values())
        )
        return sorted(
            (model for group in results for model in group if model.available),
            key=lambda model: model.id,
        )

    async def invoke(self, target: str, request: ModelInvocation) -> ModelResult:
        adapter, descriptor = await self.resolve(target)
        model = descriptor.model
        requested_modalities = input_modalities(request.input)
        unsupported = requested_modalities - descriptor.capabilities.input_modalities
        if unsupported:
            raise ModelRuntimeError(
                f"Model '{target}' does not accept: {', '.join(sorted(unsupported))}.",
                code="unsupported_input_modality",
                param="input",
            )
        if request.tools and descriptor.capabilities.tool_calling != "native":
            raise ModelRuntimeError(
                f"Model '{target}' does not support native function tools.",
                code="unsupported_tools",
                param="tools",
            )
        if request.response_format == "json" and not descriptor.capabilities.structured_output:
            raise ModelRuntimeError(
                f"Model '{target}' does not support structured output.",
                code="unsupported_response_format",
                param="response_format",
            )
        result = await adapter.invoke(model, request)
        if request.response_format == "json" and result.text:
            try:
                normalized = json.dumps(extract_json(result.text), ensure_ascii=False)
            except StructuredOutputError as exc:
                raise ModelRuntimeError(
                    "Provider did not return valid structured output.",
                    code="invalid_structured_output",
                    status_code=502,
                ) from exc
            result = ModelResult(
                text=normalized,
                effective_model=result.effective_model,
                function_calls=result.function_calls,
                usage=result.usage,
                metadata=result.metadata,
            )
        return result

    async def resolve(
        self, target: str, *, operation: str | None = None
    ) -> tuple[ProviderAdapter, ModelDescriptor]:
        provider, separator, model = target.partition(":")
        if not separator or not provider or not model:
            raise ModelRuntimeError(
                "Model must use the provider:model format.",
                code="invalid_model",
                param="model",
            )
        adapter = self._adapters.get(provider)
        if adapter is None:
            raise ModelRuntimeError(
                f"Provider '{provider}' is not configured.",
                code="provider_not_configured",
                param="model",
            )
        descriptors = {item.model: item for item in await self._models_for(adapter, refresh=False) if item.available}
        descriptor = descriptors.get(model)
        if descriptor is None:
            raise ModelRuntimeError(
                f"Model '{target}' is not available to the configured account.",
                code="model_not_available",
                param="model",
            )
        if operation and operation not in descriptor.capabilities.operations:
            raise ModelRuntimeError(
                f"Model '{target}' does not support operation '{operation}'.",
                code="unsupported_operation",
                param="model",
            )
        return adapter, descriptor
