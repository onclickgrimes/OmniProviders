from __future__ import annotations

from typing import Protocol

from app.domain.models import ModelDescriptor, ModelInvocation, ModelResult


class ProviderAdapter(Protocol):
    @property
    def provider_id(self) -> str: ...

    async def list_models(self, *, refresh: bool = False) -> list[ModelDescriptor]: ...

    async def invoke(self, model: str, request: ModelInvocation) -> ModelResult: ...
