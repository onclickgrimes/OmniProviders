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
from app.providers.google_genai.transport import GeminiApiError, GeminiApiService
from app.providers.google_genai.media import GoogleMediaService
from app.providers.google_genai.voice import GeminiVoiceError, GeminiVoiceService


THINKING_LEVELS: tuple[tuple[str, str], ...] = (
    ("low", "Baixo"),
    ("medium", "Médio"),
    ("high", "Alto"),
)

CONVERSATIONAL_MODELS: dict[str, str] = {
    "gemini-3.6-flash": "Gemini 3.6 Flash",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro Preview",
}

TTS_MODELS_BY_PROVIDER: dict[str, tuple[tuple[str, str], ...]] = {
    "gemini": (
        ("gemini-3.1-flash-tts-preview", "Gemini 3.1 Flash TTS Preview"),
        ("gemini-2.5-flash-preview-tts", "Gemini 2.5 Flash Preview TTS"),
        ("gemini-2.5-pro-preview-tts", "Gemini 2.5 Pro Preview TTS"),
    ),
    "vertex": (
        ("gemini-3.1-flash-tts-preview", "Gemini 3.1 Flash TTS Preview"),
        ("gemini-2.5-flash-tts", "Gemini 2.5 Flash TTS"),
        ("gemini-2.5-flash-lite-preview-tts", "Gemini 2.5 Flash Lite Preview TTS"),
        ("gemini-2.5-pro-tts", "Gemini 2.5 Pro TTS"),
    ),
}

VERTEX_VERIFIED_MODELS: tuple[dict[str, Any], ...] = (
    {
        "id": "gemini-3.6-flash",
        "label": "Gemini 3.6 Flash",
        "supportedActions": ["generateContent"],
    },
    {
        "id": "gemini-3.1-pro-preview",
        "label": "Gemini 3.1 Pro Preview",
        "supportedActions": ["generateContent"],
    },
    {
        "id": "gemini-3.1-flash-image-preview",
        "label": "Gemini 3.1 Flash Image Preview",
        "supportedActions": ["generateContent"],
    },
    {
        "id": "veo-2.0-generate-001",
        "label": "Veo 2",
        "supportedActions": ["generateVideos"],
    },
    {
        "id": "veo-3.1-generate-001",
        "label": "Veo 3.1",
        "supportedActions": ["generateVideos"],
    },
    {
        "id": "veo-3.1-fast-generate-001",
        "label": "Veo 3.1 Fast",
        "supportedActions": ["generateVideos"],
    },
    {
        "id": "veo-3.1-lite-generate-001",
        "label": "Veo 3.1 Lite",
        "supportedActions": ["generateVideos"],
    },
)


def _capabilities(item: dict[str, Any]) -> ModelCapabilities:
    model = str(item.get("id") or item.get("name") or "").lower()
    actions = {str(action).lower() for action in item.get("supportedActions") or []}
    if "generatevideos" in actions or model.startswith("veo-"):
        return ModelCapabilities(
            input_modalities=frozenset({"text", "image"}),
            output_modalities=frozenset({"video"}),
            operations=frozenset({"videos.generate"}),
        )
    if "tts" in model:
        return ModelCapabilities(
            input_modalities=frozenset({"text"}),
            output_modalities=frozenset({"audio"}),
            operations=frozenset({"audio.speech"}),
        )
    if (
        "image" in model
        or model.startswith("imagen-")
        or model.startswith("nano-banana")
    ):
        return ModelCapabilities(
            input_modalities=frozenset({"text", "image"}),
            output_modalities=frozenset({"image"}),
            operations=frozenset({"images.generate"}),
        )
    return ModelCapabilities(
        input_modalities=frozenset({"text", "image", "video", "audio"}),
        output_modalities=frozenset({"text"}),
        structured_output=True,
        streaming=False,
        tool_calling="native",
        operations=frozenset({"responses", "chat.completions"}),
    )


def _model_id(item: dict[str, Any]) -> str:
    return str(item.get("id") or item.get("name") or "").strip().removeprefix("models/")


def _metadata(item: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        **{
            key: item[key]
            for key in ("description", "version", "supportedActions", "catalogSource")
            if item.get(key) is not None
        },
        **extra,
    }


def _thinking_variant(model: str) -> tuple[str, str | None]:
    base_model, separator, level = str(model or "").rpartition("@")
    if (
        separator
        and base_model in CONVERSATIONAL_MODELS
        and level in {value for value, _label in THINKING_LEVELS}
    ):
        return base_model, level
    return str(model or ""), None


class GoogleGenAIProviderAdapter:
    def __init__(
        self,
        *,
        provider: str,
        transport: Any | None = None,
        media: Any | None = None,
        voice: Any | None = None,
    ) -> None:
        self._provider = provider
        self._transport = transport or GeminiApiService(backend=provider)
        self._media = media or GoogleMediaService(backend=provider)
        self._voice = voice or GeminiVoiceService(backend=provider)

    @property
    def provider_id(self) -> str:
        return self._provider

    async def list_models(self, *, refresh: bool = False) -> list[ModelDescriptor]:
        del refresh
        has_auth = getattr(self._transport, "has_auth_config", None)
        if callable(has_auth) and not has_auth():
            return []
        try:
            raw_models = await self._transport.list_models()
        except Exception:
            raw_models = []
        live_discovery = "account_live" if self.provider_id == "vertex" else "provider_live"
        raw_models = [
            {**item, "_discovery": live_discovery}
            for item in raw_models
            if isinstance(item, dict)
        ]
        if self.provider_id == "vertex":
            live_ids = {
                _model_id(item)
                for item in raw_models
            }
            raw_models = [
                *raw_models,
                *(
                    {
                        **item,
                        "_discovery": "static_verified",
                        "catalogSource": "vertex_verified_fallback",
                    }
                    for item in VERTEX_VERIFIED_MODELS
                    if item["id"] not in live_ids
                ),
            ]
        descriptors: list[ModelDescriptor] = []
        for item in raw_models:
            model = _model_id(item)
            if not model:
                continue
            discovery = str(item.get("_discovery") or live_discovery)
            if model in CONVERSATIONAL_MODELS:
                for thinking_level, thinking_label in THINKING_LEVELS:
                    descriptors.append(
                        ModelDescriptor(
                            provider=self.provider_id,
                            model=f"{model}@{thinking_level}",
                            label=f"{CONVERSATIONAL_MODELS[model]} ({thinking_label})",
                            available=bool(item.get("available", True)),
                            discovery=discovery,
                            effective_model=model,
                            capabilities=_capabilities({**item, "id": model}),
                            metadata=_metadata(
                                item,
                                thinkingLevel=thinking_level,
                            ),
                        )
                    )
                continue
            capabilities = _capabilities({**item, "id": model})
            if not (
                capabilities.operations
                & frozenset({"images.generate", "videos.generate", "audio.speech"})
            ):
                continue
            descriptors.append(
                ModelDescriptor(
                    provider=self.provider_id,
                    model=model,
                    label=str(item.get("label") or item.get("displayName") or model),
                    available=bool(item.get("available", True)),
                    discovery=discovery,
                    capabilities=capabilities,
                    metadata=_metadata(item),
                )
            )
        known_models = {descriptor.model for descriptor in descriptors}
        descriptors.extend(
            ModelDescriptor(
                provider=self.provider_id,
                model=model,
                label=label,
                available=True,
                discovery="static_verified",
                capabilities=_capabilities({"id": model}),
                metadata={"catalogSource": "google_tts_documentation"},
            )
            for model, label in TTS_MODELS_BY_PROVIDER.get(self.provider_id, ())
            if model not in known_models
        )
        return descriptors

    async def invoke(self, model: str, request: ModelInvocation) -> ModelResult:
        messages = messages_from_openai_input(request.input)
        effective_model, thinking_level = _thinking_variant(model)
        try:
            if request.tools:
                native = await self._transport.generate_native(
                    messages=messages,
                    model=effective_model,
                    tools=list(request.tools),
                    tool_choice=request.tool_choice,
                    temperature=request.temperature,
                    thinking_level=thinking_level,
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
                    effective_model=str(native.get("effectiveModel") or effective_model),
                    function_calls=calls,
                    usage=dict(native.get("usage") or {}),
                )
            text = await self._transport.generate_text_from_messages(
                messages,
                model=effective_model,
                temperature=request.temperature if request.temperature is not None else 0.4,
                response_json=request.response_format == "json",
                thinking_level=thinking_level,
            )
        except GeminiApiError as exc:
            raise ModelRuntimeError(
                str(exc), code="provider_error", status_code=502
            ) from exc
        return ModelResult(text=text, effective_model=effective_model)

    async def generate_images(self, model: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return await self._media.generate_images(model, payload)
        except GeminiApiError as exc:
            raise ModelRuntimeError(str(exc), code="provider_error", status_code=502) from exc

    async def generate_video(self, model: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return await self._media.generate_video(model, payload)
        except GeminiApiError as exc:
            raise ModelRuntimeError(str(exc), code="provider_error", status_code=502) from exc

    async def generate_speech(self, model: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return await self._voice.generate_speech({**payload, "model": model})
        except GeminiVoiceError as exc:
            raise ModelRuntimeError(str(exc), code="provider_error", status_code=502) from exc

    async def status(self, *, validate: bool = False) -> dict[str, Any]:
        config = dict(self._transport.check_config())
        if not validate or not config.get("success"):
            return config
        try:
            validation_model = str(config.get("model") or "").strip()
            if not validation_model:
                raise GeminiApiError("Nenhum modelo de texto configurado para validar a conta.")
            validation_text = await self._transport.generate_text_from_messages(
                [{"role": "user", "content": "Responda somente OK."}],
                model=validation_model,
                temperature=0.0,
                response_json=False,
            )
            return {
                **config,
                "success": True,
                "isLoggedIn": True,
                "canGenerate": True,
                "accountStatus": "validated",
                "validationModel": validation_model,
                "effectiveModel": validation_model,
                "validationText": str(validation_text or "").strip(),
            }
        except Exception as exc:
            return {
                **config,
                "success": False,
                "isLoggedIn": bool(config.get("isLoggedIn")),
                "canGenerate": False,
                "accountStatus": "configured",
                "error": str(exc),
            }


class GeminiProviderAdapter(GoogleGenAIProviderAdapter):
    def __init__(self, transport: Any | None = None, *, media: Any | None = None, voice: Any | None = None) -> None:
        super().__init__(provider="gemini", transport=transport, media=media, voice=voice)


class VertexProviderAdapter(GoogleGenAIProviderAdapter):
    def __init__(self, transport: Any | None = None, *, media: Any | None = None, voice: Any | None = None) -> None:
        super().__init__(provider="vertex", transport=transport, media=media, voice=voice)
