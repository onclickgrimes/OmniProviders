from __future__ import annotations

import json
import inspect
import os
import uuid
from typing import Any

from google import genai
from google.genai import errors, types

from app.infrastructure.debug import (
    make_json_safe,
    serialize_messages_for_ai_debug,
    write_ai_debug_exchange,
)
from app.persistence.credentials import get_env_or_credential, get_env_or_provider_credential
from app.media.input import load_attachment_bytes
from app.protocols.structured_output import extract_json


DEFAULT_MODEL = "gemini-3.1-pro-preview"
VERTEX_DEFAULT_MODEL = "gemini-2.5-flash"
MAX_INLINE_GEMINI_VIDEO_BYTES = 20 * 1024 * 1024
MAX_INLINE_GEMINI_AUDIO_BYTES = 20 * 1024 * 1024


class GeminiApiError(RuntimeError):
    pass


def _recover_malformed_function_call_text(raw_response: Any) -> str:
    response = raw_response if isinstance(raw_response, dict) else make_json_safe(raw_response)
    candidates = response.get("candidates") if isinstance(response.get("candidates"), list) else []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        finish_reason = str(candidate.get("finish_reason") or candidate.get("finishReason") or "").strip()
        finish_message = str(candidate.get("finish_message") or candidate.get("finishMessage") or "").strip()
        if finish_reason != "MALFORMED_FUNCTION_CALL" or "{" not in finish_message:
            continue
        try:
            recovered = extract_json(finish_message)
        except Exception:
            recovered = None
        if recovered is not None:
            return json.dumps(recovered, ensure_ascii=False)
        return finish_message[finish_message.find("{") :].strip()
    return ""


def _read_gemini_response_text(response: Any) -> tuple[str, dict[str, Any], Any]:
    raw_response = make_json_safe(response)
    text = (response.text or "").strip()
    if text:
        return text, {"source": "response.text", "recovered": False}, raw_response

    recovered = _recover_malformed_function_call_text(raw_response).strip()
    if recovered:
        return (
            recovered,
            {
                "source": "candidate.finish_message",
                "recovered": True,
                "reason": "MALFORMED_FUNCTION_CALL",
            },
            raw_response,
        )

    return "", {"source": "response.text", "recovered": False}, raw_response


def _part_from_bytes(data: bytes, mime_type: str) -> types.Part:
    if hasattr(types.Part, "from_bytes"):
        return types.Part.from_bytes(data=data, mime_type=mime_type)
    return types.Part(
        inline_data=types.Blob(
            data=data,
            mime_type=mime_type,
        )
    )


def resolve_video_genai_backend() -> str:
    backend = str(get_env_or_credential("GENAI_BACKEND") or "").strip().lower()
    return "vertex" if backend == "vertex" else "gemini"


def create_genai_client(backend: str, *, api_version: str = "v1") -> genai.Client:
    if backend == "vertex":
        vertex_project = get_env_or_provider_credential("vertex", "VERTEX_PROJECT")
        if not vertex_project:
            raise GeminiApiError("Projeto do Vertex não configurado. Adicione-o em Configurações > Vertex API.")
            
        location = get_env_or_provider_credential("vertex", "VERTEX_LOCATION") or "global"
        credentials_path = get_env_or_provider_credential("vertex", "VERTEX_CREDENTIALS_PATH")
        
        if credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path.strip().strip('"')
            
        return genai.Client(
            vertexai=True,
            project=vertex_project,
            location=location,
            http_options=types.HttpOptions(api_version=api_version),
        )
    
    api_key = get_env_or_provider_credential("gemini", "GEMINI_API_KEY", "GOOGLE_API_KEY")
    if not api_key:
        raise GeminiApiError("Chave do Gemini não configurada.")
        
    return genai.Client(api_key=api_key)


def create_video_genai_client() -> genai.Client:
    return create_genai_client(resolve_video_genai_backend())


class GeminiApiService:
    def __init__(self, *, backend: str | None = None, client: Any | None = None) -> None:
        if backend not in {None, "gemini", "vertex"}:
            raise ValueError(f"Unsupported Google GenAI backend: {backend}")
        self._forced_backend = backend
        self._provided_client = client
        self._client: genai.Client | None = client
        self._client_state_hash: str | None = "provided" if client is not None else None

    def _backend(self) -> str:
        return self._forced_backend or resolve_video_genai_backend()

    def _default_model(self) -> str:
        configured = get_env_or_provider_credential(self._backend(), "GEMINI_MODEL")
        if configured:
            return configured
        return VERTEX_DEFAULT_MODEL if self._backend() == "vertex" else DEFAULT_MODEL

    def _resolve_model(self, options: dict[str, Any] | None = None) -> str:
        model = str((options or {}).get("model") or self._default_model()).strip()
        if not model or model == "gemini-web-auto":
            return self._default_model()
        return model

    def _get_client_state_hash(self) -> str:
        backend = self._backend()
        if backend == "vertex":
            return f"vertex:{get_env_or_provider_credential('vertex', 'VERTEX_PROJECT')}:{get_env_or_provider_credential('vertex', 'VERTEX_LOCATION')}:{get_env_or_provider_credential('vertex', 'VERTEX_CREDENTIALS_PATH')}"
        return f"gemini:{get_env_or_provider_credential('gemini', 'GEMINI_API_KEY', 'GOOGLE_API_KEY')}"

    def has_auth_config(self) -> bool:
        backend = self._backend()
        if backend == "vertex":
            return bool(get_env_or_provider_credential("vertex", "VERTEX_PROJECT"))
        return bool(get_env_or_provider_credential("gemini", "GEMINI_API_KEY", "GOOGLE_API_KEY"))

    def check_config(self) -> dict[str, Any]:
        if not self.has_auth_config():
            return {
                "success": False,
                "isLoggedIn": False,
                "env": ["GEMINI_API_KEY"] if self._backend() == "gemini" else ["VERTEX_PROJECT"],
                "message": "Configure o Vertex ou Gemini para usar a API.",
            }
        return {
            "success": True,
            "isLoggedIn": True,
            "model": self._default_model(),
        }

    def _get_client(self) -> genai.Client:
        if self._provided_client is not None:
            return self._provided_client
        state_hash = self._get_client_state_hash()
        if self._client is None or self._client_state_hash != state_hash:
            self._client = create_genai_client(self._backend())
            self._client_state_hash = state_hash
        return self._client

    async def list_models(self) -> list[dict[str, Any]]:
        models: list[dict[str, Any]] = []
        pager = self._get_client().aio.models.list()
        if inspect.isawaitable(pager):
            pager = await pager
        async for model in pager:
            name = str(getattr(model, "name", "") or "").strip()
            model_id = name.removeprefix("models/")
            if not model_id:
                continue
            models.append(
                {
                    "id": model_id,
                    "name": name,
                    "label": str(getattr(model, "display_name", "") or model_id),
                    "displayName": getattr(model, "display_name", None),
                    "description": getattr(model, "description", None),
                    "version": getattr(model, "version", None),
                    "supportedActions": list(getattr(model, "supported_actions", None) or []),
                    "inputTokenLimit": getattr(model, "input_token_limit", None),
                    "outputTokenLimit": getattr(model, "output_token_limit", None),
                    "available": True,
                }
            )
        return models

    async def generate_text(
        self,
        prompt: str,
        *,
        model: str | None = None,
        response_json: bool = False,
        temperature: float = 0.4,
        debug_context: dict[str, Any] | None = None,
    ) -> str:
        resolved_model = model or self._default_model()
        config = types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json" if response_json else None,
        )
        debug_request = {
            "contents": prompt,
            "config": {
                "temperature": temperature,
                "response_mime_type": "application/json" if response_json else None,
            },
        }
        metadata = {
            "backend": self._backend(),
            **(debug_context or {}),
        }

        try:
            response = await self._get_client().aio.models.generate_content(
                model=resolved_model,
                contents=prompt,
                config=config,
            )
        except errors.APIError as exc:
            write_ai_debug_exchange(
                provider="gemini",
                model=resolved_model,
                operation="gemini.generate_text",
                request=debug_request,
                error=exc,
                metadata=metadata,
            )
            raise GeminiApiError(f"Gemini API error: {exc}") from exc
        except Exception as exc:
            write_ai_debug_exchange(
                provider="gemini",
                model=resolved_model,
                operation="gemini.generate_text",
                request=debug_request,
                error=exc,
                metadata=metadata,
            )
            raise GeminiApiError(f"Gemini SDK error: {exc}") from exc

        try:
            text, text_read, raw_response = _read_gemini_response_text(response)
        except Exception as exc:
            write_ai_debug_exchange(
                provider="gemini",
                model=resolved_model,
                operation="gemini.generate_text",
                request=debug_request,
                response={"raw": make_json_safe(response)},
                error=exc,
                metadata=metadata,
            )
            raise GeminiApiError(f"Gemini response text error: {exc}") from exc

        write_ai_debug_exchange(
            provider="gemini",
            model=resolved_model,
            operation="gemini.generate_text",
            request=debug_request,
            response={
                "text": text,
                "textRead": text_read,
                "raw": raw_response,
            },
            metadata=metadata,
        )
        if not text:
            raise GeminiApiError("Gemini returned an empty response.")
        return text

    def _build_content_parts(self, message: dict[str, Any]) -> list[types.Part]:
        raw_parts = message.get("parts") if isinstance(message.get("parts"), list) else None
        if not raw_parts:
            return [types.Part(text=str(message.get("content") or ""))]

        parts: list[types.Part] = []
        for part in raw_parts:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "image":
                image_bytes, mime_type = load_attachment_bytes(part, fallback_mime_type="image/png")
                parts.append(_part_from_bytes(image_bytes, mime_type))
                continue

            if part.get("type") == "video":
                video_bytes, mime_type = load_attachment_bytes(part, fallback_mime_type="video/webm")
                if len(video_bytes) > MAX_INLINE_GEMINI_VIDEO_BYTES:
                    size_mb = len(video_bytes) / (1024 * 1024)
                    raise GeminiApiError(
                        f"Preview video attachment is {size_mb:.1f}MB; compress it below 20MB before sending to Gemini API."
                    )
                parts.append(_part_from_bytes(video_bytes, mime_type))
                continue

            if part.get("type") == "audio":
                audio_bytes, mime_type = load_attachment_bytes(part, fallback_mime_type="audio/wav")
                if len(audio_bytes) > MAX_INLINE_GEMINI_AUDIO_BYTES:
                    size_mb = len(audio_bytes) / (1024 * 1024)
                    raise GeminiApiError(
                        f"Audio attachment is {size_mb:.1f}MB; reduce it below 20MB before sending to Gemini API."
                    )
                parts.append(_part_from_bytes(audio_bytes, mime_type))
                continue

            text = str(part.get("text") or "")
            if text:
                parts.append(types.Part(text=text))

        return parts or [types.Part(text=str(message.get("content") or ""))]

    def _build_native_contents(
        self, messages: list[dict[str, Any]]
    ) -> tuple[str, list[types.Content]]:
        system_parts: list[str] = []
        contents: list[types.Content] = []
        call_names: dict[str, str] = {}

        for message in messages:
            if not isinstance(message, dict):
                continue
            message_type = str(message.get("type") or "").strip()
            if message_type == "function_call":
                call_id = str(message.get("call_id") or message.get("id") or "").strip()
                name = str(message.get("name") or "").strip()
                if not name:
                    continue
                arguments = message.get("arguments") or {}
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {"value": arguments}
                if not isinstance(arguments, dict):
                    arguments = {"value": arguments}
                if call_id:
                    call_names[call_id] = name
                contents.append(
                    types.Content(
                        role="model",
                        parts=[
                            types.Part(
                                function_call=types.FunctionCall(
                                    id=call_id or None,
                                    name=name,
                                    args=arguments,
                                )
                            )
                        ],
                    )
                )
                continue

            if message_type == "function_call_output":
                call_id = str(message.get("call_id") or "").strip()
                name = str(message.get("name") or call_names.get(call_id) or "").strip()
                if not name:
                    raise GeminiApiError(
                        f"Function output {call_id or '<without id>'} has no matching function name."
                    )
                output = message.get("output")
                if isinstance(output, str):
                    try:
                        output = json.loads(output)
                    except json.JSONDecodeError:
                        pass
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(
                                function_response=types.FunctionResponse(
                                    id=call_id or None,
                                    name=name,
                                    response={"output": output},
                                )
                            )
                        ],
                    )
                )
                continue

            role = str(message.get("role") or "user").strip().lower()
            if role == "system":
                if message.get("parts"):
                    system_parts.extend(
                        str(part.get("text") or "")
                        for part in message.get("parts")
                        if isinstance(part, dict) and part.get("type") == "text"
                    )
                else:
                    system_parts.append(str(message.get("content") or ""))
                continue
            contents.append(
                types.Content(
                    role="model" if role == "assistant" else "user",
                    parts=self._build_content_parts(message),
                )
            )

        return "\n".join(part for part in system_parts if part).strip(), contents

    @staticmethod
    def _build_native_tools(tools: list[dict[str, Any]]) -> list[types.Tool]:
        declarations: list[types.FunctionDeclaration] = []
        for item in tools:
            if not isinstance(item, dict) or item.get("type") != "function":
                continue
            definition = item.get("function") if isinstance(item.get("function"), dict) else item
            name = str(definition.get("name") or "").strip()
            if not name:
                continue
            schema = definition.get("parameters")
            declarations.append(
                types.FunctionDeclaration(
                    name=name,
                    description=str(definition.get("description") or "") or None,
                    parameters_json_schema=schema if isinstance(schema, dict) else {"type": "object"},
                )
            )
        return [types.Tool(function_declarations=declarations)] if declarations else []

    @staticmethod
    def _build_tool_config(tool_choice: Any) -> types.ToolConfig | None:
        if tool_choice in {None, "auto"}:
            mode = "AUTO"
            names = None
        elif tool_choice == "required":
            mode = "ANY"
            names = None
        elif tool_choice == "none":
            return types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="NONE")
            )
        elif isinstance(tool_choice, dict):
            definition = (
                tool_choice.get("function")
                if isinstance(tool_choice.get("function"), dict)
                else tool_choice
            )
            selected = str(definition.get("name") or "").strip()
            mode = "ANY"
            names = [selected] if selected else None
        else:
            mode = "AUTO"
            names = None
        return types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode=mode,
                allowed_function_names=names,
            )
        )

    async def generate_native(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = "auto",
        temperature: float | None = 0.4,
        response_json: bool = False,
        debug_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        system_instruction, contents = self._build_native_contents(messages)
        if not contents:
            raise GeminiApiError("Gemini request did not contain user content.")
        native_tools = self._build_native_tools(tools or [])
        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction or None,
            response_mime_type="application/json" if response_json else None,
            tools=native_tools or None,
            tool_config=self._build_tool_config(tool_choice) if native_tools else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        resolved_model = model or self._default_model()
        debug_request = {
            "contents": serialize_messages_for_ai_debug(messages),
            "tools": make_json_safe(native_tools),
            "toolChoice": make_json_safe(tool_choice),
        }
        metadata = {"backend": self._backend(), **(debug_context or {})}
        try:
            response = await self._get_client().aio.models.generate_content(
                model=resolved_model,
                contents=contents,
                config=config,
            )
        except errors.APIError as exc:
            write_ai_debug_exchange(
                provider="gemini",
                model=resolved_model,
                operation="gemini.generate_native",
                request=debug_request,
                error=exc,
                metadata=metadata,
            )
            raise GeminiApiError(f"Gemini API error: {exc}") from exc
        except Exception as exc:
            write_ai_debug_exchange(
                provider="gemini",
                model=resolved_model,
                operation="gemini.generate_native",
                request=debug_request,
                error=exc,
                metadata=metadata,
            )
            raise GeminiApiError(f"Gemini SDK error: {exc}") from exc

        text_parts: list[str] = []
        function_calls: list[dict[str, Any]] = []
        for candidate in getattr(response, "candidates", None) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                if getattr(part, "thought", False):
                    continue
                part_text = getattr(part, "text", None)
                if part_text:
                    text_parts.append(str(part_text))
                function_call = getattr(part, "function_call", None)
                if function_call and getattr(function_call, "name", None):
                    function_calls.append(
                        {
                            "call_id": str(
                                getattr(function_call, "id", None)
                                or f"call_{uuid.uuid4().hex}"
                            ),
                            "name": str(function_call.name),
                            "arguments": dict(getattr(function_call, "args", None) or {}),
                        }
                    )

        usage_metadata = getattr(response, "usage_metadata", None)
        input_tokens = int(getattr(usage_metadata, "prompt_token_count", 0) or 0)
        output_tokens = int(
            getattr(usage_metadata, "response_token_count", None)
            or getattr(usage_metadata, "candidates_token_count", 0)
            or 0
        )
        usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": int(
                getattr(usage_metadata, "total_token_count", 0)
                or input_tokens + output_tokens
            ),
        }
        result = {
            "text": "\n".join(text_parts).strip(),
            "functionCalls": function_calls,
            "effectiveModel": str(getattr(response, "model_version", None) or resolved_model),
            "usage": usage,
        }
        write_ai_debug_exchange(
            provider="gemini",
            model=resolved_model,
            operation="gemini.generate_native",
            request=debug_request,
            response={"result": result, "raw": make_json_safe(response)},
            metadata=metadata,
        )
        if not result["text"] and not function_calls:
            raise GeminiApiError("Gemini returned an empty response.")
        return result

    async def generate_text_from_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float = 0.4,
        response_json: bool = False,
        debug_context: dict[str, Any] | None = None,
    ) -> str:
        system_instruction = ""
        contents: list[types.Content] = []

        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "user").strip().lower()
            if role == "system":
                if message.get("parts"):
                    system_instruction = "\n".join(
                        str(part.get("text") or "")
                        for part in message.get("parts")
                        if isinstance(part, dict) and part.get("type") == "text"
                    ).strip()
                else:
                    system_instruction = str(message.get("content") or "").strip()
                continue

            contents.append(
                types.Content(
                    role="model" if role == "assistant" else "user",
                    parts=self._build_content_parts(message),
                )
            )

        if not contents:
            raise GeminiApiError("Gemini request did not contain user content.")

        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction or None,
            response_mime_type="application/json" if response_json else None,
        )
        resolved_model = model or self._default_model()
        debug_request = {
            "systemInstruction": system_instruction or None,
            "contents": serialize_messages_for_ai_debug(
                [
                    message
                    for message in messages
                    if isinstance(message, dict) and str(message.get("role") or "user").strip().lower() != "system"
                ]
            ),
            "config": {
                "temperature": temperature,
                "response_mime_type": "application/json" if response_json else None,
            },
        }
        metadata = {
            "backend": self._backend(),
            **(debug_context or {}),
        }

        try:
            response = await self._get_client().aio.models.generate_content(
                model=resolved_model,
                contents=contents,
                config=config,
            )
        except errors.APIError as exc:
            write_ai_debug_exchange(
                provider="gemini",
                model=resolved_model,
                operation="gemini.generate_text_from_messages",
                request=debug_request,
                error=exc,
                metadata=metadata,
            )
            raise GeminiApiError(f"Gemini API error: {exc}") from exc
        except Exception as exc:
            write_ai_debug_exchange(
                provider="gemini",
                model=resolved_model,
                operation="gemini.generate_text_from_messages",
                request=debug_request,
                error=exc,
                metadata=metadata,
            )
            raise GeminiApiError(f"Gemini SDK error: {exc}") from exc

        try:
            text, text_read, raw_response = _read_gemini_response_text(response)
        except Exception as exc:
            write_ai_debug_exchange(
                provider="gemini",
                model=resolved_model,
                operation="gemini.generate_text_from_messages",
                request=debug_request,
                response={"raw": make_json_safe(response)},
                error=exc,
                metadata=metadata,
            )
            raise GeminiApiError(f"Gemini response text error: {exc}") from exc

        write_ai_debug_exchange(
            provider="gemini",
            model=resolved_model,
            operation="gemini.generate_text_from_messages",
            request=debug_request,
            response={
                "text": text,
                "textRead": text_read,
                "raw": raw_response,
            },
            metadata=metadata,
        )
        if not text:
            raise GeminiApiError("Gemini returned an empty response.")
        return text

    async def generate_json(
        self, prompt: str, *, options: dict[str, Any] | None = None
    ) -> Any:
        text = await self.generate_text(
            prompt,
            model=self._resolve_model(options),
            response_json=True,
        )
        return extract_json(text)

    async def generate_json_from_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        options: dict[str, Any] | None = None,
    ) -> Any:
        text = await self.generate_text_from_messages(
            messages,
            model=self._resolve_model(options),
            response_json=True,
        )
        return extract_json(text)


gemini_api_service = GeminiApiService()
