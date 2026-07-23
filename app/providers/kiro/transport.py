from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import struct
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import httpx

from app.infrastructure.debug import serialize_messages_for_ai_debug, write_ai_debug_exchange
from app.persistence.credentials import get_env_or_credential, update_credentials
from app.protocols.structured_output import StructuredOutputError, extract_json


SERVICE_ID = "kiro-oauth"
DEFAULT_KIRO_MODEL = "auto"
DEFAULT_KIRO_REGION = "us-east-1"
KIRO_TOKEN_EXPIRY_BUFFER_SECONDS = 60
KIRO_ORIGIN = "AI_EDITOR"
KIRO_AGENT_MODE = "vibe"
KIRO_USER_AGENT = "KiroIDE 1.0.52 OmniProviders"
KIRO_LOCAL_TOKEN_PATH = Path.home() / ".aws" / "sso" / "cache" / "kiro-auth-token.json"
KIRO_EFFORT_SEPARATOR = "@"
KIRO_SELECTABLE_EFFORT_LEVELS = ("low", "medium", "high", "max")
KIRO_EFFORT_LABELS = {
    "low": "Low",
    "medium": "Medium",
    "high": "High",
    "max": "Max",
}


class KiroOAuthError(RuntimeError):
    pass


def _json_value(payload: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = str(payload.get(key) or "").strip()
        if value:
            return value
    return ""


def _load_json_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _configured_token_path() -> Path:
    raw_path = str(get_env_or_credential("KIRO_CREDS_FILE") or "").strip()
    return Path(raw_path).expanduser() if raw_path else KIRO_LOCAL_TOKEN_PATH


def _parse_profile_region(profile_arn: str | None) -> str:
    match = re.match(r"^arn:[^:]*:[^:]*:([^:]*):", str(profile_arn or ""))
    if match and match.group(1):
        return match.group(1)
    return DEFAULT_KIRO_REGION


def _parse_expiry(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        number = float(value)
        return number / 1000 if number > 10_000_000_000 else number

    text = str(value or "").strip()
    if not text:
        return 0.0
    try:
        number = float(text)
        return number / 1000 if number > 10_000_000_000 else number
    except ValueError:
        pass

    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return 0.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _token_is_fresh(expires_at: float) -> bool:
    return expires_at > time.time() + KIRO_TOKEN_EXPIRY_BUFFER_SECONDS


def _normalize_kiro_model(value: str | None) -> str:
    model = str(value or "").strip() or DEFAULT_KIRO_MODEL
    if ":" in model:
        provider, raw_model = model.split(":", 1)
        if provider.strip().lower().replace("-", "_") in {"kiro", "kiro_oauth"}:
            model = raw_model.strip()
    return model or DEFAULT_KIRO_MODEL


def _normalize_effort_level(value: Any) -> str | None:
    effort = str(value or "").strip().lower().replace("_", "").replace("-", "")
    return effort if effort in KIRO_SELECTABLE_EFFORT_LEVELS else None


def _split_kiro_model_and_effort(value: str | None) -> tuple[str, str | None]:
    model = _normalize_kiro_model(value)
    if KIRO_EFFORT_SEPARATOR not in model:
        return model, None
    base_model, raw_effort = model.rsplit(KIRO_EFFORT_SEPARATOR, 1)
    effort = _normalize_effort_level(raw_effort)
    base_model = base_model.strip()
    if not base_model or not effort:
        return model, None
    return base_model, effort


def _extract_error_message(payload: Any) -> str:
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error.get("code") or error).strip()
        if error:
            return str(error).strip()
        for key in ("message", "detail", "Message", "__type"):
            if payload.get(key):
                return str(payload.get(key)).strip()
    return str(payload or "").strip()


def _strip_data_url(value: str) -> tuple[str, str | None]:
    text = str(value or "").strip()
    if not text.lower().startswith("data:"):
        return text, None
    header, _, data = text.partition(",")
    match = re.match(r"data:([^;]+);base64", header, re.IGNORECASE)
    return data.strip(), match.group(1) if match else None


def _mime_to_image_format(mime_type: str | None) -> str:
    mime = str(mime_type or "image/png").strip().lower()
    if "/" in mime:
        mime = mime.split("/", 1)[1]
    if mime in {"jpeg", "jpg"}:
        return "jpeg"
    if mime in {"png", "gif", "webp"}:
        return mime
    return "png"


def _is_probably_local_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
    except Exception:
        return False
    return parsed.hostname in {"127.0.0.1", "localhost", "::1"}


def _path_to_image_part(path_text: str, fallback_mime_type: str = "image/png") -> dict[str, Any]:
    path = Path(path_text)
    if not path.exists() or not path.is_file():
        raise KiroOAuthError(f"Imagem local nao encontrada para Kiro: {path_text}")
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    mime_type = mimetypes.guess_type(path.name)[0] or fallback_mime_type
    return {
        "format": _mime_to_image_format(mime_type),
        "source": {"bytes": data},
        "mimeType": mime_type,
    }


def _image_part_from_prompt_part(part: dict[str, Any]) -> dict[str, Any]:
    data = str(part.get("data") or "").strip()
    mime_type = str(part.get("mimeType") or part.get("mime_type") or "image/png").strip()
    if data:
        data, data_url_mime = _strip_data_url(data)
        mime_type = data_url_mime or mime_type
        return {
            "format": _mime_to_image_format(mime_type),
            "source": {"bytes": data},
            "mimeType": mime_type,
        }

    url = str(part.get("url") or part.get("sourceUrl") or part.get("httpUrl") or "").strip()
    if url and not _is_probably_local_url(url):
        return {
            "format": _mime_to_image_format(mime_type),
            "source": {"url": url},
            "mimeType": mime_type,
        }

    path_text = str(part.get("path") or "").strip()
    if path_text:
        return _path_to_image_part(path_text, mime_type)

    raise KiroOAuthError("Parte de imagem sem data, url ou path para enviar ao Kiro.")


def _eventstream_payloads(data: bytes) -> list[bytes]:
    payloads: list[bytes] = []
    offset = 0
    while len(data) - offset >= 16:
        total_len, headers_len = struct.unpack(">II", data[offset : offset + 8])
        if total_len < 16 or headers_len < 0:
            break
        message_end = offset + total_len
        if message_end > len(data):
            break
        payload_start = offset + 12 + headers_len
        payload_end = message_end - 4
        if payload_start <= payload_end:
            payloads.append(data[payload_start:payload_end])
        offset = message_end
    return payloads


def _extract_text_from_eventstream(data: bytes) -> tuple[str, list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    text_parts: list[str] = []
    for payload in _eventstream_payloads(data):
        try:
            event = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(event, dict):
            continue
        events.append(event)
        content = event.get("content")
        if isinstance(content, str) and content:
            text_parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("text"):
                    text_parts.append(str(item.get("text") or ""))
        if event.get("assistantResponseMessage"):
            nested = event.get("assistantResponseMessage")
            if isinstance(nested, dict) and isinstance(nested.get("content"), str):
                text_parts.append(nested["content"])
    return "".join(text_parts).strip(), events


def _summarize_kiro_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    content_chunk_count = 0
    content_char_count = 0
    signature_count = 0
    signature_char_count = 0
    summary: dict[str, Any] = {}

    for event in events:
        if not isinstance(event, dict):
            continue
        content = event.get("content")
        if isinstance(content, str):
            content_chunk_count += 1
            content_char_count += len(content)
            continue
        if isinstance(content, list):
            content_chunk_count += len(content)
            content_char_count += sum(len(str(item)) for item in content)
            continue

        signature = event.get("signature")
        if isinstance(signature, str):
            signature_count += 1
            signature_char_count += len(signature)
            continue

        if event.get("stopReason") is not None:
            summary["stopReason"] = event.get("stopReason")
            continue
        if event.get("contextUsagePercentage") is not None:
            summary["contextUsagePercentage"] = event.get("contextUsagePercentage")
            continue
        if event.get("usage") is not None or event.get("unit") is not None:
            summary["metering"] = {
                "usage": event.get("usage"),
                "unit": event.get("unit"),
                "unitPlural": event.get("unitPlural"),
            }
            continue

    summary["contentChunkCount"] = content_chunk_count
    summary["contentCharCount"] = content_char_count
    if signature_count:
        summary["signatureOmitted"] = True
        summary["signatureCount"] = signature_count
        summary["signatureCharCount"] = signature_char_count
    return summary


def _normalize_tool_schema(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"type": "object", "properties": {}}
    normalized: dict[str, Any] = {}
    for key, item in value.items():
        if key == "additionalProperties":
            continue
        if key == "required" and isinstance(item, list) and not item:
            continue
        if isinstance(item, dict):
            normalized[key] = _normalize_tool_schema(item)
        elif isinstance(item, list):
            normalized[key] = [
                _normalize_tool_schema(entry) if isinstance(entry, dict) else entry
                for entry in item
            ]
        else:
            normalized[key] = item
    return normalized


def _kiro_tool_specifications(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specifications: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict) or str(tool.get("type") or "function") != "function":
            continue
        function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        description = str(function.get("description") or f"Tool: {name}").strip()
        if len(description) > 10000:
            description = description[:9997] + "..."
        schema = function.get("parameters") or function.get("input_schema") or {}
        specifications.append(
            {
                "toolSpecification": {
                    "name": name,
                    "description": description,
                    "inputSchema": {"json": _normalize_tool_schema(schema)},
                }
            }
        )
    return specifications


def _parse_tool_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _kiro_user_message(
    *,
    content: str,
    model: str,
    images: list[dict[str, Any]] | None = None,
    tool_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    user_input: dict[str, Any] = {
        "content": content,
        "modelId": model,
        "origin": KIRO_ORIGIN,
    }
    if images:
        user_input["images"] = images
    if tool_results:
        user_input["userInputMessageContext"] = {"toolResults": tool_results}
    return {"userInputMessage": user_input}


def _native_conversation_state(
    messages: list[dict[str, Any]],
    *,
    model: str,
    tools: list[dict[str, Any]],
    tool_choice: Any = None,
) -> dict[str, Any]:
    turns: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        item_type = str(message.get("type") or "").strip()
        if item_type == "function_call":
            call_id = str(message.get("call_id") or message.get("id") or uuid4())
            turns.append(
                {
                    "assistantResponseMessage": {
                        "content": "",
                        "toolUses": [
                            {
                                "toolUseId": call_id,
                                "name": str(message.get("name") or ""),
                                "input": _parse_tool_arguments(message.get("arguments")),
                            }
                        ],
                    }
                }
            )
            continue
        if item_type == "function_call_output":
            output = message.get("output")
            output_text = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)
            turns.append(
                _kiro_user_message(
                    content="",
                    model=model,
                    tool_results=[
                        {
                            "toolUseId": str(message.get("call_id") or ""),
                            "status": "success",
                            "content": [{"text": output_text or "(no output)"}],
                        }
                    ],
                )
            )
            continue

        role = str(message.get("role") or "user").lower()
        parts = message.get("parts") if isinstance(message.get("parts"), list) else []
        text_parts: list[str] = []
        images: list[dict[str, Any]] = []
        content = message.get("content")
        if isinstance(content, str) and content:
            text_parts.append(content)
        for part in parts:
            if not isinstance(part, dict):
                continue
            part_type = str(part.get("type") or "text").lower()
            if part_type == "text" and part.get("text"):
                text_parts.append(str(part["text"]))
            elif part_type == "image":
                images.append(_image_part_from_prompt_part(part))
            elif part_type in {"video", "audio", "file"}:
                raise KiroOAuthError(f"Kiro does not accept {part_type} input.")
        text = "\n\n".join(text_parts).strip()
        if role == "system":
            text = f"<system-reminder>\n{text}\n</system-reminder>"
            role = "user"
        if role == "assistant":
            turns.append({"assistantResponseMessage": {"content": text}})
        else:
            turns.append(
                _kiro_user_message(
                    content=text or ("" if images else "(empty)"),
                    model=model,
                    images=images,
                )
            )

    if turns and "userInputMessage" in turns[-1]:
        current_message = turns.pop()
    else:
        current_message = _kiro_user_message(content="...", model=model)

    specifications = [] if tool_choice == "none" else _kiro_tool_specifications(tools)
    if specifications:
        context = current_message["userInputMessage"].setdefault(
            "userInputMessageContext", {}
        )
        context["tools"] = specifications

    return {
        "agentContinuationId": str(uuid4()),
        "agentTaskType": KIRO_AGENT_MODE,
        "chatTriggerType": "MANUAL",
        "conversationId": str(uuid4()),
        "currentMessage": current_message,
        "history": turns,
    }


def _coerce_schema(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}
    return {}


def _effort_schema_info(model: dict[str, Any]) -> dict[str, Any] | None:
    schema = _coerce_schema(model.get("additionalModelRequestFieldsSchema"))
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    schema_paths = [
        ("output_config", properties.get("output_config")),
        ("reasoning", properties.get("reasoning")),
    ]
    for schema_path, container in schema_paths:
        container_properties = container.get("properties") if isinstance(container, dict) else {}
        effort_schema = container_properties.get("effort") if isinstance(container_properties, dict) else None
        if not isinstance(effort_schema, dict):
            continue
        raw_levels = effort_schema.get("enum")
        if not isinstance(raw_levels, list):
            continue
        available = {_normalize_effort_level(level) for level in raw_levels}
        levels = [level for level in KIRO_SELECTABLE_EFFORT_LEVELS if level in available]
        if not levels:
            continue
        default_level = _normalize_effort_level(effort_schema.get("default"))
        if default_level not in levels:
            default_level = "high" if "high" in levels else levels[0]
        return {
            "schemaPath": schema_path,
            "levels": levels,
            "defaultLevel": default_level,
        }
    return None


def _effort_variant_model_id(model_id: str, effort: str) -> str:
    return f"{model_id}{KIRO_EFFORT_SEPARATOR}{effort}"


def _additional_model_request_fields(effort: str | None, schema_path: str = "output_config") -> dict[str, Any] | None:
    effort_level = _normalize_effort_level(effort)
    if not effort_level:
        return None
    if schema_path == "reasoning":
        return {"reasoning": {"effort": effort_level}}
    return {
        "thinking": {
            "type": "adaptive",
            "display": "summarized",
        },
        "output_config": {"effort": effort_level},
    }


def _serialize_model_option(model: dict[str, Any], fallback_index: int) -> dict[str, Any] | None:
    model_id = str(
        model.get("modelId")
        or model.get("id")
        or model.get("model")
        or model.get("name")
        or ""
    ).strip()
    if not model_id:
        return None
    model_name = str(model.get("modelName") or model.get("displayName") or model_id).strip()
    label = model_name if model_name.lower().startswith("kiro") else f"Kiro {model_name}"
    return {
        "id": model_id,
        "value": model_id,
        "model": model_id,
        "label": label,
        "provider": "kiro",
        "description": model.get("description") or "",
        "sortOrder": fallback_index,
    }


def _serialize_model_options(model: dict[str, Any], fallback_index: int) -> list[dict[str, Any]]:
    base_option = _serialize_model_option(model, fallback_index)
    if not base_option:
        return []
    effort_info = _effort_schema_info(model)
    if not effort_info:
        return []

    base_label = str(base_option.get("label") or base_option["model"]).strip()
    options: list[dict[str, Any]] = []
    for effort_index, effort in enumerate(effort_info["levels"]):
        model_id = _effort_variant_model_id(base_option["model"], effort)
        options.append(
            {
                **base_option,
                "id": model_id,
                "value": model_id,
                "model": model_id,
                "label": f"{base_label} - {KIRO_EFFORT_LABELS[effort]}",
                "sortOrder": fallback_index + ((effort_index + 1) / 10),
                "baseModel": base_option["model"],
                "effort": effort,
                "effortSchemaPath": effort_info["schemaPath"],
            }
        )
    return options


def _expanded_default_model(default_model: str, raw_models: list[Any]) -> str:
    normalized_default = _normalize_kiro_model(default_model)
    for item in raw_models:
        if not isinstance(item, dict):
            continue
        option = _serialize_model_option(item, 0)
        if not option or option["model"] != normalized_default:
            continue
        effort_info = _effort_schema_info(item)
        if not effort_info:
            return normalized_default
        return _effort_variant_model_id(normalized_default, effort_info["defaultLevel"])
    return normalized_default


def _fallback_default_model(default_model: str, models: list[dict[str, Any]]) -> str:
    available_models = {str(option.get("model") or "").strip() for option in models}
    if default_model in available_models:
        return default_model
    for option in models:
        if option.get("effort") == "high":
            return str(option.get("model") or default_model or DEFAULT_KIRO_MODEL)
    return str((models[0] if models else {}).get("model") or default_model or DEFAULT_KIRO_MODEL)


class KiroOAuthService:
    def _raw_credentials(self) -> dict[str, Any]:
        file_payload = _load_json_file(_configured_token_path())
        access_token = str(get_env_or_credential("KIRO_ACCESS_TOKEN") or "").strip()
        refresh_token = str(get_env_or_credential("KIRO_REFRESH_TOKEN") or "").strip()
        profile_arn = str(get_env_or_credential("KIRO_PROFILE_ARN") or "").strip()
        region = str(get_env_or_credential("KIRO_REGION") or "").strip()
        expires_at = str(get_env_or_credential("KIRO_TOKEN_EXPIRES_AT") or "").strip()
        auth_method = str(get_env_or_credential("KIRO_AUTH_METHOD") or "").strip()
        provider = str(get_env_or_credential("KIRO_PROVIDER") or "").strip()

        return {
            "accessToken": access_token or _json_value(file_payload, "accessToken", "access_token"),
            "refreshToken": refresh_token or _json_value(file_payload, "refreshToken", "refresh_token"),
            "expiresAt": expires_at or file_payload.get("expiresAt") or file_payload.get("expires_at"),
            "profileArn": profile_arn or _json_value(file_payload, "profileArn", "profile_arn"),
            "region": region or _json_value(file_payload, "region"),
            "authMethod": auth_method or _json_value(file_payload, "authMethod", "auth_method"),
            "provider": provider or _json_value(file_payload, "provider"),
            "clientId": _json_value(file_payload, "clientId", "client_id"),
            "clientSecret": _json_value(file_payload, "clientSecret", "client_secret"),
            "tokenPath": str(_configured_token_path()),
        }

    def _region(self, credentials: dict[str, Any] | None = None) -> str:
        payload = credentials or self._raw_credentials()
        return str(payload.get("region") or "").strip() or _parse_profile_region(payload.get("profileArn"))

    def _runtime_endpoint(self, region: str) -> str:
        return f"https://runtime.{region}.kiro.dev"

    def _model_endpoints(self, region: str) -> list[str]:
        return [
            f"https://management.{region}.kiro.dev",
            f"https://q.{region}.amazonaws.com",
            f"https://codewhisperer.{region}.amazonaws.com",
        ]

    def _default_model(self) -> str:
        return _normalize_kiro_model(get_env_or_credential("KIRO_MODEL") or DEFAULT_KIRO_MODEL)

    def has_auth_config(self) -> bool:
        credentials = self._raw_credentials()
        return bool(credentials.get("accessToken") or credentials.get("refreshToken"))

    def check_config(self) -> dict[str, Any]:
        credentials = self._raw_credentials()
        is_configured = self.has_auth_config()
        return {
            "success": is_configured,
            "isLoggedIn": is_configured,
            "canGenerate": is_configured,
            "validationModel": self._default_model(),
            "effectiveModel": self._default_model(),
            "accountStatus": credentials.get("authMethod") or "local-session",
            "userId": credentials.get("provider") or "",
            "userIdSource": "kiro-token-cache",
            "profileArnConfigured": bool(credentials.get("profileArn")),
            "region": self._region(credentials),
            **(
                {}
                if is_configured
                else {
                    "message": (
                        "Abra o Kiro externo e faca login, ou configure KIRO_CREDS_FILE "
                        "apontando para kiro-auth-token.json."
                    )
                }
            ),
        }

    def start_login(self) -> dict[str, Any]:
        try:
            kwargs: dict[str, Any] = {}
            if os.name == "nt":
                kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            subprocess.Popen(["kiro"], **kwargs)
            return {
                "success": True,
                "isLoggedIn": self.has_auth_config(),
                "externalOpened": True,
                "message": "Kiro externo aberto. Faca login nele e depois clique em Validar salvo.",
            }
        except Exception as exc:
            return {
                "success": True,
                "isLoggedIn": self.has_auth_config(),
                "externalOpened": False,
                "message": (
                    "Abra o Kiro manualmente, faca login e depois valide. "
                    f"Nao consegui abrir pelo comando kiro: {exc}"
                ),
            }

    def _headers(self, access_token: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "User-Agent": KIRO_USER_AGENT,
            "x-amzn-kiro-agent-mode": KIRO_AGENT_MODE,
        }

    async def _refresh_desktop_token(self, credentials: dict[str, Any]) -> dict[str, Any]:
        refresh_token = str(credentials.get("refreshToken") or "").strip()
        if not refresh_token:
            raise KiroOAuthError("Refresh token Kiro nao configurado.")
        region = self._region(credentials)
        url = f"https://prod.{region}.auth.desktop.kiro.dev/refreshToken"
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
            response = await client.post(
                url,
                headers={"Content-Type": "application/json", "User-Agent": KIRO_USER_AGENT},
                json={"refreshToken": refresh_token},
            )
        payload = response.json() if response.headers.get("content-type", "").startswith("application/json") else {"text": response.text}
        if response.status_code >= 400:
            raise KiroOAuthError(_extract_error_message(payload) or f"Kiro refresh HTTP {response.status_code}")
        return payload if isinstance(payload, dict) else {}

    async def _refresh_oidc_token(self, credentials: dict[str, Any]) -> dict[str, Any]:
        refresh_token = str(credentials.get("refreshToken") or "").strip()
        client_id = str(credentials.get("clientId") or "").strip()
        client_secret = str(credentials.get("clientSecret") or "").strip()
        if not refresh_token or not client_id or not client_secret:
            raise KiroOAuthError("Credenciais OIDC Kiro incompletas para refresh.")
        region = self._region(credentials)
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
            response = await client.post(
                f"https://oidc.{region}.amazonaws.com/token",
                data={
                    "grant_type": "refresh_token",
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "refresh_token": refresh_token,
                },
            )
        try:
            payload = response.json()
        except Exception:
            payload = {"text": response.text}
        if response.status_code >= 400:
            raise KiroOAuthError(_extract_error_message(payload) or f"Kiro OIDC refresh HTTP {response.status_code}")
        return payload if isinstance(payload, dict) else {}

    async def _refresh_access_token(self, credentials: dict[str, Any]) -> str:
        if credentials.get("clientId") and credentials.get("clientSecret"):
            payload = await self._refresh_oidc_token(credentials)
            access_token = str(payload.get("access_token") or payload.get("accessToken") or "").strip()
            refresh_token = str(payload.get("refresh_token") or payload.get("refreshToken") or credentials.get("refreshToken") or "").strip()
            expires_in = float(payload.get("expires_in") or payload.get("expiresIn") or 3600)
            profile_arn = str(payload.get("profileArn") or credentials.get("profileArn") or "").strip()
        else:
            payload = await self._refresh_desktop_token(credentials)
            access_token = str(payload.get("accessToken") or payload.get("access_token") or "").strip()
            refresh_token = str(payload.get("refreshToken") or payload.get("refresh_token") or credentials.get("refreshToken") or "").strip()
            expires_in = float(payload.get("expiresIn") or payload.get("expires_in") or 3600)
            profile_arn = str(payload.get("profileArn") or credentials.get("profileArn") or "").strip()

        if not access_token:
            raise KiroOAuthError("Refresh Kiro nao retornou access token.")

        expires_at = time.time() + max(60.0, expires_in)
        update_credentials(
            SERVICE_ID,
            {
                "KIRO_ACCESS_TOKEN": access_token,
                "KIRO_REFRESH_TOKEN": refresh_token,
                "KIRO_TOKEN_EXPIRES_AT": str(expires_at),
                "KIRO_PROFILE_ARN": profile_arn,
                "KIRO_REGION": self._region({**credentials, "profileArn": profile_arn}),
                "KIRO_AUTH_METHOD": str(credentials.get("authMethod") or ""),
                "KIRO_PROVIDER": str(credentials.get("provider") or ""),
            },
        )
        return access_token

    async def _get_access_token(self, *, force_refresh: bool = False) -> str:
        credentials = self._raw_credentials()
        access_token = str(credentials.get("accessToken") or "").strip()
        expires_at = _parse_expiry(credentials.get("expiresAt"))
        if access_token and not force_refresh and _token_is_fresh(expires_at):
            return access_token
        if not credentials.get("refreshToken"):
            if access_token:
                return access_token
            raise KiroOAuthError("Sessao Kiro nao encontrada. Abra o Kiro externo e faca login.")
        return await self._refresh_access_token(credentials)

    def _flatten_messages(self, messages: list[dict[str, Any]], *, response_json: bool = False) -> tuple[str, list[dict[str, Any]]]:
        blocks: list[str] = []
        images: list[dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "user").strip().upper()
            content = str(message.get("content") or "").strip()
            parts = message.get("parts") if isinstance(message.get("parts"), list) else []
            text_parts: list[str] = []
            if content:
                text_parts.append(content)
            for part in parts:
                if not isinstance(part, dict):
                    continue
                part_type = str(part.get("type") or "text").strip().lower()
                if part_type == "text":
                    text = str(part.get("text") or "").strip()
                    if text:
                        text_parts.append(text)
                    continue
                if part_type == "image":
                    images.append(_image_part_from_prompt_part(part))
                    continue
                if part_type in {"video", "audio", "file"}:
                    raise KiroOAuthError(
                        "Kiro OAuth v1 nao reconhece video/audio/file bruto neste runtime. "
                        "Os payloads brutos testados foram rejeitados ou ignorados pelo endpoint Kiro."
                    )
            if text_parts:
                blocks.append(f"[{role}]\n" + "\n\n".join(text_parts))

        if response_json:
            blocks.append(
                "[SYSTEM]\nResponda somente com JSON valido, sem markdown e sem texto fora do JSON."
            )
        prompt = "\n\n".join(blocks).strip()
        if not prompt:
            raise KiroOAuthError("Kiro request did not contain prompt text.")
        return prompt, images

    def _request_payload(
        self,
        prompt: str,
        images: list[dict[str, Any]],
        model: str,
        profile_arn: str | None,
        effort: str | None = None,
    ) -> dict[str, Any]:
        user_input: dict[str, Any] = {
            "content": prompt,
            "modelId": model,
            "origin": KIRO_ORIGIN,
            "userInputMessageContext": {"tools": []},
        }
        if images:
            user_input["images"] = images

        payload: dict[str, Any] = {
            "conversationState": {
                "agentContinuationId": str(uuid4()),
                "agentTaskType": KIRO_AGENT_MODE,
                "chatTriggerType": "MANUAL",
                "conversationId": str(uuid4()),
                "currentMessage": {"userInputMessage": user_input},
                "history": [],
            },
        }
        if profile_arn:
            payload["profileArn"] = profile_arn
        additional_model_request_fields = _additional_model_request_fields(effort)
        if additional_model_request_fields:
            payload["additionalModelRequestFields"] = additional_model_request_fields
        return payload

    def _debug_request(
        self,
        *,
        endpoint: str,
        model: str,
        selected_model: str,
        effort: str | None,
        messages: list[dict[str, Any]],
        prompt: str,
        images: list[dict[str, Any]],
        profile_arn: str | None,
    ) -> dict[str, Any]:
        return {
            "endpoint": endpoint,
            "model": model,
            "selectedModel": selected_model,
            "effort": effort,
            "messages": serialize_messages_for_ai_debug(messages),
            "promptPreview": prompt[:2000],
            "imageCount": len(images),
            "images": [
                {
                    "format": image.get("format"),
                    "mimeType": image.get("mimeType"),
                    "source": "bytes" if isinstance(image.get("source"), dict) and image["source"].get("bytes") else "url",
                    "base64Length": len(str((image.get("source") or {}).get("bytes") or "")),
                }
                for image in images
            ],
            "profileArnConfigured": bool(profile_arn),
        }

    async def _post_generate(
        self,
        *,
        access_token: str,
        endpoint: str,
        payload: dict[str, Any],
    ) -> tuple[int, bytes, Any]:
        async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=10.0)) as client:
            async with client.stream(
                "POST",
                f"{endpoint}/generateAssistantResponse",
                headers=self._headers(access_token),
                json=payload,
            ) as response:
                chunks = [chunk async for chunk in response.aiter_bytes()]
                raw = b"".join(chunks)
                response_payload: Any
                if response.headers.get("content-type", "").startswith("application/json"):
                    try:
                        response_payload = json.loads(raw.decode("utf-8"))
                    except Exception:
                        response_payload = {"text": raw.decode("utf-8", errors="replace")}
                else:
                    response_payload = raw
                return response.status_code, raw, response_payload

    async def generate_text_from_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        response_json: bool = False,
        temperature: float | None = None,
        debug_context: dict[str, Any] | None = None,
    ) -> str:
        del temperature
        selected_model = _normalize_kiro_model(model or self._default_model())
        resolved_model, effort = _split_kiro_model_and_effort(selected_model)
        credentials = self._raw_credentials()
        region = self._region(credentials)
        endpoint = self._runtime_endpoint(region)
        profile_arn = str(credentials.get("profileArn") or "").strip()
        prompt, images = self._flatten_messages(messages, response_json=response_json)
        payload = self._request_payload(prompt, images, resolved_model, profile_arn, effort)
        debug_request = self._debug_request(
            endpoint=endpoint,
            model=resolved_model,
            selected_model=selected_model,
            effort=effort,
            messages=messages,
            prompt=prompt,
            images=images,
            profile_arn=profile_arn,
        )

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                access_token = await self._get_access_token(force_refresh=attempt == 1)
                status_code, raw_response, response_payload = await self._post_generate(
                    access_token=access_token,
                    endpoint=endpoint,
                    payload=payload,
                )
                if status_code in {401, 403} and attempt == 0:
                    continue
                if status_code == 403 and profile_arn and attempt == 1:
                    payload.pop("profileArn", None)
                    debug_request["profileArnRetriedWithout"] = True
                    continue
                if status_code >= 400:
                    message = _extract_error_message(response_payload)
                    raise KiroOAuthError(message or f"Kiro HTTP {status_code}")

                text, events = _extract_text_from_eventstream(raw_response)
                write_ai_debug_exchange(
                    provider="kiro",
                    model=selected_model,
                    operation="kiro.generate_assistant_response",
                    request=debug_request,
                    response={
                        "text": text,
                        "eventSummary": _summarize_kiro_events(events),
                    },
                    metadata=debug_context,
                )
                if not text:
                    raise KiroOAuthError("Kiro retornou resposta vazia.")
                return text
            except KiroOAuthError as exc:
                last_error = exc
                break
            except Exception as exc:
                last_error = exc
                break

        write_ai_debug_exchange(
            provider="kiro",
            model=selected_model,
            operation="kiro.generate_assistant_response",
            request=debug_request,
            error=last_error,
            metadata=debug_context,
        )
        raise KiroOAuthError(str(last_error or "Falha desconhecida no Kiro."))

    async def generate_text(self, prompt: str, *, model: str | None = None, response_json: bool = False) -> str:
        return await self.generate_text_from_messages(
            [{"role": "user", "content": prompt}],
            model=model,
            response_json=response_json,
        )

    async def generate_native(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
    ) -> dict[str, Any]:
        selected_model = _normalize_kiro_model(model or self._default_model())
        resolved_model, effort = _split_kiro_model_and_effort(selected_model)
        credentials = self._raw_credentials()
        region = self._region(credentials)
        endpoint = self._runtime_endpoint(region)
        profile_arn = str(credentials.get("profileArn") or "").strip()
        payload: dict[str, Any] = {
            "conversationState": _native_conversation_state(
                messages,
                model=resolved_model,
                tools=tools or [],
                tool_choice=tool_choice,
            )
        }
        if profile_arn:
            payload["profileArn"] = profile_arn
        additional_fields = _additional_model_request_fields(effort)
        if additional_fields:
            payload["additionalModelRequestFields"] = additional_fields

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                access_token = await self._get_access_token(force_refresh=attempt == 1)
                status_code, raw_response, response_payload = await self._post_generate(
                    access_token=access_token,
                    endpoint=endpoint,
                    payload=payload,
                )
                if status_code in {401, 403} and attempt == 0:
                    continue
                if status_code == 403 and payload.get("profileArn") and attempt == 1:
                    payload.pop("profileArn", None)
                    continue
                if status_code >= 400:
                    raise KiroOAuthError(
                        _extract_error_message(response_payload)
                        or f"Kiro HTTP {status_code}"
                    )
                text, events = _extract_text_from_eventstream(raw_response)
                usage: dict[str, int] = {}
                for event in events:
                    raw_usage = event.get("usageEvent") or event.get("usage")
                    if not isinstance(raw_usage, dict):
                        continue
                    input_tokens = int(raw_usage.get("inputTokens") or 0)
                    output_tokens = int(raw_usage.get("outputTokens") or 0)
                    usage = {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                    }
                return {
                    "text": text,
                    "events": events,
                    "usage": usage,
                    "effectiveModel": resolved_model,
                }
            except KiroOAuthError as exc:
                last_error = exc
                break
            except Exception as exc:
                last_error = exc
                break
        raise KiroOAuthError(str(last_error or "Unknown Kiro native generation failure."))

    async def generate_json_from_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        debug_context: dict[str, Any] | None = None,
    ) -> Any:
        text = await self.generate_text_from_messages(
            messages,
            model=model,
            response_json=True,
            debug_context=debug_context,
        )
        try:
            return extract_json(text)
        except StructuredOutputError as exc:
            raise KiroOAuthError(str(exc)) from exc

    async def validate_credentials(self) -> dict[str, Any]:
        if not self.has_auth_config():
            return {
                "success": False,
                "isLoggedIn": False,
                "canGenerate": False,
                "error": "Nenhuma sessao Kiro local configurada.",
            }
        try:
            text = await self.generate_text(
                "Responda exatamente: ok",
                model=self._default_model(),
            )
            credentials = self._raw_credentials()
            return {
                "success": True,
                "isLoggedIn": True,
                "canGenerate": True,
                "validationModel": self._default_model(),
                "effectiveModel": self._default_model(),
                "accountStatus": credentials.get("authMethod") or "authenticated",
                "validationText": text[:180],
                "userId": credentials.get("provider") or "",
                "userIdSource": "kiro-token-cache",
                "userIdVerified": bool(credentials.get("provider")),
                "profileArnConfigured": bool(credentials.get("profileArn")),
                "region": self._region(credentials),
            }
        except Exception as exc:
            return {
                "success": False,
                "isLoggedIn": False,
                "canGenerate": False,
                "validationModel": self._default_model(),
                "effectiveModel": self._default_model(),
                "error": str(exc),
                "region": self._region(),
            }

    async def send_message(self, prompt: str, *, model: str | None = None) -> dict[str, Any]:
        text = await self.generate_text(prompt, model=model)
        return {
            "success": True,
            "provider": "kiro",
            "text": text,
            "response": text,
            "modelUsed": _normalize_kiro_model(model or self._default_model()),
        }

    async def stream_message(self, prompt: str, *, model: str | None = None):
        yield await self.generate_text(prompt, model=model)

    async def list_models(self) -> dict[str, Any]:
        if not self.has_auth_config():
            return {
                "success": False,
                "isLoggedIn": False,
                "canGenerate": False,
                "error": "Nenhuma sessao Kiro local configurada.",
                "models": [],
                "defaultModel": self._default_model(),
            }

        credentials = self._raw_credentials()
        region = self._region(credentials)
        profile_arn = str(credentials.get("profileArn") or "").strip()
        last_error = ""

        for token_attempt in range(2):
            access_token = await self._get_access_token(force_refresh=token_attempt == 1)
            for endpoint in self._model_endpoints(region):
                for include_profile in ([True, False] if profile_arn else [False]):
                    params = {"origin": KIRO_ORIGIN}
                    if include_profile and profile_arn:
                        params["profileArn"] = profile_arn
                    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                        response = await client.get(
                            f"{endpoint}/ListAvailableModels",
                            headers=self._headers(access_token),
                            params=params,
                        )
                    try:
                        payload = response.json()
                    except Exception:
                        payload = {"text": response.text}
                    if response.status_code in {401, 403} and token_attempt == 0:
                        last_error = _extract_error_message(payload)
                        break
                    if response.status_code >= 400:
                        last_error = _extract_error_message(payload) or f"Kiro models HTTP {response.status_code}"
                        continue

                    raw_models = payload.get("models") if isinstance(payload, dict) else None
                    raw_model_list = raw_models if isinstance(raw_models, list) else []
                    models = [
                        option
                        for index, item in enumerate(raw_model_list)
                        if isinstance(item, dict)
                        for option in _serialize_model_options(item, index)
                        if option
                    ]
                    default_model = DEFAULT_KIRO_MODEL
                    if isinstance(payload, dict) and isinstance(payload.get("defaultModel"), dict):
                        default_model = str(payload["defaultModel"].get("modelId") or default_model).strip()
                    default_model = _expanded_default_model(default_model, raw_model_list)
                    default_model = _fallback_default_model(default_model, models)
                    return {
                        "success": True,
                        "isLoggedIn": True,
                        "canGenerate": True,
                        "provider": "kiro",
                        "models": models,
                        "defaultModel": default_model or self._default_model(),
                        "endpoint": endpoint,
                    }

        return {
            "success": False,
            "isLoggedIn": True,
            "canGenerate": False,
            "error": last_error or "Nao foi possivel listar modelos Kiro.",
            "models": [],
            "defaultModel": self._default_model(),
        }


kiro_oauth_service = KiroOAuthService()
