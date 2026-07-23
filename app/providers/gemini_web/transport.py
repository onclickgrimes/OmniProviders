from __future__ import annotations

import asyncio
import os
import json
import logging
import hashlib
import mimetypes
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator

try:
    from gemini_webapi import GeminiClient, set_log_level
    from gemini_webapi.exceptions import GeminiError
except Exception:  # pragma: no cover - dependency is validated in runtime env.
    GeminiClient = None  # type: ignore[assignment]
    set_log_level = None  # type: ignore[assignment]
    GeminiError = Exception  # type: ignore[assignment]

from app import config
from app.infrastructure.debug import (
    serialize_messages_for_ai_debug,
    write_ai_debug_exchange,
)
from app.media.input import load_attachment_bytes
from app.persistence.credentials import get_env_or_credential
from app.protocols.structured_output import extract_json


DEFAULT_TIMEOUT = 450
DEFAULT_CLOSE_DELAY = 300
DEFAULT_VALIDATION_TIMEOUT = 90
DEFAULT_MODEL = "gemini-3-pro"
LEGACY_AUTO_MODEL = "gemini-web-auto"
DEFAULT_SDK_LOG_LEVEL = "ERROR"
logger = logging.getLogger(__name__)


def _flatten_messages(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("type") in {"function_call", "function_call_output"}:
            continue
        role = str(message.get("role") or "user").strip().upper()
        content = str(message.get("content") or "").strip()
        if not content:
            content = "\n".join(
                str(part.get("text") or "")
                for part in message.get("parts") or []
                if isinstance(part, dict) and part.get("type") == "text"
            ).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n\n".join(lines)


def _attachment_sources(messages: list[dict[str, Any]]) -> list[Any]:
    sources: list[Any] = []
    for message in messages:
        for part in message.get("parts") or [] if isinstance(message, dict) else []:
            if isinstance(part, dict) and part.get("type") in {"image", "video", "audio", "file"}:
                sources.append(part)
    return sources


class GeminiScrapingError(RuntimeError):
    pass


def _first_env(*names: str) -> str | None:
    return get_env_or_credential(*names)


def _stringify_model(model: Any) -> str | None:
    if not model:
        return None
    value = str(model).strip()
    if not value or value == LEGACY_AUTO_MODEL:
        return None
    return value


def _display_model(model: Any) -> str:
    return _stringify_model(model) or DEFAULT_MODEL


def _effective_model(model: Any) -> str:
    return _stringify_model(model) or DEFAULT_MODEL


def _account_status_info(client: Any) -> dict[str, str | None]:
    status = getattr(client, "account_status", None)
    if status is None:
        return {"name": None, "description": None}

    name = getattr(status, "name", None)
    if not name:
        name = str(status).split(".")[-1] if str(status) else None

    description = getattr(status, "description", None)
    if description is not None:
        description = str(description)

    return {"name": str(name) if name else None, "description": description}


def _is_authenticated_status(account_status: dict[str, str | None]) -> bool:
    name = account_status.get("name")
    return not name or name == "AVAILABLE"


def _account_status_error(account_status: dict[str, str | None]) -> str:
    name = account_status.get("name") or "UNKNOWN"
    description = account_status.get("description")
    if name == "UNAUTHENTICATED":
        return (
            "Gemini Web session is not authenticated or cookies have expired. "
            "Update GEMINI_WEB_SECURE_1PSID/GEMINI_WEB_SECURE_1PSIDTS."
        )
    if description:
        return f"Gemini Web account status is {name}: {description}"
    return f"Gemini Web account status is {name}."


def _safe_session_fingerprint(secure_1psid: str | None) -> str | None:
    if not secure_1psid:
        return None
    digest = hashlib.sha256(secure_1psid.encode("utf-8")).hexdigest()
    return f"gemini:{digest[:16]}"


def _env_float(name: str, default: float, *, minimum: float | None = None) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    if minimum is not None:
        value = max(value, minimum)
    return value


def _bool_from_value(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on", "sim"}:
        return True
    if normalized in {"0", "false", "no", "n", "off", "nao", "não"}:
        return False
    return default


def _build_image_prompt(prompt: str, count: int, aspect_ratio: str | None) -> str:
    details = [
        "Generate an original image using Gemini's image generation tool.",
        "Return generated image output, not a web image search result.",
        f"Prompt: {prompt}",
    ]
    if aspect_ratio:
        details.append(f"Aspect ratio: {aspect_ratio}.")
    if count > 1:
        details.append(f"Create option {count} as a distinct variation.")
    return "\n".join(details)


def _build_video_prompt(prompt: str, duration_seconds: int | None) -> str:
    details = [
        "Generate a short video using Gemini's video generation tool.",
        f"Prompt: {prompt}",
    ]
    if duration_seconds:
        details.append(f"Target duration: {duration_seconds} seconds.")
    return "\n".join(details)


class GeminiScrapingService:
    def __init__(self, *, client: Any | None = None) -> None:
        self._provided_client = client
        self._client: Any | None = client
        self._client_identity: tuple[str | None, str | None, str | None] | None = None
        self.cookie_cache_dir = config.DATA_DIR / "gemini-webapi"
        self.cookie_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("GEMINI_COOKIE_PATH", str(self.cookie_cache_dir))
        self._configure_sdk_logging()

    def _configure_sdk_logging(self) -> None:
        if not set_log_level:
            return
        level = str(os.environ.get("GEMINI_WEB_LOG_LEVEL") or DEFAULT_SDK_LOG_LEVEL).strip() or DEFAULT_SDK_LOG_LEVEL
        try:
            set_log_level(level)
        except Exception:
            logger.debug("[GeminiWeb] failed to configure gemini-webapi log level", exc_info=True)

    def _dependency_available(self) -> bool:
        return GeminiClient is not None

    def _cookie_values(self) -> tuple[str | None, str | None]:
        secure_1psid = _first_env(
            "GEMINI_WEB_SECURE_1PSID",
            "GEMINI_SECURE_1PSID",
            "SECURE_1PSID",
            "__Secure-1PSID",
        )
        secure_1psidts = _first_env(
            "GEMINI_WEB_SECURE_1PSIDTS",
            "GEMINI_SECURE_1PSIDTS",
            "SECURE_1PSIDTS",
            "__Secure-1PSIDTS",
        )
        return secure_1psid, secure_1psidts

    def _proxy(self) -> str | None:
        return _first_env("GEMINI_WEB_PROXY", "HTTPS_PROXY", "HTTP_PROXY")

    def _default_temporary(self) -> bool:
        return _bool_from_value(
            _first_env("GEMINI_WEB_TEMPORARY", "GEMINI_SCRAPING_TEMPORARY"),
            True,
        )

    def _resolve_temporary(self, value: Any = None) -> bool:
        return _bool_from_value(value, self._default_temporary())

    def _identity(self) -> tuple[str | None, str | None, str | None]:
        secure_1psid, secure_1psidts = self._cookie_values()
        return secure_1psid, secure_1psidts, self._proxy()

    def _sync_cookie_cache(self, secure_1psid: str, secure_1psidts: str | None) -> None:
        if not secure_1psidts:
            return

        cache_path = self.cookie_cache_dir / f".cached_cookies_{secure_1psid}.json"
        cookies = [
            {
                "name": "__Secure-1PSID",
                "value": secure_1psid,
                "domain": ".google.com",
                "path": "/",
                "expires": None,
            },
            {
                "name": "__Secure-1PSIDTS",
                "value": secure_1psidts,
                "domain": ".google.com",
                "path": "/",
                "expires": None,
            },
        ]
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cookies), encoding="utf-8")
            cache_path.chmod(0o600)
        except Exception:
            logger.debug("[GeminiWeb] failed to sync explicit cookies into cache", exc_info=True)

    def _session_identity_info(self, verified: bool) -> dict[str, Any]:
        secure_1psid, _secure_1psidts = self._cookie_values()
        fingerprint = _safe_session_fingerprint(secure_1psid)
        if not fingerprint:
            return {}
        return {
            "userId": fingerprint,
            "userIdVerified": verified,
            "userIdSource": "__Secure-1PSID sha256",
        }

    def _missing_auth_message(self) -> str:
        return (
            "Set GEMINI_WEB_SECURE_1PSID and optionally "
            "GEMINI_WEB_SECURE_1PSIDTS with cookies from gemini.google.com."
        )

    async def _ensure_client(self) -> Any:
        if self._provided_client is not None:
            return self._provided_client
        if not self._dependency_available():
            raise GeminiScrapingError(
                "gemini-webapi is not installed. Run pip install -r core/requirements.txt."
            )

        secure_1psid, secure_1psidts, proxy = self._identity()
        identity = (secure_1psid, secure_1psidts, proxy)
        if self._client and self._client_identity == identity:
            return self._client

        if self._client:
            await self.close()

        if not secure_1psid:
            raise GeminiScrapingError(self._missing_auth_message())

        client = None
        try:
            self._sync_cookie_cache(secure_1psid, secure_1psidts)
            client = GeminiClient(secure_1psid, secure_1psidts, proxy=proxy)
            await client.init(
                timeout=_env_float("GEMINI_WEB_TIMEOUT", DEFAULT_TIMEOUT, minimum=10),
                auto_close=True,
                close_delay=_env_float("GEMINI_WEB_CLOSE_DELAY", DEFAULT_CLOSE_DELAY, minimum=10),
                auto_refresh=True,
            )
        except asyncio.CancelledError:
            if client:
                try:
                    await client.close()
                except Exception:
                    logger.debug("[GeminiWeb] failed to close cancelled client", exc_info=True)
            raise
        except ValueError as exc:
            raise GeminiScrapingError(self._missing_auth_message()) from exc
        except GeminiError as exc:
            raise GeminiScrapingError(f"Gemini Web error: {exc}") from exc
        except Exception as exc:
            raise GeminiScrapingError(f"Gemini Web SDK error: {exc}") from exc

        account_status = _account_status_info(client)
        if _is_authenticated_status(account_status):
            logger.info(
                "[GeminiWeb] client initialized accountStatus=%s",
                account_status.get("name") or "UNKNOWN",
            )
        else:
            logger.info(
                "[GeminiWeb] client initialized but accountStatus=%s description=%s",
                account_status.get("name") or "UNKNOWN",
                account_status.get("description") or "",
            )

        self._client = client
        self._client_identity = identity
        return client

    def _client_account_status(self, client: Any) -> dict[str, str | None]:
        account_status = _account_status_info(client)
        if not _is_authenticated_status(account_status):
            logger.debug(
                "[GeminiWeb] accountStatus=%s description=%s; continuing request because gemini-webapi can still generate after this warning.",
                account_status.get("name") or "UNKNOWN",
                account_status.get("description") or "",
            )
        return account_status

    async def close(self) -> None:
        if self._client and self._provided_client is None:
            await self._client.close()
        self._client = self._provided_client
        self._client_identity = None

    def has_auth_config(self) -> bool:
        secure_1psid, _secure_1psidts = self._cookie_values()
        return bool(secure_1psid)

    async def check_login(self) -> dict[str, Any]:
        if not self._dependency_available():
            return {
                "success": False,
                "isLoggedIn": False,
                "error": "gemini-webapi is not installed.",
            }
        try:
            client = await self._ensure_client()
            models = self._serialize_models(client.list_models() or [])
            account_status = _account_status_info(client)
            is_account_available = _is_authenticated_status(account_status)
            result = {
                "success": is_account_available,
                "isLoggedIn": is_account_available,
                "isAccountAvailable": is_account_available,
                "models": models if is_account_available else [],
                "accountStatus": account_status.get("name"),
                "accountStatusDescription": account_status.get("description"),
            }
            if is_account_available:
                result.update(self._session_identity_info(is_account_available))
            if not is_account_available:
                result["error"] = _account_status_error(account_status)
            return result
        except GeminiScrapingError as exc:
            return {"success": False, "isLoggedIn": False, "error": str(exc)}

    async def validate_login(self) -> dict[str, Any]:
        timeout = _env_float(
            "GEMINI_WEB_VALIDATION_TIMEOUT",
            DEFAULT_VALIDATION_TIMEOUT,
            minimum=15,
        )
        try:
            return await asyncio.wait_for(self._validate_login_once(), timeout=timeout)
        except asyncio.TimeoutError:
            await self.close()
            return {
                "success": False,
                "isLoggedIn": False,
                "canGenerate": False,
                "validationModel": DEFAULT_MODEL,
                "error": f"Gemini Web validation timed out after {timeout:g}s.",
            }

    async def _validate_login_once(self) -> dict[str, Any]:
        if not self._dependency_available():
            return {
                "success": False,
                "isLoggedIn": False,
                "canGenerate": False,
                "validationModel": DEFAULT_MODEL,
                "error": "gemini-webapi is not installed.",
            }
        if not self.has_auth_config():
            return {
                "success": False,
                "isLoggedIn": False,
                "canGenerate": False,
                "validationModel": DEFAULT_MODEL,
                "error": self._missing_auth_message(),
            }

        await self.close()

        try:
            result = await self.send_message(
                "Responda apenas: OK",
                model=DEFAULT_MODEL,
                temporary=True,
            )
        except GeminiScrapingError as exc:
            return {
                "success": False,
                "isLoggedIn": False,
                "canGenerate": False,
                "validationModel": DEFAULT_MODEL,
                "error": str(exc),
            }

        text = str(result.get("text") or result.get("response") or "").strip()
        can_generate = bool(text)
        result_account = {
            "name": result.get("accountStatus"),
            "description": result.get("accountStatusDescription"),
        }
        sdk_logged_in = _is_authenticated_status(result_account)
        response = {
            "success": can_generate,
            "isLoggedIn": sdk_logged_in,
            "isAccountAvailable": sdk_logged_in,
            "canGenerate": can_generate,
            "validationModel": DEFAULT_MODEL,
            "validationText": text[:120],
            "modelUsed": result.get("modelUsed"),
            "effectiveModel": result.get("effectiveModel"),
            "accountStatus": result_account.get("name"),
            "accountStatusDescription": result_account.get("description"),
            **({} if can_generate else {"error": "Gemini Web validation returned an empty response."}),
        }
        if can_generate or sdk_logged_in:
            response.update(self._session_identity_info(sdk_logged_in))
        if not sdk_logged_in and result_account.get("name"):
            response["warning"] = _account_status_error(result_account)
        return response

    def login_instructions(self) -> dict[str, Any]:
        return {
            "success": True,
            "isLoggedIn": False,
            "loginUrl": "https://gemini.google.com",
            "env": ["GEMINI_WEB_SECURE_1PSID", "GEMINI_WEB_SECURE_1PSIDTS"],
            "message": self._missing_auth_message(),
        }

    def _resolve_files(self, values: Any) -> list[Path]:
        files: list[Path] = []
        if not values:
            return files
        items = values if isinstance(values, list) else [values]
        for item in items:
            if not item:
                continue
            files.append(self._materialize_file(item))
        return files

    def _materialize_file(self, value: Any) -> Path:
        if isinstance(value, (str, Path)):
            candidate = Path(str(value)).expanduser()
            if candidate.is_file():
                return candidate.resolve()
        data, mime_type = load_attachment_bytes(value)
        suffix = mimetypes.guess_extension(mime_type) or ".bin"
        target_dir = config.ARTIFACTS_DIR / "inputs"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"input_{uuid.uuid4().hex}{suffix}"
        target.write_bytes(data)
        return target

    def _serialize_models(self, models: list[Any]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for model in models:
            result.append(
                {
                    "modelName": getattr(model, "model_name", None),
                    "displayName": getattr(model, "display_name", None),
                    "description": getattr(model, "description", None),
                    "available": getattr(model, "is_available", True),
                }
            )
        return result

    def _serialize_output(
        self,
        output: Any,
        *,
        requested_model: Any = None,
        effective_model: str | None = None,
        account_status: dict[str, str | None] | None = None,
    ) -> dict[str, Any]:
        return {
            "success": True,
            "provider": "gemini-scraping",
            "modelUsed": _display_model(requested_model),
            "requestedModel": _display_model(requested_model),
            "effectiveModel": effective_model or _effective_model(requested_model),
            "accountStatus": (account_status or {}).get("name"),
            "accountStatusDescription": (account_status or {}).get("description"),
            "response": output.text,
            "text": output.text,
            "metadata": output.metadata,
            "rcid": output.rcid,
            "images": [
                {
                    "url": getattr(image, "url", None),
                    "title": getattr(image, "title", None),
                    "alt": getattr(image, "alt", None),
                    "kind": type(image).__name__,
                }
                for image in output.images
            ],
            "videos": [
                {
                    "url": getattr(video, "url", None),
                    "title": getattr(video, "title", None),
                    "kind": type(video).__name__,
                }
                for video in output.videos
            ],
            "media": [
                {
                    "url": getattr(media, "url", None),
                    "title": getattr(media, "title", None),
                    "kind": type(media).__name__,
                }
                for media in output.media
            ],
        }

    async def send_message(
        self,
        message: str,
        *,
        model: str | None = None,
        files: Any = None,
        temporary: bool | None = None,
    ) -> dict[str, Any]:
        client = await self._ensure_client()
        account_status = self._client_account_status(client)
        resolved_files = self._resolve_files(files) or None
        effective_model = _effective_model(model)
        effective_temporary = self._resolve_temporary(temporary)
        logger.info(
            "[GeminiWeb] send_message start model=%s effectiveModel=%s temporary=%s files=%d promptChars=%d accountStatus=%s",
            _display_model(model),
            effective_model,
            effective_temporary,
            len(resolved_files or []),
            len(message or ""),
            account_status.get("name") or "UNKNOWN",
        )
        try:
            output = await client.generate_content(
                message,
                files=resolved_files,
                model=effective_model,
                temporary=effective_temporary,
            )
            logger.info(
                "[GeminiWeb] send_message complete model=%s effectiveModel=%s textChars=%d images=%d videos=%d media=%d rcid=%s",
                _display_model(model),
                effective_model,
                len(getattr(output, "text", "") or ""),
                len(getattr(output, "images", []) or []),
                len(getattr(output, "videos", []) or []),
                len(getattr(output, "media", []) or []),
                getattr(output, "rcid", None),
            )
            return self._serialize_output(
                output,
                requested_model=model,
                effective_model=effective_model,
                account_status=account_status,
            )
        except GeminiError as exc:
            raise GeminiScrapingError(f"Gemini Web error: {exc}") from exc
        except Exception as exc:
            raise GeminiScrapingError(f"Gemini Web SDK error: {exc}") from exc

    async def stream_message(
        self,
        message: str,
        *,
        model: str | None = None,
        files: Any = None,
        temporary: bool | None = None,
    ) -> AsyncGenerator[str, None]:
        client = await self._ensure_client()
        account_status = self._client_account_status(client)
        resolved_files = self._resolve_files(files) or None
        effective_model = _effective_model(model)
        effective_temporary = self._resolve_temporary(temporary)
        logger.info(
            "[GeminiWeb] stream_message start model=%s effectiveModel=%s temporary=%s files=%d promptChars=%d accountStatus=%s",
            _display_model(model),
            effective_model,
            effective_temporary,
            len(resolved_files or []),
            len(message or ""),
            account_status.get("name") or "UNKNOWN",
        )
        chunks = 0
        text_chars = 0
        try:
            async for chunk in client.generate_content_stream(
                message,
                files=resolved_files,
                model=effective_model,
                temporary=effective_temporary,
            ):
                if chunk.text_delta:
                    chunks += 1
                    text_chars += len(chunk.text_delta)
                    yield chunk.text_delta
            logger.info(
                "[GeminiWeb] stream_message complete model=%s effectiveModel=%s chunks=%d textChars=%d",
                _display_model(model),
                effective_model,
                chunks,
                text_chars,
            )
        except GeminiError as exc:
            raise GeminiScrapingError(f"Gemini Web error: {exc}") from exc
        except Exception as exc:
            raise GeminiScrapingError(f"Gemini Web SDK error: {exc}") from exc

    async def generate_json(
        self, prompt: str, *, options: dict[str, Any] | None = None
    ) -> Any:
        output = await self.send_message(
            prompt,
            model=(options or {}).get("model"),
            temporary=(options or {}).get("temporary"),
        )
        return extract_json(str(output.get("text") or output.get("response") or ""))

    async def generate_text_from_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        response_json: bool = False,
        debug_context: dict[str, Any] | None = None,
    ) -> str:
        prompt = _flatten_messages(messages)
        if response_json:
            prompt = f"{prompt}\n\nReturn only valid JSON."
        files = _attachment_sources(messages)
        output = await self.send_message(prompt, model=model, files=files or None)
        text = str(output.get("text") or output.get("response") or "").strip()
        if not text:
            raise GeminiScrapingError("Gemini Web retornou resposta vazia.")
        return text

    async def generate_json_from_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        options: dict[str, Any] | None = None,
    ) -> Any:
        resolved_options = options or {}
        model = _effective_model(resolved_options.get("model"))
        operation = str(
            resolved_options.get("debugOperation")
            or "gemini_scraping.generate_json_from_messages"
        )
        debug_request = {
            "messages": serialize_messages_for_ai_debug(messages),
            "options": resolved_options,
        }
        try:
            text = await self.generate_text_from_messages(
                messages,
                model=model,
                response_json=True,
                debug_context=resolved_options,
            )
            result = extract_json(text)
        except Exception as exc:
            write_ai_debug_exchange(
                provider="gemini_scraping",
                model=model,
                operation=operation,
                request=debug_request,
                error=exc,
                metadata=resolved_options,
            )
            raise
        write_ai_debug_exchange(
            provider="gemini_scraping",
            model=model,
            operation=operation,
            request=debug_request,
            response=result,
            metadata=resolved_options,
        )
        return result

    async def generate_images(self, payload: dict[str, Any]) -> dict[str, Any]:
        prompt = str(payload.get("prompt") or "").strip()
        if not prompt:
            raise GeminiScrapingError("Image prompt is required.")

        count = max(1, min(int(payload.get("count") or 1), 4))
        aspect_ratio = str(payload.get("aspectRatio") or "").strip() or None
        files = self._resolve_files(payload.get("ingredientImagePaths"))
        client = await self._ensure_client()
        account_status = self._client_account_status(client)
        effective_model = _effective_model(payload.get("model"))
        effective_temporary = self._resolve_temporary(payload.get("temporary"))

        started_at = time.perf_counter()
        generated_media: list[Any] = []
        logger.info(
            "[GeminiWeb] generate_images start model=%s effectiveModel=%s count=%d files=%d aspectRatio=%s promptChars=%d accountStatus=%s",
            _display_model(payload.get("model")),
            effective_model,
            count,
            len(files),
            aspect_ratio or "",
            len(prompt),
            account_status.get("name") or "UNKNOWN",
        )
        try:
            for index in range(count):
                output = await client.generate_content(
                    _build_image_prompt(prompt, index + 1, aspect_ratio),
                    files=files or None,
                    model=effective_model,
                    temporary=effective_temporary,
                )
                generated_media.extend(output.images)
                if len(generated_media) >= count:
                    break
        except GeminiError as exc:
            raise GeminiScrapingError(f"Gemini Web error: {exc}") from exc
        except Exception as exc:
            raise GeminiScrapingError(f"Gemini Web SDK error: {exc}") from exc

        if not generated_media:
            raise GeminiScrapingError(
                "Gemini Web did not return generated images. Try a prompt that explicitly asks to generate an image."
            )

        duration_ms = int((time.perf_counter() - started_at) * 1000)
        logger.info(
            "[GeminiWeb] generate_images complete model=%s effectiveModel=%s generated=%d durationMs=%d",
            _display_model(payload.get("model")),
            effective_model,
            len(generated_media),
            duration_ms,
        )
        return {
            "success": True,
            "provider": "gemini-scraping",
            "modelUsed": _display_model(payload.get("model")),
            "requestedModel": _display_model(payload.get("model")),
            "effectiveModel": effective_model,
            "accountStatus": account_status.get("name"),
            "accountStatusDescription": account_status.get("description"),
            "media": generated_media[:count],
            "mediaClient": getattr(client, "client", None),
            "durationMs": duration_ms,
        }

    async def generate_video(self, payload: dict[str, Any]) -> dict[str, Any]:
        prompt = str(payload.get("prompt") or "").strip()
        if not prompt:
            raise GeminiScrapingError("Video prompt is required.")

        files = self._resolve_files(
            payload.get("ingredientImagePaths")
            or payload.get("referenceImagePaths")
            or payload.get("referenceImagePath")
        )
        duration_seconds = payload.get("durationSeconds")
        duration_seconds = int(duration_seconds) if duration_seconds else None
        client = await self._ensure_client()
        account_status = self._client_account_status(client)
        effective_model = _effective_model(payload.get("model"))
        effective_temporary = self._resolve_temporary(payload.get("temporary"))

        started_at = time.perf_counter()
        generated_media: list[Any] = []
        logger.info(
            "[GeminiWeb] generate_video start model=%s effectiveModel=%s files=%d durationSeconds=%s promptChars=%d accountStatus=%s",
            _display_model(payload.get("model")),
            effective_model,
            len(files),
            duration_seconds if duration_seconds is not None else "",
            len(prompt),
            account_status.get("name") or "UNKNOWN",
        )
        try:
            output = await client.generate_content(
                _build_video_prompt(prompt, duration_seconds),
                files=files or None,
                model=effective_model,
                temporary=effective_temporary,
            )
            generated_media.extend([*output.videos, *output.media])
        except GeminiError as exc:
            raise GeminiScrapingError(f"Gemini Web error: {exc}") from exc
        except Exception as exc:
            raise GeminiScrapingError(f"Gemini Web SDK error: {exc}") from exc

        if not generated_media:
            raise GeminiScrapingError(
                f"Gemini Web did not return generated video/media. Response text: {output.text[:500]}"
            )

        duration_ms = int((time.perf_counter() - started_at) * 1000)
        logger.info(
            "[GeminiWeb] generate_video complete model=%s effectiveModel=%s generated=%d durationMs=%d",
            _display_model(payload.get("model")),
            effective_model,
            len(generated_media),
            duration_ms,
        )
        return {
            "success": True,
            "provider": "gemini-scraping",
            "modelUsed": _display_model(payload.get("model")),
            "requestedModel": _display_model(payload.get("model")),
            "effectiveModel": effective_model,
            "accountStatus": account_status.get("name"),
            "accountStatusDescription": account_status.get("description"),
            "media": generated_media,
            "mediaClient": getattr(client, "client", None),
            "durationMs": duration_ms,
        }

gemini_scraping_service = GeminiScrapingService()
