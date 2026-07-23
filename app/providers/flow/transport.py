from __future__ import annotations

import asyncio
import base64
import contextvars
import hashlib
import json
import os
import random
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from curl_cffi.requests import AsyncSession

from app import config
from app.media.input import load_attachment_bytes
from app.persistence.credentials import (
    get_env_or_credential,
    list_flow_cookie_runtime_accounts,
    update_flow_user_account_session,
)
from app.providers.flow.browser_captcha import (
    FlowBrowserCaptchaError,
    flow_browser_captcha_service,
)


def data_dir() -> Path:
    return config.DATA_DIR


FLOW_LABS_BASE_URL = "https://labs.google/fx/api"
FLOW_API_BASE_URL = "https://aisandbox-pa.googleapis.com/v1"
FLOW_RECAPTCHA_SITE_KEY = "6LdsFiUsAAAAAIjVDZcuLhaHiDn5nnHVXVRQGeMV"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36"
)
FLOW_IMAGE_MODEL_NANO_BANANA_2 = "NARWHAL"
FLOW_IMAGE_MODEL_NANO_BANANA_PRO = "GEM_PIX_2"
FLOW_IMAGE_PRO_FALLBACK_SECONDS = 6 * 60 * 60
FLOW_IMAGE_API_REQUIRED_ERROR_CODE = "FLOW_IMAGE_FALLBACK_DAILY_QUOTA_REACHED"
FLOW_CAPTCHA_SOLVERS: dict[str, tuple[str, ...]] = {
    "yescaptcha": ("FLOW_YESCAPTCHA_API_KEY", "YESCAPTCHA_API_KEY"),
    "capmonster": ("FLOW_CAPMONSTER_API_KEY", "CAPMONSTER_API_KEY"),
    "ezcaptcha": ("FLOW_EZCAPTCHA_API_KEY", "EZCAPTCHA_API_KEY"),
    "capsolver": ("FLOW_CAPSOLVER_API_KEY", "CAPSOLVER_API_KEY"),
}
FLOW_BROWSER_CAPTCHA_METHODS = frozenset({"browser", "personal"})


class FlowScrapingError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.metadata = metadata or {}


@dataclass(frozen=True)
class FlowVideoPlan:
    video_type: str
    model_key: str
    aspect_ratio: str
    image_paths: list[str]
    use_v2_model_config: bool = False


@dataclass(frozen=True)
class FlowAccount:
    key: str
    label: str
    id: str | None = None
    session_token: str | None = None
    access_token: str | None = None
    project_id: str | None = None
    cookie_header: str | None = None
    source_field: str | None = None


def _first_env(*names: str) -> str | None:
    return get_env_or_credential(*names)


def _strip_cookie_value(value: str, cookie_name: str) -> str:
    value = value.strip().strip(";")
    marker = f"{cookie_name}="
    if marker not in value:
        return value
    for part in value.split(";"):
        key, _, cookie_value = part.strip().partition("=")
        if key == cookie_name and cookie_value:
            return cookie_value.strip()
    return value


def _normalize_count(value: Any, *, limit: int = 4) -> int:
    try:
        count = int(value or 1)
    except (TypeError, ValueError):
        count = 1
    return max(1, min(count, limit))


def _normalize_ratio(value: Any, *, default: str = "9:16") -> str:
    candidate = str(value or "").strip().lower().replace(" ", "")
    aliases = {
        "16:9": "16:9",
        "landscape": "16:9",
        "horizontal": "16:9",
        "9:16": "9:16",
        "portrait": "9:16",
        "vertical": "9:16",
        "1:1": "1:1",
        "square": "1:1",
        "4:3": "4:3",
        "3:4": "3:4",
    }
    return aliases.get(candidate, default)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    candidate = str(value).strip().lower()
    if candidate in {"1", "true", "yes", "on"}:
        return True
    if candidate in {"0", "false", "no", "off"}:
        return False
    return None


def _image_aspect_enum(value: Any) -> str:
    ratio = _normalize_ratio(value, default="16:9")
    return {
        "16:9": "IMAGE_ASPECT_RATIO_LANDSCAPE",
        "9:16": "IMAGE_ASPECT_RATIO_PORTRAIT",
        "1:1": "IMAGE_ASPECT_RATIO_SQUARE",
        "4:3": "IMAGE_ASPECT_RATIO_LANDSCAPE_FOUR_THREE",
        "3:4": "IMAGE_ASPECT_RATIO_PORTRAIT_THREE_FOUR",
    }.get(ratio, "IMAGE_ASPECT_RATIO_LANDSCAPE")


def _video_aspect_enum(value: Any) -> str:
    ratio = _normalize_ratio(value, default="9:16")
    return "VIDEO_ASPECT_RATIO_LANDSCAPE" if ratio == "16:9" else "VIDEO_ASPECT_RATIO_PORTRAIT"


def _mime_ext(content_type: str, fallback: str = ".bin") -> str:
    content_type = (content_type or "").split(";", 1)[0].strip().lower()
    return {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/gif": ".gif",
        "video/mp4": ".mp4",
        "video/webm": ".webm",
    }.get(content_type, fallback)


def _detect_image_mime(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\x89PNG"):
        return "image/png"
    if image_bytes[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    if image_bytes[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return "image/jpeg"


class FlowScrapingService:
    """Direct Flow client extracted from flow2api service logic.

    This keeps only authentication, project creation, media upload, generation,
    polling and captcha API integration. It intentionally does not bring the
    flow2api admin panel, DB, token pool, Docker setup or OpenAI-compatible API.
    """

    def __init__(self, *, browser_captcha: Any | None = None) -> None:
        self._browser_captcha = browser_captcha or flow_browser_captcha_service
        self._access_token: str | None = None
        self._access_expires_at: float = 0
        self._project_id: str | None = None
        self._cancel_requested = False
        self._account_access_cache: dict[str, tuple[str, float]] = {}
        self._account_session_overrides: dict[str, str] = {}
        self._account_project_cache: dict[str, str] = {}
        self._account_project_pools: dict[str, list[str]] = {}
        self._account_project_rr: dict[str, int] = {}
        self._account_inflight: dict[str, dict[str, int]] = {}
        self._account_access_locks: dict[str, asyncio.Lock] = {}
        self._access_token_account_keys: dict[str, str] = {}
        self._access_lock_guard = asyncio.Lock()
        self._account_lock = asyncio.Lock()
        self._project_lock = asyncio.Lock()
        self._round_robin_index = 0
        self._user_agent_cache: dict[str, str] = {}
        self._request_fingerprint_ctx: contextvars.ContextVar[dict[str, Any] | None] = (
            contextvars.ContextVar("flow_request_fingerprint", default=None)
        )

    def has_auth_config(self) -> bool:
        return bool(self._flow_accounts())

    def login_instructions(self) -> dict[str, Any]:
        return {
            "success": True,
            "isLoggedIn": False,
            "loginUrl": "https://labs.google/fx",
            "env": [
                "FLOW_SESSION_TOKEN",
                "FLOW_PROJECT_ID",
                "FLOW_CAPTCHA_METHOD",
                "FLOW_YESCAPTCHA_API_KEY",
            ],
            "message": (
                "Abra o login do Flow para autenticar no navegador persistente do "
                "OmniProviders, ou configure uma conta por cookies. O captcha pode usar "
                "personal/browser ou um solver externo."
            ),
            "browser": self._browser_captcha.status(),
        }

    async def check_login(self, *, validate: bool = False) -> dict[str, Any]:
        if not self.has_auth_config():
            return self.login_instructions() | {
                "canGenerate": False,
                "captchaConfigured": self.captcha_configuration()["configured"],
            }
        try:
            credits = await self.get_credits()
            captcha = self.captcha_configuration()
            result = {
                "success": True,
                "isLoggedIn": True,
                "canGenerate": bool(captcha["configured"]),
                "credits": credits.get("credits"),
                "userPaygateTier": credits.get("userPaygateTier"),
                "creditAccounts": credits.get("accounts", []),
                "projectId": self._project_id or get_env_or_credential("FLOW_PROJECT_ID"),
                "captchaMethod": captcha["method"],
                "captchaConfigured": captcha["configured"],
                "browser": self._browser_captcha.status(),
                "accounts": self._account_status(),
                "models": self.supported_models(),
            }
            if captcha.get("error"):
                result["warning"] = captcha["error"]
                if validate:
                    result["success"] = False
                    result["error"] = captcha["error"]
            return result
        except Exception as exc:
            return {
                "success": False,
                "isLoggedIn": False,
                "canGenerate": False,
                "error": str(exc),
            }

    def captcha_configuration(self) -> dict[str, Any]:
        if _first_env("FLOW_RECAPTCHA_TOKEN", "FLOW_CAPTCHA_TOKEN"):
            return {"method": "explicit_token", "configured": True}

        method = self._captcha_method()
        if method in FLOW_BROWSER_CAPTCHA_METHODS:
            browser = self._browser_captcha.status()
            available = bool(browser.get("available"))
            return {
                "method": method,
                "configured": available,
                "browser": browser,
                **(
                    {}
                    if available
                    else {
                        "error": (
                            "O navegador do OmniProviders não está disponível. "
                            "Verifique Playwright Python, Chrome instalado e o driver "
                            "Electron/Node externo."
                        )
                    }
                ),
            }
        if method == "remote_browser":
            return {
                "method": method,
                "configured": False,
                "error": "remote_browser não é suportado; use personal ou browser.",
            }
        key_names = FLOW_CAPTCHA_SOLVERS.get(method)
        if key_names:
            if _first_env(*key_names):
                return {"method": method, "configured": True}
            return {
                "method": method,
                "configured": False,
                "error": f"A API key do {method} não está configurada.",
            }
        return {
            "method": method,
            "configured": False,
            "error": (
                "Selecione um captcha solver compatível com o OmniProviders: "
                "YesCaptcha, CapMonster, EZCaptcha ou CapSolver."
            ),
        }

    async def close(self) -> None:
        self._cancel_requested = False
        await self._browser_captcha.close()

    def cancel(self) -> dict[str, Any]:
        self._cancel_requested = True
        return {"success": True, "cancelled": True}

    async def open_login_window(self) -> dict[str, Any]:
        try:
            result = await self._browser_captcha.open_login_window()
            return result | {"captchaMethod": self._captcha_method()}
        except FlowBrowserCaptchaError as exc:
            raise FlowScrapingError(str(exc)) from exc

    async def refresh_session_token_from_browser(self) -> dict[str, Any]:
        registered_accounts = list_flow_cookie_runtime_accounts()
        if not registered_accounts:
            raise FlowScrapingError(
                "Cadastre uma conta Flow antes de atualizar a sessão pelo navegador."
            )
        if len(registered_accounts) > 1:
            raise FlowScrapingError(
                "Há mais de uma conta Flow cadastrada. Atualize os cookies da conta "
                "desejada para evitar substituir a sessão de outra identidade."
            )
        account = registered_accounts[0]
        project_id = (
            str(account.get("projectId") or "").strip()
            or self._project_id
        )
        try:
            session_token = await self._browser_captcha.refresh_session_token(project_id)
        except FlowBrowserCaptchaError as exc:
            raise FlowScrapingError(str(exc)) from exc
        if not session_token:
            raise FlowScrapingError(
                "Não foi possível encontrar o cookie "
                "__Secure-next-auth.session-token no navegador do Flow."
            )
        update_flow_user_account_session(
            str(account["id"]),
            session_token,
        )
        self._access_token = None
        self._access_expires_at = 0
        self._account_access_cache.clear()
        self._account_session_overrides.clear()
        return {
            "success": True,
            "sessionTokenUpdated": True,
            "accountId": account["id"],
            "accountLabel": account.get("label") or account["id"],
            "browser": self._browser_captcha.status(),
        }

    def browser_status(self) -> dict[str, Any]:
        return self._browser_captcha.status() | {
            "captchaMethod": self._captcha_method(),
            "accounts": self._account_status(),
        }

    def supported_models(self) -> list[dict[str, Any]]:
        return [
            {"id": "gemini-3.1-flash-image", "name": "Nano Banana 2 / NARWHAL", "type": "image"},
            {"id": "gemini-3.0-pro-image", "name": "Nano Banana Pro / GEM_PIX_2", "type": "image"},
            {"id": "veo_3_1_t2v_fast", "name": "Veo 3.1 Fast text-to-video", "type": "video"},
            {"id": "veo_3_1_i2v_s_fast", "name": "Veo 3.1 Fast image-to-video", "type": "video"},
            {"id": "veo_3_1_r2v_fast", "name": "Veo 3.1 Fast references-to-video", "type": "video"},
            {"id": "veo_3_1_t2v_lite", "name": "Veo 3.1 Lite text-to-video", "type": "video"},
            {"id": "veo_3_1_i2v_lite", "name": "Veo 3.1 Lite image-to-video", "type": "video"},
        ]

    def _flow_accounts(self) -> list[FlowAccount]:
        accounts: list[FlowAccount] = []
        seen: set[str] = set()

        for entry in list_flow_cookie_runtime_accounts():
            account_id = str(entry.get("id") or "").strip()
            session_token = str(entry.get("sessionToken") or "").strip()
            if not account_id or not session_token:
                continue
            account = FlowAccount(
                key=f"FLOW_COOKIE_ACCOUNTS:{account_id}",
                id=account_id,
                label=str(entry.get("label") or account_id).strip() or account_id,
                session_token=session_token,
                project_id=str(entry.get("projectId") or "").strip() or None,
                cookie_header=str(entry.get("cookieHeader") or "").strip() or None,
                source_field="FLOW_COOKIE_ACCOUNTS",
            )
            if account.key not in seen:
                seen.add(account.key)
                accounts.append(account)
        return accounts

    def _account_status(self) -> list[dict[str, Any]]:
        accounts = self._flow_accounts()
        return [
            {
                "id": account.id or account.key,
                "key": account.key,
                "label": account.label,
                "source": account.source_field,
                "hasSessionToken": bool(account.session_token),
                "hasAccessToken": bool(account.access_token),
                "projectId": account.project_id or self._account_project_cache.get(account.key),
                "projectPoolSize": len(self._account_project_pools.get(account.key, [])),
                "inflight": self._account_inflight.get(account.key, {"image": 0, "video": 0}),
            }
            for account in accounts
        ]

    def _can_refresh_session_from_browser(self, account: FlowAccount) -> bool:
        accounts = self._flow_accounts()
        return bool(
            account.id
            and len(accounts) == 1
            and accounts[0].id == account.id
        )

    def _account_concurrency_limit(self, kind: str) -> int | None:
        env_name = "FLOW_IMAGE_CONCURRENCY" if kind == "image" else "FLOW_VIDEO_CONCURRENCY"
        value = get_env_or_credential(env_name) or "-1"
        try:
            parsed = int(value)
        except ValueError:
            parsed = -1
        return parsed if parsed > 0 else None

    def _token_wait_timeout(self) -> float:
        value = get_env_or_credential("FLOW_TOKEN_WAIT_TIMEOUT") or "120"
        try:
            return max(1.0, float(value))
        except ValueError:
            return 120.0

    async def _acquire_flow_account(self, kind: str) -> FlowAccount:
        accounts = self._flow_accounts()
        if not accounts:
            raise FlowScrapingError(
                "Configure FLOW_SESSION_TOKEN or FLOW_ACCESS_TOKEN before using Flow."
            )
        deadline = time.monotonic() + self._token_wait_timeout()
        limit = self._account_concurrency_limit(kind)

        while True:
            async with self._account_lock:
                for offset in range(len(accounts)):
                    index = (self._round_robin_index + offset) % len(accounts)
                    account = accounts[index]
                    inflight = self._account_inflight.setdefault(account.key, {"image": 0, "video": 0})
                    current = int(inflight.get(kind, 0))
                    if limit is not None and current >= limit:
                        continue
                    inflight[kind] = current + 1
                    self._round_robin_index = (index + 1) % len(accounts)
                    return account

            if time.monotonic() > deadline:
                raise FlowScrapingError(
                    f"No Flow token slot became available within {self._token_wait_timeout():g}s."
                )
            print(f"[FlowScraping] Aguardando slot de conta livre para '{kind}'...")
            await asyncio.sleep(0.5)

    async def _release_flow_account(self, account: FlowAccount, kind: str) -> None:
        async with self._account_lock:
            inflight = self._account_inflight.setdefault(account.key, {"image": 0, "video": 0})
            inflight[kind] = max(0, int(inflight.get(kind, 0)) - 1)

    async def generate_images(self, payload: dict[str, Any]) -> dict[str, Any]:
        prompt = str(payload.get("prompt") or "").strip()
        if not prompt:
            raise FlowScrapingError("Flow image prompt is required.")

        print(f"[FlowScraping] Iniciando geração de imagem: '{prompt[:50]}...'")
        self._cancel_requested = False
        self._set_request_fingerprint(None)
        account = await self._acquire_flow_account("image")
        print(f"[FlowScraping] Usando conta: {account.label}")
        try:
            started_at = time.perf_counter()
            count = _normalize_count(payload.get("count"))
            captcha_headless = _optional_bool(payload.get("browserHeadless"))
            at = await self._access_token_for_account(account)
            project_id = await self._project_id_for_account(account)
            model_name, upsample_resolution, initial_fallback = self._resolve_image_model(payload)
            requested_model_name = self._requested_image_model(payload)
            active_model_name = model_name
            aspect_ratio = _image_aspect_enum(payload.get("aspectRatio"))

            image_inputs = []
            warnings: list[str] = []
            reference_values = (
                payload.get("ingredientImagePaths")
                or payload.get("referenceImagePaths")
                or payload.get("referenceImagePath")
                or []
            )
            for path in self._iter_reference_values(reference_values):
                self._raise_if_cancelled()
                print(f"[FlowScraping] Fazendo upload de referência: {path}")
                media_id = await self.upload_image(
                    at,
                    await self._read_reference_bytes(path),
                    aspect_ratio,
                    project_id,
                )
                image_inputs.append({"name": media_id, "imageInputType": "IMAGE_INPUT_TYPE_REFERENCE"})

            generated_media: list[dict[str, Any]] = []
            source_urls: list[str] = []
            generation_failures: list[str] = []
            fallback_events: list[dict[str, Any]] = []
            if initial_fallback:
                fallback_events.append(initial_fallback)
                print(
                    "[FlowScraping] Nano Banana Pro em cooldown; "
                    "usando Nano Banana 2 para esta geração."
                )
            print(f"[FlowScraping] Iniciando batch de {count} imagem(ns)...")
            for _index in range(count):
                self._raise_if_cancelled()
                saved_before = len(generated_media)
                try:
                    try:
                        result, session_id = await self._generate_image_once(
                            at=at,
                            project_id=project_id,
                            prompt=prompt,
                            model_name=active_model_name,
                            aspect_ratio=aspect_ratio,
                            image_inputs=image_inputs,
                            captcha_headless=captcha_headless,
                        )
                    except FlowScrapingError as exc:
                        if self._should_fallback_flow_image_model(active_model_name, exc):
                            fallback_state = self._activate_flow_image_pro_fallback(str(exc))
                            fallback_event = self._flow_image_fallback_event(
                                fallback_state,
                                reason="quota",
                            )
                            fallback_events.append(fallback_event)
                            warnings.append(
                                "Nano Banana Pro atingiu a quota diaria no Flow; "
                                "retry automatico com Nano Banana 2."
                            )
                            print(
                                "[FlowScraping] Nano Banana Pro atingiu quota diaria; "
                                "fazendo retry automatico com Nano Banana 2."
                            )
                            active_model_name = FLOW_IMAGE_MODEL_NANO_BANANA_2
                            try:
                                result, session_id = await self._generate_image_once(
                                    at=at,
                                    project_id=project_id,
                                    prompt=prompt,
                                    model_name=active_model_name,
                                    aspect_ratio=aspect_ratio,
                                    image_inputs=image_inputs,
                                    captcha_headless=captcha_headless,
                                )
                            except FlowScrapingError as fallback_exc:
                                if self._is_flow_image_daily_quota_error(fallback_exc):
                                    raise self._flow_image_api_required_error(
                                        fallback_exc,
                                        fallback_state,
                                    ) from fallback_exc
                                raise
                        elif (
                            active_model_name == FLOW_IMAGE_MODEL_NANO_BANANA_2
                            and self._is_flow_image_daily_quota_error(exc)
                        ):
                            raise self._flow_image_api_required_error(exc, None) from exc
                        else:
                            raise
                    for media in result.get("media", []):
                        image = media.get("image", {}).get("generatedImage", {})
                        image_url = image.get("fifeUrl")
                        media_id = media.get("name")
                        if not image_url:
                            continue
                        source_urls.append(image_url)
                        if upsample_resolution and media_id:
                            encoded = await self.upsample_image(
                                at=at,
                                project_id=project_id,
                                media_id=media_id,
                                target_resolution=upsample_resolution,
                                session_id=session_id,
                                captcha_headless=captcha_headless,
                            )
                            if encoded:
                                generated_media.append(
                                    {
                                        "bytes": base64.b64decode(encoded),
                                        "mime_type": "image/jpeg",
                                        "filename": "flow-image-upsampled.jpg",
                                    }
                                )
                                continue
                        generated_media.append(
                            await self._download_media(
                                image_url,
                                fallback_mime="image/jpeg",
                                filename="flow-image.jpg",
                            )
                        )
                        if len(generated_media) >= count:
                            break
                    if len(generated_media) == saved_before:
                        raise FlowScrapingError("Flow did not return generated image URLs.")
                    print(f"[FlowScraping] Imagem {len(generated_media)}/{count} gerada com sucesso.")
                except FlowScrapingError as exc:
                    if getattr(exc, "code", None) == FLOW_IMAGE_API_REQUIRED_ERROR_CODE:
                        raise
                    print(f"[FlowScraping] Falha ao gerar imagem {_index + 1}: {exc}")
                    generation_failures.append(f"Image {_index + 1}/{count}: {exc}")
                    continue
                if len(generated_media) >= count:
                    break

            if generation_failures:
                warnings.extend(generation_failures)

            if not generated_media:
                detail = " | ".join(warnings) if warnings else "Flow did not return generated image URLs."
                raise FlowScrapingError(detail)

            return {
                "success": True,
                "media": generated_media,
                "sourceUrls": source_urls,
                "warnings": warnings,
                "error": " | ".join(warnings) if warnings else None,
                "projectId": project_id,
                "flowAccount": account.label,
                "flowImageRequestedModel": requested_model_name,
                "flowImageModel": active_model_name,
                "flowImageFallback": fallback_events[-1] if fallback_events else None,
                "flowImageFallbacks": fallback_events,
                "durationMs": int((time.perf_counter() - started_at) * 1000),
            }
        finally:
            await self._release_flow_account(account, "image")

    async def generate_video(self, payload: dict[str, Any]) -> dict[str, Any]:
        prompt = str(payload.get("prompt") or "").strip()
        if not prompt:
            raise FlowScrapingError("Flow video prompt is required.")

        self._cancel_requested = False
        self._set_request_fingerprint(None)
        account = await self._acquire_flow_account("video")
        try:
            started_at = time.perf_counter()
            count = _normalize_count(payload.get("count"))
            captcha_headless = _optional_bool(payload.get("browserHeadless"))
            at = await self._access_token_for_account(account)
            project_id = await self._project_id_for_account(account)
            plan = self._build_video_plan(payload)
            credits = await self._safe_credits(at)
            user_tier = (
                get_env_or_credential("FLOW_PAYGATE_TIER")
                or credits.get("userPaygateTier")
                or "PAYGATE_TIER_ONE"
            )

            uploaded = await self._upload_video_inputs(at, project_id, plan)
            generated_media: list[dict[str, Any]] = []
            source_urls: list[str] = []
            operations_seen: list[dict[str, Any]] = []

            for _index in range(count):
                self._raise_if_cancelled()
                result = await self._submit_video_generation(
                    at=at,
                    project_id=project_id,
                    prompt=prompt,
                    plan=plan,
                    uploaded=uploaded,
                    user_paygate_tier=user_tier,
                    captcha_headless=captcha_headless,
                )
                operations = result.get("operations") or []
                if not operations:
                    raise FlowScrapingError(f"Flow video task creation failed: {result}")
                operations_seen.extend(operations)
                video = await self._poll_video_result(at, operations)
                source_urls.append(video["url"])
                generated_media.append(
                    await self._download_media(
                        video["url"],
                        fallback_mime="video/mp4",
                        filename="flow-video.mp4",
                    )
                )

            return {
                "success": True,
                "media": generated_media,
                "sourceUrls": source_urls,
                "operations": operations_seen,
                "credits": credits.get("credits"),
                "projectId": project_id,
                "flowAccount": account.label,
                "modelKey": plan.model_key,
                "durationMs": int((time.perf_counter() - started_at) * 1000),
            }
        finally:
            await self._release_flow_account(account, "video")

    async def generate_veo2_flow(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise FlowScrapingError(
            "Veo 2 Flow is not exposed by the flow2api service code present in this workspace. "
            "Use Veo 3.1 Flow or the official Veo 2 API endpoint."
        )

    async def get_credits(self, at: str | None = None) -> dict[str, Any]:
        if at:
            return await self._request("GET", f"{self._api_base_url()}/credits", use_at=True, at_token=at)

        accounts = self._flow_accounts()
        if len(accounts) <= 1:
            account = accounts[0] if accounts else None
            access_token = (
                await self._access_token_for_account(account)
                if account
                else await self._access_token_value()
            )
            credits = await self._request("GET", f"{self._api_base_url()}/credits", use_at=True, at_token=access_token)
            if account:
                credits["accounts"] = [
                    {
                        "id": account.id or account.key,
                        "label": account.label,
                        "projectId": account.project_id or self._account_project_cache.get(account.key),
                        "credits": credits.get("credits"),
                        "userPaygateTier": credits.get("userPaygateTier"),
                    }
                ]
            return credits

        account_credits: list[dict[str, Any]] = []
        total_credits = 0
        has_numeric_total = False
        first_tier: Any = None
        for account in accounts:
            try:
                access_token = await self._access_token_for_account(account)
                credits = await self._request("GET", f"{self._api_base_url()}/credits", use_at=True, at_token=access_token)
                value = credits.get("credits")
                if isinstance(value, (int, float)):
                    total_credits += int(value)
                    has_numeric_total = True
                first_tier = first_tier or credits.get("userPaygateTier")
                account_credits.append(
                    {
                        "id": account.id or account.key,
                        "label": account.label,
                        "projectId": account.project_id or self._account_project_cache.get(account.key),
                        "credits": value,
                        "userPaygateTier": credits.get("userPaygateTier"),
                    }
                )
            except Exception as exc:
                account_credits.append({"id": account.id or account.key, "label": account.label, "error": str(exc)})

        return {
            "credits": total_credits if has_numeric_total else None,
            "userPaygateTier": first_tier,
            "accounts": account_credits,
        }

    async def test_account(self, account_id: str) -> dict[str, Any]:
        normalized_id = str(account_id or "").strip()
        if not normalized_id:
            raise FlowScrapingError("Flow account id is required.")
        account = next(
            (
                candidate
                for candidate in self._flow_accounts()
                if candidate.id == normalized_id or candidate.key == normalized_id
            ),
            None,
        )
        if not account:
            raise FlowScrapingError("Flow account not found.")

        access_token = await self._access_token_for_account(account)
        credits = await self._request(
            "GET",
            f"{self._api_base_url()}/credits",
            use_at=True,
            at_token=access_token,
        )
        return {
            "success": True,
            "id": account.id or account.key,
            "label": account.label,
            "projectId": account.project_id or self._account_project_cache.get(account.key),
            "credits": credits.get("credits"),
            "userPaygateTier": credits.get("userPaygateTier"),
            "raw": credits,
        }

    async def upload_image(
        self,
        at: str,
        image_bytes: bytes,
        aspect_ratio: str,
        project_id: str | None = None,
    ) -> str:
        if aspect_ratio.startswith("VIDEO_"):
            aspect_ratio = aspect_ratio.replace("VIDEO_", "IMAGE_")
        mime_type = _detect_image_mime(image_bytes)
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        file_ext = "png" if mime_type == "image/png" else "jpg"

        client_context = {"tool": "PINHOLE"}
        if project_id:
            client_context["projectId"] = project_id

        last_error: FlowScrapingError | None = None
        max_retries = self._max_retries()
        for attempt in range(max_retries):
            filename = f"flow-upload-{int(time.time() * 1000)}-{attempt + 1}.{file_ext}"
            try:
                result = await self._request(
                    "POST",
                    f"{self._api_base_url()}/flow/uploadImage",
                    json_data={
                        "clientContext": client_context,
                        "fileName": filename,
                        "imageBytes": image_base64,
                        "isHidden": False,
                        "isUserUploaded": True,
                        "mimeType": mime_type,
                    },
                    use_at=True,
                    at_token=at,
                    use_media_proxy=True,
                )
                media_id = result.get("media", {}).get("name") or result.get("mediaGenerationId", {}).get("mediaGenerationId")
                if media_id:
                    return media_id
                raise FlowScrapingError(f"Flow image upload response did not include a media id: {result}")
            except FlowScrapingError as exc:
                last_error = exc
                retry_reason = self._get_retry_reason(str(exc))
                if not retry_reason or attempt >= max_retries - 1:
                    raise
                await self._sleep_before_retry(attempt, retry_reason)

        raise last_error or FlowScrapingError("Flow image upload failed.")

    async def upsample_image(
        self,
        at: str,
        project_id: str,
        media_id: str,
        target_resolution: str,
        session_id: str | None = None,
        captcha_headless: bool | None = None,
    ) -> str:
        recaptcha_token = await self._recaptcha_token(
            project_id,
            "IMAGE_GENERATION",
            headless=captcha_headless,
        )
        try:
            result = await self._request(
                "POST",
                f"{self._api_base_url()}/flow/upsampleImage",
                json_data={
                    "mediaId": media_id,
                    "targetResolution": target_resolution,
                    "clientContext": {
                        "recaptchaContext": self._recaptcha_context(recaptcha_token),
                        "sessionId": session_id or self._session_id(),
                        "projectId": project_id,
                        "tool": "PINHOLE",
                        "userPaygateTier": get_env_or_credential("FLOW_PAYGATE_TIER") or "PAYGATE_TIER_ONE",
                    },
                },
                use_at=True,
                at_token=at,
                timeout=self._upsample_timeout(),
            )
        except FlowScrapingError as exc:
            await self._report_browser_captcha_error(str(exc))
            raise
        finally:
            await self._report_browser_captcha_finished()
        return str(result.get("encodedImage") or "")

    async def _generate_image_once(
        self,
        *,
        at: str,
        project_id: str,
        prompt: str,
        model_name: str,
        aspect_ratio: str,
        image_inputs: list[dict[str, Any]],
        captcha_headless: bool | None = None,
    ) -> tuple[dict[str, Any], str]:
        last_error: FlowScrapingError | None = None
        max_retries = self._max_retries()
        for attempt in range(max_retries):
            self._raise_if_cancelled()
            try:
                recaptcha_token = await self._recaptcha_token(
                    project_id,
                    "IMAGE_GENERATION",
                    headless=captcha_headless,
                )
                session_id = self._session_id()
                client_context = {
                    "recaptchaContext": self._recaptcha_context(recaptcha_token),
                    "sessionId": session_id,
                    "projectId": project_id,
                    "tool": "PINHOLE",
                }
                request_data = {
                    "clientContext": client_context,
                    "seed": random.randint(1, 999999),
                    "imageModelName": model_name,
                    "imageAspectRatio": aspect_ratio,
                    "structuredPrompt": {"parts": [{"text": prompt}]},
                    "imageInputs": image_inputs,
                }
                try:
                    result = await self._submit_image_generation_request(
                        at=at,
                        project_id=project_id,
                        client_context=client_context,
                        request_data=request_data,
                    )
                finally:
                    await self._report_browser_captcha_finished()
                return result, session_id
            except FlowScrapingError as exc:
                last_error = exc
                retry_reason = self._get_retry_reason(str(exc))
                await self._report_browser_captcha_error(str(exc))
                if not retry_reason or attempt >= max_retries - 1:
                    raise
                await self._sleep_before_retry(attempt, retry_reason)

        raise last_error or FlowScrapingError("Flow image generation failed.")

    async def _submit_image_generation_request(
        self,
        *,
        at: str,
        project_id: str,
        client_context: dict[str, Any],
        request_data: dict[str, Any],
    ) -> dict[str, Any]:
        url = f"{self._api_base_url()}/projects/{project_id}/flowMedia:batchGenerateImages"
        payload = {
            "clientContext": client_context,
            "mediaGenerationContext": {"batchId": str(uuid.uuid4())},
            "useNewMedia": True,
            "requests": [request_data],
        }
        attempts = self._image_timeout_retry_count() + 1
        last_error: FlowScrapingError | None = None
        for attempt in range(attempts):
            try:
                return await self._request(
                    "POST",
                    url,
                    json_data=payload,
                    use_at=True,
                    at_token=at,
                    timeout=self._image_timeout(),
                )
            except FlowScrapingError as exc:
                last_error = exc
                if not self._is_timeout_error(exc):
                    raise
                if attempt >= attempts - 1:
                    raise FlowScrapingError(
                        "Flow image submit timed out after dispatch. "
                        "The generation may still complete in Flow; increase FLOW_IMAGE_TIMEOUT "
                        "if this keeps happening."
                    ) from exc
                delay = self._image_timeout_retry_delay()
                print(
                    "[FlowScraping] Timeout no submit de imagem; "
                    f"repetindo a mesma requisicao ({attempt + 2}/{attempts})..."
                )
                if delay > 0:
                    await asyncio.sleep(delay)
        raise last_error or FlowScrapingError("Flow image submit failed.")

    async def _submit_video_generation(
        self,
        *,
        at: str,
        project_id: str,
        prompt: str,
        plan: FlowVideoPlan,
        uploaded: dict[str, Any],
        user_paygate_tier: str,
        captcha_headless: bool | None = None,
    ) -> dict[str, Any]:
        recaptcha_token = await self._recaptcha_token(
            project_id,
            "VIDEO_GENERATION",
            headless=captcha_headless,
        )
        session_id = self._session_id()
        scene_id = str(uuid.uuid4())
        client_context = {
            "recaptchaContext": self._recaptcha_context(recaptcha_token),
            "sessionId": session_id,
            "projectId": project_id,
            "tool": "PINHOLE",
            "userPaygateTier": user_paygate_tier,
        }
        base_request = {
            "aspectRatio": plan.aspect_ratio,
            "seed": random.randint(1, 99999),
            "textInput": self._video_text_input(prompt, plan.use_v2_model_config),
            "videoModelKey": plan.model_key,
            "metadata": {"sceneId": scene_id},
        }

        if plan.video_type == "i2v-start-end":
            url = f"{self._api_base_url()}/video:batchAsyncGenerateVideoStartAndEndImage"
            request = {
                **base_request,
                "startImage": {"mediaId": uploaded["start_media_id"]},
                "endImage": {"mediaId": uploaded["end_media_id"]},
            }
        elif plan.video_type == "i2v-start":
            url = f"{self._api_base_url()}/video:batchAsyncGenerateVideoStartImage"
            model_key = plan.model_key.replace("_fl_", "_")
            if model_key.endswith("_fl"):
                model_key = model_key[:-3]
            request = {**base_request, "videoModelKey": model_key, "startImage": {"mediaId": uploaded["start_media_id"]}}
        elif plan.video_type == "r2v":
            url = f"{self._api_base_url()}/video:batchAsyncGenerateVideoReferenceImages"
            request = {
                **base_request,
                "textInput": {"structuredPrompt": {"parts": [{"text": prompt}]}},
                "referenceImages": uploaded["reference_images"],
            }
            plan = FlowVideoPlan(**{**plan.__dict__, "use_v2_model_config": True})
        else:
            url = f"{self._api_base_url()}/video:batchAsyncGenerateVideoText"
            request = base_request

        json_data: dict[str, Any] = {"clientContext": client_context, "requests": [request]}
        if plan.use_v2_model_config or plan.video_type == "r2v":
            json_data["mediaGenerationContext"] = {"batchId": str(uuid.uuid4())}
            json_data["useV2ModelConfig"] = True

        try:
            return await self._request(
                "POST",
                url,
                json_data=json_data,
                use_at=True,
                at_token=at,
                timeout=self._video_submit_timeout(),
            )
        except FlowScrapingError as exc:
            await self._report_browser_captcha_error(str(exc))
            raise
        finally:
            await self._report_browser_captcha_finished()

    async def _poll_video_result(self, at: str, operations: list[dict[str, Any]]) -> dict[str, str]:
        max_attempts = self._poll_attempts()
        interval = self._poll_interval()
        last_status = ""
        last_error = ""

        for _attempt in range(max_attempts):
            self._raise_if_cancelled()
            await asyncio.sleep(interval)
            result = await self._request(
                "POST",
                f"{self._api_base_url()}/video:batchCheckAsyncVideoGenerationStatus",
                json_data={"operations": operations},
                use_at=True,
                at_token=at,
                timeout=self._video_poll_timeout(),
            )
            checked = result.get("operations") or []
            if not checked:
                continue
            operation = checked[0]
            status = str(operation.get("status") or "")
            last_status = status
            if status == "MEDIA_GENERATION_STATUS_SUCCESSFUL":
                metadata = operation.get("operation", {}).get("metadata", {})
                video = metadata.get("video") or {}
                url = str(video.get("fifeUrl") or "")
                media_id = str(video.get("mediaGenerationId") or "")
                if url:
                    return {"url": url, "mediaId": media_id}
                raise FlowScrapingError(f"Flow video finished without a fifeUrl: {operation}")
            if status == "MEDIA_GENERATION_STATUS_FAILED" or status.startswith("MEDIA_GENERATION_STATUS_ERROR"):
                error = operation.get("operation", {}).get("error", {})
                message = error.get("message") or status
                raise FlowScrapingError(f"Flow video generation failed: {message}")
            error = operation.get("operation", {}).get("error", {})
            if error:
                last_error = str(error)

        detail = f" Last status: {last_status}." if last_status else ""
        if last_error:
            detail += f" Last error: {last_error[:300]}."
        raise FlowScrapingError(f"Flow video generation timed out after {max_attempts} polls.{detail}")

    async def _upload_video_inputs(self, at: str, project_id: str, plan: FlowVideoPlan) -> dict[str, Any]:
        if not plan.image_paths:
            return {}
        if plan.video_type == "r2v":
            reference_images = []
            for path in plan.image_paths[:3]:
                media_id = await self.upload_image(at, await self._read_reference_bytes(path), plan.aspect_ratio, project_id)
                reference_images.append({"imageUsageType": "IMAGE_USAGE_TYPE_ASSET", "mediaId": media_id})
            return {"reference_images": reference_images}

        start_media_id = await self.upload_image(at, await self._read_reference_bytes(plan.image_paths[0]), plan.aspect_ratio, project_id)
        payload = {"start_media_id": start_media_id}
        if len(plan.image_paths) > 1:
            payload["end_media_id"] = await self.upload_image(at, await self._read_reference_bytes(plan.image_paths[1]), plan.aspect_ratio, project_id)
        return payload

    async def _access_token_value(self) -> str:
        accounts = self._flow_accounts()
        if not accounts:
            raise FlowScrapingError(
                "FLOW_SESSION_TOKEN is required unless FLOW_ACCESS_TOKEN is set."
            )
        return await self._access_token_for_account(accounts[0])

    async def _access_token_for_account(
        self,
        account: FlowAccount,
        *,
        allow_browser_refresh: bool = True,
    ) -> str:
        if account.access_token and not account.session_token:
            self._access_token_account_keys[account.access_token] = account.key
            return account.access_token

        now = time.time()
        cached = self._account_access_cache.get(account.key)
        if cached and cached[1] > now + self._access_refresh_margin_seconds():
            self._access_token_account_keys[cached[0]] = account.key
            return cached[0]

        access_lock = await self._access_lock_for_account(account.key)
        async with access_lock:
            now = time.time()
            cached = self._account_access_cache.get(account.key)
            if cached and cached[1] > now + self._access_refresh_margin_seconds():
                self._access_token_account_keys[cached[0]] = account.key
                return cached[0]

            st = self._session_token_for_account(account)
            if not st:
                if account.access_token:
                    self._access_token_account_keys[account.access_token] = account.key
                    return account.access_token
                raise FlowScrapingError(
                    "FLOW_SESSION_TOKEN is required unless FLOW_ACCESS_TOKEN is set."
                )

            try:
                result = await self._session_to_access_token(
                    st,
                    cookie_header=account.cookie_header,
                )
            except FlowScrapingError:
                if (
                    not allow_browser_refresh
                    or not self._can_refresh_session_from_browser(account)
                    or self._captcha_method() not in FLOW_BROWSER_CAPTCHA_METHODS
                ):
                    raise
                refreshed = await self._refresh_session_for_account(account)
                if not refreshed:
                    raise
                result = await self._session_to_access_token(refreshed)
            access_token = str(result.get("access_token") or "")
            if (
                not access_token
                and allow_browser_refresh
                and self._can_refresh_session_from_browser(account)
                and self._captcha_method() in FLOW_BROWSER_CAPTCHA_METHODS
            ):
                refreshed = await self._refresh_session_for_account(account)
                if refreshed:
                    result = await self._session_to_access_token(refreshed)
                    access_token = str(result.get("access_token") or "")

            if not access_token:
                raise FlowScrapingError(f"Flow session response did not include access_token: {result}")

            expires_at = self._parse_expires(result.get("expires"))
            self._account_access_cache[account.key] = (access_token, expires_at)
            self._access_token_account_keys[access_token] = account.key
            if account.key == "default":
                self._access_token = access_token
                self._access_expires_at = expires_at
            return access_token

    async def _refresh_session_for_account(
        self,
        account: FlowAccount,
    ) -> str | None:
        project_id = (
            account.project_id
            or self._account_project_cache.get(account.key)
            or self._project_id
        )
        try:
            session_token = await self._browser_captcha.refresh_session_token(project_id)
        except FlowBrowserCaptchaError:
            return None
        if not session_token:
            return None
        self._account_session_overrides[account.key] = session_token
        if account.id:
            try:
                update_flow_user_account_session(account.id, session_token)
            except KeyError:
                return None
        self._account_access_cache.pop(account.key, None)
        return session_token

    async def _session_to_access_token(
        self,
        session_token: str,
        *,
        cookie_header: str | None = None,
    ) -> dict[str, Any]:
        try:
            return await self._request(
                "GET",
                f"{self._labs_base_url()}/auth/session",
                use_st=True,
                st_token=session_token,
                cookie_header=cookie_header,
                timeout=self._control_timeout(),
            )
        except FlowScrapingError as exc:
            if not self._is_proxy_connection_error(exc):
                raise
            return await self._request(
                "GET",
                f"{self._labs_base_url()}/auth/session",
                use_st=True,
                st_token=session_token,
                cookie_header=cookie_header,
                timeout=self._control_timeout(),
                force_no_proxy=True,
            )

    async def _project_id_value(self) -> str:
        accounts = self._flow_accounts()
        if not accounts:
            raise FlowScrapingError(
                "FLOW_PROJECT_ID is required when using FLOW_ACCESS_TOKEN without FLOW_SESSION_TOKEN."
            )
        return await self._project_id_for_account(accounts[0])

    async def _project_id_for_account(self, account: FlowAccount) -> str:
        if account.project_id:
            return account.project_id
        async with self._project_lock:
            pool = self._account_project_pools.get(account.key)
            if pool:
                return self._select_project_from_pool(account)

            cached = self._account_project_cache.get(account.key)
            if cached:
                self._account_project_pools[account.key] = [cached]
                return self._select_project_from_pool(account)
            if account.key == "default" and self._project_id:
                self._account_project_pools[account.key] = [self._project_id]
                return self._select_project_from_pool(account)

            st = self._session_token_for_account(account)
            if not st:
                explicit = _first_env("FLOW_PROJECT_ID")
                if explicit:
                    return explicit
                raise FlowScrapingError(
                    "FLOW_PROJECT_ID is required when using FLOW_ACCESS_TOKEN without FLOW_SESSION_TOKEN."
                )

            desired_pool_size = self._project_pool_size()
            created: list[str] = []
            while len(created) < desired_pool_size:
                project_index = len(created) + 1
                try:
                    created.append(
                        await self._create_project_for_account(account, st, project_index)
                    )
                except FlowScrapingError:
                    if (
                        created
                        or not self._can_refresh_session_from_browser(account)
                        or self._captcha_method() not in FLOW_BROWSER_CAPTCHA_METHODS
                    ):
                        raise
                    refreshed = await self._refresh_session_for_account(account)
                    if not refreshed:
                        raise
                    st = refreshed
                    created.append(
                        await self._create_project_for_account(account, st, project_index)
                    )

            self._account_project_pools[account.key] = created
            return self._select_project_from_pool(account)

    async def _create_project_for_account(self, account: FlowAccount, session_token: str, pool_index: int) -> str:
        suffix = f" P{pool_index}" if self._project_pool_size() > 1 else ""
        title = datetime.now().strftime(f"Uno Studio %Y-%m-%d %H:%M{suffix}")
        result = await self._request(
            "POST",
            f"{self._labs_base_url()}/trpc/project.createProject",
            json_data={"json": {"projectTitle": title, "toolName": "PINHOLE"}},
            use_st=True,
            st_token=session_token,
            cookie_header=account.cookie_header if session_token == account.session_token else None,
            timeout=self._control_timeout(),
        )
        project = result.get("result", {}).get("data", {}).get("json", {}).get("result", {})
        project_id = str(project.get("projectId") or "")
        if not project_id:
            raise FlowScrapingError(f"Flow project.createProject response did not include projectId: {result}")
        print(f"[FlowScraping] Projeto Flow criado para {account.label}: {project_id}")
        return project_id

    def _select_project_from_pool(self, account: FlowAccount) -> str:
        pool = self._account_project_pools.get(account.key) or []
        if not pool:
            raise FlowScrapingError("Flow project pool is empty.")
        index = self._account_project_rr.get(account.key, 0) % len(pool)
        self._account_project_rr[account.key] = index + 1
        project_id = pool[index]
        self._account_project_cache[account.key] = project_id
        if account.key == "default":
            self._project_id = project_id
        return project_id

    def _session_token_for_account(self, account: FlowAccount) -> str | None:
        return self._account_session_overrides.get(account.key) or account.session_token

    async def _access_lock_for_account(self, account_key: str) -> asyncio.Lock:
        async with self._access_lock_guard:
            lock = self._account_access_locks.get(account_key)
            if lock is None:
                lock = asyncio.Lock()
                self._account_access_locks[account_key] = lock
            return lock

    async def _safe_credits(self, at: str) -> dict[str, Any]:
        try:
            return await self.get_credits(at)
        except Exception:
            return {}

    async def _request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json_data: dict[str, Any] | None = None,
        use_st: bool = False,
        st_token: str | None = None,
        cookie_header: str | None = None,
        use_at: bool = False,
        at_token: str | None = None,
        timeout: float | None = None,
        use_media_proxy: bool = False,
        respect_fingerprint_proxy: bool = True,
        force_no_proxy: bool = False,
    ) -> dict[str, Any]:
        auth_cookie = ""
        auth_bearer = at_token
        if use_st:
            if cookie_header:
                auth_cookie = cookie_header
            elif st_token:
                auth_cookie = f"__Secure-next-auth.session-token={st_token}"
            else:
                cookie = _first_env("FLOW_COOKIE", "FLOW_LABS_COOKIE")
                auth_cookie = cookie or f"__Secure-next-auth.session-token={self._session_token()}"
        if use_at:
            auth_bearer = at_token or await self._access_token_value()

        account_id = None
        if st_token:
            account_id = st_token[:16]
        elif auth_bearer:
            account_id = self._access_token_account_keys.get(auth_bearer) or auth_bearer[:16]

        request_headers = self._headers(headers, account_id=account_id)
        if auth_cookie:
            request_headers["Cookie"] = auth_cookie
        if auth_bearer:
            request_headers["authorization"] = f"Bearer {auth_bearer}"

        fingerprint = self._request_fingerprint_ctx.get()
        if isinstance(fingerprint, dict):
            self._apply_request_fingerprint(request_headers, fingerprint)
        else:
            self._sync_client_hints_with_user_agent(request_headers)

        proxy_url = None if force_no_proxy else self._proxy_url(media=use_media_proxy)
        if (
            not force_no_proxy
            and respect_fingerprint_proxy
            and isinstance(fingerprint, dict)
            and "proxy_url" in fingerprint
        ):
            proxy_url = str(fingerprint.get("proxy_url") or "").strip() or None
        try:
            async with AsyncSession(trust_env=False) as session:
                kwargs = {
                    "headers": request_headers,
                    "timeout": timeout or self._timeout(),
                    "proxy": proxy_url,
                    "impersonate": self._http_impersonation(),
                }
                if method.upper() == "GET":
                    response = await session.get(url, **kwargs)
                else:
                    response = await session.post(url, json=json_data or {}, **kwargs)
        except Exception as exc:
            raise FlowScrapingError(f"Flow request failed: {exc}") from exc

        text = response.text or ""
        if response.status_code >= 400:
            raise FlowScrapingError(
                self._format_flow_http_error(response.status_code, url, text)
            )
        try:
            return response.json()
        except Exception as exc:
            raise FlowScrapingError(f"Flow returned invalid JSON at {url}: {text[:500]}") from exc

    def _format_flow_http_error(self, status_code: int, url: str, text: str) -> str:
        reason = ""
        message = ""
        try:
            payload = json.loads(text) if text else {}
        except Exception:
            payload = {}
        error = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(error, dict):
            message = str(error.get("message") or "").strip()
            details = error.get("details")
            if isinstance(details, list):
                for detail in details:
                    if isinstance(detail, dict) and detail.get("reason"):
                        reason = str(detail.get("reason") or "").strip()
                        break
            reason = reason or str(error.get("status") or "").strip()
        suffix = f" [{reason}]" if reason else ""
        body = text[:500]
        if message and message not in body:
            body = f"{message} {body}".strip()
        return f"Flow HTTP {status_code}{suffix} at {url}: {body}"

    def _max_retries(self) -> int:
        value = get_env_or_credential("FLOW_MAX_RETRIES") or os.environ.get("FLOW_MAX_RETRIES") or "3"
        try:
            return max(1, min(8, int(value)))
        except ValueError:
            return 3

    def _retry_base_delay(self) -> float:
        value = get_env_or_credential("FLOW_RETRY_BASE_DELAY") or os.environ.get("FLOW_RETRY_BASE_DELAY") or "1.5"
        try:
            return max(0.2, min(30.0, float(value)))
        except ValueError:
            return 1.5

    async def _sleep_before_retry(self, attempt: int, reason: str) -> None:
        base_delay = self._retry_base_delay()
        reason_lower = reason.lower()
        if "recaptcha" in reason_lower or "unusual" in reason_lower:
            base_delay = max(base_delay, 3.0)
        delay = min(30.0, base_delay * (2 ** attempt)) + random.uniform(0.0, 0.6)
        await asyncio.sleep(delay)

    def _is_timeout_error(self, error: BaseException | str) -> bool:
        message = str(error).lower()
        return any(
            marker in message
            for marker in [
                "timed out",
                "timeout",
                "curl: (28)",
                "operation timed out",
                "connection timed out",
            ]
        )

    def _is_retryable_network_error(self, error: BaseException | str) -> bool:
        message = str(error).lower()
        return any(
            marker in message
            for marker in [
                "curl: (6)",
                "curl: (7)",
                "curl: (28)",
                "curl: (35)",
                "curl: (52)",
                "curl: (56)",
                "ssl_error_syscall",
                "tls connect error",
                "ssl connect error",
                "connection reset",
                "connection aborted",
                "unexpected eof",
                "empty reply from server",
                "recv failure",
                "send failure",
                "connection refused",
                "network is unreachable",
                "remote host closed connection",
            ]
        )

    def _get_retry_reason(self, error: BaseException | str) -> str | None:
        message = str(error)
        lower = message.lower()
        if any(
            marker in lower
            for marker in [
                "api key is not configured",
                "requires recaptcha",
                "not embedded in this app",
                "configure flow_captcha_method",
                "configure flow_session_token",
                "flow session response did not include access_token",
                "flow image submit timed out after dispatch",
                "public_error_per_model_daily_quota_reached",
                "public_error_unsafe_generation",
            ]
        ):
            return None
        if self._is_timeout_error(message):
            return "network timeout"
        if self._is_retryable_network_error(message):
            return "network/TLS error"
        if "429" in lower or "too many requests" in lower:
            return "rate limit"
        if "recaptcha evaluation failed" in lower or "public_error_unusual_activity" in lower:
            return "reCAPTCHA verification failed"
        if "403" in lower or "recaptcha" in lower:
            return "reCAPTCHA/403 error"
        if any(
            marker in lower
            for marker in [
                "public_error_minor_upload",
                "public_error",
                "http 500",
                "http 502",
                "http 503",
                "http 504",
                "internal error",
                "reason=internal",
                "reason: internal",
                "\"reason\":\"internal\"",
                "server error",
                "upstream error",
            ]
        ):
            return "Flow transient error"
        return None

    def _is_proxy_connection_error(self, error: BaseException | str) -> bool:
        message = str(error).lower()
        return any(
            marker in message
            for marker in [
                "failed to connect to 127.0.0.1 port",
                "failed to connect to localhost port",
                "proxyerror",
                "proxy error",
                "failed to connect to proxy",
                "couldn't connect to server",
                "curl: (7)",
            ]
        )

    def _headers(self, headers: dict[str, str] | None = None, *, account_id: str | None = None) -> dict[str, str]:
        user_agent = os.environ.get("FLOW_USER_AGENT") or self._generate_user_agent(account_id)
        chrome_major = self._chrome_major_from_user_agent(user_agent)
        result = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Accept-Language": os.environ.get(
                "FLOW_ACCEPT_LANGUAGE",
                "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
            ),
            "User-Agent": user_agent,
            "Origin": "https://labs.google",
            "Referer": "https://labs.google/",
            "sec-ch-ua": f'"Chromium";v="{chrome_major}", "Google Chrome";v="{chrome_major}", "Not A(Brand";v="99"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "cross-site",
            "x-browser-channel": "stable",
            "x-browser-copyright": "Copyright 2026 Google LLC. All Rights reserved.",
            "x-browser-year": "2026",
        }
        self._sync_client_hints_with_user_agent(result)
        if headers:
            result.update(headers)
            self._sync_client_hints_with_user_agent(result)
        return result

    def _generate_user_agent(self, account_id: str | None = None) -> str:
        account_key = account_id or "default"
        cached = self._user_agent_cache.get(account_key)
        if cached:
            return cached
        seed = int(hashlib.md5(account_key.encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(seed)
        chrome_versions = ["147.0.7727.56", "146.0.7688.92", "145.0.7649.100"]
        version = rng.choice(chrome_versions)
        user_agent = (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            f"AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{version} Safari/537.36"
        )
        self._user_agent_cache[account_key] = user_agent
        return user_agent

    def _chrome_major_from_user_agent(self, user_agent: str) -> str:
        match = re.search(r"Chrome/(\d+)", user_agent or "")
        return match.group(1) if match else "124"

    def _sync_client_hints_with_user_agent(self, headers: dict[str, str]) -> None:
        ua_lower = str(headers.get("User-Agent") or "").lower()
        if "android" in ua_lower:
            headers["sec-ch-ua-platform"] = '"Android"'
            headers["sec-ch-ua-mobile"] = "?1"
        elif "mac" in ua_lower:
            headers["sec-ch-ua-platform"] = '"macOS"'
            headers["sec-ch-ua-mobile"] = "?0"
        elif "linux" in ua_lower or "x11" in ua_lower:
            headers["sec-ch-ua-platform"] = '"Linux"'
            headers["sec-ch-ua-mobile"] = "?0"
        else:
            headers["sec-ch-ua-platform"] = '"Windows"'
            headers["sec-ch-ua-mobile"] = "?0"

    def _apply_request_fingerprint(self, headers: dict[str, str], fingerprint: dict[str, Any]) -> None:
        user_agent = str(fingerprint.get("user_agent") or "").strip()
        if user_agent:
            headers["User-Agent"] = user_agent
            self._sync_client_hints_with_user_agent(headers)
        accept_language = str(fingerprint.get("accept_language") or "").strip()
        if accept_language:
            headers["Accept-Language"] = accept_language
        header_map = {
            "sec_ch_ua": "sec-ch-ua",
            "sec_ch_ua_mobile": "sec-ch-ua-mobile",
            "sec_ch_ua_platform": "sec-ch-ua-platform",
        }
        for source_key, header_name in header_map.items():
            value = str(fingerprint.get(source_key) or "").strip()
            if value:
                headers[header_name] = value

    def _http_impersonation(self) -> str:
        return (
            get_env_or_credential("FLOW_HTTP_IMPERSONATE")
            or os.environ.get("FLOW_BROWSER_IMPERSONATE")
            or os.environ.get("FLOW_BROWSER")
            or "chrome124"
        )

    def _captcha_provider_proxy_kwargs(self) -> dict[str, Any]:
        proxy_url = self._proxy_url()
        if not proxy_url:
            return {}
        if proxy_url.startswith("socks5://"):
            return {"proxy": proxy_url}
        return {"proxies": {"http": proxy_url, "https": proxy_url}}

    def _captcha_task_min_score(self, task_type: str) -> float | None:
        normalized = str(task_type or "").strip()
        if normalized.endswith("S9"):
            return 0.9
        if normalized.endswith("S7"):
            return 0.7
        return None

    def _set_request_fingerprint(self, fingerprint: dict[str, Any] | None) -> None:
        self._request_fingerprint_ctx.set(
            dict(fingerprint) if isinstance(fingerprint, dict) and fingerprint else None
        )

    async def _report_browser_captcha_error(self, error_reason: str) -> None:
        if self._captcha_method() not in FLOW_BROWSER_CAPTCHA_METHODS:
            return
        try:
            await self._browser_captcha.report_error(error_reason)
        except Exception:
            pass

    async def _report_browser_captcha_finished(self) -> None:
        if self._captcha_method() not in FLOW_BROWSER_CAPTCHA_METHODS:
            return
        try:
            await self._browser_captcha.report_request_finished()
        except Exception:
            pass

    async def _recaptcha_token(
        self,
        project_id: str,
        action: str,
        *,
        headless: bool | None = None,
    ) -> str:
        explicit = _first_env("FLOW_RECAPTCHA_TOKEN", "FLOW_CAPTCHA_TOKEN")
        if explicit:
            self._set_request_fingerprint(None)
            return explicit

        method = self._captcha_method()
        if method in FLOW_CAPTCHA_SOLVERS:
            proxy_url = self._proxy_url()
            self._set_request_fingerprint({"proxy_url": proxy_url} if proxy_url else None)
            token = await self._api_captcha_token(method, project_id, action)
            if token:
                return token
            raise FlowScrapingError(f"{method} did not return a reCAPTCHA token.")
        if method in FLOW_BROWSER_CAPTCHA_METHODS:
            try:
                token = await self._browser_captcha.get_token(
                    project_id,
                    action,
                    headless=headless,
                )
                self._set_request_fingerprint(
                    self._browser_captcha.get_last_fingerprint()
                )
                return token
            except FlowBrowserCaptchaError as exc:
                self._set_request_fingerprint(None)
                raise FlowScrapingError(str(exc)) from exc
        if method == "remote_browser":
            raise FlowScrapingError(
                "FLOW_CAPTCHA_METHOD=remote_browser não é suportado. "
                "Use personal/browser ou um solver externo."
            )
        raise FlowScrapingError(
            "Flow generation requires reCAPTCHA. Configure FLOW_CAPTCHA_METHOD "
            "with personal/browser or yescaptcha/capmonster/ezcaptcha/capsolver."
        )

    async def _api_captcha_token(self, method: str, project_id: str, action: str) -> str | None:
        configs = {
            "yescaptcha": (
                _first_env("FLOW_YESCAPTCHA_API_KEY", "YESCAPTCHA_API_KEY"),
                os.environ.get("FLOW_YESCAPTCHA_BASE_URL", "https://api.yescaptcha.com"),
                _first_env("FLOW_YESCAPTCHA_TASK_TYPE", "YESCAPTCHA_TASK_TYPE") or "RecaptchaV3TaskProxylessM1",
            ),
            "capmonster": (
                _first_env("FLOW_CAPMONSTER_API_KEY", "CAPMONSTER_API_KEY"),
                os.environ.get("FLOW_CAPMONSTER_BASE_URL", "https://api.capmonster.cloud"),
                "RecaptchaV3TaskProxyless",
            ),
            "ezcaptcha": (
                _first_env("FLOW_EZCAPTCHA_API_KEY", "EZCAPTCHA_API_KEY"),
                os.environ.get("FLOW_EZCAPTCHA_BASE_URL", "https://api.ez-captcha.com"),
                "ReCaptchaV3TaskProxylessS9",
            ),
            "capsolver": (
                _first_env("FLOW_CAPSOLVER_API_KEY", "CAPSOLVER_API_KEY"),
                os.environ.get("FLOW_CAPSOLVER_BASE_URL", "https://api.capsolver.com"),
                "ReCaptchaV3EnterpriseTaskProxyLess",
            ),
        }
        client_key, base_url, task_type = configs[method]
        if not client_key:
            raise FlowScrapingError(f"{method} API key is not configured.")

        website_url = f"https://labs.google/fx/tools/flow/project/{project_id}"
        proxy_kwargs = self._captcha_provider_proxy_kwargs()
        async with AsyncSession(trust_env=False) as session:
            task: dict[str, Any] = {
                "websiteURL": website_url,
                "websiteKey": FLOW_RECAPTCHA_SITE_KEY,
                "type": task_type,
                "pageAction": action,
            }
            min_score = self._captcha_task_min_score(task_type)
            if min_score is not None:
                task["minScore"] = min_score
            create_response = await session.post(
                f"{base_url.rstrip('/')}/createTask",
                json={
                    "clientKey": client_key,
                    "task": task,
                },
                timeout=self._control_timeout(),
                impersonate=self._http_impersonation(),
                **proxy_kwargs,
            )
            create_data = create_response.json()
            task_id = create_data.get("taskId")
            if not task_id:
                raise FlowScrapingError(f"{method} createTask failed: {create_data}")

            print(f"[Captcha] Tarefa criada ({method}): {task_id}. Aguardando solução...")
            poll_url = f"{base_url.rstrip('/')}/getTaskResult"
            for poll_idx in range(self._captcha_poll_attempts()):
                self._raise_if_cancelled()
                result_response = await session.post(
                    poll_url,
                    json={"clientKey": client_key, "taskId": task_id},
                    timeout=self._control_timeout(),
                    impersonate=self._http_impersonation(),
                    **proxy_kwargs,
                )
                result_data = result_response.json()
                if result_data.get("status") == "ready":
                    print(f"[Captcha] Solucionado em {poll_idx + 1} tentativas.")
                    token = result_data.get("solution", {}).get("gRecaptchaResponse")
                    return str(token) if token else None
                await asyncio.sleep(self._captcha_poll_interval())
        return None

    def _recaptcha_context(self, token: str) -> dict[str, str]:
        return {"token": token, "applicationType": "RECAPTCHA_APPLICATION_TYPE_WEB"}

    def _build_video_plan(self, payload: dict[str, Any]) -> FlowVideoPlan:
        aspect_ratio = _video_aspect_enum(payload.get("aspectRatio"))
        is_portrait = aspect_ratio == "VIDEO_ASPECT_RATIO_PORTRAIT"
        model = str(payload.get("model") or "").lower()
        is_lite = "lite" in model

        ingredient_paths = list(self._iter_reference_values(payload.get("ingredientImagePaths") or []))
        reference_paths = list(
            self._iter_reference_values(
                payload.get("referenceImagePaths") or payload.get("referenceImagePath") or []
            )
        )
        final_path = str(payload.get("finalImagePath") or "").strip()
        if final_path:
            reference_paths.append(final_path)

        if ingredient_paths and not is_lite:
            return FlowVideoPlan(
                video_type="r2v",
                model_key="veo_3_1_r2v_fast_portrait" if is_portrait else "veo_3_1_r2v_fast_landscape",
                aspect_ratio=aspect_ratio,
                image_paths=ingredient_paths[:3],
                use_v2_model_config=True,
            )

        image_paths = reference_paths[:2]
        if len(image_paths) >= 2:
            if is_lite:
                return FlowVideoPlan(
                    video_type="i2v-start-end",
                    model_key="veo_3_1_interpolation_lite",
                    aspect_ratio=aspect_ratio,
                    image_paths=image_paths,
                    use_v2_model_config=True,
                )
            return FlowVideoPlan(
                video_type="i2v-start-end",
                model_key="veo_3_1_i2v_s_fast_portrait_fl" if is_portrait else "veo_3_1_i2v_s_fast_fl",
                aspect_ratio=aspect_ratio,
                image_paths=image_paths,
            )
        if len(image_paths) == 1:
            if is_lite:
                return FlowVideoPlan(
                    video_type="i2v-start",
                    model_key="veo_3_1_i2v_lite",
                    aspect_ratio=aspect_ratio,
                    image_paths=image_paths,
                    use_v2_model_config=True,
                )
            return FlowVideoPlan(
                video_type="i2v-start",
                model_key="veo_3_1_i2v_s_fast_portrait_fl" if is_portrait else "veo_3_1_i2v_s_fast_fl",
                aspect_ratio=aspect_ratio,
                image_paths=image_paths,
            )

        if is_lite:
            return FlowVideoPlan(
                video_type="t2v",
                model_key="veo_3_1_t2v_lite",
                aspect_ratio=aspect_ratio,
                image_paths=[],
                use_v2_model_config=True,
            )
        return FlowVideoPlan(
            video_type="t2v",
            model_key="veo_3_1_t2v_fast_portrait" if is_portrait else "veo_3_1_t2v_fast",
            aspect_ratio=aspect_ratio,
            image_paths=[],
        )

    def _requested_image_model(self, payload: dict[str, Any]) -> str:
        raw_model = str(payload.get("model") or "").strip().lower()
        if "imagen" in raw_model:
            return "IMAGEN_3_5"
        elif "2.5" in raw_model:
            return "GEM_PIX"
        elif "pro" in raw_model or "3.0" in raw_model or "gemini-3-" in raw_model:
            return FLOW_IMAGE_MODEL_NANO_BANANA_PRO
        return FLOW_IMAGE_MODEL_NANO_BANANA_2

    def _resolve_image_model(
        self,
        payload: dict[str, Any],
    ) -> tuple[str, str | None, dict[str, Any] | None]:
        model_name = self._requested_image_model(payload)
        raw_size = str(
            payload.get("imageSize")
            or payload.get("resolution")
            or payload.get("quality")
            or ""
        ).lower()
        fallback_event = None
        if model_name == FLOW_IMAGE_MODEL_NANO_BANANA_PRO:
            fallback_state = self._active_flow_image_pro_fallback()
            if fallback_state:
                model_name = FLOW_IMAGE_MODEL_NANO_BANANA_2
                fallback_event = self._flow_image_fallback_event(
                    fallback_state,
                    reason="cooldown",
                )
        if "4k" in raw_size or raw_size in {"high", "hd", "ultra"}:
            return model_name, "UPSAMPLE_IMAGE_RESOLUTION_4K", fallback_event
        if "2k" in raw_size or raw_size == "medium":
            return model_name, "UPSAMPLE_IMAGE_RESOLUTION_2K", fallback_event
        return model_name, None, fallback_event

    def _flow_image_pro_fallback_seconds(self) -> int:
        value = get_env_or_credential("FLOW_IMAGE_PRO_FALLBACK_SECONDS") or os.environ.get(
            "FLOW_IMAGE_PRO_FALLBACK_SECONDS"
        )
        if not value:
            return FLOW_IMAGE_PRO_FALLBACK_SECONDS
        try:
            return max(60, int(float(value)))
        except ValueError:
            return FLOW_IMAGE_PRO_FALLBACK_SECONDS

    def _flow_image_pro_fallback_path(self) -> Path:
        return data_dir() / "flow-image-model-fallback.json"

    def _flow_image_fallback_date_key(self) -> str:
        return datetime.now().astimezone().date().isoformat()

    def _load_flow_image_pro_fallback(self) -> dict[str, Any] | None:
        path = self._flow_image_pro_fallback_path()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def _active_flow_image_pro_fallback(self) -> dict[str, Any] | None:
        state = self._load_flow_image_pro_fallback()
        if not state:
            return None
        try:
            expires_at = float(state.get("expiresAt") or 0)
        except (TypeError, ValueError):
            return None
        if str(state.get("date") or "") != self._flow_image_fallback_date_key():
            return None
        if expires_at <= time.time():
            return None
        if str(state.get("blockedModel") or "") != FLOW_IMAGE_MODEL_NANO_BANANA_PRO:
            return None
        if str(state.get("fallbackModel") or "") != FLOW_IMAGE_MODEL_NANO_BANANA_2:
            return None
        return state

    def _activate_flow_image_pro_fallback(self, reason: str) -> dict[str, Any]:
        now = time.time()
        expires_at = now + self._flow_image_pro_fallback_seconds()
        state = {
            "date": self._flow_image_fallback_date_key(),
            "blockedModel": FLOW_IMAGE_MODEL_NANO_BANANA_PRO,
            "blockedModelLabel": "Nano Banana Pro",
            "fallbackModel": FLOW_IMAGE_MODEL_NANO_BANANA_2,
            "fallbackModelLabel": "Nano Banana 2",
            "activatedAt": now,
            "activatedAtIso": datetime.fromtimestamp(now).astimezone().isoformat(),
            "expiresAt": expires_at,
            "expiresAtIso": datetime.fromtimestamp(expires_at).astimezone().isoformat(),
            "reason": reason[:1000],
        }
        path = self._flow_image_pro_fallback_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(state, ensure_ascii=True, indent=2), encoding="utf-8")
        except Exception as exc:
            print(f"[FlowScraping] Falha ao persistir fallback Nano Banana Pro: {exc}")
        return state

    def _flow_image_fallback_event(
        self,
        state: dict[str, Any],
        *,
        reason: str,
    ) -> dict[str, Any]:
        return {
            "fromModel": FLOW_IMAGE_MODEL_NANO_BANANA_PRO,
            "fromModelLabel": "Nano Banana Pro",
            "toModel": FLOW_IMAGE_MODEL_NANO_BANANA_2,
            "toModelLabel": "Nano Banana 2",
            "reason": reason,
            "activeForSeconds": self._flow_image_pro_fallback_seconds(),
            "expiresAt": state.get("expiresAt"),
            "expiresAtIso": state.get("expiresAtIso"),
            "date": state.get("date"),
        }

    def _is_flow_image_daily_quota_error(self, error: BaseException | str) -> bool:
        message = str(error).lower()
        return (
            "public_error_per_model_daily_quota_reached" in message
            or (
                "resource_exhausted" in message
                and "per_model_daily_quota" in message
            )
        )

    def _should_fallback_flow_image_model(
        self,
        model_name: str,
        error: BaseException | str,
    ) -> bool:
        return (
            model_name == FLOW_IMAGE_MODEL_NANO_BANANA_PRO
            and self._is_flow_image_daily_quota_error(error)
        )

    def _flow_image_api_required_error(
        self,
        error: FlowScrapingError,
        fallback_state: dict[str, Any] | None,
    ) -> FlowScrapingError:
        fallback_event = self._flow_image_fallback_event(
            fallback_state or {},
            reason="fallback-quota",
        )
        return FlowScrapingError(
            (
                "Nano Banana 2 tambem atingiu a quota diaria no Flow. "
                "Altere o Servico de geracao para um modelo de imagem da API Gemini "
                "ou Vertex e tente novamente."
            ),
            code=FLOW_IMAGE_API_REQUIRED_ERROR_CODE,
            metadata={
                "flowImageModel": FLOW_IMAGE_MODEL_NANO_BANANA_2,
                "flowImageFallback": fallback_event,
                "flowImageFallbacks": [fallback_event],
                "recommendedImageServices": ["flow-image-api", "vertex-image"],
                "cause": str(error),
            },
        )

    def _video_text_input(self, prompt: str, use_v2_model_config: bool) -> dict[str, Any]:
        if use_v2_model_config:
            return {"structuredPrompt": {"parts": [{"text": prompt}]}}
        return {"prompt": prompt}

    async def _download_media(
        self, url: str, *, fallback_mime: str, filename: str
    ) -> dict[str, Any]:
        async with AsyncSession(trust_env=False) as session:
            response = await session.get(
                url,
                headers={"User-Agent": os.environ.get("FLOW_USER_AGENT") or self._generate_user_agent()},
                proxy=self._proxy_url(media=True),
                timeout=self._download_timeout(),
                impersonate=self._http_impersonation(),
            )
        if response.status_code >= 400:
            raise FlowScrapingError(f"Flow asset download failed HTTP {response.status_code}: {response.text[:300]}")
        return {
            "bytes": response.content,
            "mime_type": (response.headers.get("content-type") or fallback_mime).split(";", 1)[0],
            "filename": filename,
        }

    async def _read_reference_bytes(self, value: Any) -> bytes:
        try:
            data, _mime_type = load_attachment_bytes(value, fallback_mime_type="image/jpeg")
            return data
        except Exception as exc:
            raise FlowScrapingError(f"Could not load Flow reference image: {exc}") from exc

    def _iter_reference_values(self, values: Any):
        if not values:
            return
        if isinstance(values, (str, Path)):
            yield str(values)
            return
        if isinstance(values, dict):
            for key in ("path", "url", "httpUrl", "imageUrl", "src"):
                if values.get(key):
                    yield str(values[key])
                    return
        if isinstance(values, (list, tuple, set)):
            for value in values:
                yield from self._iter_reference_values(value)

    def _raise_if_cancelled(self) -> None:
        if self._cancel_requested:
            self._cancel_requested = False
            raise FlowScrapingError("Flow generation was cancelled.")

    def _session_token(self) -> str | None:
        value = _first_env("FLOW_SESSION_TOKEN", "FLOW_ST_TOKEN", "FLOW_NEXT_AUTH_SESSION_TOKEN")
        if value:
            return _strip_cookie_value(value, "__Secure-next-auth.session-token")
        cookie = _first_env("FLOW_COOKIE", "FLOW_LABS_COOKIE")
        if cookie:
            return _strip_cookie_value(cookie, "__Secure-next-auth.session-token")
        return None

    def _captcha_method(self) -> str:
        return str(get_env_or_credential("FLOW_CAPTCHA_METHOD") or "manual").strip().lower().replace("-", "_")

    def _labs_base_url(self) -> str:
        return os.environ.get("FLOW_LABS_BASE_URL", FLOW_LABS_BASE_URL).rstrip("/")

    def _api_base_url(self) -> str:
        return os.environ.get("FLOW_API_BASE_URL", FLOW_API_BASE_URL).rstrip("/")

    def _proxy_url(self, *, media: bool = False) -> str | None:
        if media:
            value = _first_env("FLOW_MEDIA_PROXY", "FLOW_PROXY")
        else:
            value = _first_env("FLOW_PROXY")
        return value or None

    def _access_refresh_margin_seconds(self) -> float:
        value = get_env_or_credential("FLOW_ACCESS_REFRESH_MARGIN_SECONDS") or "3600"
        try:
            return max(60.0, min(7200.0, float(value)))
        except ValueError:
            return 3600.0

    def _project_pool_size(self) -> int:
        value = get_env_or_credential("FLOW_PROJECT_POOL_SIZE") or "4"
        try:
            return max(1, min(50, int(value)))
        except ValueError:
            return 4

    def _session_id(self) -> str:
        return f";{int(time.time() * 1000)}"

    def _parse_expires(self, value: Any) -> float:
        if not value:
            return time.time() + 50 * 60
        try:
            text = str(value).replace("Z", "+00:00")
            dt = datetime.fromisoformat(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except Exception:
            return time.time() + 50 * 60

    def _timeout(self) -> float:
        return float(os.environ.get("FLOW_TIMEOUT", "120"))

    def _control_timeout(self) -> float:
        return float(os.environ.get("FLOW_CONTROL_TIMEOUT", "10"))

    def _image_timeout(self) -> float:
        return float(os.environ.get("FLOW_IMAGE_TIMEOUT", "300"))

    def _image_timeout_retry_count(self) -> int:
        value = os.environ.get("FLOW_IMAGE_TIMEOUT_RETRY_COUNT") or "1"
        try:
            return max(0, min(3, int(value)))
        except ValueError:
            return 1

    def _image_timeout_retry_delay(self) -> float:
        value = os.environ.get("FLOW_IMAGE_TIMEOUT_RETRY_DELAY") or "0.8"
        try:
            return max(0.0, min(10.0, float(value)))
        except ValueError:
            return 0.8

    def _upsample_timeout(self) -> float:
        return float(os.environ.get("FLOW_UPSAMPLE_TIMEOUT", "300"))

    def _video_submit_timeout(self) -> float:
        return float(os.environ.get("FLOW_VIDEO_SUBMIT_TIMEOUT", "120"))

    def _video_poll_timeout(self) -> float:
        return float(os.environ.get("FLOW_VIDEO_POLL_TIMEOUT", "60"))

    def _download_timeout(self) -> float:
        return float(os.environ.get("FLOW_DOWNLOAD_TIMEOUT", "180"))

    def _poll_interval(self) -> float:
        return float(os.environ.get("FLOW_POLL_INTERVAL", "3"))

    def _poll_attempts(self) -> int:
        return int(os.environ.get("FLOW_MAX_POLL_ATTEMPTS", "200"))

    def _captcha_poll_interval(self) -> float:
        return float(os.environ.get("FLOW_CAPTCHA_POLL_INTERVAL", "3"))

    def _captcha_poll_attempts(self) -> int:
        return int(os.environ.get("FLOW_CAPTCHA_POLL_ATTEMPTS", "40"))


flow_scraping_service = FlowScrapingService()
