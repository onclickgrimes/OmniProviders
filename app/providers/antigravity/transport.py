from __future__ import annotations

import base64
import hashlib
import html
import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse
from uuid import uuid4

import httpx

from app.infrastructure.debug import (
    make_json_safe,
    serialize_messages_for_ai_debug,
    write_ai_debug_exchange,
)
from app.persistence.credentials import get_env_or_credential, update_credentials
from app.media.input import ModelMediaError, load_attachment_bytes
from app.protocols.structured_output import StructuredOutputError, extract_json


SERVICE_ID = "antigravity-oauth"
DEFAULT_ANTIGRAVITY_MODEL = "antigravity-gemini-3.5-flash-medium"
ANTIGRAVITY_REDIRECT_URI = "http://localhost:51121/oauth-callback"
ANTIGRAVITY_CALLBACK_HOST = "127.0.0.1"
ANTIGRAVITY_CALLBACK_PORT = 51121
ANTIGRAVITY_VERSION = "1.18.3"
ANTIGRAVITY_DEFAULT_PROJECT_ID = "rising-fact-p41fc"
GOOGLE_OAUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_OAUTH_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v1/userinfo?alt=json"
GEMINI_CLI_USER_AGENT = "google-api-nodejs-client/9.15.1"
ANTIGRAVITY_SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/cclog",
    "https://www.googleapis.com/auth/experimentsandconfigs",
]
ANTIGRAVITY_ENDPOINT_DAILY = "https://daily-cloudcode-pa.googleapis.com"
ANTIGRAVITY_ENDPOINT_DAILY_SANDBOX = "https://daily-cloudcode-pa.sandbox.googleapis.com"
ANTIGRAVITY_ENDPOINT_AUTOPUSH = "https://autopush-cloudcode-pa.sandbox.googleapis.com"
ANTIGRAVITY_ENDPOINT_PROD = "https://cloudcode-pa.googleapis.com"
ANTIGRAVITY_AVAILABLE_MODELS_ENDPOINT = f"{ANTIGRAVITY_ENDPOINT_DAILY}/v1internal:fetchAvailableModels"
ANTIGRAVITY_ENDPOINT_FALLBACKS = [
    ANTIGRAVITY_ENDPOINT_DAILY,
    ANTIGRAVITY_ENDPOINT_DAILY_SANDBOX,
    ANTIGRAVITY_ENDPOINT_PROD,
    ANTIGRAVITY_ENDPOINT_AUTOPUSH,
]
ANTIGRAVITY_LOAD_ENDPOINTS = [
    ANTIGRAVITY_ENDPOINT_PROD,
    ANTIGRAVITY_ENDPOINT_DAILY,
    ANTIGRAVITY_ENDPOINT_DAILY_SANDBOX,
    ANTIGRAVITY_ENDPOINT_AUTOPUSH,
]
TOKEN_EXPIRY_BUFFER_MS = 60 * 1000
# Compatibility aliases for values already saved in projects/settings. The select catalog is
# loaded from fetchAvailableModels instead of this list.
ANTIGRAVITY_LEGACY_MODEL_ALIASES = [
    {
        "id": "antigravity-gemini-3.5-flash-medium",
        "model": "antigravity-gemini-3.5-flash-medium",
        "label": "Antigravity Gemini 3.5 Flash (Medium)",
    },
    {
        "id": "antigravity-gemini-3.5-flash-high",
        "model": "antigravity-gemini-3.5-flash-high",
        "label": "Antigravity Gemini 3.5 Flash (High)",
    },
    {
        "id": "antigravity-gemini-3.5-flash-low",
        "model": "antigravity-gemini-3.5-flash-low",
        "label": "Antigravity Gemini 3.5 Flash (Low)",
    },
    {
        "id": "antigravity-gemini-3.1-pro-low",
        "model": "antigravity-gemini-3.1-pro-low",
        "label": "Antigravity Gemini 3.1 Pro (Low)",
    },
    {
        "id": "antigravity-gemini-3.1-pro-high",
        "model": "antigravity-gemini-3.1-pro-high",
        "label": "Antigravity Gemini 3.1 Pro (High)",
    },
    {
        "id": "antigravity-claude-sonnet-4.6-thinking",
        "model": "antigravity-claude-sonnet-4.6-thinking",
        "label": "Antigravity Claude Sonnet 4.6 (Thinking)",
    },
    {
        "id": "antigravity-claude-opus-4.6-thinking",
        "model": "antigravity-claude-opus-4.6-thinking",
        "label": "Antigravity Claude Opus 4.6 (Thinking)",
    },
]
ANTIGRAVITY_DISPLAY_MODEL_IDS = {
    option["label"].removeprefix("Antigravity ").lower(): option["id"]
    for option in ANTIGRAVITY_LEGACY_MODEL_ALIASES
}
ANTIGRAVITY_DISPLAY_MODEL_IDS.update(
    {
        option["label"].lower(): option["id"]
        for option in ANTIGRAVITY_LEGACY_MODEL_ALIASES
    }
)
ANTIGRAVITY_GEMINI_35_FLASH_BY_TIER = {
    # Internal IDs returned by v1internal:fetchAvailableModels for the current labels.
    "low": "gemini-3.5-flash-extra-low",
    "medium": "gemini-3.5-flash-low",
    "high": "gemini-3-flash-agent",
}


class AntigravityOAuthError(RuntimeError):
    pass


def _client_config_message() -> str:
    return "Credenciais internas do Antigravity OAuth indisponiveis no banco local."


def _antigravity_client_id() -> str:
    return str(get_env_or_credential("ANTIGRAVITY_CLIENT_ID") or "").strip()


def _antigravity_client_secret() -> str:
    return str(get_env_or_credential("ANTIGRAVITY_CLIENT_SECRET") or "").strip()


def _require_antigravity_client_id() -> str:
    client_id = _antigravity_client_id()
    if not client_id:
        raise AntigravityOAuthError(_client_config_message())
    return client_id


def _require_antigravity_client_credentials() -> tuple[str, str]:
    client_id = _require_antigravity_client_id()
    client_secret = _antigravity_client_secret()
    if not client_secret:
        raise AntigravityOAuthError(_client_config_message())
    return client_id, client_secret


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64url_json(payload: dict[str, Any]) -> str:
    return _b64url(json.dumps(payload, separators=(",", ":")).encode("utf-8"))


def _decode_b64url_json(value: str) -> dict[str, Any]:
    normalized = str(value or "").replace("-", "+").replace("_", "/")
    padded = normalized + ("=" * ((4 - len(normalized) % 4) % 4))
    decoded = base64.b64decode(padded).decode("utf-8")
    payload = json.loads(decoded)
    if not isinstance(payload, dict):
        raise AntigravityOAuthError("OAuth state invalido.")
    return payload


def _pkce_pair() -> tuple[str, str]:
    verifier = _b64url(os.urandom(64))
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


def _now_ms() -> int:
    return int(time.time() * 1000)


def _expiry_ms(start_ms: int, expires_in: Any) -> int:
    try:
        seconds = int(expires_in)
    except (TypeError, ValueError):
        seconds = 3600
    if seconds <= 0:
        return start_ms
    return start_ms + seconds * 1000


def _metadata_platform() -> str:
    return "PLATFORM_UNSPECIFIED"


def _client_metadata() -> str:
    return json.dumps(
        {
            "ideType": "ANTIGRAVITY",
            "platform": _metadata_platform(),
            "pluginType": "GEMINI",
        },
        separators=(",", ":"),
    )


def _load_headers(access_token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "User-Agent": GEMINI_CLI_USER_AGENT,
        "Client-Metadata": _client_metadata(),
    }


def _content_headers(access_token: str) -> dict[str, str]:
    platform = "windows/amd64" if os.name == "nt" else "darwin/arm64"
    return {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "User-Agent": f"antigravity/{ANTIGRAVITY_VERSION} {platform}",
    }


def _extract_error_message(payload: Any) -> str:
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error.get("status") or error.get("code") or "").strip()
        if error:
            return str(error).strip()
        for key in ("detail", "message"):
            if payload.get(key):
                return str(payload.get(key)).strip()
    return str(payload or "").strip()


def _parse_refresh_parts(raw: str | None) -> tuple[str, str, str]:
    parts = str(raw or "").strip().split("|")
    refresh_token = parts[0].strip() if len(parts) >= 1 else ""
    project_id = parts[1].strip() if len(parts) >= 2 else ""
    managed_project_id = parts[2].strip() if len(parts) >= 3 else ""
    return refresh_token, project_id, managed_project_id


def _format_refresh(refresh_token: str, project_id: str = "", managed_project_id: str = "") -> str:
    base = f"{refresh_token}|{project_id or ''}"
    return f"{base}|{managed_project_id}" if managed_project_id else base


def _coerce_float(value: str | None) -> float:
    try:
        return float(str(value or "").strip())
    except (TypeError, ValueError):
        return 0.0


def _normalize_requested_model(raw_model: str | None) -> str:
    model = str(raw_model or DEFAULT_ANTIGRAVITY_MODEL).strip()
    if not model or model == "gemini-web-auto":
        model = DEFAULT_ANTIGRAVITY_MODEL
    if model.lower().startswith("antigravity:"):
        model = model.split(":", 1)[1].strip()
    model = ANTIGRAVITY_DISPLAY_MODEL_IDS.get(model.lower(), model)
    model = re.sub(r"^antigravity-", "", model, flags=re.IGNORECASE)
    model = re.sub(r"-preview(?:-customtools)?$", "", model, flags=re.IGNORECASE)
    return model


def _dedupe_model_specs(specs: list[tuple[str, str | None]]) -> list[tuple[str, str | None]]:
    unique: list[tuple[str, str | None]] = []
    seen: set[tuple[str, str | None]] = set()
    for spec in specs:
        key = (spec[0], spec[1])
        if key in seen:
            continue
        seen.add(key)
        unique.append(spec)
    return unique


def _candidate_antigravity_models(raw_model: str | None) -> list[tuple[str, str | None]]:
    model = _normalize_requested_model(raw_model)

    tier_match = re.search(r"-(minimal|low|medium|high)$", model, flags=re.IGNORECASE)
    tier = tier_match.group(1).lower() if tier_match else None
    base_model = re.sub(r"-(minimal|low|medium|high)$", "", model, flags=re.IGNORECASE) if tier else model

    if re.match(r"^gemini-3\.5-flash", base_model, flags=re.IGNORECASE):
        effective_tier = tier if tier in ANTIGRAVITY_GEMINI_35_FLASH_BY_TIER else "medium"
        return _dedupe_model_specs(
            [
                (ANTIGRAVITY_GEMINI_35_FLASH_BY_TIER[effective_tier], effective_tier),
            ]
        )

    is_gemini_3_pro = bool(re.match(r"^gemini-3(?:\.\d+)?-pro", base_model, flags=re.IGNORECASE))
    is_gemini_3_flash = bool(re.match(r"^gemini-3(?:\.\d+)?-flash", base_model, flags=re.IGNORECASE))
    is_claude_sonnet_46 = bool(re.match(r"^claude-sonnet-4(?:\.|-)?6(?:-thinking)?$", base_model, flags=re.IGNORECASE))
    is_claude_opus_46 = bool(re.match(r"^claude-opus-4(?:\.|-)?6(?:-thinking)?$", base_model, flags=re.IGNORECASE))

    if is_gemini_3_pro:
        effective_tier = tier if tier in {"low", "high"} else "low"
        if effective_tier == "high":
            return _dedupe_model_specs(
                [
                    ("gemini-3.1-pro-low", "high"),
                    (f"{base_model}-high", "high"),
                ]
            )
        return _dedupe_model_specs(
            [
                ("gemini-3.1-pro-low", "low"),
                (f"{base_model}-low", "low"),
            ]
        )
    if is_gemini_3_flash:
        return _dedupe_model_specs([(base_model, tier or "medium")])
    if is_claude_sonnet_46:
        return _dedupe_model_specs(
            [
                ("claude-sonnet-4-6", "high"),
                ("claude-sonnet-4-6", None),
            ]
        )
    if is_claude_opus_46:
        return _dedupe_model_specs(
            [
                ("claude-opus-4-6-thinking", None),
                ("claude-opus-4-6", "high"),
            ]
        )
    if "gemini-3" in base_model.lower():
        return _dedupe_model_specs([(base_model, tier or "low")])
    return _dedupe_model_specs([(model, None)])


def _resolve_antigravity_model(raw_model: str | None) -> tuple[str, str | None]:
    return _candidate_antigravity_models(raw_model)[0]


def _slug_antigravity_model_label(label: str, fallback: str) -> str:
    source = str(label or fallback or "").strip().lower()
    slug = re.sub(r"[^a-z0-9.]+", "-", source).strip("-")
    slug = re.sub(r"-+", "-", slug)
    return f"antigravity-{slug}" if slug else DEFAULT_ANTIGRAVITY_MODEL


def _available_model_supported(meta: dict[str, Any]) -> bool:
    return bool(meta.get("supportsImages") or meta.get("supportsVideo") or meta.get("supportsThinking"))


def _available_model_display_rank(key: str, meta: dict[str, Any]) -> tuple[int, int, str]:
    provider = str(meta.get("modelProvider") or "")
    provider_rank = 0 if provider == "MODEL_PROVIDER_GOOGLE" else 1 if provider == "MODEL_PROVIDER_ANTHROPIC" else 2
    agent_rank = 0 if str(key or "").endswith("-agent") else 1
    return (provider_rank, agent_rank, str(key or ""))


def _available_model_sort_key(option: dict[str, Any]) -> tuple[int, str, str]:
    label = str(option.get("label") or "").lower()
    provider = str(option.get("modelProvider") or "")
    provider_rank = 0 if provider == "MODEL_PROVIDER_GOOGLE" else 1 if provider == "MODEL_PROVIDER_ANTHROPIC" else 2
    return (provider_rank, label, str(option.get("backendModel") or ""))


def _serialize_available_models(
    raw_models: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, tuple[str, str | None]]]:
    best_by_model_id: dict[str, tuple[tuple[int, int, str], dict[str, Any]]] = {}

    for key, raw_meta in raw_models.items():
        if not isinstance(raw_meta, dict):
            continue
        backend_model = str(key or "").strip()
        display_name = str(raw_meta.get("displayName") or "").strip()
        haystack = f"{backend_model} {display_name}".lower()
        if not backend_model or not display_name or "gpt-oss" in haystack:
            continue
        if not _available_model_supported(raw_meta):
            continue

        model_id = _slug_antigravity_model_label(display_name, backend_model)
        label = display_name if display_name.lower().startswith("antigravity") else f"Antigravity {display_name}"
        option = {
            "id": model_id,
            "value": model_id,
            "provider": "antigravity",
            "model": model_id,
            "backendModel": backend_model,
            "label": label,
            "displayName": display_name,
            "supportsImages": bool(raw_meta.get("supportsImages")),
            "supportsVideo": bool(raw_meta.get("supportsVideo")),
            "supportsThinking": bool(raw_meta.get("supportsThinking")),
            "thinkingBudget": raw_meta.get("thinkingBudget"),
            "maxTokens": raw_meta.get("maxTokens"),
            "maxOutputTokens": raw_meta.get("maxOutputTokens"),
            "tagTitle": raw_meta.get("tagTitle"),
            "modelProvider": raw_meta.get("modelProvider"),
            "apiProvider": raw_meta.get("apiProvider"),
        }
        rank = _available_model_display_rank(backend_model, raw_meta)
        current = best_by_model_id.get(model_id)
        if current and current[0] <= rank:
            continue
        best_by_model_id[model_id] = (rank, option)

    options = [item[1] for item in best_by_model_id.values()]
    options.sort(key=_available_model_sort_key)

    aliases: dict[str, tuple[str, str | None]] = {}
    for option in options:
        backend_model = str(option.get("backendModel") or "").strip()
        model_id = str(option.get("model") or "").strip()
        if not backend_model or not model_id:
            continue
        aliases[model_id.lower()] = (backend_model, None)
        aliases[_normalize_requested_model(model_id).lower()] = (backend_model, None)
    return options, aliases


class AntigravityOAuthService:
    def __init__(self) -> None:
        self._server: ThreadingHTTPServer | None = None
        self._server_thread: threading.Thread | None = None
        self._server_lock = threading.Lock()
        self._model_resolution_cache: dict[str, tuple[str, str | None]] = {}
        self._dynamic_model_aliases: dict[str, tuple[str, str | None]] = {}

    def _default_model(self) -> str:
        return get_env_or_credential("ANTIGRAVITY_MODEL") or DEFAULT_ANTIGRAVITY_MODEL

    def _candidate_models_for_request(self, requested_model: str) -> list[tuple[str, str | None]]:
        cache_key = _normalize_requested_model(requested_model).lower()
        dynamic_model = (
            self._dynamic_model_aliases.get(str(requested_model or "").strip().lower())
            or self._dynamic_model_aliases.get(cache_key)
        )
        candidates = _dedupe_model_specs([dynamic_model]) if dynamic_model else _candidate_antigravity_models(requested_model)
        cached = self._model_resolution_cache.get(cache_key)
        if cached and cached in candidates:
            return _dedupe_model_specs([cached, *candidates])
        return candidates

    def _refresh_value(self) -> str:
        return str(get_env_or_credential("ANTIGRAVITY_REFRESH_TOKEN") or "").strip()

    def _refresh_parts(self) -> tuple[str, str, str]:
        return _parse_refresh_parts(self._refresh_value())

    def _project_id(self) -> str:
        _refresh, packed_project, managed_project = self._refresh_parts()
        return (
            str(get_env_or_credential("ANTIGRAVITY_PROJECT_ID") or "").strip()
            or packed_project
            or managed_project
            or ANTIGRAVITY_DEFAULT_PROJECT_ID
        )

    def _project_hint(self) -> str:
        _refresh, packed_project, managed_project = self._refresh_parts()
        return (
            str(get_env_or_credential("ANTIGRAVITY_PROJECT_ID") or "").strip()
            or packed_project
            or managed_project
        )

    def has_auth_config(self) -> bool:
        refresh_token, _project, _managed = self._refresh_parts()
        if refresh_token:
            return True
        access_token = str(get_env_or_credential("ANTIGRAVITY_ACCESS_TOKEN") or "").strip()
        expires_at = _coerce_float(get_env_or_credential("ANTIGRAVITY_TOKEN_EXPIRES_AT"))
        return bool(access_token and expires_at > _now_ms() + TOKEN_EXPIRY_BUFFER_MS)

    def check_config(self) -> dict[str, Any]:
        configured = self.has_auth_config()
        return {
            "success": configured,
            "isLoggedIn": configured,
            "canGenerate": configured,
            "validationModel": self._default_model(),
            "effectiveModel": _resolve_antigravity_model(self._default_model())[0],
            "accountStatus": "configured" if configured else "missing",
            "userId": get_env_or_credential("ANTIGRAVITY_EMAIL") or "",
            "userIdVerified": bool(get_env_or_credential("ANTIGRAVITY_EMAIL")),
            "userIdSource": "oauth",
            "projectId": self._project_id() if configured else "",
            **(
                {}
                if configured
                else {"message": "Faca login Google em Configuracoes > API e Modelos > Antigravity OAuth."}
            ),
        }

    def _ensure_callback_server(self) -> None:
        with self._server_lock:
            if self._server:
                return

            service = self

            class OAuthCallbackHandler(BaseHTTPRequestHandler):
                def log_message(self, _format: str, *args: Any) -> None:
                    return

                def _html_response(self, status: int, title: str, message: str) -> None:
                    body = (
                        "<!doctype html><html><head><meta charset='utf-8'>"
                        f"<title>{html.escape(title)}</title>"
                        "<style>body{font-family:system-ui;margin:48px;line-height:1.5}"
                        "code{background:#f3f3f3;padding:2px 4px;border-radius:4px}</style>"
                        "</head><body>"
                        f"<h1>{html.escape(title)}</h1>"
                        f"<p>{html.escape(message)}</p>"
                        "<p>Voce pode voltar para o Uno Studio.</p>"
                        "</body></html>"
                    ).encode("utf-8")
                    self.send_response(status)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

                def do_GET(self) -> None:
                    parsed = urlparse(self.path)
                    if parsed.path != "/oauth-callback":
                        self._html_response(404, "Rota invalida", "Callback OAuth nao reconhecido.")
                        return

                    query = parse_qs(parsed.query)
                    error = str(query.get("error", [""])[0] or "").strip()
                    if error:
                        description = str(query.get("error_description", [""])[0] or error).strip()
                        self._html_response(400, "Login Antigravity falhou", description)
                        return

                    code = str(query.get("code", [""])[0] or "").strip()
                    state = str(query.get("state", [""])[0] or "").strip()
                    if not code or not state:
                        self._html_response(400, "Login Antigravity falhou", "Callback sem code/state.")
                        return

                    try:
                        result = service.complete_login_sync(code, state)
                    except Exception as exc:
                        self._html_response(500, "Login Antigravity falhou", str(exc))
                        return

                    email = str(result.get("email") or "conta Google").strip()
                    project_id = str(result.get("projectId") or "").strip()
                    suffix = f" Projeto: {project_id}." if project_id else ""
                    self._html_response(
                        200,
                        "Login Antigravity concluido",
                        f"Credenciais salvas para {email}.{suffix}",
                    )

            try:
                self._server = ThreadingHTTPServer(
                    (ANTIGRAVITY_CALLBACK_HOST, ANTIGRAVITY_CALLBACK_PORT),
                    OAuthCallbackHandler,
                )
            except OSError as exc:
                raise AntigravityOAuthError(
                    f"Nao foi possivel abrir o callback OAuth em {ANTIGRAVITY_REDIRECT_URI}: {exc}"
                ) from exc

            self._server_thread = threading.Thread(
                target=self._server.serve_forever,
                name="antigravity-oauth-callback",
                daemon=True,
            )
            self._server_thread.start()

    def start_login(self) -> dict[str, Any]:
        client_id = _require_antigravity_client_id()
        self._ensure_callback_server()
        verifier, challenge = _pkce_pair()
        state = _b64url_json({"verifier": verifier, "projectId": self._project_hint()})
        auth_params = {
            "client_id": client_id,
            "response_type": "code",
            "redirect_uri": ANTIGRAVITY_REDIRECT_URI,
            "scope": " ".join(ANTIGRAVITY_SCOPES),
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": state,
            "access_type": "offline",
            "prompt": "consent",
        }
        auth_url = f"{GOOGLE_OAUTH_AUTH_URL}?{urlencode(auth_params)}"
        return {
            "success": True,
            "isLoggedIn": False,
            "authUrl": auth_url,
            "callbackUrl": ANTIGRAVITY_REDIRECT_URI,
            "message": "Abra o login Google e conclua o callback local.",
        }

    def complete_login_sync(self, code: str, state: str) -> dict[str, Any]:
        state_payload = _decode_b64url_json(state)
        verifier = str(state_payload.get("verifier") or "").strip()
        if not verifier:
            raise AntigravityOAuthError("OAuth state sem PKCE verifier.")
        project_hint = str(state_payload.get("projectId") or "").strip()
        client_id, client_secret = _require_antigravity_client_credentials()
        start_ms = _now_ms()
        headers = {
            "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
            "Accept": "*/*",
            "User-Agent": GEMINI_CLI_USER_AGENT,
        }
        data = {
            "client_id": client_id,
            "client_secret": client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": ANTIGRAVITY_REDIRECT_URI,
            "code_verifier": verifier,
        }

        with httpx.Client(timeout=30.0) as client:
            response = client.post(GOOGLE_OAUTH_TOKEN_URL, headers=headers, data=data)
            try:
                payload = response.json()
            except Exception:
                payload = {"text": response.text}
            if response.status_code >= 400:
                raise AntigravityOAuthError(
                    _extract_error_message(payload) or f"Token exchange HTTP {response.status_code}"
                )

            access_token = str(payload.get("access_token") or "").strip()
            refresh_token = str(payload.get("refresh_token") or "").strip()
            if not access_token:
                raise AntigravityOAuthError("Google nao retornou access token.")
            if not refresh_token:
                raise AntigravityOAuthError("Google nao retornou refresh token.")

            email = ""
            userinfo_response = client.get(
                GOOGLE_USERINFO_URL,
                headers={"Authorization": f"Bearer {access_token}", "User-Agent": GEMINI_CLI_USER_AGENT},
            )
            if userinfo_response.status_code < 400:
                try:
                    userinfo = userinfo_response.json()
                    email = str(userinfo.get("email") or "").strip()
                except Exception:
                    email = ""

            project_id = project_hint
            if not project_id:
                project_id = self._discover_project_id_sync(access_token, client)

        packed_refresh = _format_refresh(refresh_token, project_id)
        expires_at = _expiry_ms(start_ms, payload.get("expires_in"))
        update_credentials(
            SERVICE_ID,
            {
                "ANTIGRAVITY_REFRESH_TOKEN": packed_refresh,
                "ANTIGRAVITY_ACCESS_TOKEN": access_token,
                "ANTIGRAVITY_TOKEN_EXPIRES_AT": str(expires_at),
                "ANTIGRAVITY_PROJECT_ID": project_id,
                "ANTIGRAVITY_EMAIL": email,
            },
        )
        return {
            "success": True,
            "refreshToken": packed_refresh,
            "accessToken": access_token,
            "expiresAt": expires_at,
            "email": email,
            "projectId": project_id,
        }

    def _discover_project_id_sync(self, access_token: str, client: httpx.Client) -> str:
        for endpoint in ANTIGRAVITY_LOAD_ENDPOINTS:
            try:
                response = client.post(
                    f"{endpoint}/v1internal:loadCodeAssist",
                    headers=_load_headers(access_token),
                    json={
                        "metadata": {
                            "ideType": "ANTIGRAVITY",
                            "platform": _metadata_platform(),
                            "pluginType": "GEMINI",
                        }
                    },
                )
            except Exception:
                continue
            if response.status_code >= 400:
                continue
            try:
                payload = response.json()
            except Exception:
                continue
            project = payload.get("cloudaicompanionProject") if isinstance(payload, dict) else None
            if isinstance(project, str) and project:
                return project
            if isinstance(project, dict) and project.get("id"):
                return str(project.get("id") or "").strip()
        return ""

    async def _discover_project_id(self, access_token: str) -> str:
        async with httpx.AsyncClient(timeout=30.0) as client:
            for endpoint in ANTIGRAVITY_LOAD_ENDPOINTS:
                try:
                    response = await client.post(
                        f"{endpoint}/v1internal:loadCodeAssist",
                        headers=_load_headers(access_token),
                        json={
                            "metadata": {
                                "ideType": "ANTIGRAVITY",
                                "platform": _metadata_platform(),
                                "pluginType": "GEMINI",
                            }
                        },
                    )
                except Exception:
                    continue
                if response.status_code >= 400:
                    continue
                try:
                    payload = response.json()
                except Exception:
                    continue
                project = payload.get("cloudaicompanionProject") if isinstance(payload, dict) else None
                if isinstance(project, str) and project:
                    return project
                if isinstance(project, dict) and project.get("id"):
                    return str(project.get("id") or "").strip()
        return ""

    async def _get_access_token(self, *, force_refresh: bool = False) -> str:
        access_token = str(get_env_or_credential("ANTIGRAVITY_ACCESS_TOKEN") or "").strip()
        expires_at = _coerce_float(get_env_or_credential("ANTIGRAVITY_TOKEN_EXPIRES_AT"))
        if not force_refresh and access_token and expires_at > _now_ms() + TOKEN_EXPIRY_BUFFER_MS:
            return access_token

        refresh_token, packed_project, managed_project = self._refresh_parts()
        if not refresh_token:
            if access_token:
                return access_token
            raise AntigravityOAuthError("Antigravity OAuth nao configurado. Faca login nas Configuracoes.")

        client_id, client_secret = _require_antigravity_client_credentials()
        start_ms = _now_ms()
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                GOOGLE_OAUTH_TOKEN_URL,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": client_id,
                    "client_secret": client_secret,
                },
            )
        try:
            payload = response.json()
        except Exception:
            payload = {"text": response.text}
        if response.status_code >= 400:
            raise AntigravityOAuthError(
                _extract_error_message(payload) or f"Refresh token HTTP {response.status_code}"
            )

        new_access_token = str(payload.get("access_token") or "").strip()
        if not new_access_token:
            raise AntigravityOAuthError("Google nao retornou access token no refresh.")
        next_refresh = str(payload.get("refresh_token") or "").strip() or refresh_token
        project_id = str(get_env_or_credential("ANTIGRAVITY_PROJECT_ID") or "").strip() or packed_project
        if not project_id:
            project_id = await self._discover_project_id(new_access_token)
        packed_refresh = _format_refresh(next_refresh, project_id, managed_project)
        expires_at_ms = _expiry_ms(start_ms, payload.get("expires_in"))
        update_credentials(
            SERVICE_ID,
            {
                "ANTIGRAVITY_REFRESH_TOKEN": packed_refresh,
                "ANTIGRAVITY_ACCESS_TOKEN": new_access_token,
                "ANTIGRAVITY_TOKEN_EXPIRES_AT": str(expires_at_ms),
                "ANTIGRAVITY_PROJECT_ID": project_id,
            },
        )
        return new_access_token

    async def _ensure_project_id(self) -> str:
        project_id = self._project_hint()
        if project_id:
            return project_id
        access_token = await self._get_access_token()
        project_id = await self._discover_project_id(access_token)
        if project_id:
            update_credentials(SERVICE_ID, {"ANTIGRAVITY_PROJECT_ID": project_id})
            return project_id
        return ANTIGRAVITY_DEFAULT_PROJECT_ID

    def _content_part(self, part: dict[str, Any]) -> dict[str, Any] | None:
        part_type = str(part.get("type") or "text").strip().lower()
        if part_type == "text":
            text = str(part.get("text") or "").strip()
            return {"text": text} if text else None

        fallback_mime_type = {
            "image": "image/png",
            "video": "video/webm",
            "audio": "audio/wav",
        }.get(part_type)
        if not fallback_mime_type:
            text = str(part.get("text") or "").strip()
            return {"text": text} if text else None

        try:
            media_bytes, mime_type = load_attachment_bytes(
                part,
                fallback_mime_type=fallback_mime_type,
            )
        except ModelMediaError as exc:
            raise AntigravityOAuthError(str(exc)) from exc

        return {
            "inlineData": {
                "mimeType": mime_type,
                "data": base64.b64encode(media_bytes).decode("ascii"),
            }
        }

    def _message_parts(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        raw_parts = message.get("parts") if isinstance(message.get("parts"), list) else None
        if raw_parts:
            parts = [
                built
                for built in (self._content_part(part) for part in raw_parts if isinstance(part, dict))
                if built
            ]
            if parts:
                return parts
        content = str(message.get("content") or "").strip()
        return [{"text": content}] if content else []

    def _build_request_payload(
        self,
        messages: list[dict[str, Any]],
        *,
        actual_model: str,
        thinking_level: str | None,
        temperature: float,
        response_json: bool,
    ) -> dict[str, Any]:
        system_parts: list[str] = []
        contents: list[dict[str, Any]] = []

        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "user").strip().lower()
            if role == "system":
                system_parts.extend(
                    part.get("text", "")
                    for part in (message.get("parts") or [])
                    if isinstance(part, dict) and part.get("type") == "text"
                )
                if not message.get("parts"):
                    system_parts.append(str(message.get("content") or ""))
                continue
            parts = self._message_parts(message)
            if not parts:
                continue
            contents.append(
                {
                    "role": "model" if role == "assistant" else "user",
                    "parts": parts,
                }
            )

        if not contents:
            raise AntigravityOAuthError("Antigravity request sem conteudo de usuario.")

        generation_config: dict[str, Any] = {"temperature": temperature}
        if response_json:
            generation_config["responseMimeType"] = "application/json"
        if thinking_level:
            generation_config["thinkingConfig"] = {
                "includeThoughts": False,
                "thinkingLevel": thinking_level,
            }

        request: dict[str, Any] = {
            "contents": contents,
            "generationConfig": generation_config,
            "sessionId": f"omni-providers-{uuid4()}",
        }
        system_text = "\n\n".join(part.strip() for part in system_parts if str(part or "").strip())
        if system_text:
            request["systemInstruction"] = {
                "role": "user",
                "parts": [{"text": system_text}],
            }

        wrapper = {
            "project": self._project_id(),
            "model": actual_model,
            "request": request,
            "requestType": "agent",
            "userAgent": "antigravity",
            "requestId": f"agent-{uuid4()}",
        }
        return wrapper

    def _extract_text(self, payload: Any) -> str:
        if not isinstance(payload, dict):
            return ""
        response_root = payload.get("response") if isinstance(payload.get("response"), dict) else payload
        candidates = response_root.get("candidates") if isinstance(response_root, dict) else None
        if not isinstance(candidates, list):
            return str(response_root.get("text") or payload.get("text") or "").strip() if isinstance(response_root, dict) else ""

        chunks: list[str] = []
        fallback_thought_chunks: list[str] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content") if isinstance(candidate.get("content"), dict) else {}
            parts = content.get("parts") if isinstance(content.get("parts"), list) else []
            for part in parts:
                if not isinstance(part, dict):
                    continue
                text = str(part.get("text") or "").strip()
                if not text:
                    continue
                if part.get("thought") is True:
                    fallback_thought_chunks.append(text)
                else:
                    chunks.append(text)
        return "\n".join(chunks or fallback_thought_chunks).strip()

    def _clean_tool_schema(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {"type": "object", "properties": {}}
        banned = {
            "$schema",
            "$defs",
            "definitions",
            "$ref",
            "$comment",
            "const",
            "additionalProperties",
            "propertyNames",
            "patternProperties",
            "title",
            "enumDescriptions",
        }
        result: dict[str, Any] = {}
        for key, item in value.items():
            if key in banned or key.startswith("x-"):
                continue
            if isinstance(item, dict):
                result[key] = self._clean_tool_schema(item)
            elif isinstance(item, list):
                result[key] = [
                    self._clean_tool_schema(entry) if isinstance(entry, dict) else entry
                    for entry in item
                ]
            elif key == "type" and isinstance(item, str):
                result[key] = item.lower()
            else:
                result[key] = item
        properties = result.get("properties")
        required = result.get("required")
        if isinstance(properties, dict) and isinstance(required, list):
            valid = [name for name in required if isinstance(name, str) and name in properties]
            if valid:
                result["required"] = valid
            else:
                result.pop("required", None)
        return result

    def _native_request_payload(
        self,
        messages: list[dict[str, Any]],
        *,
        actual_model: str,
        thinking_level: str | None,
        temperature: float,
        tools: list[dict[str, Any]],
        tool_choice: Any,
    ) -> dict[str, Any]:
        system_parts: list[str] = []
        contents: list[dict[str, Any]] = []
        function_names: dict[str, str] = {}
        for message in messages:
            if not isinstance(message, dict):
                continue
            item_type = str(message.get("type") or "")
            if item_type == "function_call":
                call_id = str(message.get("call_id") or message.get("id") or uuid4())
                name = str(message.get("name") or "").strip()
                function_names[call_id] = name
                arguments = message.get("arguments")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                contents.append(
                    {
                        "role": "model",
                        "parts": [
                            {
                                "functionCall": {
                                    "id": call_id,
                                    "name": name,
                                    "args": arguments if isinstance(arguments, dict) else {},
                                }
                            }
                        ],
                    }
                )
                continue
            if item_type == "function_call_output":
                call_id = str(message.get("call_id") or "")
                output = message.get("output")
                if isinstance(output, str):
                    try:
                        output_value = json.loads(output)
                    except json.JSONDecodeError:
                        output_value = output
                else:
                    output_value = output
                contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "id": call_id,
                                    "name": function_names.get(call_id) or call_id,
                                    "response": {"result": output_value},
                                }
                            }
                        ],
                    }
                )
                continue
            role = str(message.get("role") or "user").lower()
            if role == "system":
                if message.get("parts"):
                    system_parts.extend(
                        str(part.get("text") or "")
                        for part in message["parts"]
                        if isinstance(part, dict) and part.get("type") == "text"
                    )
                else:
                    system_parts.append(str(message.get("content") or ""))
                continue
            parts = self._message_parts(message)
            if parts:
                contents.append(
                    {"role": "model" if role == "assistant" else "user", "parts": parts}
                )
        if not contents:
            raise AntigravityOAuthError("Antigravity request sem conteudo de usuario.")

        generation_config: dict[str, Any] = {"temperature": temperature}
        if thinking_level:
            generation_config["thinkingConfig"] = {
                "includeThoughts": False,
                "thinkingLevel": thinking_level,
            }
        request: dict[str, Any] = {
            "contents": contents,
            "generationConfig": generation_config,
            "sessionId": f"omni-providers-{uuid4()}",
        }
        system_text = "\n\n".join(item.strip() for item in system_parts if item.strip())
        if system_text:
            request["systemInstruction"] = {
                "role": "user",
                "parts": [{"text": system_text}],
            }
        declarations: list[dict[str, Any]] = []
        if tool_choice != "none":
            for tool in tools:
                function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
                name = str(function.get("name") or "").strip()
                if not name:
                    continue
                declarations.append(
                    {
                        "name": name,
                        "description": str(function.get("description") or ""),
                        "parameters": self._clean_tool_schema(
                            function.get("parameters") or function.get("input_schema") or {}
                        ),
                    }
                )
        if declarations:
            request["tools"] = [{"functionDeclarations": declarations}]
            if tool_choice == "required":
                request["toolConfig"] = {"functionCallingConfig": {"mode": "ANY"}}
            elif isinstance(tool_choice, dict):
                selected = tool_choice.get("name")
                function_choice = tool_choice.get("function")
                if isinstance(function_choice, dict):
                    selected = function_choice.get("name")
                if selected:
                    request["toolConfig"] = {
                        "functionCallingConfig": {
                            "mode": "ANY",
                            "allowedFunctionNames": [str(selected)],
                        }
                    }
        return {
            "project": self._project_id(),
            "model": actual_model,
            "request": request,
            "requestType": "agent",
            "userAgent": "omni-providers",
            "requestId": f"agent-{uuid4()}",
        }

    async def _post_generate_content(
        self, *, endpoint: str, access_token: str, payload: dict[str, Any]
    ) -> tuple[int, Any]:
        async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=15.0)) as client:
            response = await client.post(
                f"{endpoint.rstrip('/')}/v1internal:generateContent",
                headers=_content_headers(access_token),
                json=payload,
            )
        try:
            response_payload = response.json()
        except Exception:
            response_payload = {"text": response.text}
        return response.status_code, response_payload

    async def generate_native(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        requested_model = str(model or self._default_model()).strip() or self._default_model()
        await self._ensure_project_id()
        model_candidates = self._candidate_models_for_request(requested_model)
        access_token = await self._get_access_token()
        last_error: Exception | None = None
        for actual_model, thinking_level in model_candidates:
            payload = self._native_request_payload(
                messages,
                actual_model=actual_model,
                thinking_level=thinking_level,
                temperature=temperature if temperature is not None else 0.4,
                tools=tools or [],
                tool_choice=tool_choice,
            )
            for endpoint in ANTIGRAVITY_ENDPOINT_FALLBACKS:
                for token_attempt in range(2):
                    try:
                        status_code, response_payload = await self._post_generate_content(
                            endpoint=endpoint,
                            access_token=access_token,
                            payload=payload,
                        )
                        if status_code == 401 and token_attempt == 0:
                            access_token = await self._get_access_token(force_refresh=True)
                            continue
                        if status_code >= 400:
                            last_error = AntigravityOAuthError(
                                _extract_error_message(response_payload)
                                or f"Antigravity HTTP {status_code}"
                            )
                            break
                        response_root = (
                            response_payload.get("response")
                            if isinstance(response_payload, dict)
                            and isinstance(response_payload.get("response"), dict)
                            else response_payload
                        )
                        function_calls: list[dict[str, Any]] = []
                        for candidate in (
                            response_root.get("candidates")
                            if isinstance(response_root, dict)
                            and isinstance(response_root.get("candidates"), list)
                            else []
                        ):
                            content = candidate.get("content") if isinstance(candidate, dict) else None
                            parts = content.get("parts") if isinstance(content, dict) else None
                            for part in parts if isinstance(parts, list) else []:
                                call = part.get("functionCall") if isinstance(part, dict) else None
                                if not isinstance(call, dict) or not call.get("name"):
                                    continue
                                function_calls.append(
                                    {
                                        "call_id": str(call.get("id") or f"call_{uuid4().hex}"),
                                        "name": str(call["name"]),
                                        "arguments": call.get("args") or {},
                                    }
                                )
                        usage_metadata = (
                            response_root.get("usageMetadata")
                            if isinstance(response_root, dict)
                            and isinstance(response_root.get("usageMetadata"), dict)
                            else {}
                        )
                        usage = {
                            "input_tokens": int(usage_metadata.get("promptTokenCount") or 0),
                            "output_tokens": int(usage_metadata.get("candidatesTokenCount") or 0),
                            "total_tokens": int(usage_metadata.get("totalTokenCount") or 0),
                        }
                        self._model_resolution_cache[
                            _normalize_requested_model(requested_model).lower()
                        ] = (actual_model, thinking_level)
                        return {
                            "text": self._extract_text(response_payload),
                            "functionCalls": function_calls,
                            "effectiveModel": actual_model,
                            "usage": usage,
                        }
                    except Exception as exc:
                        last_error = exc
                        break
        raise AntigravityOAuthError(
            str(last_error or "Unknown Antigravity native generation failure.")
        )

    async def generate_text_from_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float = 0.4,
        response_json: bool = False,
        debug_context: dict[str, Any] | None = None,
    ) -> str:
        requested_model = str(model or self._default_model()).strip() or self._default_model()
        await self._ensure_project_id()
        model_candidates = self._candidate_models_for_request(requested_model)
        last_actual_model = model_candidates[0][0]
        last_debug_request: dict[str, Any] = {
            "model": requested_model,
            "modelCandidates": [
                {"actualModel": actual_model, "thinkingLevel": thinking_level}
                for actual_model, thinking_level in model_candidates
            ],
            "messages": serialize_messages_for_ai_debug(messages),
        }

        endpoints = ANTIGRAVITY_ENDPOINT_FALLBACKS
        last_error: Exception | None = None
        access_token = await self._get_access_token()

        for actual_model, thinking_level in model_candidates:
            request_payload = self._build_request_payload(
                messages,
                actual_model=actual_model,
                thinking_level=thinking_level,
                temperature=temperature,
                response_json=response_json,
            )
            last_actual_model = actual_model
            debug_request = {
                **last_debug_request,
                "actualModel": actual_model,
                "thinkingLevel": thinking_level,
                "project": request_payload.get("project"),
                "request": make_json_safe(request_payload),
            }
            last_debug_request = debug_request

            for endpoint in endpoints:
                for token_attempt in range(2):
                    try:
                        async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=15.0)) as client:
                            response = await client.post(
                                f"{endpoint.rstrip('/')}/v1internal:generateContent",
                                headers=_content_headers(access_token),
                                json=request_payload,
                            )
                        try:
                            response_payload = response.json()
                        except Exception:
                            response_payload = {"text": response.text}

                        if response.status_code == 401 and token_attempt == 0:
                            access_token = await self._get_access_token(force_refresh=True)
                            continue

                        if response.status_code >= 400:
                            message = _extract_error_message(response_payload) or f"HTTP {response.status_code}"
                            error = AntigravityOAuthError(f"Antigravity {response.status_code}: {message}")
                            write_ai_debug_exchange(
                                provider="antigravity",
                                model=actual_model,
                                operation="antigravity.generate_content",
                                request={**debug_request, "endpoint": endpoint},
                                response=response_payload,
                                error=error,
                                metadata=debug_context,
                            )
                            last_error = error
                            break

                        text = self._extract_text(response_payload)
                        write_ai_debug_exchange(
                            provider="antigravity",
                            model=actual_model,
                            operation="antigravity.generate_content",
                            request={**debug_request, "endpoint": endpoint},
                            response=response_payload,
                            metadata=debug_context,
                        )
                        if not text:
                            last_error = AntigravityOAuthError("Antigravity retornou resposta vazia.")
                            break
                        self._model_resolution_cache[_normalize_requested_model(requested_model).lower()] = (
                            actual_model,
                            thinking_level,
                        )
                        return text
                    except Exception as exc:
                        last_error = exc
                        break

        write_ai_debug_exchange(
            provider="antigravity",
            model=last_actual_model,
            operation="antigravity.generate_content",
            request=last_debug_request,
            error=last_error,
            metadata=debug_context,
        )
        raise AntigravityOAuthError(str(last_error or "Falha desconhecida no Antigravity."))

    async def generate_text(
        self,
        prompt: str,
        *,
        model: str | None = None,
        response_json: bool = False,
        temperature: float = 0.4,
        debug_context: dict[str, Any] | None = None,
    ) -> str:
        return await self.generate_text_from_messages(
            [{"role": "user", "content": prompt}],
            model=model,
            response_json=response_json,
            temperature=temperature,
            debug_context=debug_context,
        )

    async def generate_json_from_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float = 0.4,
        debug_context: dict[str, Any] | None = None,
    ) -> Any:
        text = await self.generate_text_from_messages(
            messages,
            model=model,
            temperature=temperature,
            response_json=True,
            debug_context=debug_context,
        )
        try:
            return extract_json(text)
        except StructuredOutputError as exc:
            raise AntigravityOAuthError(str(exc)) from exc

    async def validate_credentials(self) -> dict[str, Any]:
        if not self.has_auth_config():
            return {
                "success": False,
                "isLoggedIn": False,
                "canGenerate": False,
                "error": "Nenhuma credencial Antigravity OAuth configurada.",
            }
        try:
            text = await self.generate_text(
                "Responda exatamente: ok",
                model=self._default_model(),
                temperature=0,
                debug_context={"callSite": "validate_antigravity_credentials"},
            )
            return {
                "success": True,
                "isLoggedIn": True,
                "canGenerate": True,
                "validationModel": self._default_model(),
                "effectiveModel": _resolve_antigravity_model(self._default_model())[0],
                "accountStatus": "authenticated",
                "validationText": text[:180],
                "userId": get_env_or_credential("ANTIGRAVITY_EMAIL") or "",
                "userIdVerified": bool(get_env_or_credential("ANTIGRAVITY_EMAIL")),
                "userIdSource": "oauth",
                "projectId": self._project_id(),
            }
        except Exception as exc:
            return {
                "success": False,
                "isLoggedIn": False,
                "canGenerate": False,
                "validationModel": self._default_model(),
                "effectiveModel": _resolve_antigravity_model(self._default_model())[0],
                "error": str(exc),
                "userId": get_env_or_credential("ANTIGRAVITY_EMAIL") or "",
                "projectId": self._project_id(),
            }

    async def send_message(self, prompt: str, *, model: str | None = None) -> dict[str, Any]:
        text = await self.generate_text(prompt, model=model)
        return {
            "success": True,
            "provider": "antigravity",
            "text": text,
            "response": text,
            "modelUsed": str(model or self._default_model()),
        }

    async def stream_message(self, prompt: str, *, model: str | None = None):
        yield await self.generate_text(prompt, model=model)

    async def list_models(self) -> dict[str, Any]:
        if not self.has_auth_config():
            self._dynamic_model_aliases = {}
            return {
                "success": False,
                "isLoggedIn": False,
                "canGenerate": False,
                "error": "Nenhuma credencial Antigravity OAuth configurada.",
                "models": [],
                "defaultModel": self._default_model(),
            }

        await self._ensure_project_id()
        access_token = await self._get_access_token()
        payload = {"project": self._project_id()}

        for token_attempt in range(2):
            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.post(
                    ANTIGRAVITY_AVAILABLE_MODELS_ENDPOINT,
                    headers=_content_headers(access_token),
                    json=payload,
                )
            try:
                response_payload = response.json()
            except Exception:
                response_payload = {"text": response.text}

            if response.status_code == 401 and token_attempt == 0:
                access_token = await self._get_access_token(force_refresh=True)
                continue

            if response.status_code >= 400:
                self._dynamic_model_aliases = {}
                return {
                    "success": False,
                    "isLoggedIn": True,
                    "canGenerate": False,
                    "error": _extract_error_message(response_payload) or f"HTTP {response.status_code}",
                    "models": [],
                    "defaultModel": self._default_model(),
                    "httpStatus": response.status_code,
                }

            raw_models = response_payload.get("models") if isinstance(response_payload, dict) else {}
            if not isinstance(raw_models, dict):
                raw_models = {}
            models, aliases = _serialize_available_models(raw_models)
            self._dynamic_model_aliases = aliases
            return {
                "success": True,
                "isLoggedIn": True,
                "canGenerate": True,
                "models": models,
                "modelIds": [str(option.get("model") or "") for option in models if option.get("model")],
                "backendModelIds": [
                    str(option.get("backendModel") or "")
                    for option in models
                    if option.get("backendModel")
                ],
                "defaultModel": self._default_model(),
                "projectId": self._project_id(),
            }

        self._dynamic_model_aliases = {}
        return {
            "success": False,
            "isLoggedIn": False,
            "canGenerate": False,
            "error": "Falha ao listar modelos Antigravity.",
            "models": [],
            "defaultModel": self._default_model(),
        }


antigravity_oauth_service = AntigravityOAuthService()
