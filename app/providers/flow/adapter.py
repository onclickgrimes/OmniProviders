from __future__ import annotations

from typing import Any

from app.domain.models import ModelCapabilities, ModelDescriptor, ModelInvocation, ModelResult, ModelRuntimeError
from app.providers.flow.transport import (
    FLOW_BROWSER_CAPTCHA_METHODS,
    FLOW_CAPTCHA_SOLVERS,
    FlowScrapingError,
    FlowScrapingService,
)


FLOW_CONFIGURATION_FIELDS: list[dict[str, Any]] = [
    {
        "key": "FLOW_CAPTCHA_METHOD",
        "label": "Captcha solver",
        "required": True,
        "defaultValue": "personal",
        "options": [
            {"value": "personal", "label": "Browser pessoal"},
            {"value": "browser", "label": "Browser"},
            {"value": "yescaptcha", "label": "YesCaptcha"},
            {"value": "capmonster", "label": "CapMonster"},
            {"value": "ezcaptcha", "label": "EZCaptcha"},
            {"value": "capsolver", "label": "CapSolver"},
        ],
    },
    {
        "key": "FLOW_BROWSER_HEADLESS",
        "label": "Browser headless",
        "defaultValue": "true",
        "options": [
            {"value": "true", "label": "Headless"},
            {"value": "false", "label": "Mostrar navegador"},
        ],
    },
    {
        "key": "FLOW_BROWSER_FOREGROUND",
        "label": "Trazer browser para frente",
        "defaultValue": "false",
        "options": [
            {"value": "false", "label": "Não roubar foco"},
            {"value": "true", "label": "Trazer para frente"},
        ],
    },
    {
        "key": "FLOW_BROWSER_CHANNEL",
        "label": "Browser",
        "defaultValue": "chrome",
        "options": [
            {"value": "chrome", "label": "Google Chrome instalado"},
            {"value": "msedge", "label": "Microsoft Edge instalado"},
            {"value": "chromium", "label": "Chromium configurado"},
        ],
    },
    {
        "key": "FLOW_BROWSER_EXECUTABLE",
        "label": "Executável do browser",
        "placeholder": "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    },
    {
        "key": "FLOW_BROWSER_COUNT",
        "label": "Browsers simultâneos",
        "defaultValue": "1",
    },
    {
        "key": "FLOW_YESCAPTCHA_API_KEY",
        "label": "YesCaptcha API key",
        "secret": True,
    },
    {
        "key": "FLOW_YESCAPTCHA_TASK_TYPE",
        "label": "YesCaptcha task type",
        "defaultValue": "RecaptchaV3TaskProxylessM1",
        "options": [
            {"value": "RecaptchaV3TaskProxyless", "label": "Proxyless"},
            {"value": "RecaptchaV3TaskProxylessM1", "label": "M1"},
            {"value": "RecaptchaV3TaskProxylessM1S7", "label": "M1S7 minScore 0.7"},
            {"value": "RecaptchaV3TaskProxylessM1S9", "label": "M1S9 minScore 0.9"},
        ],
    },
    {"key": "FLOW_CAPMONSTER_API_KEY", "label": "CapMonster API key", "secret": True},
    {"key": "FLOW_EZCAPTCHA_API_KEY", "label": "EZCaptcha API key", "secret": True},
    {"key": "FLOW_CAPSOLVER_API_KEY", "label": "CapSolver API key", "secret": True},
    {
        "key": "FLOW_IMAGE_CONCURRENCY",
        "label": "Concorrência de imagens por conta",
        "placeholder": "-1 = ilimitado",
    },
    {
        "key": "FLOW_VIDEO_CONCURRENCY",
        "label": "Concorrência de vídeos por conta",
        "placeholder": "-1 = ilimitado",
    },
    {
        "key": "FLOW_TOKEN_WAIT_TIMEOUT",
        "label": "Espera por conta livre (s)",
        "defaultValue": "120",
    },
    {
        "key": "FLOW_ACCESS_REFRESH_MARGIN_SECONDS",
        "label": "Antecedência para renovar access token (s)",
        "defaultValue": "3600",
    },
    {
        "key": "FLOW_PROJECT_POOL_SIZE",
        "label": "Projetos por conta",
        "defaultValue": "4",
    },
    {
        "key": "FLOW_PROXY",
        "label": "Proxy do Flow e captcha",
        "placeholder": "http://127.0.0.1:8080",
    },
    {
        "key": "FLOW_BROWSER_PROXY",
        "label": "Proxy do browser/captcha",
        "placeholder": "Usa o proxy do Flow quando vazio",
    },
    {
        "key": "FLOW_MEDIA_PROXY",
        "label": "Proxy de upload/download",
        "placeholder": "Usa o proxy do Flow quando vazio",
    },
    {
        "key": "FLOW_BROWSER_PAGE_HOLD_SECONDS",
        "label": "Manter aba do captcha (s)",
        "defaultValue": "20",
    },
    {
        "key": "FLOW_BROWSER_FRESH_RESTART_EVERY_N_SOLVES",
        "label": "Reiniciar browser a cada N captchas",
        "defaultValue": "10",
    },
    {
        "key": "FLOW_HTTP_IMPERSONATE",
        "label": "Perfil HTTP curl_cffi",
        "defaultValue": "chrome124",
    },
]
FLOW_ACCOUNT_FIELDS = frozenset(
    {
        "FLOW_ACCOUNT_KIND",
        "FLOW_COOKIE",
        "FLOW_PROJECT_ID",
        "FLOW_SESSION_TOKEN",
    }
)


class FlowProviderAdapter:
    provider_id = "flow"

    def __init__(self, transport: Any | None = None) -> None:
        self._transport = transport or FlowScrapingService()

    def configuration(self) -> dict[str, Any]:
        return {
            "label": "Flow",
            "fields": FLOW_CONFIGURATION_FIELDS,
            "features": {
                "browserAutomation": True,
                "browserDriver": "external_node",
                "bundledBrowser": False,
                "cookieAccounts": True,
                "captchaSolvers": sorted(
                    {*FLOW_CAPTCHA_SOLVERS, *FLOW_BROWSER_CAPTCHA_METHODS}
                ),
            },
        }

    def validate_settings_fields(self, fields: dict[str, Any]) -> None:
        method = str(fields.get("FLOW_CAPTCHA_METHOD") or "").strip().lower().replace("-", "_")
        if method and method not in {*FLOW_CAPTCHA_SOLVERS, *FLOW_BROWSER_CAPTCHA_METHODS}:
            message = f"FLOW_CAPTCHA_METHOD={method} não é suportado pelo OmniProviders."
            raise ModelRuntimeError(
                message,
                code="unsupported_captcha_method",
                param="FLOW_CAPTCHA_METHOD",
                status_code=422,
            )

    def validate_account_fields(self, fields: dict[str, Any]) -> None:
        invalid_fields = sorted(set(fields) - FLOW_ACCOUNT_FIELDS)
        if invalid_fields:
            raise ModelRuntimeError(
                (
                    "Flow account records accept identity fields only. "
                    "Save runtime configuration in /providers/flow/settings."
                ),
                code="invalid_flow_account_fields",
                param=invalid_fields[0],
                status_code=422,
            )

    async def list_models(self, *, refresh: bool = False) -> list[ModelDescriptor]:
        del refresh
        if not self._transport.has_auth_config():
            return []
        captcha_configuration = getattr(self._transport, "captcha_configuration", None)
        if callable(captcha_configuration):
            captcha = captcha_configuration()
            if not bool(captcha.get("configured")):
                return []
        result: list[ModelDescriptor] = []
        for item in self._transport.supported_models():
            model = str(item.get("id") or "").strip()
            media_type = str(item.get("type") or "").strip()
            if not model or media_type not in {"image", "video"}:
                continue
            result.append(
                ModelDescriptor(
                    provider=self.provider_id,
                    model=model,
                    label=str(item.get("name") or model),
                    discovery="account_live",
                    capabilities=ModelCapabilities(
                        input_modalities=frozenset({"text", "image"}),
                        output_modalities=frozenset({media_type}),
                        operations=frozenset({f"{media_type}s.generate"}),
                    ),
                    metadata={"captcha": "browser_or_external", "browserAutomation": True},
                )
            )
        return result

    async def invoke(self, model: str, request: ModelInvocation) -> ModelResult:
        del model, request
        raise ModelRuntimeError("Flow only supports image and video generation.", code="unsupported_operation")

    async def generate_images(self, model: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return await self._transport.generate_images({**payload, "model": model})
        except FlowScrapingError as exc:
            raise ModelRuntimeError(
                str(exc),
                code=exc.code or "provider_error",
                status_code=502,
            ) from exc

    async def generate_video(self, model: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return await self._transport.generate_video({**payload, "model": model})
        except FlowScrapingError as exc:
            raise ModelRuntimeError(str(exc), code=exc.code or "provider_error", status_code=502) from exc

    async def status(self, *, validate: bool = False) -> dict[str, Any]:
        return await self._transport.check_login(validate=validate)

    async def start_login(self) -> dict[str, Any]:
        handler = getattr(self._transport, "open_login_window", None)
        if callable(handler):
            try:
                return await handler()
            except FlowScrapingError as exc:
                raise ModelRuntimeError(
                    str(exc),
                    code=exc.code or "browser_login_failed",
                    status_code=502,
                ) from exc
        return self._transport.login_instructions()

    async def refresh_session(self) -> dict[str, Any]:
        handler = getattr(self._transport, "refresh_session_token_from_browser", None)
        if not callable(handler):
            raise ModelRuntimeError(
                "Flow browser session refresh is unavailable.",
                code="unsupported_auth_flow",
            )
        try:
            return await handler()
        except FlowScrapingError as exc:
            raise ModelRuntimeError(
                str(exc),
                code=exc.code or "browser_session_refresh_failed",
                status_code=502,
            ) from exc

    async def close(self) -> None:
        await self._transport.close()
