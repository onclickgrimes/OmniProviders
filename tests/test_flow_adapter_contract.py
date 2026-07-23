from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import create_app
from app.persistence.credentials import (
    FLOW_ACCOUNT_KIND_CONFIGURATION,
    FLOW_ACCOUNT_KIND_FIELD,
    FLOW_ACCOUNT_KIND_USER,
    CredentialStore,
    list_flow_cookie_runtime_accounts,
)
from app.providers.flow import FlowProviderAdapter
from app.providers.flow.transport import FlowScrapingError, FlowScrapingService
from app.runtime.registry import ProviderRegistry


class FlowTransportFake:
    def __init__(self) -> None:
        self.login_opened = False
        self.session_refreshed = False

    def has_auth_config(self):
        return True

    def captcha_configuration(self):
        return {"method": "personal", "configured": True}

    def supported_models(self):
        return [
            {"id": "gemini-3.1-flash-image", "name": "Nano Banana 2", "type": "image"},
            {"id": "veo_3_1_t2v_fast", "name": "Veo 3.1 Fast", "type": "video"},
        ]

    async def open_login_window(self):
        self.login_opened = True
        return {"success": True, "isOpen": True}

    async def refresh_session_token_from_browser(self):
        self.session_refreshed = True
        return {"success": True, "sessionTokenUpdated": True}


class FlowAdapterContractTest(unittest.TestCase):
    def test_browser_session_refresh_updates_the_registered_account(self) -> None:
        class BrowserFake:
            async def refresh_session_token(self, project_id):
                self.project_id = project_id
                return "new-session-token"

            def status(self):
                return {"available": True}

        with tempfile.TemporaryDirectory() as directory:
            store = CredentialStore(Path(directory) / "accounts.sqlite3")
            store.upsert_account(
                account_id="user-account",
                provider="flow",
                label="User account",
                fields={
                    FLOW_ACCOUNT_KIND_FIELD: FLOW_ACCOUNT_KIND_USER,
                    "FLOW_SESSION_TOKEN": "old-session-token",
                    "FLOW_PROJECT_ID": "project-1",
                    "FLOW_COOKIE": (
                        "other=value; "
                        "__Secure-next-auth.session-token=old-session-token"
                    ),
                },
            )
            browser = BrowserFake()
            with patch("app.persistence.credentials._default_store", store):
                result = asyncio.run(
                    FlowScrapingService(
                        browser_captcha=browser
                    ).refresh_session_token_from_browser()
                )
            saved = store.get_account("user-account", include_secrets=True)
            settings = store.get_provider_settings("flow", include_secrets=True)

        self.assertEqual("user-account", result["accountId"])
        self.assertEqual("project-1", browser.project_id)
        self.assertEqual(
            "new-session-token",
            saved["fields"]["FLOW_SESSION_TOKEN"],
        )
        self.assertIn(
            "__Secure-next-auth.session-token=new-session-token",
            saved["fields"]["FLOW_COOKIE"],
        )
        self.assertIsNone(settings)

    def test_browser_session_refresh_requires_one_registered_account(self) -> None:
        class BrowserFake:
            async def refresh_session_token(self, _project_id):
                raise AssertionError("browser must not be called without an account")

            def status(self):
                return {"available": True}

        with tempfile.TemporaryDirectory() as directory:
            store = CredentialStore(Path(directory) / "accounts.sqlite3")
            with patch("app.persistence.credentials._default_store", store):
                with self.assertRaisesRegex(
                    FlowScrapingError,
                    "Cadastre uma conta Flow",
                ):
                    asyncio.run(
                        FlowScrapingService(
                            browser_captcha=BrowserFake()
                        ).refresh_session_token_from_browser()
                    )

    def test_only_user_registered_accounts_enter_generation_pool(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = CredentialStore(Path(directory) / "accounts.sqlite3")
            store.upsert_account(
                account_id="flow-scraping",
                provider="flow",
                label="Flow runtime",
                fields={
                    FLOW_ACCOUNT_KIND_FIELD: FLOW_ACCOUNT_KIND_CONFIGURATION,
                    "FLOW_SESSION_TOKEN": "technical-token",
                },
            )
            store.upsert_account(
                account_id="user-account",
                provider="flow",
                label="User account",
                fields={
                    FLOW_ACCOUNT_KIND_FIELD: FLOW_ACCOUNT_KIND_USER,
                    "FLOW_SESSION_TOKEN": "user-token",
                    "FLOW_COOKIE": "__Secure-next-auth.session-token=user-token",
                },
            )
            store.upsert_account(
                account_id="unclassified-account",
                provider="flow",
                label="Unclassified",
                fields={"FLOW_SESSION_TOKEN": "unclassified-token"},
            )

            accounts = list_flow_cookie_runtime_accounts(store)

        self.assertEqual(["user-account"], [account["id"] for account in accounts])

    def test_legacy_registered_cookie_account_is_migrated_but_runtime_is_not(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = CredentialStore(Path(directory) / "accounts.sqlite3")
            store.upsert_account(
                account_id="flow-scraping",
                provider="flow",
                label="Flow runtime",
                fields={
                    "FLOW_SESSION_TOKEN": "technical-token",
                    "FLOW_COOKIE": "__Secure-next-auth.session-token=technical-token",
                },
            )
            store.upsert_account(
                account_id="legacy-user-account",
                provider="flow",
                label="Legacy user",
                fields={
                    "FLOW_SESSION_TOKEN": "user-token",
                    "FLOW_COOKIE": "__Secure-next-auth.session-token=user-token",
                    "FLOW_PROJECT_ID": "project-1",
                },
            )

            accounts = list_flow_cookie_runtime_accounts(store)
            migrated = store.get_account("legacy-user-account", include_secrets=True)
            runtime = store.get_account("flow-scraping", include_secrets=True)

        self.assertEqual(["legacy-user-account"], [account["id"] for account in accounts])
        self.assertEqual(
            FLOW_ACCOUNT_KIND_USER,
            migrated["fields"][FLOW_ACCOUNT_KIND_FIELD],
        )
        self.assertEqual(
            FLOW_ACCOUNT_KIND_CONFIGURATION,
            runtime["fields"][FLOW_ACCOUNT_KIND_FIELD],
        )

    def test_environment_token_sources_do_not_create_generation_accounts(self) -> None:
        def legacy_token_source(*names: str) -> str | None:
            if "FLOW_TOKEN_POOL" in names:
                return "legacy-pool-token"
            if "FLOW_SESSION_TOKENS" in names:
                return "legacy-session-token"
            return None

        with (
            patch(
                "app.providers.flow.transport.list_flow_cookie_runtime_accounts",
                return_value=[],
            ),
            patch(
                "app.providers.flow.transport._first_env",
                side_effect=legacy_token_source,
            ),
        ):
            accounts = FlowScrapingService()._flow_accounts()

        self.assertEqual([], accounts)

    def test_exposes_media_operations_with_browser_automation(self) -> None:
        models = asyncio.run(FlowProviderAdapter(FlowTransportFake()).list_models())
        self.assertEqual({"images.generate"}, set(models[0].capabilities.operations))
        self.assertEqual({"videos.generate"}, set(models[1].capabilities.operations))
        self.assertTrue(models[0].metadata["browserAutomation"])

    def test_does_not_publish_models_when_captcha_configuration_is_invalid(self) -> None:
        class InvalidCaptchaTransport(FlowTransportFake):
            def captcha_configuration(self):
                return {"method": "browser", "configured": False}

        models = asyncio.run(
            FlowProviderAdapter(InvalidCaptchaTransport()).list_models()
        )

        self.assertEqual([], models)

    def test_configuration_exposes_browser_and_external_captcha_solvers(self) -> None:
        configuration = FlowProviderAdapter(FlowTransportFake()).configuration()
        fields = configuration["fields"]
        field_keys = {field["key"] for field in fields}
        captcha = next(field for field in fields if field["key"] == "FLOW_CAPTCHA_METHOD")

        self.assertEqual(
            {"personal", "browser", "yescaptcha", "capmonster", "ezcaptcha", "capsolver"},
            {option["value"] for option in captcha["options"]},
        )
        self.assertIn("FLOW_BROWSER_HEADLESS", field_keys)
        self.assertIn("FLOW_BROWSER_FOREGROUND", field_keys)
        self.assertIn("FLOW_BROWSER_CHANNEL", field_keys)
        self.assertIn("FLOW_BROWSER_EXECUTABLE", field_keys)
        self.assertTrue(configuration["features"]["browserAutomation"])
        self.assertEqual("external_node", configuration["features"]["browserDriver"])

    def test_configuration_is_published_by_provider_endpoint(self) -> None:
        client = TestClient(
            create_app(
                registry=ProviderRegistry([FlowProviderAdapter(FlowTransportFake())]),
                require_api_key=False,
            )
        )

        response = client.get("/providers/flow/configuration")

        self.assertEqual(200, response.status_code)
        self.assertEqual("flow", response.json()["provider"])
        self.assertTrue(response.json()["features"]["cookieAccounts"])

    def test_browser_captcha_modes_are_accepted_when_settings_are_saved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = TestClient(
                create_app(
                    registry=ProviderRegistry([FlowProviderAdapter(FlowTransportFake())]),
                    credential_store=CredentialStore(Path(directory) / "accounts.sqlite3"),
                    require_api_key=False,
                )
            )

            response = client.put(
                "/providers/flow/settings",
                json={
                    "fields": {"FLOW_CAPTCHA_METHOD": "browser"},
                },
            )

        self.assertEqual(200, response.status_code)
        self.assertEqual("browser", response.json()["fields"]["FLOW_CAPTCHA_METHOD"])

    def test_browser_configuration_is_valid_when_runtime_is_available(self) -> None:
        class BrowserFake:
            def status(self):
                return {"available": True, "isOpen": False}

        service = FlowScrapingService(browser_captcha=BrowserFake())
        original = service._captcha_method
        service._captcha_method = lambda: "browser"
        try:
            status = service.captcha_configuration()
        finally:
            service._captcha_method = original

        self.assertTrue(status["configured"])
        self.assertNotIn("error", status)

    def test_browser_captcha_is_used_for_recaptcha(self) -> None:
        class BrowserFake:
            def __init__(self):
                self.request = None

            def status(self):
                return {"available": True}

            async def get_token(self, project_id, action, *, headless=None):
                self.request = (project_id, action, headless)
                return "browser-token"

            def get_last_fingerprint(self):
                return {"user_agent": "Browser UA", "proxy_url": ""}

        browser = BrowserFake()
        service = FlowScrapingService(browser_captcha=browser)
        original = service._captcha_method
        service._captcha_method = lambda: "personal"

        async def solve():
            token = await service._recaptcha_token(
                "project-2", "IMAGE_GENERATION", headless=False
            )
            return token, service._request_fingerprint_ctx.get()

        try:
            token, fingerprint = asyncio.run(solve())
        finally:
            service._captcha_method = original

        self.assertEqual("browser-token", token)
        self.assertEqual(("project-2", "IMAGE_GENERATION", False), browser.request)
        self.assertEqual(
            "Browser UA",
            fingerprint["user_agent"],
        )

    def test_provider_login_opens_flow_browser(self) -> None:
        transport = FlowTransportFake()
        result = asyncio.run(FlowProviderAdapter(transport).start_login())

        self.assertTrue(result["success"])
        self.assertTrue(transport.login_opened)

    def test_provider_session_refresh_delegates_to_flow_browser(self) -> None:
        transport = FlowTransportFake()
        result = asyncio.run(FlowProviderAdapter(transport).refresh_session())

        self.assertTrue(result["sessionTokenUpdated"])
        self.assertTrue(transport.session_refreshed)

    def test_validate_flag_is_forwarded_to_flow_transport(self) -> None:
        class StatusTransport(FlowTransportFake):
            def __init__(self) -> None:
                self.validate = None

            async def check_login(self, *, validate=False):
                self.validate = validate
                return {"success": False, "canGenerate": False}

        transport = StatusTransport()
        asyncio.run(FlowProviderAdapter(transport).status(validate=True))

        self.assertTrue(transport.validate)


if __name__ == "__main__":
    unittest.main()
