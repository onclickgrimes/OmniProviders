from __future__ import annotations

import json
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app
from app.persistence.credentials import CredentialStore
from app.runtime.registry import ProviderRegistry


class ProviderAccountsContractTest(unittest.TestCase):
    def test_provider_settings_are_persisted_outside_provider_accounts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = CredentialStore(Path(directory) / "accounts.sqlite3")
            client = TestClient(
                create_app(
                    registry=ProviderRegistry(),
                    credential_store=store,
                    require_api_key=False,
                )
            )

            saved = client.put(
                "/providers/flow/settings",
                json={
                    "fields": {
                        "FLOW_CAPTCHA_METHOD": "browser",
                        "FLOW_YESCAPTCHA_API_KEY": "solver-secret",
                    }
                },
            )
            listed_accounts = client.get("/providers/flow/accounts")
            loaded = client.get("/providers/flow/settings")

        self.assertEqual(200, saved.status_code)
        self.assertEqual(
            "********",
            saved.json()["fields"]["FLOW_YESCAPTCHA_API_KEY"],
        )
        self.assertEqual([], listed_accounts.json()["data"])
        self.assertEqual("browser", loaded.json()["fields"]["FLOW_CAPTCHA_METHOD"])
        self.assertNotIn("solver-secret", loaded.text)

    def test_legacy_flow_configuration_account_is_migrated_and_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "accounts.sqlite3"
            now = time.time()
            connection = sqlite3.connect(database_path)
            try:
                connection.execute(
                    """
                    CREATE TABLE provider_accounts (
                        id TEXT PRIMARY KEY,
                        provider TEXT NOT NULL,
                        label TEXT NOT NULL,
                        fields_json TEXT NOT NULL,
                        enabled INTEGER NOT NULL DEFAULT 1,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    INSERT INTO provider_accounts
                        (id, provider, label, fields_json, enabled, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "flow-scraping",
                        "flow",
                        "Flow runtime",
                        json.dumps(
                            {
                                "FLOW_CAPTCHA_METHOD": "browser",
                                "FLOW_BROWSER_HEADLESS": "true",
                                "FLOW_SESSION_TOKEN": "must-not-become-setting",
                                "FLOW_COOKIE": "must-not-become-setting",
                                "FLOW_ACCOUNT_KIND": "configuration",
                            }
                        ),
                        1,
                        now,
                        now,
                    ),
                )
                connection.commit()
            finally:
                connection.close()

            store = CredentialStore(database_path)
            settings = store.get_provider_settings("flow", include_secrets=True)
            legacy_account = store.get_account(
                "flow-scraping",
                include_secrets=True,
            )

        self.assertIsNone(legacy_account)
        self.assertEqual("browser", settings["fields"]["FLOW_CAPTCHA_METHOD"])
        self.assertEqual("true", settings["fields"]["FLOW_BROWSER_HEADLESS"])
        self.assertNotIn("FLOW_SESSION_TOKEN", settings["fields"])
        self.assertNotIn("FLOW_COOKIE", settings["fields"])
        self.assertNotIn("FLOW_ACCOUNT_KIND", settings["fields"])

    def test_reserved_flow_configuration_id_cannot_be_created_as_account(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = TestClient(
                create_app(
                    registry=ProviderRegistry(),
                    credential_store=CredentialStore(
                        Path(directory) / "accounts.sqlite3"
                    ),
                    require_api_key=False,
                )
            )

            response = client.post(
                "/providers/flow/accounts",
                json={
                    "id": "flow-scraping",
                    "fields": {"FLOW_SESSION_TOKEN": "technical-token"},
                },
            )

        self.assertEqual(409, response.status_code)
        self.assertEqual("reserved_provider_settings_id", response.json()["error"]["code"])

    def test_saves_provider_account_and_never_returns_secret_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = CredentialStore(Path(directory) / "accounts.sqlite3")
            client = TestClient(
                create_app(
                    registry=ProviderRegistry(),
                    credential_store=store,
                    require_api_key=False,
                )
            )

            created = client.post(
                "/providers/kiro/accounts",
                json={
                    "id": "kiro-default",
                    "label": "Conta Kiro",
                    "fields": {
                        "KIRO_REFRESH_TOKEN": "super-secret",
                        "KIRO_REGION": "us-east-1",
                    },
                },
            )

            self.assertEqual(200, created.status_code)
            self.assertEqual("********", created.json()["fields"]["KIRO_REFRESH_TOKEN"])
            self.assertEqual("us-east-1", created.json()["fields"]["KIRO_REGION"])

            listed = client.get("/providers/kiro/accounts")

            self.assertEqual(200, listed.status_code)
            self.assertEqual("********", listed.json()["data"][0]["fields"]["KIRO_REFRESH_TOKEN"])
            self.assertNotIn("super-secret", listed.text)

    def test_provider_scoped_values_do_not_leak_between_gemini_and_vertex(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = CredentialStore(Path(directory) / "accounts.sqlite3")
            store.upsert_account(
                account_id="gemini-api",
                provider="gemini",
                label="Gemini API",
                fields={"GEMINI_MODEL": "gemini-only", "GEMINI_TTS_VOICE": "Kore"},
            )
            store.upsert_account(
                account_id="vertex-api",
                provider="vertex",
                label="Vertex API",
                fields={"GEMINI_MODEL": "vertex-only", "GEMINI_TTS_VOICE": "Sadaltager"},
            )

            self.assertEqual("gemini-only", store.get_provider_value("gemini", "GEMINI_MODEL"))
            self.assertEqual("vertex-only", store.get_provider_value("vertex", "GEMINI_MODEL"))
            self.assertEqual("Sadaltager", store.get_provider_value("vertex", "GEMINI_TTS_VOICE"))


if __name__ == "__main__":
    unittest.main()
