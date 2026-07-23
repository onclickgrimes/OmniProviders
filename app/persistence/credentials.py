from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from app import config


MASK = "********"
_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "COOKIE", "CREDENTIAL")
FLOW_ACCOUNT_KIND_FIELD = "FLOW_ACCOUNT_KIND"
FLOW_ACCOUNT_KIND_USER = "user"
FLOW_ACCOUNT_KIND_CONFIGURATION = "configuration"
_FLOW_CONFIGURATION_ACCOUNT_IDS = frozenset({"flow", "flow-scraping"})
_FLOW_IDENTITY_FIELDS = frozenset(
    {
        FLOW_ACCOUNT_KIND_FIELD,
        "FLOW_ACCESS_TOKEN",
        "FLOW_ACCESS_TOKENS",
        "FLOW_AT_TOKEN",
        "FLOW_COOKIE",
        "FLOW_COOKIE_ACCOUNTS",
        "FLOW_LABS_COOKIE",
        "FLOW_NEXT_AUTH_SESSION_TOKEN",
        "FLOW_PROJECT_ID",
        "FLOW_SESSION_TOKEN",
        "FLOW_SESSION_TOKENS",
        "FLOW_ST_TOKEN",
        "FLOW_TOKEN_POOL",
        "accessToken",
        "cookieHeader",
        "projectId",
        "sessionToken",
    }
)


def _flow_settings_fields(fields: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in fields.items()
        if str(key) not in _FLOW_IDENTITY_FIELDS
    }


def _replace_flow_session_cookie(cookie_header: str, session_token: str) -> str:
    session_cookie = "__Secure-next-auth.session-token"
    parts = [
        part.strip()
        for part in str(cookie_header or "").split(";")
        if part.strip()
    ]
    replaced = False
    for index, part in enumerate(parts):
        name, separator, _value = part.partition("=")
        if separator and name.strip() == session_cookie:
            parts[index] = f"{session_cookie}={session_token}"
            replaced = True
    if not replaced:
        parts.append(f"{session_cookie}={session_token}")
    return "; ".join(parts)


def _is_secret_field(name: str) -> bool:
    normalized = name.upper()
    if normalized.endswith("_PATH") or normalized.endswith("_FILE"):
        return False
    return any(marker in normalized for marker in _SECRET_MARKERS)


def _public_fields(fields: dict[str, Any]) -> dict[str, Any]:
    return {
        key: MASK if _is_secret_field(key) and value not in (None, "") else value
        for key, value in fields.items()
    }


class CredentialStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _ensure_schema(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS provider_accounts (
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
                "CREATE INDEX IF NOT EXISTS idx_provider_accounts_provider "
                "ON provider_accounts(provider, enabled)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS provider_settings (
                    provider TEXT PRIMARY KEY,
                    fields_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            self._migrate_legacy_flow_configuration(connection)

    @staticmethod
    def _migrate_legacy_flow_configuration(
        connection: sqlite3.Connection,
    ) -> None:
        legacy = connection.execute(
            "SELECT * FROM provider_accounts WHERE id = ?",
            ("flow-scraping",),
        ).fetchone()
        if not legacy:
            return

        legacy_fields = _flow_settings_fields(json.loads(legacy["fields_json"]))
        current = connection.execute(
            "SELECT * FROM provider_settings WHERE provider = ?",
            ("flow",),
        ).fetchone()
        if legacy_fields:
            current_fields = json.loads(current["fields_json"]) if current else {}
            merged_fields = {**legacy_fields, **current_fields}
            created_at = (
                float(current["created_at"])
                if current
                else float(legacy["created_at"])
            )
            updated_at = max(
                float(legacy["updated_at"]),
                float(current["updated_at"]) if current else 0,
            )
            connection.execute(
                """
                INSERT INTO provider_settings
                    (provider, fields_json, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(provider) DO UPDATE SET
                    fields_json = excluded.fields_json,
                    updated_at = excluded.updated_at
                """,
                (
                    "flow",
                    json.dumps(merged_fields, ensure_ascii=False),
                    created_at,
                    updated_at,
                ),
            )
        connection.execute(
            "DELETE FROM provider_accounts WHERE id = ?",
            ("flow-scraping",),
        )

    def upsert_account(
        self,
        *,
        account_id: str,
        provider: str,
        label: str,
        fields: dict[str, Any],
        enabled: bool = True,
    ) -> dict[str, Any]:
        now = time.time()
        with self._connection() as connection:
            current = connection.execute(
                "SELECT fields_json, created_at FROM provider_accounts WHERE id = ?",
                (account_id,),
            ).fetchone()
            current_fields = json.loads(current["fields_json"]) if current else {}
            merged_fields = dict(current_fields)
            for key, value in fields.items():
                if value == MASK and key in current_fields:
                    continue
                merged_fields[str(key)] = value
            created_at = float(current["created_at"]) if current else now
            connection.execute(
                """
                INSERT INTO provider_accounts
                    (id, provider, label, fields_json, enabled, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    provider = excluded.provider,
                    label = excluded.label,
                    fields_json = excluded.fields_json,
                    enabled = excluded.enabled,
                    updated_at = excluded.updated_at
                """,
                (
                    account_id,
                    provider,
                    label,
                    json.dumps(merged_fields, ensure_ascii=False),
                    1 if enabled else 0,
                    created_at,
                    now,
                ),
            )
        return self.get_account(account_id, include_secrets=False) or {}

    def get_account(
        self, account_id: str, *, include_secrets: bool = False
    ) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM provider_accounts WHERE id = ?", (account_id,)
            ).fetchone()
        return self._serialize_row(row, include_secrets=include_secrets) if row else None

    def list_accounts(
        self, provider: str | None = None, *, include_secrets: bool = False
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM provider_accounts"
        params: tuple[Any, ...] = ()
        if provider:
            query += " WHERE provider = ?"
            params = (provider,)
        query += " ORDER BY created_at, id"
        with self._connection() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._serialize_row(row, include_secrets=include_secrets) for row in rows]

    def delete_account(self, account_id: str) -> bool:
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM provider_accounts WHERE id = ?", (account_id,)
            )
            return cursor.rowcount > 0

    def upsert_provider_settings(
        self,
        *,
        provider: str,
        fields: dict[str, Any],
    ) -> dict[str, Any]:
        canonical_provider = provider.strip().lower().replace("_", "-")
        persisted_fields = (
            _flow_settings_fields(fields)
            if canonical_provider == "flow"
            else dict(fields)
        )
        now = time.time()
        with self._connection() as connection:
            current = connection.execute(
                "SELECT fields_json, created_at FROM provider_settings WHERE provider = ?",
                (canonical_provider,),
            ).fetchone()
            current_fields = json.loads(current["fields_json"]) if current else {}
            merged_fields = dict(current_fields)
            for key, value in persisted_fields.items():
                if value == MASK and key in current_fields:
                    continue
                merged_fields[str(key)] = value
            created_at = float(current["created_at"]) if current else now
            connection.execute(
                """
                INSERT INTO provider_settings
                    (provider, fields_json, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(provider) DO UPDATE SET
                    fields_json = excluded.fields_json,
                    updated_at = excluded.updated_at
                """,
                (
                    canonical_provider,
                    json.dumps(merged_fields, ensure_ascii=False),
                    created_at,
                    now,
                ),
            )
        return self.get_provider_settings(canonical_provider) or {}

    def get_provider_settings(
        self,
        provider: str,
        *,
        include_secrets: bool = False,
    ) -> dict[str, Any] | None:
        canonical_provider = provider.strip().lower().replace("_", "-")
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM provider_settings WHERE provider = ?",
                (canonical_provider,),
            ).fetchone()
        if not row:
            return None
        fields = json.loads(row["fields_json"])
        return {
            "provider": row["provider"],
            "fields": fields if include_secrets else _public_fields(fields),
            "createdAt": row["created_at"],
            "updatedAt": row["updated_at"],
        }

    def delete_provider_settings(self, provider: str) -> bool:
        canonical_provider = provider.strip().lower().replace("_", "-")
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM provider_settings WHERE provider = ?",
                (canonical_provider,),
            )
            return cursor.rowcount > 0

    def get_value(self, *names: str) -> str | None:
        for name in names:
            value = os.environ.get(name)
            if value not in (None, ""):
                return value
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT fields_json FROM provider_settings ORDER BY provider"
            ).fetchall()
        for row in rows:
            fields = json.loads(row["fields_json"])
            for name in names:
                value = fields.get(name)
                if value not in (None, ""):
                    return str(value)
        for account in self.list_accounts(include_secrets=True):
            if not account["enabled"]:
                continue
            fields = account["fields"]
            for name in names:
                value = fields.get(name)
                if value not in (None, ""):
                    return str(value)
        return None

    def get_provider_value(self, provider: str, *names: str) -> str | None:
        for name in names:
            value = os.environ.get(name)
            if value not in (None, ""):
                return value
        canonical_provider = provider.strip().lower().replace("_", "-")
        settings = self.get_provider_settings(
            canonical_provider,
            include_secrets=True,
        )
        settings_fields = dict((settings or {}).get("fields") or {})
        for name in names:
            value = settings_fields.get(name)
            if value not in (None, ""):
                return str(value)
        for account in self.list_accounts(provider=canonical_provider, include_secrets=True):
            if not account["enabled"]:
                continue
            fields = account["fields"]
            for name in names:
                value = fields.get(name)
                if value not in (None, ""):
                    return str(value)
        return None

    @staticmethod
    def _serialize_row(
        row: sqlite3.Row, *, include_secrets: bool
    ) -> dict[str, Any]:
        fields = json.loads(row["fields_json"])
        return {
            "id": row["id"],
            "provider": row["provider"],
            "label": row["label"],
            "enabled": bool(row["enabled"]),
            "fields": fields if include_secrets else _public_fields(fields),
            "createdAt": row["created_at"],
            "updatedAt": row["updated_at"],
        }


_default_store: CredentialStore | None = None


def default_credential_store() -> CredentialStore:
    global _default_store
    if _default_store is None:
        _default_store = CredentialStore(config.DATABASE_PATH)
    return _default_store


def get_env_or_credential(*names: str) -> str | None:
    return default_credential_store().get_value(*names)


def get_env_or_provider_credential(provider: str, *names: str) -> str | None:
    return default_credential_store().get_provider_value(provider, *names)


def update_credentials(provider: str, fields: dict[str, Any]) -> dict[str, Any]:
    canonical_provider = {
        "kiro-oauth": "kiro",
        "antigravity-oauth": "antigravity",
        "gemini-scraping": "gemini-web",
        "flow-scraping": "flow",
        "deepgram-api": "deepgram",
        "gemini-api": "gemini",
        "vertex-api": "vertex",
    }.get(provider.replace("_", "-"), provider.replace("_", "-"))
    if canonical_provider == "flow":
        return default_credential_store().upsert_provider_settings(
            provider="flow",
            fields=fields,
        )
    return default_credential_store().upsert_account(
        account_id=provider.replace("_", "-"),
        provider=canonical_provider,
        label=provider.replace("_", " ").title(),
        fields=fields,
    )


def is_reserved_provider_settings_id(provider: str, account_id: str) -> bool:
    canonical_provider = provider.strip().lower().replace("_", "-")
    normalized_id = account_id.strip().lower().replace("_", "-")
    return (
        canonical_provider == "flow"
        and normalized_id in _FLOW_CONFIGURATION_ACCOUNT_IDS
    )


def update_flow_user_account_session(
    account_id: str,
    session_token: str,
    *,
    store: CredentialStore | None = None,
) -> dict[str, Any]:
    credential_store = store or default_credential_store()
    account = credential_store.get_account(account_id, include_secrets=True)
    if not account or _flow_account_kind(credential_store, account) != FLOW_ACCOUNT_KIND_USER:
        raise KeyError(account_id)
    fields = dict(account.get("fields") or {})
    cookie_header = _replace_flow_session_cookie(
        str(fields.get("FLOW_COOKIE") or fields.get("cookieHeader") or ""),
        session_token,
    )
    return credential_store.upsert_account(
        account_id=str(account["id"]),
        provider="flow",
        label=str(account.get("label") or account["id"]),
        fields={
            FLOW_ACCOUNT_KIND_FIELD: FLOW_ACCOUNT_KIND_USER,
            "FLOW_SESSION_TOKEN": session_token,
            "FLOW_COOKIE": cookie_header,
        },
        enabled=bool(account.get("enabled", True)),
    )


def _flow_account_kind(
    store: CredentialStore,
    account: dict[str, Any],
) -> str | None:
    account_id = str(account.get("id") or "").strip().lower().replace("_", "-")
    fields = dict(account.get("fields") or {})
    if account_id in _FLOW_CONFIGURATION_ACCOUNT_IDS:
        kind = FLOW_ACCOUNT_KIND_CONFIGURATION
    else:
        explicit_kind = str(fields.get(FLOW_ACCOUNT_KIND_FIELD) or "").strip().lower()
        if explicit_kind in {
            FLOW_ACCOUNT_KIND_USER,
            FLOW_ACCOUNT_KIND_CONFIGURATION,
        }:
            kind = explicit_kind
        else:
            has_session = bool(
                str(
                    fields.get("FLOW_SESSION_TOKEN")
                    or fields.get("sessionToken")
                    or ""
                ).strip()
            )
            has_registered_account_data = bool(
                str(
                    fields.get("FLOW_COOKIE")
                    or fields.get("cookieHeader")
                    or fields.get("FLOW_PROJECT_ID")
                    or fields.get("projectId")
                    or ""
                ).strip()
            )
            kind = (
                FLOW_ACCOUNT_KIND_USER
                if has_session and has_registered_account_data
                else None
            )

    if kind and fields.get(FLOW_ACCOUNT_KIND_FIELD) != kind:
        store.upsert_account(
            account_id=str(account.get("id") or ""),
            provider="flow",
            label=str(account.get("label") or account.get("id") or "Flow"),
            fields={FLOW_ACCOUNT_KIND_FIELD: kind},
            enabled=bool(account.get("enabled", True)),
        )
    return kind


def list_flow_cookie_runtime_accounts(
    store: CredentialStore | None = None,
) -> list[dict[str, str]]:
    credential_store = store or default_credential_store()
    accounts: list[dict[str, str]] = []
    for account in credential_store.list_accounts(include_secrets=True):
        if str(account.get("provider") or "").replace("_", "-") not in {"flow", "flow-scraping"}:
            continue
        if not account.get("enabled", True):
            continue
        if _flow_account_kind(credential_store, account) != FLOW_ACCOUNT_KIND_USER:
            continue
        fields = account.get("fields") or {}
        session_token = str(fields.get("FLOW_SESSION_TOKEN") or fields.get("sessionToken") or "").strip()
        if not session_token:
            continue
        accounts.append(
            {
                "id": str(account.get("id") or ""),
                "label": str(account.get("label") or account.get("id") or "Flow"),
                "sessionToken": session_token,
                "projectId": str(fields.get("FLOW_PROJECT_ID") or fields.get("projectId") or ""),
                "cookieHeader": str(fields.get("FLOW_COOKIE") or fields.get("cookieHeader") or ""),
            }
        )
    return accounts
