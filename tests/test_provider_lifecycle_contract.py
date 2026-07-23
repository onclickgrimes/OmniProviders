from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from app.domain.models import ModelResult
from app.main import create_app
from app.runtime.registry import ProviderRegistry


class LifecycleAdapterFake:
    provider_id = "oauth"

    async def list_models(self, *, refresh=False):
        return []

    async def invoke(self, model, request):
        return ModelResult()

    async def status(self, *, validate=False):
        return {"success": True, "isLoggedIn": validate}

    def start_login(self):
        return {"success": True, "authorizationUrl": "https://example.test/oauth"}

    def complete_login(self, payload):
        return {"success": payload == {"code": "x", "state": "y"}, "isLoggedIn": True}


class ProviderLifecycleContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(
            create_app(
                registry=ProviderRegistry([LifecycleAdapterFake()]),
                require_api_key=False,
            )
        )

    def test_provider_status_login_and_oauth_callback_are_sidecar_owned(self) -> None:
        self.assertEqual(["oauth"], [item["id"] for item in self.client.get("/providers").json()["data"]])
        self.assertTrue(self.client.get("/providers/oauth/status?validate=true").json()["isLoggedIn"])
        self.assertIn("authorizationUrl", self.client.post("/providers/oauth/login").json())
        callback = self.client.post(
            "/providers/oauth/oauth/callback", json={"code": "x", "state": "y"}
        )
        self.assertTrue(callback.json()["isLoggedIn"])


if __name__ == "__main__":
    unittest.main()
