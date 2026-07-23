from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from app.main import create_app
from app.providers.gemini_web import GeminiWebProviderAdapter
from app.providers.gemini_web.transport import GeminiScrapingService
from app.runtime.registry import ProviderRegistry


class GeminiWebTransportFake:
    async def check_login(self):
        return {
            "success": True,
            "accountStatus": "AVAILABLE",
            "models": [
                {
                    "modelName": "gemini-3-pro",
                    "displayName": "Gemini 3 Pro",
                    "available": True,
                }
            ],
        }

    async def generate_text_from_messages(self, messages, *, model, response_json=False):
        self.messages = messages
        self.model = model
        self.response_json = response_json
        return "resposta web"


class GeminiWebAdapterContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.transport = GeminiWebTransportFake()
        self.client = TestClient(
            create_app(
                registry=ProviderRegistry([GeminiWebProviderAdapter(self.transport)]),
                require_api_key=False,
            )
        )

    def test_catalog_comes_from_authenticated_web_account(self) -> None:
        response = self.client.get("/v1/models")
        self.assertEqual(200, response.status_code)
        model = response.json()["data"][0]
        self.assertEqual("gemini-web:gemini-3-pro", model["id"])
        self.assertIn("videos.generate", model["x_omni"]["capabilities"]["operations"])

    def test_text_uses_openai_responses_contract(self) -> None:
        response = self.client.post(
            "/v1/responses",
            json={"model": "gemini-web:gemini-3-pro", "input": "olá"},
        )
        self.assertEqual(200, response.status_code)
        self.assertEqual("resposta web", response.json()["output"][0]["content"][0]["text"])

    def test_native_tools_are_rejected_before_web_transport(self) -> None:
        response = self.client.post(
            "/v1/responses",
            json={
                "model": "gemini-web:gemini-3-pro",
                "input": "olá",
                "tools": [{"type": "function", "name": "x", "parameters": {}}],
            },
        )
        self.assertEqual(400, response.status_code)
        self.assertEqual("unsupported_tools", response.json()["error"]["code"])


class _ExpiredAccountStatus:
    name = "UNAUTHENTICATED"
    description = "cookies expired"


class _ExpiredGeminiClient:
    account_status = _ExpiredAccountStatus()

    def list_models(self):
        return [{"model_name": "gemini-3-pro"}]


class GeminiWebExpiredSessionTest(unittest.IsolatedAsyncioTestCase):
    async def test_expired_sdk_session_is_not_reported_as_logged_in(self) -> None:
        transport = GeminiScrapingService(client=_ExpiredGeminiClient())
        transport._dependency_available = lambda: True

        result = await transport.check_login()

        self.assertFalse(result["success"])
        self.assertFalse(result["isLoggedIn"])
        self.assertEqual("UNAUTHENTICATED", result["accountStatus"])


if __name__ == "__main__":
    unittest.main()
