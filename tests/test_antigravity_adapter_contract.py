from __future__ import annotations

import asyncio
import unittest

from fastapi.testclient import TestClient

from app.main import create_app
from app.providers.antigravity import AntigravityProviderAdapter
from app.providers.antigravity.transport import AntigravityOAuthService
from app.runtime.registry import ProviderRegistry


class AntigravityTransportFake:
    async def list_models(self) -> dict:
        return {
            "success": True,
            "models": [
                {
                    "model": "gemini-3.5-flash-medium",
                    "backendModel": "gemini-3.5-flash",
                    "label": "Antigravity Gemini 3.5 Flash (Medium)",
                }
            ],
        }


class AntigravityToolTransportFake(AntigravityTransportFake):
    async def generate_native(self, **_kwargs) -> dict:
        return {
            "text": "",
            "functionCalls": [
                {
                    "call_id": "call_scene_2",
                    "name": "get_scene",
                    "arguments": {"number": 2},
                }
            ],
            "effectiveModel": "gemini-3.5-flash",
        }


class CapturingAntigravityTransport(AntigravityOAuthService):
    def __init__(self) -> None:
        super().__init__()
        self.payload: dict | None = None

    async def _ensure_project_id(self) -> str:
        return "project-test"

    def _project_id(self) -> str:
        return "project-test"

    async def _get_access_token(self, *, force_refresh: bool = False) -> str:
        del force_refresh
        return "access-token"

    async def _post_generate_content(self, *, endpoint: str, access_token: str, payload: dict):
        del endpoint, access_token
        self.payload = payload
        return 200, {
            "response": {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {
                                        "id": "call_scene_2",
                                        "name": "get_scene",
                                        "args": {"number": 2},
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        }


class AntigravityAdapterContractTest(unittest.TestCase):
    def test_exposes_live_models_with_multimodal_and_native_tool_capabilities(self) -> None:
        adapter = AntigravityProviderAdapter(transport=AntigravityTransportFake())
        client = TestClient(
            create_app(registry=ProviderRegistry([adapter]), require_api_key=False)
        )

        response = client.get("/v1/models?refresh=true")

        self.assertEqual(200, response.status_code)
        model = response.json()["data"][0]
        self.assertEqual("antigravity:gemini-3.5-flash-medium", model["id"])
        self.assertEqual("gemini-3.5-flash", model["x_omni"]["effective_model"])
        self.assertEqual("account_live", model["x_omni"]["discovery"])
        self.assertEqual("native", model["x_omni"]["capabilities"]["tool_calling"])
        self.assertEqual(
            ["audio", "image", "text", "video"],
            model["x_omni"]["capabilities"]["input_modalities"],
        )

    def test_translates_native_function_calls_to_responses_items(self) -> None:
        adapter = AntigravityProviderAdapter(transport=AntigravityToolTransportFake())
        client = TestClient(
            create_app(registry=ProviderRegistry([adapter]), require_api_key=False)
        )

        response = client.post(
            "/v1/responses",
            json={
                "model": "antigravity:gemini-3.5-flash-medium",
                "input": "Mostre a cena 2",
                "tools": [
                    {
                        "type": "function",
                        "name": "get_scene",
                        "description": "Obtém uma cena.",
                        "parameters": {"type": "object", "properties": {}},
                    }
                ],
            },
        )

        self.assertEqual(200, response.status_code)
        call = response.json()["output"][0]
        self.assertEqual("function_call", call["type"])
        self.assertEqual("call_scene_2", call["call_id"])
        self.assertEqual("get_scene", call["name"])

    def test_transport_sends_function_declarations_to_antigravity(self) -> None:
        transport = CapturingAntigravityTransport()

        result = asyncio.run(
            transport.generate_native(
                messages=[{"role": "user", "content": "Mostre a cena 2"}],
                model="gemini-3.5-flash-medium",
                tools=[
                    {
                        "type": "function",
                        "name": "get_scene",
                        "description": "Obtém uma cena.",
                        "parameters": {
                            "type": "object",
                            "properties": {"number": {"type": "integer"}},
                            "required": ["number"],
                        },
                    }
                ],
                tool_choice="auto",
            )
        )

        declaration = transport.payload["request"]["tools"][0]["functionDeclarations"][0]
        self.assertEqual("get_scene", declaration["name"])
        self.assertEqual(["number"], declaration["parameters"]["required"])
        self.assertEqual("call_scene_2", result["functionCalls"][0]["call_id"])


if __name__ == "__main__":
    unittest.main()
