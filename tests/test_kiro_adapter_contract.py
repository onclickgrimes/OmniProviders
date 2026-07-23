from __future__ import annotations

import json
import struct
import unittest

from fastapi.testclient import TestClient

from app.main import create_app
from app.providers.kiro import KiroProviderAdapter
from app.providers.kiro.transport import KiroOAuthService
from app.runtime.registry import ProviderRegistry


class KiroTransportFake:
    async def list_models(self) -> dict:
        return {
            "success": True,
            "models": [
                {
                    "model": "claude-sonnet-4.5",
                    "label": "Kiro Claude Sonnet 4.5",
                    "effort": "high",
                }
            ],
        }


class KiroToolTransportFake(KiroTransportFake):
    async def generate_native(self, **_kwargs) -> dict:
        return {
            "text": "",
            "events": [
                {
                    "toolUseEvent": {
                        "toolUseId": "call_scene_2",
                        "name": "get_scene",
                        "input": {"number": 2},
                    }
                }
            ],
            "effectiveModel": "claude-sonnet-4.5",
        }


def event_stream_bytes(*events: dict) -> bytes:
    chunks: list[bytes] = []
    for event in events:
        payload = json.dumps(event).encode("utf-8")
        total_length = 16 + len(payload)
        chunks.append(struct.pack(">II", total_length, 0) + b"\0\0\0\0" + payload + b"\0\0\0\0")
    return b"".join(chunks)


class CapturingKiroTransport(KiroOAuthService):
    def __init__(self) -> None:
        self.payload: dict | None = None

    def _raw_credentials(self) -> dict:
        return {"profileArn": "profile:test"}

    async def _get_access_token(self, *, force_refresh: bool = False) -> str:
        del force_refresh
        return "access-token"

    async def _post_generate(self, *, access_token: str, endpoint: str, payload: dict):
        del access_token, endpoint
        self.payload = payload
        raw = event_stream_bytes(
            {
                "toolUseEvent": {
                    "toolUseId": "call_scene_2",
                    "name": "get_scene",
                    "input": {"number": 2},
                }
            }
        )
        return 200, raw, raw


class KiroAdapterContractTest(unittest.TestCase):
    def test_exposes_only_models_discovered_for_the_kiro_account(self) -> None:
        adapter = KiroProviderAdapter(transport=KiroTransportFake())
        client = TestClient(
            create_app(
                registry=ProviderRegistry([adapter]),
                require_api_key=False,
            )
        )

        response = client.get("/v1/models?refresh=true")

        self.assertEqual(200, response.status_code)
        model = response.json()["data"][0]
        self.assertEqual("kiro:claude-sonnet-4.5", model["id"])
        self.assertEqual("account_live", model["x_omni"]["discovery"])
        self.assertEqual("native", model["x_omni"]["capabilities"]["tool_calling"])
        self.assertEqual(
            ["image", "text"],
            model["x_omni"]["capabilities"]["input_modalities"],
        )

    def test_rejects_video_before_calling_kiro_transport(self) -> None:
        adapter = KiroProviderAdapter(transport=KiroTransportFake())
        client = TestClient(
            create_app(registry=ProviderRegistry([adapter]), require_api_key=False),
            raise_server_exceptions=False,
        )

        response = client.post(
            "/v1/responses",
            json={
                "model": "kiro:claude-sonnet-4.5",
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Revise este vídeo"},
                            {"type": "input_video", "video_url": "data:video/mp4;base64,AA=="},
                        ],
                    }
                ],
            },
        )

        self.assertEqual(400, response.status_code)
        self.assertEqual("unsupported_input_modality", response.json()["error"]["code"])

    def test_translates_kiro_native_tool_use_to_openai_function_call(self) -> None:
        adapter = KiroProviderAdapter(transport=KiroToolTransportFake())
        client = TestClient(
            create_app(registry=ProviderRegistry([adapter]), require_api_key=False)
        )

        response = client.post(
            "/v1/responses",
            json={
                "model": "kiro:claude-sonnet-4.5",
                "input": "Mostre a cena 2",
                "tools": [
                    {
                        "type": "function",
                        "name": "get_scene",
                        "description": "Obtém uma cena.",
                        "parameters": {
                            "type": "object",
                            "properties": {"number": {"type": "integer"}},
                        },
                    }
                ],
            },
        )

        self.assertEqual(200, response.status_code)
        function_call = response.json()["output"][0]
        self.assertEqual("function_call", function_call["type"])
        self.assertEqual("call_scene_2", function_call["call_id"])
        self.assertEqual("get_scene", function_call["name"])
        self.assertEqual('{"number": 2}', function_call["arguments"])

    def test_kiro_transport_sends_openai_tools_as_native_tool_specifications(self) -> None:
        transport = CapturingKiroTransport()

        result = __import__("asyncio").run(
            transport.generate_native(
                messages=[{"role": "user", "content": "Mostre a cena 2"}],
                model="claude-sonnet-4.5",
                tools=[
                    {
                        "type": "function",
                        "name": "get_scene",
                        "description": "Obtém uma cena.",
                        "parameters": {
                            "type": "object",
                            "properties": {"number": {"type": "integer"}},
                        },
                    }
                ],
                tool_choice="auto",
            )
        )

        self.assertEqual("get_scene", result["events"][0]["toolUseEvent"]["name"])
        context = transport.payload["conversationState"]["currentMessage"][
            "userInputMessage"
        ]["userInputMessageContext"]
        specification = context["tools"][0]["toolSpecification"]
        self.assertEqual("get_scene", specification["name"])
        self.assertEqual("integer", specification["inputSchema"]["json"]["properties"]["number"]["type"])


if __name__ == "__main__":
    unittest.main()
