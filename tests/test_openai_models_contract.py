from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from app.main import create_app
from app.providers.fake import FakeProviderAdapter
from app.runtime.registry import ProviderRegistry


class OpenAIModelsContractTest(unittest.TestCase):
    def test_lists_only_models_available_to_the_configured_account(self) -> None:
        registry = ProviderRegistry(
            [
                FakeProviderAdapter(
                    provider="kiro",
                    models=[
                        {
                            "id": "claude-sonnet-4.5",
                            "label": "Claude Sonnet 4.5",
                            "available": True,
                        },
                        {
                            "id": "claude-opus-4.5",
                            "label": "Claude Opus 4.5",
                            "available": False,
                        },
                    ],
                )
            ]
        )
        client = TestClient(create_app(registry=registry, require_api_key=False))

        response = client.get("/v1/models")

        self.assertEqual(200, response.status_code)
        self.assertEqual(
            ["kiro:claude-sonnet-4.5"],
            [item["id"] for item in response.json()["data"]],
        )
        self.assertEqual("list", response.json()["object"])
        self.assertEqual("kiro", response.json()["data"][0]["owned_by"])

    def test_creates_a_text_response_through_the_selected_provider(self) -> None:
        adapter = FakeProviderAdapter(
            provider="kiro",
            models=[{"id": "claude-sonnet-4.5", "available": True}],
            response_text="Olá do Kiro",
        )
        registry = ProviderRegistry([adapter])
        client = TestClient(create_app(registry=registry, require_api_key=False))

        response = client.post(
            "/v1/responses",
            json={
                "model": "kiro:claude-sonnet-4.5",
                "input": "Responda em português",
                "store": False,
            },
        )

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual("response", payload["object"])
        self.assertEqual("completed", payload["status"])
        self.assertEqual("kiro:claude-sonnet-4.5", payload["model"])
        self.assertEqual("Olá do Kiro", payload["output"][0]["content"][0]["text"])

        instructed = client.post(
            "/v1/responses",
            json={
                "model": "kiro:claude-sonnet-4.5",
                "instructions": "Responda sempre em português.",
                "input": "Olá",
            },
        )
        self.assertEqual(200, instructed.status_code)
        self.assertEqual("system", adapter.last_request.input[0]["role"])
        self.assertEqual("Responda sempre em português.", adapter.last_request.input[0]["content"])

    def test_normalizes_structured_output_and_rejects_invalid_json(self) -> None:
        valid = FakeProviderAdapter(
            provider="gemini",
            models=[{"id": "flash"}],
            response_text='```json\n{"ok": true}\n```',
        )
        client = TestClient(create_app(registry=ProviderRegistry([valid]), require_api_key=False))
        response = client.post(
            "/v1/responses",
            json={
                "model": "gemini:flash",
                "input": "JSON",
                "text": {"format": {"type": "json_object"}},
            },
        )
        self.assertEqual(200, response.status_code)
        self.assertEqual('{"ok": true}', response.json()["output"][0]["content"][0]["text"])

        invalid = FakeProviderAdapter(
            provider="gemini",
            models=[{"id": "flash"}],
            response_text="não é json",
        )
        client = TestClient(create_app(registry=ProviderRegistry([invalid]), require_api_key=False))
        response = client.post(
            "/v1/responses",
            json={
                "model": "gemini:flash",
                "input": "JSON",
                "response_format": "json",
            },
        )
        self.assertEqual(502, response.status_code)
        self.assertEqual("invalid_structured_output", response.json()["error"]["code"])

    def test_returns_native_function_calls_and_accepts_their_outputs(self) -> None:
        registry = ProviderRegistry(
            [
                FakeProviderAdapter(
                    provider="kiro",
                    models=[{"id": "claude-sonnet-4.5", "available": True}],
                    response_text="A cena 2 dura seis segundos.",
                    tool_name="get_scene",
                )
            ]
        )
        client = TestClient(create_app(registry=registry, require_api_key=False))
        tools = [
            {
                "type": "function",
                "name": "get_scene",
                "description": "Obtém uma cena pelo número.",
                "parameters": {
                    "type": "object",
                    "properties": {"number": {"type": "integer"}},
                    "required": ["number"],
                },
            }
        ]

        first = client.post(
            "/v1/responses",
            json={
                "model": "kiro:claude-sonnet-4.5",
                "input": "Qual a duração da cena 2?",
                "tools": tools,
                "store": False,
            },
        )

        self.assertEqual(200, first.status_code)
        function_call = first.json()["output"][0]
        self.assertEqual("function_call", function_call["type"])
        self.assertEqual("get_scene", function_call["name"])

        second = client.post(
            "/v1/responses",
            json={
                "model": "kiro:claude-sonnet-4.5",
                "input": [
                    {"role": "user", "content": "Qual a duração da cena 2?"},
                    function_call,
                    {
                        "type": "function_call_output",
                        "call_id": function_call["call_id"],
                        "output": '{"number":2,"duration":6}',
                    },
                ],
                "tools": tools,
                "store": False,
            },
        )

        self.assertEqual(200, second.status_code)
        self.assertEqual(
            "A cena 2 dura seis segundos.",
            second.json()["output"][0]["content"][0]["text"],
        )

    def test_rejects_requests_without_the_local_api_key(self) -> None:
        client = TestClient(
            create_app(
                registry=ProviderRegistry(),
                require_api_key=True,
                api_key="local-secret",
            )
        )

        response = client.get("/v1/models")

        self.assertEqual(401, response.status_code)
        self.assertEqual("invalid_api_key", response.json()["error"]["code"])

    def test_chat_completions_uses_the_same_runtime(self) -> None:
        registry = ProviderRegistry(
            [
                FakeProviderAdapter(
                    provider="gemini",
                    models=[{"id": "gemini-flash", "available": True}],
                    response_text="Resposta Gemini",
                )
            ]
        )
        client = TestClient(create_app(registry=registry, require_api_key=False))

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gemini:gemini-flash",
                "messages": [{"role": "user", "content": "Olá"}],
            },
        )

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual("chat.completion", payload["object"])
        self.assertEqual("Resposta Gemini", payload["choices"][0]["message"]["content"])


if __name__ == "__main__":
    unittest.main()
