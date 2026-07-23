from __future__ import annotations

import unittest

from fastapi.testclient import TestClient
from google.genai import types

from app.main import create_app
from app.providers.google_genai import GeminiProviderAdapter, VertexProviderAdapter
from app.providers.google_genai.transport import GeminiApiService
from app.runtime.registry import ProviderRegistry


class GoogleGenAITransportFake:
    def __init__(self, backend: str, models: list[dict]) -> None:
        self.backend = backend
        self.models = models

    async def list_models(self) -> list[dict]:
        return self.models


class ValidatingGoogleTransportFake(GoogleGenAITransportFake):
    def check_config(self) -> dict:
        return {"success": True, "isLoggedIn": True, "model": "gemini-3.1-flash"}

    async def generate_text_from_messages(self, messages, *, model, temperature, response_json):
        self.validation_request = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "response_json": response_json,
        }
        return "OK"


class UnlistableVertexTransportFake(ValidatingGoogleTransportFake):
    async def list_models(self) -> list[dict]:
        raise RuntimeError("404 /v1/publishers/google/models")


class AsyncModelPager:
    def __init__(self, values: list[object]) -> None:
        self._values = values

    def __aiter__(self):
        async def iterate():
            for value in self._values:
                yield value

        return iterate()


class ModelValue:
    name = "models/gemini-3-flash"
    display_name = "Gemini 3 Flash"
    description = "Fast model"
    version = "3"
    supported_actions = ["generateContent"]
    input_token_limit = 1_000_000
    output_token_limit = 65_536


class ModelCollectionFake:
    async def list(self):
        return AsyncModelPager([ModelValue()])


class AioFake:
    models = ModelCollectionFake()


class GenAIClientFake:
    aio = AioFake()


class NativeModelsFake:
    def __init__(self) -> None:
        self.config = None

    async def generate_content(self, *, model, contents, config):
        del model, contents
        self.config = config
        return types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(
                        role="model",
                        parts=[
                            types.Part(
                                function_call=types.FunctionCall(
                                    id="call_scene_2",
                                    name="get_scene",
                                    args={"number": 2},
                                )
                            )
                        ],
                    )
                )
            ],
            model_version="gemini-3-flash",
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=10,
                candidates_token_count=4,
                total_token_count=14,
            ),
        )


class NativeAioFake:
    def __init__(self) -> None:
        self.models = NativeModelsFake()


class NativeGenAIClientFake:
    def __init__(self) -> None:
        self.aio = NativeAioFake()


class GoogleGenAIAdapterContractTest(unittest.TestCase):
    def test_gemini_and_vertex_are_simultaneous_providers_with_live_catalogs(self) -> None:
        gemini = GeminiProviderAdapter(
            transport=GoogleGenAITransportFake(
                "gemini",
                [
                    {
                        "id": "gemini-3-flash",
                        "label": "Gemini 3 Flash",
                        "supportedActions": ["generateContent"],
                    }
                ],
            )
        )
        vertex = VertexProviderAdapter(
            transport=GoogleGenAITransportFake(
                "vertex",
                [
                    {
                        "id": "gemini-3.1-pro",
                        "label": "Gemini 3.1 Pro",
                        "supportedActions": ["generateContent"],
                    },
                    {
                        "id": "veo-3.1-generate-001",
                        "label": "Veo 3.1",
                        "supportedActions": ["generateVideos"],
                    },
                ],
            )
        )
        client = TestClient(
            create_app(
                registry=ProviderRegistry([gemini, vertex]),
                require_api_key=False,
            )
        )

        response = client.get("/v1/models?refresh=true")

        self.assertEqual(200, response.status_code)
        models = {item["id"]: item for item in response.json()["data"]}
        self.assertTrue(
            {
                "gemini:gemini-3-flash",
                "vertex:gemini-3.1-pro",
                "vertex:veo-3.1-generate-001",
            }.issubset(models)
        )
        self.assertIn(
            "videos.generate",
            models["vertex:veo-3.1-generate-001"]["x_omni"]["capabilities"]["operations"],
        )
        self.assertEqual(
            "native",
            models["gemini:gemini-3-flash"]["x_omni"]["capabilities"]["tool_calling"],
        )

    def test_supplements_google_catalog_with_provider_specific_tts_models(self) -> None:
        gemini = GeminiProviderAdapter(transport=GoogleGenAITransportFake("gemini", []))
        vertex = VertexProviderAdapter(transport=GoogleGenAITransportFake("vertex", []))
        client = TestClient(
            create_app(
                registry=ProviderRegistry([gemini, vertex]),
                require_api_key=False,
            )
        )

        response = client.get("/v1/models?refresh=true")

        self.assertEqual(200, response.status_code)
        models = {item["id"]: item for item in response.json()["data"]}
        self.assertIn("gemini:gemini-3.1-flash-tts-preview", models)
        self.assertIn("gemini:gemini-2.5-flash-preview-tts", models)
        self.assertIn("vertex:gemini-3.1-flash-tts-preview", models)
        self.assertIn("vertex:gemini-2.5-flash-tts", models)
        self.assertNotIn("vertex:gemini-2.5-flash-preview-tts", models)
        self.assertIn(
            "audio.speech",
            models["vertex:gemini-3.1-flash-tts-preview"]["x_omni"]["capabilities"]["operations"],
        )

    def test_validate_status_performs_a_real_generation_probe(self) -> None:
        transport = ValidatingGoogleTransportFake(
            "vertex",
            [
                {
                    "id": "gemini-3.1-flash",
                    "label": "Gemini 3.1 Flash",
                    "supportedActions": ["generateContent"],
                }
            ],
        )
        adapter = VertexProviderAdapter(transport=transport)

        status = __import__("asyncio").run(adapter.status(validate=True))

        self.assertTrue(status["success"])
        self.assertTrue(status["canGenerate"])
        self.assertEqual("gemini-3.1-flash", status["validationModel"])
        self.assertEqual("gemini-3.1-flash", transport.validation_request["model"])

    def test_vertex_verified_catalog_survives_unsupported_list_endpoint(self) -> None:
        vertex = VertexProviderAdapter(
            transport=UnlistableVertexTransportFake("vertex", [])
        )
        client = TestClient(
            create_app(
                registry=ProviderRegistry([vertex]),
                require_api_key=False,
            )
        )

        response = client.get("/v1/models?refresh=true")

        self.assertEqual(200, response.status_code)
        model_ids = {item["id"] for item in response.json()["data"]}
        self.assertIn("vertex:gemini-2.5-flash", model_ids)
        self.assertIn("vertex:gemini-3.1-flash-tts-preview", model_ids)

    def test_transport_discovers_models_from_the_selected_google_backend(self) -> None:
        transport = GeminiApiService(backend="gemini", client=GenAIClientFake())

        models = __import__("asyncio").run(transport.list_models())

        self.assertEqual("gemini-3-flash", models[0]["id"])
        self.assertEqual(["generateContent"], models[0]["supportedActions"])
        self.assertEqual(1_000_000, models[0]["inputTokenLimit"])

    def test_transport_uses_google_native_function_declarations(self) -> None:
        client = NativeGenAIClientFake()
        transport = GeminiApiService(backend="gemini", client=client)

        result = __import__("asyncio").run(
            transport.generate_native(
                messages=[{"role": "user", "content": "Mostre a cena 2"}],
                model="gemini-3-flash",
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

        declaration = client.aio.models.config.tools[0].function_declarations[0]
        self.assertEqual("get_scene", declaration.name)
        self.assertEqual("call_scene_2", result["functionCalls"][0]["call_id"])
        self.assertEqual(14, result["usage"]["total_tokens"])


if __name__ == "__main__":
    unittest.main()
