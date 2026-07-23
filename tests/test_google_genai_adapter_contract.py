from __future__ import annotations

import unittest

from fastapi.testclient import TestClient
from google.genai import types

from app.domain.models import ModelInvocation
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

    async def generate_text_from_messages(
        self,
        messages,
        *,
        model,
        temperature,
        response_json,
        thinking_level=None,
    ):
        self.validation_request = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "response_json": response_json,
            "thinking_level": thinking_level,
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
    def test_conversational_catalog_is_limited_to_supported_thinking_variants(self) -> None:
        raw_models = [
            {
                "id": "gemini-3.6-flash",
                "label": "Gemini 3.6 Flash",
                "supportedActions": ["generateContent"],
            },
            {
                "id": "gemini-3.1-pro-preview",
                "label": "Gemini 3.1 Pro Preview",
                "supportedActions": ["generateContent"],
            },
            {
                "id": "aqa",
                "label": "Attributed Question Answering",
                "supportedActions": ["generateAnswer"],
            },
            {
                "id": "gemini-embedding-001",
                "label": "Gemini Embedding",
                "supportedActions": ["embedContent"],
            },
            {
                "id": "gemini-3.1-flash-image-preview",
                "label": "Gemini Image",
                "supportedActions": ["generateContent"],
            },
            {
                "id": "veo-3.1-generate-preview",
                "label": "Veo 3.1",
                "supportedActions": ["generateVideos"],
            },
        ]
        client = TestClient(
            create_app(
                registry=ProviderRegistry(
                    [
                        GeminiProviderAdapter(
                            transport=GoogleGenAITransportFake("gemini", raw_models)
                        ),
                        VertexProviderAdapter(
                            transport=GoogleGenAITransportFake("vertex", [])
                        ),
                    ]
                ),
                require_api_key=False,
            )
        )

        response = client.get("/v1/models?refresh=true")

        self.assertEqual(200, response.status_code)
        models = {item["id"]: item for item in response.json()["data"]}
        conversational_ids = {
            model_id
            for model_id, item in models.items()
            if "responses" in item["x_omni"]["capabilities"]["operations"]
        }
        self.assertEqual(
            {
                f"{provider}:{model}@{level}"
                for provider in ("gemini", "vertex")
                for model in ("gemini-3.6-flash", "gemini-3.1-pro-preview")
                for level in ("low", "medium", "high")
            },
            conversational_ids,
        )
        self.assertEqual(
            "Gemini 3.6 Flash (Baixo)",
            models["gemini:gemini-3.6-flash@low"]["x_omni"]["label"],
        )
        self.assertEqual(
            "gemini-3.6-flash",
            models["gemini:gemini-3.6-flash@high"]["x_omni"]["effective_model"],
        )
        self.assertEqual(
            "high",
            models["gemini:gemini-3.6-flash@high"]["x_omni"]["thinkingLevel"],
        )
        self.assertNotIn("gemini:aqa", models)
        self.assertNotIn("gemini:gemini-embedding-001", models)
        self.assertIn("gemini:gemini-3.1-flash-image-preview", models)
        self.assertIn("gemini:veo-3.1-generate-preview", models)

    def test_thinking_variant_routes_effective_model_and_level(self) -> None:
        transport = ValidatingGoogleTransportFake(
            "gemini",
            [
                {
                    "id": "gemini-3.6-flash",
                    "label": "Gemini 3.6 Flash",
                    "supportedActions": ["generateContent"],
                }
            ],
        )
        adapter = GeminiProviderAdapter(transport=transport)

        result = __import__("asyncio").run(
            adapter.invoke(
                "gemini-3.6-flash@high",
                ModelInvocation(input="Olá"),
            )
        )

        self.assertEqual("gemini-3.6-flash", transport.validation_request["model"])
        self.assertEqual("high", transport.validation_request["thinking_level"])
        self.assertEqual("gemini-3.6-flash", result.effective_model)

    def test_gemini_and_vertex_are_simultaneous_providers_with_live_catalogs(self) -> None:
        gemini = GeminiProviderAdapter(
            transport=GoogleGenAITransportFake(
                "gemini",
                [
                    {
                        "id": "gemini-3.6-flash",
                        "label": "Gemini 3.6 Flash",
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
                        "id": "gemini-3.1-pro-preview",
                        "label": "Gemini 3.1 Pro Preview",
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
                "gemini:gemini-3.6-flash@medium",
                "vertex:gemini-3.1-pro-preview@high",
                "vertex:veo-3.1-generate-001",
            }.issubset(models)
        )
        self.assertIn(
            "videos.generate",
            models["vertex:veo-3.1-generate-001"]["x_omni"]["capabilities"]["operations"],
        )
        self.assertEqual(
            "native",
            models["gemini:gemini-3.6-flash@medium"]["x_omni"]["capabilities"]["tool_calling"],
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
        self.assertIn("vertex:gemini-3.6-flash@medium", model_ids)
        self.assertIn("vertex:gemini-3.1-pro-preview@high", model_ids)
        self.assertNotIn("vertex:gemini-2.5-flash", model_ids)
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
                thinking_level="high",
            )
        )

        declaration = client.aio.models.config.tools[0].function_declarations[0]
        self.assertEqual("get_scene", declaration.name)
        self.assertEqual(
            types.ThinkingLevel.HIGH,
            client.aio.models.config.thinking_config.thinking_level,
        )
        self.assertEqual("call_scene_2", result["functionCalls"][0]["call_id"])
        self.assertEqual(14, result["usage"]["total_tokens"])


if __name__ == "__main__":
    unittest.main()
