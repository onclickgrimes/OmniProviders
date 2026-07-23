from __future__ import annotations

import asyncio
import unittest

import httpx

from app.providers.deepgram import DeepgramProviderAdapter
from app.providers.deepgram.transport import DeepgramTransport


class HttpClientFake:
    def __init__(self) -> None:
        self.request = None

    async def post(self, url, **kwargs):
        self.request = (url, kwargs)
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={
                "metadata": {"duration": 1.2},
                "results": {
                    "channels": [
                        {
                            "alternatives": [
                                {
                                    "transcript": "Olá.",
                                    "confidence": 0.99,
                                    "words": [
                                        {
                                            "word": "olá",
                                            "punctuated_word": "Olá.",
                                            "start": 0,
                                            "end": 1.2,
                                            "confidence": 0.99,
                                            "speaker": 0,
                                        }
                                    ],
                                }
                            ]
                        }
                    ]
                },
            },
        )


class DeepgramAdapterContractTest(unittest.TestCase):
    def test_uses_http_transport_without_deepgram_sdk(self) -> None:
        client = HttpClientFake()
        transport = DeepgramTransport(client=client, api_key="secret")

        result = asyncio.run(
            transport.transcribe(
                b"audio",
                model="nova-3",
                language="pt-BR",
                mime_type="audio/wav",
            )
        )

        url, request = client.request
        self.assertEqual("https://api.deepgram.com/v1/listen", url)
        self.assertEqual("Token secret", request["headers"]["Authorization"])
        self.assertEqual("nova-3", request["params"]["model"])
        self.assertEqual("Olá.", result["text"])
        self.assertEqual(1, len(result["segments"]))

    def test_catalog_is_available_only_with_an_account(self) -> None:
        adapter = DeepgramProviderAdapter(DeepgramTransport(api_key="secret"))
        models = asyncio.run(adapter.list_models())
        self.assertEqual(["nova-3", "nova-2"], [item.model for item in models])
        self.assertEqual(
            {"audio.transcriptions"},
            set(models[0].capabilities.operations),
        )


if __name__ == "__main__":
    unittest.main()
