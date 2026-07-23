from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.providers.google_genai.media import GoogleMediaService
from app.providers.google_genai.voice import GeminiVoiceService


class GoogleMediaModelsFake:
    def __init__(self) -> None:
        self.video_config = None

    async def generate_content(self, **_kwargs):
        inline = SimpleNamespace(data=b"PNG", mime_type="image/png")
        part = SimpleNamespace(inline_data=inline)
        return SimpleNamespace(
            candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))]
        )

    async def generate_videos(self, **kwargs):
        self.video_config = kwargs["config"]
        video = SimpleNamespace(video_bytes=b"MP4", mime_type="video/mp4")
        response = SimpleNamespace(
            generated_videos=[SimpleNamespace(video=video)],
            rai_media_filtered_reasons=[],
        )
        return SimpleNamespace(done=True, error=None, response=response)


class GoogleMediaClientFake:
    def __init__(self) -> None:
        self.aio = SimpleNamespace(
            models=GoogleMediaModelsFake(),
            operations=SimpleNamespace(get=AsyncMock()),
            files=SimpleNamespace(download=AsyncMock()),
        )


class GoogleMediaContractTest(unittest.TestCase):
    def test_images_and_veo_return_bytes_without_project_storage(self) -> None:
        client = GoogleMediaClientFake()
        service = GoogleMediaService(backend="vertex", client=client)

        image = asyncio.run(service.generate_images("gemini-image", {"prompt": "Cena"}))
        video = asyncio.run(
            service.generate_video(
                "veo-3.1-generate-001",
                {"prompt": "Cena", "durationSeconds": 6, "aspectRatio": "16:9"},
            )
        )

        self.assertEqual(b"PNG", image["media"][0]["bytes"])
        self.assertEqual(b"MP4", video["media"][0]["bytes"])
        self.assertEqual("vertex", video["provider"])
        self.assertEqual(6, client.aio.models.video_config.duration_seconds)

    def test_gemini_voice_returns_audio_bytes(self) -> None:
        service = GeminiVoiceService(backend="gemini", client=object())
        service._generate_wav = AsyncMock(return_value=(b"WAVE", "audio/L16"))  # type: ignore[method-assign]

        result = asyncio.run(
            service.generate_speech(
                {"input": "Olá", "voice": "Kore", "model": "gemini-tts"}
            )
        )

        self.assertTrue(result["success"])
        self.assertEqual(b"WAVE", result["media"][0]["bytes"])
        self.assertEqual("audio/wav", result["media"][0]["mime_type"])


if __name__ == "__main__":
    unittest.main()
