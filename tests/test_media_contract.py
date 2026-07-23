from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.domain.models import ModelCapabilities, ModelDescriptor, ModelResult
from app.main import create_app
from app.media.artifacts import ArtifactStore
from app.persistence.jobs import GenerationJobStore
from app.runtime.registry import ProviderRegistry


class MediaAdapterFake:
    provider_id = "media"

    async def list_models(self, *, refresh=False):
        del refresh
        return [
            ModelDescriptor(
                provider="media",
                model="all",
                label="All media",
                capabilities=ModelCapabilities(
                    input_modalities=frozenset({"text", "audio"}),
                    output_modalities=frozenset({"image", "video", "audio", "text"}),
                    operations=frozenset(
                        {"images.generate", "videos.generate", "audio.speech", "audio.transcriptions"}
                    ),
                ),
            )
        ]

    async def invoke(self, model, request):
        return ModelResult(text="", effective_model=model)

    async def generate_images(self, model, payload):
        return {"media": [{"bytes": b"PNG", "mime_type": "image/png"}], "effectiveModel": model}

    async def generate_video(self, model, payload):
        return {"media": [{"bytes": b"VIDEO", "mime_type": "video/mp4"}], "effectiveModel": model}

    async def generate_speech(self, model, payload):
        return {"media": [{"bytes": b"WAVE", "mime_type": "audio/wav"}]}

    async def transcribe(self, model, audio, payload):
        return {"text": f"{len(audio)} bytes", "model": model}


class MediaContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.client = TestClient(
            create_app(
                registry=ProviderRegistry([MediaAdapterFake()]),
                artifact_store=ArtifactStore(root / "artifacts"),
                job_store=GenerationJobStore(root / "jobs.sqlite3"),
                require_api_key=False,
            )
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_images_generation_returns_downloadable_artifact(self) -> None:
        response = self.client.post(
            "/v1/images/generations",
            json={"model": "media:all", "prompt": "x", "n": 1},
        )
        self.assertEqual(200, response.status_code)
        artifact = response.json()["data"][0]
        download = self.client.get(artifact["url"])
        self.assertEqual(b"PNG", download.content)
        self.assertEqual("image/png", download.headers["content-type"])

    def test_video_uses_persisted_job_and_content_endpoint(self) -> None:
        response = self.client.post(
            "/v1/videos",
            json={"model": "media:all", "prompt": "x"},
        )
        self.assertEqual(200, response.status_code)
        video_id = response.json()["id"]
        status = self.client.get(f"/v1/videos/{video_id}")
        self.assertEqual("completed", status.json()["status"])
        content = self.client.get(f"/v1/videos/{video_id}/content")
        self.assertEqual(b"VIDEO", content.content)

    def test_speech_and_transcription_use_openai_routes(self) -> None:
        speech = self.client.post(
            "/v1/audio/speech",
            json={"model": "media:all", "input": "olá", "voice": "Kore"},
        )
        self.assertEqual(b"WAVE", speech.content)
        transcript = self.client.post(
            "/v1/audio/transcriptions",
            data={"model": "media:all"},
            files={"file": ("audio.wav", b"12345", "audio/wav")},
        )
        self.assertEqual("5 bytes", transcript.json()["text"])

    def test_artifacts_and_interrupted_jobs_survive_restart_safely(self) -> None:
        root = Path(self.temp.name)
        first_store = ArtifactStore(root / "persistent-artifacts")
        artifact = first_store.save_bytes(b"persisted", mime_type="video/mp4", filename="scene.mp4")
        restarted_store = ArtifactStore(root / "persistent-artifacts")
        self.assertEqual(b"persisted", Path(restarted_store.get(artifact.id).path).read_bytes())

        job_store = GenerationJobStore(root / "recovery.sqlite3")
        job = job_store.create(operation="videos.generate", model="media:all", request={})
        job_store.update(job["id"], status="in_progress")
        self.assertEqual(1, job_store.recover_interrupted())
        recovered = job_store.get(job["id"])
        self.assertEqual("failed", recovered["status"])
        self.assertEqual("sidecar_restarted", recovered["error"]["code"])


if __name__ == "__main__":
    unittest.main()
