from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from google.genai import types

from app.media.input import load_attachment_bytes
from app.providers.google_genai.transport import GeminiApiError, create_genai_client


def _closest_duration(value: Any, allowed: tuple[int, ...], default: int) -> int:
    try:
        requested = float(value)
    except (TypeError, ValueError):
        return default
    return min(allowed, key=lambda item: abs(item - requested))


def _image(value: Any) -> types.Image:
    data, mime_type = load_attachment_bytes(value, fallback_mime_type="image/png")
    return types.Image(image_bytes=data, mime_type=mime_type)


class GoogleMediaService:
    def __init__(self, *, backend: str, client: Any | None = None) -> None:
        self.backend = backend
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = create_genai_client(self.backend)
        return self._client

    async def generate_images(self, model: str, payload: dict[str, Any]) -> dict[str, Any]:
        prompt = str(payload.get("prompt") or "").strip()
        if not prompt:
            raise GeminiApiError("Image prompt is required.")
        count = max(1, min(int(payload.get("n") or payload.get("count") or 1), 4))
        contents: list[Any] = [prompt]
        references = (
            payload.get("ingredientImagePaths")
            or payload.get("referenceImagePaths")
            or payload.get("referenceImagePath")
            or []
        )
        if not isinstance(references, list):
            references = [references]
        for reference in references:
            data, mime_type = load_attachment_bytes(reference, fallback_mime_type="image/png")
            contents.append(types.Part.from_bytes(data=data, mime_type=mime_type))

        media: list[dict[str, Any]] = []
        for _index in range(count):
            response = await self._get_client().aio.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
            )
            for candidate in getattr(response, "candidates", None) or []:
                for part in getattr(getattr(candidate, "content", None), "parts", None) or []:
                    inline_data = getattr(part, "inline_data", None)
                    if inline_data is None or getattr(inline_data, "data", None) is None:
                        continue
                    data = inline_data.data
                    media.append(
                        {
                            "bytes": bytes(data) if not isinstance(data, str) else __import__("base64").b64decode(data),
                            "mime_type": inline_data.mime_type or "image/png",
                        }
                    )
            if len(media) >= count:
                break
        if not media:
            raise GeminiApiError("Google GenAI returned no generated image.")
        return {"media": media[:count], "effectiveModel": model, "provider": self.backend}

    async def generate_video(self, model: str, payload: dict[str, Any]) -> dict[str, Any]:
        prompt = str(payload.get("prompt") or "").strip()
        if not prompt:
            raise GeminiApiError("Video prompt is required.")
        is_veo2 = "veo-2" in model.lower()
        is_lite = "lite" in model.lower()
        references = payload.get("ingredientImagePaths") or payload.get("referenceImagePaths") or []
        if not isinstance(references, list):
            references = [references]
        if is_lite:
            references = []
        reference_images = [
            types.VideoGenerationReferenceImage(image=_image(value), reference_type="asset")
            for value in references[:3]
            if value
        ]
        first_frame = payload.get("referenceImagePath") or payload.get("input_reference")
        image = _image(first_frame) if first_frame and not reference_images else None
        duration = _closest_duration(
            payload.get("durationSeconds") or payload.get("seconds"),
            (5, 6, 8) if is_veo2 else (4, 6, 8),
            8,
        )
        if reference_images:
            duration = 8
        config = types.GenerateVideosConfig(
            number_of_videos=max(1, min(int(payload.get("n") or 1), 4)),
            aspect_ratio=str(payload.get("aspectRatio") or payload.get("size") or "9:16"),
            duration_seconds=duration,
            resolution=None if image or reference_images else str(payload.get("resolution") or ("720p" if is_veo2 else "1080p")),
            person_generation="allow_adult" if image or reference_images else ("dont_allow" if is_veo2 else "allow_all"),
            negative_prompt=None if reference_images else str(payload.get("negativePrompt") or "Watermark, text, logo, bad quality, low quality"),
            reference_images=reference_images or None,
            generate_audio=payload.get("generateAudio"),
        )
        client = self._get_client()
        operation = await client.aio.models.generate_videos(
            model=model,
            prompt=prompt,
            image=image,
            config=config,
        )
        poll_interval = max(1.0, float(os.environ.get("OMNIPROVIDERS_VIDEO_POLL_SECONDS", "10")))
        timeout = max(30.0, float(os.environ.get("OMNIPROVIDERS_VIDEO_TIMEOUT_SECONDS", "600")))
        started = time.monotonic()
        while not getattr(operation, "done", False):
            if time.monotonic() - started >= timeout:
                raise GeminiApiError("Video generation timed out.")
            await asyncio.sleep(poll_interval)
            operation = await client.aio.operations.get(operation)
        if getattr(operation, "error", None):
            raise GeminiApiError(f"Video generation failed: {operation.error}")
        response = getattr(operation, "response", None) or getattr(operation, "result", None)
        generated = getattr(response, "generated_videos", None) or []
        media: list[dict[str, Any]] = []
        for generated_video in generated:
            video = getattr(generated_video, "video", None)
            if video is None:
                continue
            data = getattr(video, "video_bytes", None)
            if data is None:
                try:
                    data = await client.aio.files.download(file=video)
                except Exception as exc:
                    uri = str(getattr(video, "uri", None) or "")
                    if uri.startswith(("http://", "https://")):
                        media.append({"url": uri})
                        continue
                    raise GeminiApiError(f"Could not download generated video: {exc}") from exc
            media.append(
                {
                    "bytes": bytes(data),
                    "mime_type": getattr(video, "mime_type", None) or "video/mp4",
                    "filename": f"{model}.mp4",
                }
            )
        if not media:
            reasons = getattr(response, "rai_media_filtered_reasons", None) or []
            suffix = f" Filter reasons: {', '.join(reasons)}" if reasons else ""
            raise GeminiApiError(f"Google GenAI returned no generated video.{suffix}")
        return {"media": media, "effectiveModel": model, "provider": self.backend}
