from __future__ import annotations

import os
import re
from typing import Any

import httpx

from app.persistence.credentials import get_env_or_credential


class DeepgramError(RuntimeError):
    pass


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class DeepgramTransport:
    def __init__(self, *, client: Any | None = None, api_key: str | None = None) -> None:
        self._client = client
        self._api_key = api_key

    def has_auth_config(self) -> bool:
        return bool(self._api_key or get_env_or_credential("DEEPGRAM_API_KEY", "DEEPGRAM_TOKEN"))

    async def transcribe(
        self,
        audio: bytes,
        *,
        model: str,
        language: str | None = None,
        mime_type: str | None = None,
    ) -> dict[str, Any]:
        token = self._api_key or get_env_or_credential("DEEPGRAM_API_KEY", "DEEPGRAM_TOKEN")
        if not token:
            raise DeepgramError("Deepgram is not configured.")
        params = {
            "model": model,
            "language": language or get_env_or_credential("DEEPGRAM_LANGUAGE") or "pt-BR",
            "smart_format": "true",
            "diarize": "true",
            "punctuate": "true",
            "paragraphs": "true",
            "utterances": "true",
            "utt_split": os.environ.get("DEEPGRAM_UTT_SPLIT", "1.1"),
        }
        headers = {
            "Authorization": f"Token {token}",
            "Content-Type": mime_type or "application/octet-stream",
        }
        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(
            timeout=float(os.environ.get("DEEPGRAM_REQUEST_TIMEOUT", "300"))
        )
        try:
            response = await client.post(
                "https://api.deepgram.com/v1/listen",
                params=params,
                headers=headers,
                content=audio,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise DeepgramError(f"Deepgram transcription failed: {exc}") from exc
        finally:
            if owns_client:
                await client.aclose()
        return self._normalize(payload)

    def _normalize(self, response: dict[str, Any]) -> dict[str, Any]:
        results = response.get("results") or {}
        channels = results.get("channels") or []
        alternatives = (channels[0].get("alternatives") or []) if channels else []
        if not alternatives:
            raise DeepgramError("Deepgram returned no transcription alternative.")
        alternative = alternatives[0]
        raw_words = alternative.get("words") or []
        words = [
            {
                "word": str(item.get("word") or self._plain_word(str(item.get("punctuated_word") or ""))),
                "start": _float(item.get("start")),
                "end": _float(item.get("end")),
                "confidence": _float(item.get("confidence")),
                "speaker": _int(item.get("speaker")),
                "punctuatedWord": str(item.get("punctuated_word") or item.get("word") or ""),
            }
            for item in raw_words
            if item.get("word") or item.get("punctuated_word")
        ]
        transcript = str(alternative.get("transcript") or "")
        duration = _float((response.get("metadata") or {}).get("duration"))
        if duration <= 0 and words:
            duration = max(item["end"] for item in words)
        paragraphs = self._paragraphs(((alternative.get("paragraphs") or {}).get("paragraphs") or []))
        segments = self._segments_from_utterances(results.get("utterances") or [], words)
        if not segments:
            segments = self._segments_from_paragraphs(paragraphs, words)
        if not segments and words:
            segments = self._segments_from_words(words)
        if not segments and transcript:
            segments = [{"id": 1, "text": transcript, "start": 0, "end": duration, "speaker": 0, "words": words}]
        return {
            "text": transcript,
            "duration": duration,
            "transcript": transcript,
            "confidence": _float(alternative.get("confidence")),
            "words": words,
            "paragraphs": paragraphs,
            "segments": segments,
        }

    @staticmethod
    def _paragraphs(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "sentences": [
                    {"text": str(sentence.get("text") or ""), "start": _float(sentence.get("start")), "end": _float(sentence.get("end"))}
                    for sentence in item.get("sentences") or []
                    if sentence.get("text")
                ],
                "speaker": _int(item.get("speaker")),
                "numWords": _int(item.get("num_words")),
                "start": _float(item.get("start")),
                "end": _float(item.get("end")),
            }
            for item in values
        ]

    @staticmethod
    def _words_in_range(words: list[dict[str, Any]], start: float, end: float) -> list[dict[str, Any]]:
        return [item for item in words if item["start"] >= start and item["end"] <= end]

    def _segments_from_utterances(self, values: list[dict[str, Any]], words: list[dict[str, Any]]) -> list[dict[str, Any]]:
        segments = []
        for index, item in enumerate(values, 1):
            start, end = _float(item.get("start")), _float(item.get("end"))
            text = str(item.get("transcript") or "").strip()
            if text:
                segments.append({"id": index, "text": text, "start": start, "end": end, "speaker": _int(item.get("speaker")), "words": self._words_in_range(words, start, end)})
        return segments

    def _segments_from_paragraphs(self, paragraphs: list[dict[str, Any]], words: list[dict[str, Any]]) -> list[dict[str, Any]]:
        segments = []
        for paragraph in paragraphs:
            for sentence in paragraph["sentences"]:
                start, end = sentence["start"], sentence["end"]
                segments.append({"id": len(segments) + 1, "text": sentence["text"], "start": start, "end": end, "speaker": paragraph["speaker"], "words": self._words_in_range(words, start, end)})
        return segments

    @staticmethod
    def _segments_from_words(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
        groups: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for word in words:
            previous = current[-1] if current else None
            if previous and (word["speaker"] != previous["speaker"] or word["start"] - previous["end"] > 1 or str(previous["punctuatedWord"]).endswith((".", "!", "?")) or len(current) >= 28):
                groups.append(current)
                current = []
            current.append(word)
        if current:
            groups.append(current)
        return [
            {"id": index, "text": " ".join(item["punctuatedWord"] for item in group), "start": group[0]["start"], "end": group[-1]["end"], "speaker": group[0]["speaker"], "words": group}
            for index, group in enumerate(groups, 1)
        ]

    @staticmethod
    def _plain_word(value: str) -> str:
        return re.sub(r"^\W+|\W+$", "", value).lower()
