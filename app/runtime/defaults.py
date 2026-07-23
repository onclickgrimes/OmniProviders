from __future__ import annotations

from app.providers.antigravity import AntigravityProviderAdapter
from app.providers.deepgram import DeepgramProviderAdapter
from app.providers.flow import FlowProviderAdapter
from app.providers.gemini_web import GeminiWebProviderAdapter
from app.providers.google_genai import GeminiProviderAdapter, VertexProviderAdapter
from app.providers.kiro import KiroProviderAdapter
from app.runtime.registry import ProviderRegistry


def create_default_registry() -> ProviderRegistry:
    return ProviderRegistry(
        [
            KiroProviderAdapter(),
            AntigravityProviderAdapter(),
            GeminiProviderAdapter(),
            VertexProviderAdapter(),
            GeminiWebProviderAdapter(),
            FlowProviderAdapter(),
            DeepgramProviderAdapter(),
        ]
    )
