from __future__ import annotations

import json
import logging
from typing import Any


logger = logging.getLogger("omni-providers.exchange")


def serialize_messages_for_ai_debug(messages: Any) -> Any:
    """Return a JSON-safe, size-bounded representation without attachment bytes."""
    try:
        serialized = json.loads(json.dumps(messages, default=str))
    except Exception:
        return str(messages)[:4000]
    if isinstance(serialized, list):
        for message in serialized:
            if not isinstance(message, dict):
                continue
            for part in message.get("parts") or []:
                if isinstance(part, dict) and part.get("data"):
                    part["data"] = f"<omitted:{len(str(part['data']))}>"
    return serialized


def write_ai_debug_exchange(
    *,
    provider: str,
    model: str,
    operation: str,
    request: Any,
    response: Any = None,
    error: Any = None,
    metadata: Any = None,
) -> None:
    logger.debug(
        "provider=%s model=%s operation=%s request=%s response=%s error=%s metadata=%s",
        provider,
        model,
        operation,
        request,
        response,
        error,
        metadata,
    )


def make_json_safe(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, default=str))
    except Exception:
        return str(value)
