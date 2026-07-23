from __future__ import annotations

import json
import re
from typing import Any


class StructuredOutputError(ValueError):
    pass


def extract_json(value: str) -> Any:
    text = str(value or "").strip()
    if not text:
        raise StructuredOutputError("Model returned empty structured output.")
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    candidates = [fenced.group(1).strip()] if fenced else []
    candidates.append(text)
    starts = [position for marker in ("{", "[") if (position := text.find(marker)) >= 0]
    if starts:
        candidates.append(text[min(starts) :])
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            try:
                parsed, _end = decoder.raw_decode(candidate)
                return parsed
            except json.JSONDecodeError:
                continue
    raise StructuredOutputError("Model did not return valid JSON.")
