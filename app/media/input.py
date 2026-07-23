from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import httpx


class ModelMediaError(ValueError):
    pass


def load_attachment_bytes(
    value: Any,
    *,
    mime_type: str | None = None,
    fallback_mime_type: str | None = None,
    max_bytes: int | None = None,
) -> tuple[bytes, str]:
    mime_type = mime_type or fallback_mime_type
    source = value
    has_inline_data = False
    if isinstance(value, dict):
        inline_data = value.get("data")
        has_inline_data = inline_data is not None and inline_data != ""
        source = (
            inline_data
            if has_inline_data
            else value.get("url") or value.get("path") or value.get("source")
        )
        mime_type = str(value.get("mime_type") or value.get("mimeType") or mime_type or "") or None
    if isinstance(source, bytes):
        data = source
    else:
        text = str(source or "").strip()
        if not text:
            raise ModelMediaError("Attachment source is empty.")
        if text.startswith("data:") and "," in text:
            header, encoded = text.split(",", 1)
            declared_mime_type = header[5:].split(";", 1)[0].strip()
            if declared_mime_type:
                mime_type = declared_mime_type
            try:
                data = (
                    base64.b64decode(encoded, validate=True)
                    if ";base64" in header
                    else unquote(encoded).encode()
                )
            except Exception as exc:
                raise ModelMediaError("Attachment data URL is invalid.") from exc
        elif text.startswith(("http://", "https://")):
            response = httpx.get(text, timeout=60.0, follow_redirects=True)
            response.raise_for_status()
            data = response.content
            mime_type = mime_type or response.headers.get("content-type", "").split(";", 1)[0]
        elif has_inline_data:
            try:
                data = base64.b64decode("".join(text.split()), validate=True)
            except Exception as exc:
                raise ModelMediaError("Attachment inline data is invalid base64.") from exc
        else:
            try:
                path = Path(text).expanduser().resolve()
            except (OSError, ValueError) as exc:
                raise ModelMediaError("Attachment path is invalid or too long.") from exc
            if not path.is_file():
                raise ModelMediaError(f"Attachment file not found: {path}")
            data = path.read_bytes()
            mime_type = mime_type or mimetypes.guess_type(path.name)[0]
    if max_bytes is not None and len(data) > max_bytes:
        raise ModelMediaError(f"Attachment exceeds the {max_bytes} byte limit.")
    return data, mime_type or "application/octet-stream"
