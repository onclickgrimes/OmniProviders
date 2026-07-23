from __future__ import annotations

import base64
import unittest
from unittest.mock import Mock, patch

from app.media.input import ModelMediaError, load_attachment_bytes


class MediaInputContractTest(unittest.TestCase):
    def test_decodes_legacy_inline_base64_and_preserves_mime_type(self) -> None:
        expected = b"\xff\xd8\xff\xd9"

        payload, mime_type = load_attachment_bytes(
            {
                "data": base64.b64encode(expected).decode("ascii"),
                "mimeType": "image/jpeg",
            },
            fallback_mime_type="image/png",
        )

        self.assertEqual(expected, payload)
        self.assertEqual("image/jpeg", mime_type)

    def test_invalid_inline_base64_is_reported_as_media_error(self) -> None:
        with self.assertRaisesRegex(ModelMediaError, "base64"):
            load_attachment_bytes(
                {"data": "not valid base64!", "mimeType": "image/jpeg"},
                fallback_mime_type="image/png",
            )

    def test_data_url_mime_type_overrides_provider_fallback(self) -> None:
        payload, mime_type = load_attachment_bytes(
            "data:image/jpeg;base64,/9j/2Q==",
            fallback_mime_type="image/png",
        )

        self.assertEqual(b"\xff\xd8\xff\xd9", payload)
        self.assertEqual("image/jpeg", mime_type)

    def test_legacy_url_misfiled_as_data_is_loaded_as_url(self) -> None:
        response = Mock(
            content=b"RIFF....WAVE",
            headers={"content-type": "audio/wav"},
        )
        response.raise_for_status.return_value = None

        with patch("app.media.input.httpx.get", return_value=response) as get:
            payload, mime_type = load_attachment_bytes(
                {"data": "http://127.0.0.1:17812/assets/voice.wav"},
                fallback_mime_type="audio/wav",
            )

        get.assert_called_once()
        self.assertEqual(b"RIFF....WAVE", payload)
        self.assertEqual("audio/wav", mime_type)


if __name__ == "__main__":
    unittest.main()
