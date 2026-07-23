from __future__ import annotations

import unittest

from app.protocols.openai_input import messages_from_openai_input


class OpenAIInputContractTest(unittest.TestCase):
    def test_audio_source_is_normalized_without_becoming_inline_data(self) -> None:
        messages = messages_from_openai_input(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "path": "C:/project/audio/voice.wav",
                            "url": "http://127.0.0.1:17812/assets/voice.wav",
                            "mimeType": "audio/wav",
                        }
                    ],
                }
            ]
        )

        audio = messages[0]["parts"][0]
        self.assertEqual("audio", audio["type"])
        self.assertEqual(
            "http://127.0.0.1:17812/assets/voice.wav",
            audio["url"],
        )
        self.assertNotIn("data", audio)

    def test_chat_tool_history_becomes_responses_function_items(self) -> None:
        messages = messages_from_openai_input(
            [
                {"role": "user", "content": "Mostre a cena 2"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "get_scene", "arguments": '{"number":2}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_2", "content": '{"id":2}'},
            ]
        )

        self.assertEqual("function_call", messages[1]["type"])
        self.assertEqual("get_scene", messages[1]["name"])
        self.assertEqual("function_call_output", messages[2]["type"])
        self.assertEqual("call_2", messages[2]["call_id"])


if __name__ == "__main__":
    unittest.main()
