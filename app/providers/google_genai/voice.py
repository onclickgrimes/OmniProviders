from __future__ import annotations

import asyncio
import base64
import struct
from typing import Any

from google.genai import errors, types

from app.persistence.credentials import get_env_or_provider_credential
from app.providers.google_genai.transport import create_genai_client


TARGET_AUDIO_ENCODING = "LINEAR16"
TARGET_SAMPLE_RATE_HZ = 22050
GEMINI_TTS_DEFAULT_MIME_TYPE = "audio/L16;rate=24000"
GEMINI_31_FLASH_TTS_MODEL = "gemini-3.1-flash-tts-preview"
DEFAULT_TTS_MODEL = GEMINI_31_FLASH_TTS_MODEL
PRO_TTS_MODEL = "gemini-2.5-pro-preview-tts"
FLASH_TTS_MODEL = "gemini-2.5-flash-preview-tts"
VERTEX_PRO_TTS_MODEL = "gemini-2.5-pro-tts"
VERTEX_FLASH_TTS_MODEL = "gemini-2.5-flash-tts"

GEMINI_VOICES = [
    "Achernar",
    "Achird",
    "Algenib",
    "Algieba",
    "Alnilam",
    "Aoede",
    "Autonoe",
    "Callirrhoe",
    "Charon",
    "Despina",
    "Enceladus",
    "Erinome",
    "Fenrir",
    "Gacrux",
    "Iapetus",
    "Kore",
    "Laomedeia",
    "Leda",
    "Orus",
    "Puck",
    "Pulcherrima",
    "Rasalgethi",
    "Sadachbia",
    "Sadaltager",
    "Schedar",
    "Sulafat",
    "Umbriel",
    "Vindemiatrix",
    "Zephyr",
    "Zubenelgenubi",
]


class GeminiVoiceError(RuntimeError):
    pass


def _clamp_temperature(value: Any) -> float:
    try:
        temperature = float(value)
    except (TypeError, ValueError):
        temperature = 1.0
    return max(0.0, min(2.0, temperature))


def _is_rate_limit_error(message: str) -> bool:
    normalized = message.lower()
    return (
        "429" in normalized
        or "rate limit" in normalized
        or "quota" in normalized
        or "resource exhausted" in normalized
    )


class GeminiVoiceService:
    def __init__(self, *, backend: str, client: Any | None = None) -> None:
        self._backend = backend
        self._provided_client = client
        self._client: Any | None = client
        self._client_state_hash: str | None = "provided" if client is not None else None

    def _default_model(self) -> str:
        return get_env_or_provider_credential(self._backend, "GEMINI_TTS_MODEL") or DEFAULT_TTS_MODEL

    def _default_voice(self) -> str:
        return get_env_or_provider_credential(self._backend, "GEMINI_TTS_VOICE") or "Achernar"

    def _client_state(self) -> str:
        return ":".join(
            [
                self._backend,
                str(get_env_or_provider_credential(self._backend, "GEMINI_API_KEY", "GOOGLE_API_KEY") or ""),
                str(get_env_or_provider_credential(self._backend, "VERTEX_PROJECT") or ""),
                str(get_env_or_provider_credential(self._backend, "VERTEX_LOCATION") or ""),
                str(get_env_or_provider_credential(self._backend, "VERTEX_CREDENTIALS_PATH") or ""),
            ]
        )

    def _get_client(self) -> Any:
        if self._provided_client is not None:
            return self._provided_client
        state = self._client_state()
        if self._client is None or self._client_state_hash != state:
            self._client = create_genai_client(
                self._backend,
                api_version="v1beta1" if self._backend == "vertex" else "v1beta",
            )
            self._client_state_hash = state
        return self._client

    def available_voices(self) -> list[str]:
        return list(GEMINI_VOICES)

    async def generate_speech(self, payload: dict[str, Any]) -> dict[str, Any]:
        text = str(payload.get("input") or payload.get("text") or payload.get("script") or "").strip()
        if not text:
            return {"success": False, "error": "Texto vazio fornecido."}

        voice_name = str(
            payload.get("voiceName")
            or payload.get("voice_name")
            or payload.get("voiceId")
            or payload.get("voice_id")
            or self._default_voice()
        ).strip()
        temperature = _clamp_temperature(payload.get("temperature"))
        requested_model = str(payload.get("model") or self._default_model()).strip()
        model = requested_model or self._default_model()

        last_error = ""
        for attempt in range(3):
            try:
                wav_buffer, mime_type = await self._generate_wav(
                    text=text,
                    voice_name=voice_name,
                    temperature=temperature,
                    model=model,
                )
                result = {
                    "success": True,
                    "media": [
                        {
                            "bytes": wav_buffer,
                            "mime_type": "audio/wav",
                            "filename": f"speech-{voice_name}.wav",
                        }
                    ],
                    "sourceMimeType": mime_type,
                    "size": len(wav_buffer),
                    "voiceName": voice_name,
                    "model": model,
                }
                return result
            except Exception as exc:
                last_error = self._format_generation_error(exc)
                if _is_rate_limit_error(last_error):
                    pro_model = VERTEX_PRO_TTS_MODEL if self._backend == "vertex" else PRO_TTS_MODEL
                    flash_model = VERTEX_FLASH_TTS_MODEL if self._backend == "vertex" else FLASH_TTS_MODEL
                    model = flash_model if model == pro_model else pro_model
                if attempt < 2:
                    await asyncio.sleep(3)

        return {"success": False, "error": last_error or "Erro desconhecido ao gerar áudio no Gemini TTS."}

    async def _generate_wav(
        self,
        *,
        text: str,
        voice_name: str,
        temperature: float,
        model: str,
    ) -> tuple[bytes, str]:
        config = types.GenerateContentConfig(
            temperature=temperature,
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice_name,
                    )
                )
            ),
        )
        contents = [
            types.Content(
                role="user",
                parts=[types.Part(text=text)],
            )
        ]

        audio_buffers: list[bytes] = []
        mime_type = GEMINI_TTS_DEFAULT_MIME_TYPE
        try:
            stream = await self._get_client().aio.models.generate_content_stream(
                model=model,
                config=config,
                contents=contents,
            )
            async for chunk in stream:
                for candidate in chunk.candidates or []:
                    content = candidate.content
                    if not content:
                        continue
                    for part in content.parts or []:
                        inline_data = part.inline_data
                        if not inline_data or inline_data.data is None:
                            continue
                        if inline_data.mime_type:
                            mime_type = inline_data.mime_type
                        data = inline_data.data
                        if isinstance(data, str):
                            audio_buffers.append(base64.b64decode(data))
                        else:
                            audio_buffers.append(bytes(data))
        except errors.APIError as exc:
            raise GeminiVoiceError(f"Gemini API error: {exc}") from exc
        except Exception as exc:
            raise GeminiVoiceError(f"Gemini SDK error: {exc}") from exc

        if not audio_buffers:
            raise GeminiVoiceError("Nenhum áudio gerado.")

        raw_audio = b"".join(audio_buffers)
        return self._convert_to_wav(raw_audio, mime_type), mime_type

    def _convert_to_wav(self, raw_buffer: bytes, mime_type: str) -> bytes:
        normalized_mime = (mime_type or "").split(";", 1)[0].strip().lower()
        if normalized_mime in {"audio/wav", "audio/wave", "audio/x-wav"} and raw_buffer.startswith(b"RIFF"):
            return raw_buffer

        source_options = self._parse_mime_type(mime_type)
        target_options = {
            **source_options,
            "sample_rate": TARGET_SAMPLE_RATE_HZ,
            "bits_per_sample": 16,
        }
        linear16_buffer = self._normalize_linear16_sample_rate(
            raw_buffer,
            source_options,
            target_options,
        )
        wav_header = self._create_wav_header(linear16_buffer, target_options)
        return wav_header + linear16_buffer

    def _parse_mime_type(self, mime_type: str) -> dict[str, int]:
        file_type, *params = [part.strip() for part in (mime_type or GEMINI_TTS_DEFAULT_MIME_TYPE).split(";")]
        _, _, audio_format = file_type.partition("/")
        options = {
            "num_channels": 1,
            "sample_rate": 24000,
            "bits_per_sample": 16,
        }
        if audio_format.startswith("L"):
            try:
                options["bits_per_sample"] = int(audio_format[1:])
            except ValueError:
                pass
        for param in params:
            key, _, value = param.partition("=")
            if key.strip().lower() == "rate":
                try:
                    options["sample_rate"] = int(value.strip())
                except ValueError:
                    pass
        return options

    def _normalize_linear16_sample_rate(
        self,
        raw_buffer: bytes,
        source_options: dict[str, int],
        target_options: dict[str, int],
    ) -> bytes:
        if source_options["bits_per_sample"] != 16:
            raise GeminiVoiceError(
                f"Formato de áudio inesperado: {source_options['bits_per_sample']} bits. "
                f"{TARGET_AUDIO_ENCODING} requer PCM de 16 bits."
            )

        frame_size = max(1, source_options["num_channels"]) * 2
        aligned_length = len(raw_buffer) - (len(raw_buffer) % frame_size)
        linear16_buffer = raw_buffer[:aligned_length]
        if source_options["sample_rate"] == target_options["sample_rate"]:
            return linear16_buffer
        return self._resample_pcm16(
            linear16_buffer,
            source_options["sample_rate"],
            target_options["sample_rate"],
            target_options["num_channels"],
        )

    def _resample_pcm16(
        self,
        input_data: bytes,
        source_sample_rate: int,
        target_sample_rate: int,
        num_channels: int,
    ) -> bytes:
        bytes_per_sample = 2
        frame_size = bytes_per_sample * num_channels
        source_frame_count = len(input_data) // frame_size
        if source_frame_count == 0 or source_sample_rate <= 0 or target_sample_rate <= 0:
            return input_data

        target_frame_count = max(1, round(source_frame_count * target_sample_rate / source_sample_rate))
        output = bytearray(target_frame_count * frame_size)

        for target_frame in range(target_frame_count):
            source_position = target_frame * source_sample_rate / target_sample_rate
            source_frame = min(int(source_position), source_frame_count - 1)
            next_frame = min(source_frame + 1, source_frame_count - 1)
            fraction = source_position - source_frame
            for channel in range(num_channels):
                sample_offset = (source_frame * num_channels + channel) * bytes_per_sample
                next_sample_offset = (next_frame * num_channels + channel) * bytes_per_sample
                sample = struct.unpack_from("<h", input_data, sample_offset)[0]
                next_sample = struct.unpack_from("<h", input_data, next_sample_offset)[0]
                interpolated = round(sample + (next_sample - sample) * fraction)
                interpolated = max(-32768, min(32767, interpolated))
                output_offset = (target_frame * num_channels + channel) * bytes_per_sample
                struct.pack_into("<h", output, output_offset, interpolated)

        return bytes(output)

    def _create_wav_header(self, linear16_buffer: bytes, options: dict[str, int]) -> bytes:
        data_length = len(linear16_buffer)
        num_channels = options["num_channels"]
        sample_rate = options["sample_rate"]
        bits_per_sample = options["bits_per_sample"]
        byte_rate = sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        return struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + data_length,
            b"WAVE",
            b"fmt ",
            16,
            1,
            num_channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            b"data",
            data_length,
        )

    def _format_generation_error(self, error: Exception) -> str:
        message = str(error) or "Erro desconhecido"
        normalized = message.lower()
        missing_default_credentials = (
            "could not load the default credentials" in normalized
            or "default credentials" in normalized
        )
        if missing_default_credentials:
            credentials_path = get_env_or_provider_credential("vertex", "VERTEX_CREDENTIALS_PATH")
            hint = (
                f"Verifique se o arquivo está acessível: {credentials_path}."
                if credentials_path
                else "Configure um JSON de Service Account em Configurações > Google GenAI Engine, ou execute gcloud auth application-default login."
            )
            return f"Falha de autenticação do Vertex AI: credenciais Google Cloud não encontradas. {hint}"
        return message
