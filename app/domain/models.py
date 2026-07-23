from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ToolCallingMode = Literal["none", "native"]
DiscoveryMode = Literal["account_live", "provider_live", "static_verified"]


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    input_modalities: frozenset[str] = frozenset({"text"})
    output_modalities: frozenset[str] = frozenset({"text"})
    structured_output: bool = False
    streaming: bool = False
    tool_calling: ToolCallingMode = "none"
    operations: frozenset[str] = frozenset({"responses", "chat.completions"})

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_modalities": sorted(self.input_modalities),
            "output_modalities": sorted(self.output_modalities),
            "structured_output": self.structured_output,
            "streaming": self.streaming,
            "tool_calling": self.tool_calling,
            "operations": sorted(self.operations),
        }


@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    provider: str
    model: str
    label: str
    available: bool = True
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)
    discovery: DiscoveryMode = "static_verified"
    effective_model: str | None = None
    account_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.provider}:{self.model}"

    def to_openai_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": "model",
            "created": 0,
            "owned_by": self.provider,
            "x_omni": {
                "label": self.label,
                "effective_model": self.effective_model or self.model,
                "account_id": self.account_id,
                "discovery": self.discovery,
                "capabilities": self.capabilities.to_dict(),
                **self.metadata,
            },
        }


@dataclass(frozen=True, slots=True)
class ModelInvocation:
    input: Any
    instructions: str | None = None
    tools: tuple[dict[str, Any], ...] = ()
    tool_choice: Any = None
    temperature: float | None = None
    response_format: str = "text"


@dataclass(frozen=True, slots=True)
class FunctionCall:
    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class ModelResult:
    text: str = ""
    effective_model: str | None = None
    function_calls: tuple[FunctionCall, ...] = ()
    usage: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelRuntimeError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "model_runtime_error",
        status_code: int = 400,
        param: str | None = None,
        error_type: str = "invalid_request_error",
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.param = param
        self.error_type = error_type


@dataclass(frozen=True, slots=True)
class Artifact:
    id: str
    path: str
    mime_type: str
    filename: str
    size: int
    created_at: float
    expires_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": "artifact",
            "filename": self.filename,
            "mime_type": self.mime_type,
            "size": self.size,
            "created_at": int(self.created_at),
            "expires_at": int(self.expires_at),
            "url": f"/v1/artifacts/{self.id}",
        }
