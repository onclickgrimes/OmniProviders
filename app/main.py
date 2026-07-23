from __future__ import annotations

import time
import inspect
from uuid import uuid4

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app import config
from app.domain.models import ModelInvocation, ModelRuntimeError
from app.media.artifacts import ArtifactStore, default_artifact_store
from app.persistence.credentials import (
    CredentialStore,
    default_credential_store,
    is_reserved_provider_settings_id,
)
from app.persistence.jobs import GenerationJobStore, default_job_store
from app.runtime.media import MediaRuntime
from app.runtime.registry import ProviderRegistry
from app.runtime.defaults import create_default_registry


def create_app(
    *,
    registry: ProviderRegistry | None = None,
    credential_store: CredentialStore | None = None,
    artifact_store: ArtifactStore | None = None,
    job_store: GenerationJobStore | None = None,
    require_api_key: bool = True,
    api_key: str = "",
) -> FastAPI:
    provider_registry = registry or create_default_registry()
    accounts = credential_store or default_credential_store()
    artifacts = artifact_store or default_artifact_store()
    jobs = job_store or default_job_store()
    jobs.recover_interrupted()
    media_runtime = MediaRuntime(provider_registry, artifacts, jobs)
    app = FastAPI(title=config.APP_NAME, version=config.APP_VERSION)

    @app.exception_handler(ModelRuntimeError)
    async def model_runtime_error_handler(
        _request: Request, error: ModelRuntimeError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=error.status_code,
            content={
                "error": {
                    "message": str(error),
                    "type": error.error_type,
                    "param": error.param,
                    "code": error.code,
                }
            },
        )

    async def authorize(authorization: str | None = Header(default=None)) -> None:
        if not require_api_key:
            return
        expected = f"Bearer {api_key}"
        if not api_key or authorization != expected:
            raise ModelRuntimeError(
                "Invalid OmniProviders API key.",
                code="invalid_api_key",
                status_code=401,
                error_type="authentication_error",
            )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {
            "status": "ok",
            "service": "omni-providers",
            "version": config.APP_VERSION,
        }

    @app.get("/version")
    async def version() -> dict[str, str]:
        return {"service": "omni-providers", "version": config.APP_VERSION}

    @app.get("/runtime", dependencies=[Depends(authorize)])
    async def runtime() -> dict[str, object]:
        return {
            "service": "omni-providers",
            "version": config.APP_VERSION,
            "host": config.HOST,
            "port": config.PORT,
            "dataDir": str(config.DATA_DIR),
            "databasePath": str(config.DATABASE_PATH),
            "artifactTtlSeconds": config.ARTIFACT_TTL_SECONDS,
            "modelCacheTtlSeconds": config.MODEL_CACHE_TTL_SECONDS,
        }

    @app.get("/providers/{provider}/accounts", dependencies=[Depends(authorize)])
    async def list_provider_accounts(provider: str) -> dict[str, object]:
        return {"object": "list", "data": accounts.list_accounts(provider)}

    @app.post("/providers/{provider}/accounts", dependencies=[Depends(authorize)])
    async def save_provider_account(
        provider: str, payload: dict[str, object]
    ) -> dict[str, object]:
        account_id = str(payload.get("id") or f"{provider}-default").strip()
        if is_reserved_provider_settings_id(provider, account_id):
            raise ModelRuntimeError(
                (
                    f"{account_id} is reserved for {provider} settings. "
                    f"Use /providers/{provider}/settings."
                ),
                code="reserved_provider_settings_id",
                param="id",
                status_code=409,
            )
        try:
            adapter = provider_registry.get_adapter(provider)
        except ModelRuntimeError:
            adapter = None
        label = str(payload.get("label") or provider).strip()
        fields = payload.get("fields") if isinstance(payload.get("fields"), dict) else {}
        current = accounts.get_account(account_id, include_secrets=True)
        merged_fields = dict((current or {}).get("fields") or {})
        merged_fields.update(fields)
        validator = getattr(adapter, "validate_account_fields", None) if adapter else None
        if callable(validator):
            validation = validator(merged_fields)
            if inspect.isawaitable(validation):
                await validation
        saved = accounts.upsert_account(
            account_id=account_id,
            provider=provider,
            label=label,
            fields=fields,
            enabled=bool(payload.get("enabled", True)),
        )
        provider_registry.invalidate(provider)
        return saved

    @app.get("/providers/{provider}/settings", dependencies=[Depends(authorize)])
    async def get_provider_settings(provider: str) -> dict[str, object]:
        saved = accounts.get_provider_settings(provider)
        if not saved:
            return {"provider": provider, "fields": {}, "configured": False}
        return {**saved, "configured": True}

    @app.put("/providers/{provider}/settings", dependencies=[Depends(authorize)])
    async def save_provider_settings(
        provider: str,
        payload: dict[str, object],
    ) -> dict[str, object]:
        try:
            adapter = provider_registry.get_adapter(provider)
        except ModelRuntimeError:
            adapter = None
        fields = payload.get("fields") if isinstance(payload.get("fields"), dict) else {}
        current = accounts.get_provider_settings(provider, include_secrets=True)
        merged_fields = dict((current or {}).get("fields") or {})
        merged_fields.update(fields)
        validator = (
            getattr(adapter, "validate_settings_fields", None)
            if adapter
            else None
        )
        if callable(validator):
            validation = validator(merged_fields)
            if inspect.isawaitable(validation):
                await validation
        saved = accounts.upsert_provider_settings(
            provider=provider,
            fields=fields,
        )
        provider_registry.invalidate(provider)
        return {**saved, "configured": True}

    @app.delete(
        "/providers/{provider}/settings",
        dependencies=[Depends(authorize)],
    )
    async def delete_provider_settings(provider: str) -> dict[str, object]:
        deleted = accounts.delete_provider_settings(provider)
        provider_registry.invalidate(provider)
        return {"deleted": deleted, "provider": provider}

    @app.get(
        "/providers/{provider}/configuration",
        dependencies=[Depends(authorize)],
    )
    async def provider_configuration(provider: str) -> dict[str, object]:
        adapter = provider_registry.get_adapter(provider)
        handler = getattr(adapter, "configuration", None)
        if not callable(handler):
            return {"provider": provider, "fields": [], "features": {}}
        result = handler()
        if inspect.isawaitable(result):
            result = await result
        return {"provider": provider, **dict(result)}

    @app.delete(
        "/providers/{provider}/accounts/{account_id}",
        dependencies=[Depends(authorize)],
    )
    async def delete_provider_account(provider: str, account_id: str) -> dict[str, object]:
        deleted = accounts.delete_account(account_id)
        provider_registry.invalidate(provider)
        return {"deleted": deleted, "id": account_id}

    @app.get("/providers", dependencies=[Depends(authorize)])
    async def list_providers() -> dict[str, object]:
        return {
            "object": "list",
            "data": [{"id": adapter.provider_id, "object": "provider"} for adapter in provider_registry.adapters()],
        }

    @app.get("/providers/{provider}/status", dependencies=[Depends(authorize)])
    async def provider_status(provider: str, validate: bool = Query(default=False)) -> dict[str, object]:
        adapter = provider_registry.get_adapter(provider)
        handler = getattr(adapter, "status", None)
        if not callable(handler):
            return {"success": True, "provider": provider, "isLoggedIn": False}
        result = handler(validate=validate)
        if inspect.isawaitable(result):
            result = await result
        return {"provider": provider, **dict(result)}

    @app.post("/providers/{provider}/login", dependencies=[Depends(authorize)])
    async def provider_login(provider: str) -> dict[str, object]:
        adapter = provider_registry.get_adapter(provider)
        handler = getattr(adapter, "start_login", None)
        if not callable(handler):
            return await provider_status(provider)
        result = handler()
        if inspect.isawaitable(result):
            result = await result
        return {"provider": provider, **dict(result)}

    @app.post("/providers/{provider}/oauth/callback", dependencies=[Depends(authorize)])
    async def provider_oauth_callback(provider: str, payload: dict[str, object]) -> dict[str, object]:
        adapter = provider_registry.get_adapter(provider)
        handler = getattr(adapter, "complete_login", None)
        if not callable(handler):
            raise ModelRuntimeError(
                f"Provider '{provider}' does not use an OAuth callback.",
                code="unsupported_auth_flow",
            )
        result = handler(dict(payload))
        if inspect.isawaitable(result):
            result = await result
        provider_registry.invalidate(provider)
        return {"provider": provider, **dict(result)}

    @app.post(
        "/providers/{provider}/session/refresh",
        dependencies=[Depends(authorize)],
    )
    async def refresh_provider_session(provider: str) -> dict[str, object]:
        adapter = provider_registry.get_adapter(provider)
        handler = getattr(adapter, "refresh_session", None)
        if not callable(handler):
            raise ModelRuntimeError(
                f"Provider '{provider}' does not support browser session refresh.",
                code="unsupported_auth_flow",
            )
        result = handler()
        if inspect.isawaitable(result):
            result = await result
        provider_registry.invalidate(provider)
        return {"provider": provider, **dict(result)}

    @app.post("/providers/{provider}/close", dependencies=[Depends(authorize)])
    async def close_provider(provider: str) -> dict[str, object]:
        adapter = provider_registry.get_adapter(provider)
        handler = getattr(adapter, "close", None)
        if callable(handler):
            result = handler()
            if inspect.isawaitable(result):
                await result
        return {"success": True, "provider": provider}

    @app.get("/v1/models", dependencies=[Depends(authorize)])
    async def list_models(refresh: bool = Query(default=False)) -> dict[str, object]:
        models = await provider_registry.list_models(refresh=refresh)
        return {"object": "list", "data": [model.to_openai_dict() for model in models]}

    @app.get("/providers/{provider}/models", dependencies=[Depends(authorize)])
    async def list_provider_models(provider: str, refresh: bool = Query(default=False)) -> dict[str, object]:
        provider_registry.get_adapter(provider)
        models = [
            model.to_openai_dict()
            for model in await provider_registry.list_models(refresh=refresh)
            if model.provider == provider
        ]
        return {"success": True, "provider": provider, "models": models}

    @app.post("/v1/responses", dependencies=[Depends(authorize)])
    async def create_response(payload: dict[str, object]) -> dict[str, object]:
        model = str(payload.get("model") or "").strip()
        text_config = payload.get("text") if isinstance(payload.get("text"), dict) else {}
        format_config = text_config.get("format") if isinstance(text_config.get("format"), dict) else {}
        response_format = str(format_config.get("type") or payload.get("response_format") or "text")
        model_input = payload.get("input", "")
        instructions = str(payload.get("instructions") or "").strip()
        if instructions:
            if isinstance(model_input, list):
                model_input = [{"role": "system", "content": instructions}, *model_input]
            else:
                model_input = [
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": str(model_input or "")},
                ]
        result = await provider_registry.invoke(
            model,
            ModelInvocation(
                input=model_input,
                instructions=instructions or None,
                tools=tuple(payload.get("tools") or ()),
                tool_choice=payload.get("tool_choice"),
                temperature=(
                    float(payload["temperature"])
                    if payload.get("temperature") is not None
                    else None
                ),
                response_format="json" if response_format in {"json_object", "json_schema", "json"} else "text",
            ),
        )
        response_id = f"resp_{uuid4().hex}"
        output: list[dict[str, object]] = []
        for function_call in result.function_calls:
            output.append(
                {
                    "id": f"fc_{uuid4().hex}",
                    "type": "function_call",
                    "status": "completed",
                    "call_id": function_call.call_id,
                    "name": function_call.name,
                    "arguments": function_call.arguments,
                }
            )
        if result.text:
            output.append(
                {
                    "id": f"msg_{uuid4().hex}",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": result.text, "annotations": []}
                    ],
                }
            )
        return {
            "id": response_id,
            "object": "response",
            "created_at": int(time.time()),
            "status": "completed",
            "model": model,
            "output": output,
            "usage": result.usage,
            "metadata": {
                "effective_model": result.effective_model,
                **result.metadata,
            },
        }

    @app.post("/v1/chat/completions", dependencies=[Depends(authorize)])
    async def create_chat_completion(payload: dict[str, object]) -> dict[str, object]:
        model = str(payload.get("model") or "").strip()
        result = await provider_registry.invoke(
            model,
            ModelInvocation(
                input=payload.get("messages") or [],
                tools=tuple(payload.get("tools") or ()),
                tool_choice=payload.get("tool_choice"),
                temperature=(
                    float(payload["temperature"])
                    if payload.get("temperature") is not None
                    else None
                ),
                response_format=(
                    "json"
                    if isinstance(payload.get("response_format"), dict)
                    and payload["response_format"].get("type") in {"json_object", "json_schema"}
                    else "text"
                ),
            ),
        )
        message: dict[str, object] = {"role": "assistant", "content": result.text or None}
        if result.function_calls:
            message["tool_calls"] = [
                {
                    "id": item.call_id,
                    "type": "function",
                    "function": {"name": item.name, "arguments": item.arguments},
                }
                for item in result.function_calls
            ]
        return {
            "id": f"chatcmpl_{uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if result.function_calls else "stop",
                }
            ],
            "usage": result.usage,
        }

    @app.post("/v1/images/generations", dependencies=[Depends(authorize)])
    async def generate_images(payload: dict[str, object]) -> dict[str, object]:
        model = str(payload.get("model") or "").strip()
        normalized = dict(payload)
        normalized.setdefault("count", payload.get("n", 1))
        normalized.setdefault("aspectRatio", payload.get("size"))
        return await media_runtime.generate_images(model, normalized)

    @app.post("/v1/videos", dependencies=[Depends(authorize)])
    async def generate_video(
        payload: dict[str, object], background_tasks: BackgroundTasks
    ) -> dict[str, object]:
        model = str(payload.get("model") or "").strip()
        job = media_runtime.create_video_job(model, dict(payload))
        background_tasks.add_task(
            media_runtime.run_video_job,
            str(job["id"]),
            model,
            dict(payload),
        )
        return job

    @app.get("/v1/videos/{video_id}", dependencies=[Depends(authorize)])
    async def get_video(video_id: str) -> dict[str, object]:
        return jobs.get(video_id)

    @app.get("/v1/videos/{video_id}/content", dependencies=[Depends(authorize)])
    async def get_video_content(video_id: str) -> FileResponse:
        job = jobs.get(video_id)
        artifacts_data = ((job.get("result") or {}).get("artifacts") or [])
        if job.get("status") != "completed" or not artifacts_data:
            raise ModelRuntimeError(
                "Video content is not available yet.",
                code="video_not_ready",
                status_code=409,
            )
        artifact = artifacts.get(str(artifacts_data[0]["id"]))
        return FileResponse(
            artifact.path,
            media_type=artifact.mime_type,
            filename=artifact.filename,
        )

    @app.post("/v1/audio/speech", dependencies=[Depends(authorize)])
    async def generate_speech(payload: dict[str, object]) -> FileResponse:
        model = str(payload.get("model") or "").strip()
        artifact = await media_runtime.generate_speech(model, dict(payload))
        return FileResponse(
            artifact.path,
            media_type=artifact.mime_type,
            filename=artifact.filename,
        )

    @app.post("/v1/audio/transcriptions", dependencies=[Depends(authorize)])
    async def transcribe_audio(
        file: UploadFile = File(...),
        model: str = Form(...),
        language: str | None = Form(default=None),
        response_format: str | None = Form(default=None),
    ) -> dict[str, object]:
        audio = await file.read()
        return await media_runtime.transcribe(
            model,
            audio,
            {
                "filename": file.filename,
                "mime_type": file.content_type,
                "language": language,
                "response_format": response_format,
            },
        )

    @app.get("/v1/artifacts/{artifact_id}", dependencies=[Depends(authorize)])
    async def get_artifact(artifact_id: str) -> FileResponse:
        artifact = artifacts.get(artifact_id)
        return FileResponse(
            artifact.path,
            media_type=artifact.mime_type,
            filename=artifact.filename,
        )

    return app


config.ensure_runtime_directories()
app = create_app(
    require_api_key=bool(config.API_KEY),
    api_key=config.API_KEY,
)
