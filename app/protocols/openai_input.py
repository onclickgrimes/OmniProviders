from __future__ import annotations

from typing import Any


def messages_from_openai_input(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    if not isinstance(value, list):
        return [{"role": "user", "content": str(value or "")}]

    messages: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "")
        if item_type in {"function_call", "function_call_output"}:
            messages.append(dict(item))
            continue
        role = str(item.get("role") or "user")
        content = item.get("content", "")
        if role == "tool":
            messages.append(
                {
                    "type": "function_call_output",
                    "call_id": str(item.get("tool_call_id") or item.get("call_id") or ""),
                    "output": content if isinstance(content, str) else str(content or ""),
                }
            )
            continue
        if role == "assistant" and isinstance(item.get("tool_calls"), list):
            if content:
                messages.append({"role": role, "content": content})
            for tool_call in item["tool_calls"]:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
                messages.append(
                    {
                        "type": "function_call",
                        "call_id": str(tool_call.get("id") or ""),
                        "name": str(function.get("name") or tool_call.get("name") or ""),
                        "arguments": function.get("arguments") or tool_call.get("arguments") or "{}",
                    }
                )
            continue
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        parts: list[dict[str, Any]] = []
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = str(part.get("type") or "")
                if part_type in {"input_text", "output_text", "text"}:
                    parts.append({"type": "text", "text": str(part.get("text") or "")})
                elif part_type in {"input_image", "image_url"}:
                    image_url = part.get("image_url") or part.get("url")
                    if isinstance(image_url, dict):
                        image_url = image_url.get("url")
                    parts.append({"type": "image", "url": image_url})
                elif part_type in {"input_video", "video_url"}:
                    parts.append(
                        {
                            "type": "video",
                            "url": part.get("video_url") or part.get("url"),
                        }
                    )
                elif part_type in {"input_audio", "audio"}:
                    parts.append({**part, "type": "audio"})
                elif part_type in {"input_file", "file"}:
                    parts.append({**part, "type": "file"})
        messages.append({"role": role, "parts": parts})
    return messages


def input_modalities(value: Any) -> frozenset[str]:
    modalities = {"text"}

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            part_type = str(node.get("type") or "").lower()
            if "image" in part_type:
                modalities.add("image")
            if "video" in part_type:
                modalities.add("video")
            if "audio" in part_type:
                modalities.add("audio")
            if "file" in part_type:
                modalities.add("file")
            for nested in node.values():
                visit(nested)
        elif isinstance(node, list):
            for nested in node:
                visit(nested)

    visit(value)
    return frozenset(modalities)
