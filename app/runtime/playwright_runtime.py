from __future__ import annotations

import os
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _existing_executable(value: str | os.PathLike[str] | None) -> str | None:
    if not value:
        return None
    candidate = Path(value).expanduser().resolve()
    return str(candidate) if candidate.is_file() else None


def _default_external_node_candidates() -> list[Path]:
    python_project = PROJECT_ROOT.parent / "python-project"
    return [
        python_project / "node_modules" / "electron" / "dist" / "electron.exe",
    ]


def resolve_playwright_node_path() -> str | None:
    explicit = _existing_executable(
        os.environ.get("OMNIPROVIDERS_PLAYWRIGHT_NODE_PATH")
    )
    if explicit:
        return explicit

    inherited = _existing_executable(os.environ.get("PLAYWRIGHT_NODEJS_PATH"))
    if inherited:
        return inherited

    for candidate in _default_external_node_candidates():
        existing = _existing_executable(candidate)
        if existing:
            return existing

    for command in ("node.exe", "node"):
        existing = _existing_executable(shutil.which(command))
        if existing:
            return existing
    return None


def configure_external_playwright_node() -> str | None:
    node_path = resolve_playwright_node_path()
    if not node_path:
        return None
    os.environ["PLAYWRIGHT_NODEJS_PATH"] = node_path
    if Path(node_path).name.lower() == "electron.exe":
        os.environ["ELECTRON_RUN_AS_NODE"] = "1"
    return node_path
