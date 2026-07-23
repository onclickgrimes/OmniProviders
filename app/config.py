from __future__ import annotations

import os
from pathlib import Path


APP_NAME = "OmniProviders"
APP_VERSION = "0.1.0"
HOST = os.environ.get("OMNIPROVIDERS_HOST", "127.0.0.1")
PORT = int(os.environ.get("OMNIPROVIDERS_PORT", "7814"))
API_KEY = os.environ.get("OMNIPROVIDERS_API_KEY", "").strip()
DATA_DIR = Path(
    os.environ.get("OMNIPROVIDERS_DATA_DIR", Path(__file__).resolve().parents[1] / "data")
).resolve()
DATABASE_PATH = DATA_DIR / "omni-providers.sqlite3"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
MODEL_CACHE_TTL_SECONDS = max(
    30, int(os.environ.get("OMNIPROVIDERS_MODEL_CACHE_TTL_SECONDS", "300"))
)
ARTIFACT_TTL_SECONDS = max(
    300, int(os.environ.get("OMNIPROVIDERS_ARTIFACT_TTL_SECONDS", "86400"))
)


def ensure_runtime_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
