from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
RUNTIME_DIR = APP_ROOT / "runtime"
UPLOADS_DIR = RUNTIME_DIR / "uploads"
RESULTS_DIR = RUNTIME_DIR / "results"
DATABASE_PATH = RUNTIME_DIR / "conversations.sqlite3"


@dataclass(frozen=True)
class Settings:
    ollama_url: str = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/v1/")
    model: str | None = os.getenv("TOOTH_VLM_MODEL") or "qwen3.8:27b"
    max_history_messages: int = int(os.getenv("TOOTH_VLM_MAX_HISTORY", "30"))


settings = Settings()


def ensure_runtime_directories() -> None:
    for directory in (RUNTIME_DIR, UPLOADS_DIR, RESULTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
