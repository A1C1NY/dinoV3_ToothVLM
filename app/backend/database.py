from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .settings import DATABASE_PATH, ensure_runtime_directories


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@contextmanager
def connection():
    ensure_runtime_directories()
    database = sqlite3.connect(DATABASE_PATH)
    database.row_factory = sqlite3.Row
    try:
        yield database
        database.commit()
    finally:
        database.close()


def initialize() -> None:
    with connection() as database:
        database.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                images_json TEXT NOT NULL DEFAULT '[]',
                report_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id)
            );
            CREATE INDEX IF NOT EXISTS idx_messages_conversation_created
                ON messages(conversation_id, created_at);
            """
        )


def create_conversation(title: str = "新对话") -> dict:
    conversation_id = str(uuid4())
    now = utc_now()
    with connection() as database:
        database.execute(
            "INSERT INTO conversations(id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (conversation_id, title, now, now),
        )
    return get_conversation(conversation_id)


def get_conversation(conversation_id: str) -> dict | None:
    with connection() as database:
        row = database.execute(
            "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
    return dict(row) if row else None


def list_conversations() -> list[dict]:
    with connection() as database:
        rows = database.execute(
            "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def delete_conversation(conversation_id: str) -> bool:
    with connection() as database:
        database.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        result = database.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    return result.rowcount > 0


def update_title_if_default(conversation_id: str, prompt: str) -> None:
    title = " ".join(prompt.split())[:36] or "图片诊断"
    with connection() as database:
        database.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ? AND title = '新对话'",
            (title, utc_now(), conversation_id),
        )


def add_message(
    conversation_id: str,
    role: str,
    content: str,
    images: list[str] | None = None,
    report: dict | None = None,
) -> dict:
    message = {
        "id": str(uuid4()),
        "conversation_id": conversation_id,
        "role": role,
        "content": content,
        "images": images or [],
        "report": report,
        "created_at": utc_now(),
    }
    with connection() as database:
        database.execute(
            """
            INSERT INTO messages(id, conversation_id, role, content, images_json, report_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message["id"], conversation_id, role, content,
                json.dumps(message["images"], ensure_ascii=False),
                json.dumps(report, ensure_ascii=False) if report else None,
                message["created_at"],
            ),
        )
        database.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (message["created_at"], conversation_id),
        )
    return message


def list_messages(conversation_id: str) -> list[dict]:
    with connection() as database:
        rows = database.execute(
            """
            SELECT id, conversation_id, role, content, images_json, report_json, created_at
            FROM messages WHERE conversation_id = ? ORDER BY created_at ASC
            """,
            (conversation_id,),
        ).fetchall()
    messages = []
    for row in rows:
        message = dict(row)
        message["images"] = json.loads(message.pop("images_json"))

        report_json = message.pop("report_json")
        message["report"] = json.loads(report_json) if report_json else None
        messages.append(message)
    return messages
