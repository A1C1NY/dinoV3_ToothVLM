from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import database
from .services.detector import detector_service
from .services.ollama import ollama_service
from .settings import RESULTS_DIR, UPLOADS_DIR, ensure_runtime_directories, settings


ensure_runtime_directories()
app = FastAPI(title="Tooth VLM API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CreateConversationRequest(BaseModel):
    title: str = "新对话"


@app.on_event("startup")
def startup() -> None:
    ensure_runtime_directories()
    database.initialize()


app.mount("/media/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")
app.mount("/media/results", StaticFiles(directory=RESULTS_DIR), name="results")


def conversation_or_404(conversation_id: str) -> dict:
    conversation = database.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="找不到该对话。")
    return conversation


@app.get("/api/health")
async def health() -> dict:
    try:
        models = await asyncio.to_thread(ollama_service.list_models)
        return {"ollama": "connected", "models": models, "selected_model": settings.model}
    except Exception as error:
        return {"ollama": "unavailable", "detail": str(error), "models": []}


@app.get("/api/conversations")
def conversations() -> list[dict]:
    return database.list_conversations()


@app.post("/api/conversations", status_code=201)
def create_conversation(request: CreateConversationRequest) -> dict:
    return database.create_conversation(request.title)


@app.delete("/api/conversations/{conversation_id}", status_code=204)
def remove_conversation(conversation_id: str) -> None:
    if not database.delete_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="找不到该对话。")


@app.get("/api/conversations/{conversation_id}/messages")
def messages(conversation_id: str) -> list[dict]:
    conversation_or_404(conversation_id)
    return database.list_messages(conversation_id)


def save_upload(upload: UploadFile, conversation_id: str) -> tuple[Path, str]:
    if not upload.content_type or not upload.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail=f"{upload.filename or '文件'} 不是图片。")
    suffix = Path(upload.filename or "image.jpg").suffix.lower() or ".jpg"
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(status_code=415, detail="仅支持 JPG、PNG、WebP 图片。")
    relative = Path(conversation_id) / f"{uuid4()}{suffix}"
    destination = UPLOADS_DIR / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as target:
        shutil.copyfileobj(upload.file, target)
    return destination, f"/media/uploads/{relative.as_posix()}"


def build_llm_history(messages: list[dict]) -> list[dict]:
    history: list[dict] = []
    for message in messages[-settings.max_history_messages :]:
        content = message["content"]
        if message["report"]:
            report = message["report"]
            content = f"{content}\n\n[检测工具结果]\n{report['report']}"
        history.append({"role": message["role"], "content": content})
    return history


@app.post("/api/conversations/{conversation_id}/messages")
async def send_message(
    conversation_id: str,
    prompt: Annotated[str, Form()] = "",
    images: Annotated[list[UploadFile], File()] = [],
) -> dict:
    conversation_or_404(conversation_id)
    prompt = prompt.strip()
    if not prompt and not images:
        raise HTTPException(status_code=422, detail="请输入内容或选择至少一张图片。")

    image_urls: list[str] = []
    analyses: list[dict] = []
    for image in images:
        image_path, image_url = save_upload(image, conversation_id)
        image_urls.append(image_url)
        try:
            analysis = await asyncio.to_thread(detector_service.analyze, image_path)
            relative_result = analysis["annotated_image"].relative_to(RESULTS_DIR).as_posix()
            analysis["annotated_image_url"] = f"/media/results/{relative_result}"
            analysis.pop("annotated_image")
            analyses.append(analysis)
        except Exception as error:
            analyses.append({"error": f"图片检测失败：{error}"})

    report = None
    if analyses:
        report = {
            "report": "\n\n".join(item.get("report", item.get("error", "")) for item in analyses),
            "analyses": analyses,
        }
    user_content = prompt or "请分析我上传的口腔图片。"
    database.update_title_if_default(conversation_id, user_content)
    user_message = database.add_message(conversation_id, "user", user_content, image_urls, report)

    try:
        history = build_llm_history(database.list_messages(conversation_id))
        response, model = await asyncio.to_thread(ollama_service.respond, history)
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"Ollama 调用失败：{error}") from error

    assistant_message = database.add_message(conversation_id, "assistant", response)
    return {"user_message": user_message, "assistant_message": assistant_message, "model": model}
