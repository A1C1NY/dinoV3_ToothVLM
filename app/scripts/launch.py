from __future__ import annotations

import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT / "app" / "backend"
FRONTEND_DIR = ROOT / "app" / "frontend"
API_URL = "http://127.0.0.1:8000/api/health"
UI_URL = "http://127.0.0.1:5173"
NPM_COMMAND = "npm.cmd" if os.name == "nt" else "npm"


def missing_dependencies() -> list[str]:
    missing = []
    for module, package in {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn[standard]",
        "multipart": "python-multipart",
        "openai": "openai",
        "ultralytics": "ultralytics",
    }.items():
        if importlib.util.find_spec(module) is None:
            missing.append(f"Python 包 {package}")
    if not shutil.which("node"):
        missing.append("Node.js")
    if not shutil.which(NPM_COMMAND):
        missing.append("npm")
    if not (FRONTEND_DIR / "node_modules").is_dir():
        missing.append("前端 node_modules")
    return missing


def ollama_ready() -> tuple[bool, str]:
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not payload.get("models"):
            return False, "Ollama 已启动，但还没有下载任何模型。"
        return True, ""
    except (urllib.error.URLError, TimeoutError) as error:
        return False, f"无法连接 Ollama 服务（{error}）。"


def wait_for(url: str, seconds: int = 20) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1).close()
            return True
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(0.25)
    return False


def main() -> int:
    conda_prefix = os.getenv("CONDA_PREFIX")
    if not conda_prefix:
        print("无法启动 Tooth VLM：请先在当前终端手动运行 conda activate dino_VLM。")
        return 1

    if Path(sys.prefix).resolve() != Path(conda_prefix).resolve():
        print("无法启动 Tooth VLM：启动器使用的 Python 与当前激活的 Conda 环境不一致。")
        print(f"  当前 Conda 环境: {conda_prefix}")
        print(f"  启动器 Python: {sys.executable}")
        print("请在已手动激活 dino_VLM 的终端执行：")
        print("  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass")
        print("  .\\app\\scripts\\install_active_conda_command.ps1")
        return 1

    missing = missing_dependencies()
    if missing:
        print("无法启动 Tooth VLM。当前激活的 Conda 环境仍缺少以下依赖：")
        for item in missing:
            print(f"  - {item}")
        print("\n请执行：")
        print("  python -m pip install -r app/backend/requirements.txt")
        print("  npm.cmd --prefix app/frontend install")
        return 1

    ready, detail = ollama_ready()
    if not ready:
        print(f"无法启动 Tooth VLM：{detail}")
        print("请启动 Ollama，并至少执行一次：ollama pull qwen3.8:27b")
        return 1

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.backend.main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=ROOT, env=environment,
    )
    frontend = None
    try:
        if not wait_for(API_URL):
            print("FastAPI 启动失败，请检查上方错误信息。")
            return 1
        frontend = subprocess.Popen(
            [NPM_COMMAND, "run", "dev", "--", "--host", "127.0.0.1"], cwd=FRONTEND_DIR, env=environment,
        )
        if not wait_for(UI_URL):
            print("Vue 前端启动失败，请检查上方错误信息。")
            return 1
        print(f"Tooth VLM 已启动：{UI_URL}")
        print("按 Ctrl+C 可关闭前后端服务。")
        webbrowser.open(UI_URL)
        while True:
            if backend.poll() is not None or frontend.poll() is not None:
                print("服务意外停止，请检查上方错误信息。")
                return 1
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n正在关闭 Tooth VLM...")
        return 0
    finally:
        for process in (frontend, backend):
            if process and process.poll() is None:
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
