# Tooth VLM Web App

This directory is intentionally separate from model training and evaluation code.

## Start from Conda

The launcher never activates Conda or chooses a Python interpreter. Open PowerShell and activate the training/inference environment yourself before every run:

```powershell
conda activate dino_VLM
```

`tooth_vlm` then uses the `python` from that active environment, so FastAPI and the detector share the same PyTorch installation.

## One-time setup

1. Install [Ollama](https://ollama.com/) and download a tool-capable model, for example:

   ```powershell
   ollama pull qwen3:8b
   ```

2. After manually activating `dino_VLM`, install the backend packages into that same environment:

   ```powershell
   python -m pip install -r app/backend/requirements.txt
   ```

3. Install the frontend packages:

   ```powershell
   npm.cmd --prefix app/frontend install
   ```

4. Still in that manually activated terminal, install the command into the active Conda environment once:

   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\app\scripts\install_active_conda_command.ps1
   ```

   This does not activate Conda or alter the global user PATH. It only creates the `tooth_vlm` command inside the current environment.

Then, whenever you have manually activated `dino_VLM`, run:

```powershell
tooth_vlm
```

The command validates dependencies in the active Conda environment, starts FastAPI and the Vue development server, and opens the browser. Press `Ctrl+C` in that terminal to stop both servers.

## Configuration

Set `TOOTH_VLM_MODEL` before starting to select a specific installed Ollama model. If omitted, the app uses the first available local model.

```powershell
$env:TOOTH_VLM_MODEL = "qwen3:8b"
tooth_vlm
```

Uploaded images, annotated results, and conversation history are stored under `app/runtime/`, which is excluded from source control.
