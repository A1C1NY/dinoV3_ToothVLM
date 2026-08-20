from __future__ import annotations

from openai import OpenAI

from ..settings import settings


SYSTEM_PROMPT = """你是 Tooth VLM 口腔健康对话助手。请用中文回答，表达清晰、克制。
当上下文中存在“检测工具结果”时，结合它解释结果，但必须说明这只是辅助筛查，不替代牙科医生诊断。
不要捏造未提供的检测发现。用户可以继续追问同一张图或此前结果。"""


class OllamaService:
    def __init__(self) -> None:
        self.client = OpenAI(base_url=settings.ollama_url, api_key="ollama")

    def list_models(self) -> list[str]:
        return [model.id for model in self.client.models.list().data]

    def selected_model(self) -> str:
        models = self.list_models()
        if settings.model:
            if settings.model not in models:
                raise RuntimeError(f"配置的模型 {settings.model} 未安装到 Ollama。")
            return settings.model
        if not models:
            raise RuntimeError("Ollama 中没有本地模型。请先运行 ollama pull qwen3:8b。")
        return models[0]

    def respond(self, history: list[dict]) -> tuple[str, str]:
        model = self.selected_model()
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history]
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.4,
        )
        content = response.choices[0].message.content
        return (content or "模型没有返回文本。", model)


ollama_service = OllamaService()
