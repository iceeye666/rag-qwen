"""通义千问生成模型：qwen-plus，经 DashScope OpenAI 兼容接口调用。"""

from __future__ import annotations

from openai import OpenAI

from .config import Settings


class QwenChat:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = settings.chat_model
        self.client = OpenAI(
            api_key=settings.require_api_key(), base_url=settings.base_url
        )

    def chat(self, messages: list[dict], temperature: float | None = None) -> str:
        # 通义千问兼容 OpenAI 的 chat/completions 协议，因此直接复用 openai 客户端，
        # 只需把 base_url 指到 DashScope 的 compatible-mode 地址即可。
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=(
                self.settings.temperature if temperature is None else temperature
            ),
        )
        return (resp.choices[0].message.content or "").strip()
