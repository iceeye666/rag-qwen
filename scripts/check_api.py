"""API 连通性自检：验证 Key 有效、embedding 与 chat 两个接口都通。

用法：python scripts/check_api.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import settings
from rag.embeddings import QwenEmbeddings
from rag.llm import QwenChat


def main() -> int:
    print(f"base_url   : {settings.base_url}")
    print(f"embed_model: {settings.embed_model}")
    print(f"chat_model : {settings.chat_model}\n")

    try:
        emb = QwenEmbeddings(settings)
        vec = emb.embed_query("连通性测试")
        print(f"[ok] embedding 调用成功，维度 = {len(vec)}")
    except Exception as e:  # noqa: BLE001
        print(f"[fail] embedding 调用失败：{type(e).__name__}: {e}")
        return 1

    try:
        chat = QwenChat(settings)
        reply = chat.chat([{"role": "user", "content": "只回复两个字：正常"}])
        print(f"[ok] chat 调用成功，回复 = {reply[:50]}")
    except Exception as e:  # noqa: BLE001
        print(f"[fail] chat 调用失败：{type(e).__name__}: {e}")
        return 1

    print("\n全部接口连通 ✅ 可以执行：python main.py ingest docs/财务管理制度样例.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
