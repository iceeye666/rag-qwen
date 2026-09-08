"""通义千问 Embedding：text-embedding-v2，经 DashScope OpenAI 兼容接口调用。"""

from __future__ import annotations

from openai import OpenAI

from .config import Settings


class QwenEmbeddings:
    """把文本转成向量。DashScope 有单次条数上限，这里自动分批。"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = settings.embed_model
        self.client = OpenAI(
            api_key=settings.require_api_key(), base_url=settings.base_url
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量向量化，保持与输入顺序一致。"""
        if not texts:
            return []

        # 记录非空白项位置，空白项用零向量占位（避免接口报错）
        valid_idx = [i for i, t in enumerate(texts) if t and t.strip()]
        vectors: list[list[float]] = [[] for _ in texts]

        # 分批调用：DashScope 对单次 embedding 的条数有上限（EMBED_BATCH_SIZE）。
        # 每批取一段 valid_idx，调用后按“原始下标”回填，保证 vectors 顺序与输入一致。
        batch = self.settings.embed_batch_size
        for start in range(0, len(valid_idx), batch):
            idx_slice = valid_idx[start : start + batch]
            batch_texts = [texts[i].strip() for i in idx_slice]
            resp = self.client.embeddings.create(model=self.model, input=batch_texts)
            # resp.data 的顺序与输入 batch_texts 一一对应，回填到对应位置
            for i, item in enumerate(resp.data):
                vectors[idx_slice[i]] = list(item.embedding)

        # 确定维度（取第一个有效向量），空白/空文本项统一补零向量。
        # 补零而非抛错：避免单个空块导致整批入库失败。
        dim = len(next((v for v in vectors if v), [])) or 1536
        return [v if v else [0.0] * dim for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
