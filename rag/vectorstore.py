"""Chroma 持久化向量库封装。"""

from __future__ import annotations

from dataclasses import dataclass

import chromadb
from chromadb.config import Settings as ChromaSettings

from .config import Settings


@dataclass
class Hit:
    chunk_id: str
    text: str
    source: str
    score: float  # 余弦相似度，越大越相关


class ChromaStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        settings.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(settings.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, ids: list[str], texts: list[str], vectors: list[list[float]],
            metadatas: list[dict]) -> int:
        if not ids:
            return 0
        self.collection.upsert(
            ids=ids, documents=texts, embeddings=vectors, metadatas=metadatas
        )
        return len(ids)

    def query(self, vector: list[float], top_k: int) -> list[Hit]:
        count = self.collection.count()
        if count == 0:
            return []
        # n_results 不能超过集合内实际条数，否则 Chroma 会报错，故取 min。
        res = self.collection.query(
            query_embeddings=[vector],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        hits: list[Hit] = []
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            hits.append(
                Hit(
                    chunk_id=meta.get("chunk_id", ""),
                    text=doc,
                    source=meta.get("source", ""),
                    # cosine 距离 = 1 - 余弦相似度
                    score=round(1.0 - float(dist), 4),
                )
            )
        return hits

    def count(self) -> int:
        return self.collection.count()

    def clear(self) -> None:
        # Chroma 没有“清空单集合”的 API，这里直接删除整个集合再重建。
        # 重建时必须重新指定 hnsw:space=cosine，否则退回默认 L2 距离，
        # 与写入时的相似度口径不一致，检索结果会失真。
        self.client.delete_collection(self.settings.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
