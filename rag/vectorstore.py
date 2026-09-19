"""向量库封装：Chroma（默认）与 Milvus（Lite / 远程服务）两种后端。

两个后端对外暴露完全一致的接口（鸭子类型，见 ChromaStore）：
    add(ids, texts, vectors, metadatas) -> int
    query(vector, top_k)                 -> list[Hit]
    count()                              -> int
    clear()                              -> None
    describe()                           -> dict[str, str]

上层 RAGPipeline 只依赖这四个方法，因此可通过 `create_store(settings)`
按 VECTOR_BACKEND 配置在两种实现之间自由切换。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

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
    """Chroma 本地持久化封装（cosine 空间）。"""

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

    def describe(self) -> dict[str, str]:
        return {
            "向量后端": "chroma",
            "实现库": "chromadb",
            "数据目录": str(self.settings.persist_dir),
            "集合名称": self.settings.collection_name,
        }


class MilvusStore:
    """Milvus 向量库封装，兼容本地 Milvus Lite 文件与远程服务。

    连接目标由 ``Settings.resolve_milvus_uri()`` 决定：
    MILVUS_URI（文件路径或 http(s) 地址）→ MILVUS_HOST:MILVUS_PORT → 本地 Lite 文件。

    说明：``pymilvus`` 在实例化时才导入（延迟导入），这样未安装 pymilvus 的
    环境仍可正常使用 Chroma 后端与离线测试。
    """

    # 单批 upsert 条数上限：避免超长文档一次性提交导致请求体过大
    UPSERT_BATCH_SIZE = 500

    def __init__(self, settings: Settings):
        self.settings = settings
        self.collection_name = settings.milvus_collection
        self.dim = settings.milvus_dim

        # pymilvus 内部的 legacy ORM 会读取同名环境变量 MILVUS_URI，并要求它
        # 是 http(s):// 形式；若我们把它设为本地文件路径，import pymilvus 会直接
        # 抛 ConnectionConfigException。这里在导入前临时摘掉该变量，连接信息由
        # Settings 显式传给 MilvusClient(uri=...)，导入完成后原样恢复。
        _saved_uri = os.environ.pop("MILVUS_URI", None)
        try:
            from pymilvus import DataType, MilvusClient
        except ImportError as e:  # pragma: no cover - 依赖缺失时的友好提示
            raise RuntimeError(
                "未安装 pymilvus，无法使用 Milvus 后端。请执行：\n"
                "  pip install pymilvus\n"
                "本地 Lite 模式还需安装可选依赖：\n"
                '  pip install "pymilvus[milvus_lite]"\n'
                "（Milvus Lite 支持 macOS/Linux；Windows 请连接远程 Milvus 服务）"
            ) from e
        finally:
            if _saved_uri is not None:
                os.environ["MILVUS_URI"] = _saved_uri

        self._DataType = DataType
        self.uri = settings.resolve_milvus_uri()
        self._is_local = "://" not in self.uri
        self.client = MilvusClient(
            uri=self.uri, token=settings.milvus_token or None
        )

    # ---------- 内部工具 ----------
    def _exists(self) -> bool:
        return bool(self.client.has_collection(self.collection_name))

    def _create_collection(self) -> None:
        """按配置维度创建集合：COSINE 度量 + FLAT 索引（Lite 与自建服务通吃）。"""
        DataType = self._DataType
        schema = self.client.create_schema(
            auto_id=False, enable_dynamic_field=False
        )
        schema.add_field(
            field_name="id", datatype=DataType.VARCHAR,
            is_primary=True, max_length=512,
        )
        schema.add_field(
            field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.dim
        )
        schema.add_field(
            field_name="text", datatype=DataType.VARCHAR, max_length=65535
        )
        schema.add_field(
            field_name="chunk_id", datatype=DataType.VARCHAR, max_length=512
        )
        schema.add_field(
            field_name="source", datatype=DataType.VARCHAR, max_length=1024
        )
        schema.add_field(field_name="start", datatype=DataType.INT64)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector", index_type="FLAT", metric_type="COSINE"
        )
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        """检索前确保集合已加载。

        Milvus（含 Lite 落盘模式）的集合在每个新进程中默认处于 released 状态，
        未 load 就 search 会报 code=101。load_collection 幂等且开销极小，因此
        每次检索前直接调用，不做状态缓存，避免缓存与真实状态不一致。
        集合尚不存在时不操作（首次 add 前的场景）。
        """
        if not self._exists():
            return
        try:
            self.client.load_collection(self.collection_name)
        except Exception as e:  # noqa: BLE001 - 不阻断，交由后续 search 报错
            print(
                f"[警告] 加载 Milvus 集合失败（{self.collection_name}）："
                f"{type(e).__name__}: {e}"
            )

    def _ensure_collection(self, vector_dim: int | None = None) -> None:
        """集合不存在则创建；存在时校验向量维度是否与配置一致。"""
        if vector_dim is not None and vector_dim != self.dim:
            raise ValueError(
                f"向量维度不匹配：MILVUS_DIM={self.dim}，"
                f"但 embedding 实际输出 {vector_dim} 维。"
                "请把 MILVUS_DIM 调整为与 EMBED_MODEL 一致"
                "（text-embedding-v2 → 1536，text-embedding-v3 → 1024），"
                "或更换 embedding 模型。"
            )
        if not self._exists():
            self._create_collection()

    # ---------- 对外接口（与 ChromaStore 对齐） ----------
    def add(self, ids: list[str], texts: list[str], vectors: list[list[float]],
            metadatas: list[dict]) -> int:
        if not ids:
            return 0
        self._ensure_collection(len(vectors[0]) if vectors else None)

        rows: list[dict[str, Any]] = []
        for i, text, vec, meta in zip(ids, texts, vectors, metadatas):
            rows.append(
                {
                    "id": i,
                    "vector": list(vec),
                    "text": text,
                    "chunk_id": str(meta.get("chunk_id", i)),
                    "source": str(meta.get("source", "")),
                    "start": int(meta.get("start", 0)),
                }
            )

        for start in range(0, len(rows), self.UPSERT_BATCH_SIZE):
            self.client.upsert(
                collection_name=self.collection_name,
                data=rows[start : start + self.UPSERT_BATCH_SIZE],
            )
        return len(rows)

    def query(self, vector: list[float], top_k: int) -> list[Hit]:
        if not self._exists():
            return []
        count = self.count()
        if count == 0:
            return []
        # 新进程中集合默认 released，检索前必须先 load（Milvus Lite 同样如此）
        self._ensure_loaded()
        # limit 不能超过集合内实际条数，与 Chroma 保持一致，避免越界报错。
        res = self.client.search(
            collection_name=self.collection_name,
            data=[list(vector)],
            limit=min(top_k, count),
            output_fields=["text", "chunk_id", "source"],
            search_params={"metric_type": "COSINE"},
        )
        hits: list[Hit] = []
        for item in res[0] if res else []:
            entity = item.get("entity") or {}
            hits.append(
                Hit(
                    chunk_id=str(entity.get("chunk_id", "")),
                    text=entity.get("text", "") or "",
                    source=str(entity.get("source", "")),
                    # COSINE 度量下 distance 即余弦相似度，越大越相关
                    score=round(float(item.get("distance", 0.0)), 4),
                )
            )
        return hits

    def count(self) -> int:
        if not self._exists():
            return 0
        stats = self.client.get_collection_stats(self.collection_name)
        return int(stats.get("row_count", 0))

    def clear(self) -> None:
        # Milvus 支持直接删集合；下次 add 时按配置维度自动重建。
        if self._exists():
            self.client.drop_collection(self.collection_name)

    def describe(self) -> dict[str, str]:
        return {
            "向量后端": "milvus",
            "部署形态": "本地 Milvus Lite" if self._is_local else "远程 Milvus 服务",
            "连接目标": self.uri,
            "集合名称": self.collection_name,
            "向量维度": str(self.dim),
        }


def create_store(settings: Settings):
    """按 settings.vector_backend 创建对应向量库实现。"""
    backend = (settings.vector_backend or "chroma").strip().lower()
    if backend == "chroma":
        return ChromaStore(settings)
    if backend == "milvus":
        return MilvusStore(settings)
    raise ValueError(
        f"未知的 VECTOR_BACKEND={backend!r}，可选值：chroma、milvus"
    )
