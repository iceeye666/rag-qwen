"""全局配置：所有可调参数集中在这里，支持 .env 覆盖。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 加载项目根目录的 .env（若存在）
load_dotenv(PROJECT_ROOT / ".env")


def _as_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _as_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    """RAG 系统的运行参数（集中管理，便于 .env 覆盖与测试注入）。

    设计为 frozen 的好处：配置在进程内不可变，避免运行期被意外篡改；
    同时天然可哈希，必要时可作缓存键。
    """

    # ---- DashScope / 通义千问 ----
    api_key: str = os.getenv("DASHSCOPE_API_KEY", "")
    base_url: str = os.getenv(
        "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    embed_model: str = os.getenv("EMBED_MODEL", "text-embedding-v2")
    chat_model: str = os.getenv("CHAT_MODEL", "qwen-plus")
    embed_batch_size: int = _as_int("EMBED_BATCH_SIZE", 10)
    temperature: float = _as_float("TEMPERATURE", 0.1)

    # ---- 切分策略 ----
    chunk_size: int = _as_int("CHUNK_SIZE", 250)
    chunk_stride: int = _as_int("CHUNK_STRIDE", 100)
    min_chunk_chars: int = _as_int("MIN_CHUNK_CHARS", 20)

    # ---- 检索 ----
    top_k: int = _as_int("TOP_K", 4)
    # 余弦相似度阈值：低于此值判定为“文档中无答案”，直接拒答
    score_threshold: float = _as_float("SCORE_THRESHOLD", 0.30)

    # ---- 存储 ----
    # 向量库后端：chroma（默认）| milvus
    vector_backend: str = os.getenv("VECTOR_BACKEND", "chroma").strip().lower()

    # Chroma 配置
    persist_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("CHROMA_DIR", PROJECT_ROOT / "storage" / "chroma")
        )
    )
    collection_name: str = os.getenv("CHROMA_COLLECTION", "rag_qwen_docs")

    # Milvus 配置（VECTOR_BACKEND=milvus 时生效）
    # 连接目标优先级：milvus_uri > milvus_host:milvus_port > 本地 Lite 文件
    milvus_uri: str = os.getenv("MILVUS_URI", "").strip()
    milvus_host: str = os.getenv("MILVUS_HOST", "").strip()
    milvus_port: int = _as_int("MILVUS_PORT", 19530)
    milvus_token: str = os.getenv("MILVUS_TOKEN", "").strip()  # Zilliz Cloud 等需要
    milvus_collection: str = os.getenv("MILVUS_COLLECTION", "rag_qwen_docs")
    # 向量维度须与 embed_model 输出一致：text-embedding-v2 → 1536
    milvus_dim: int = _as_int("MILVUS_DIM", 1536)

    def resolve_milvus_uri(self) -> str:
        """解析 Milvus 连接目标。

        MILVUS_URI 优先（可为本地文件路径或 http(s) 地址）；
        其次由 MILVUS_HOST/MILVUS_PORT 拼装；都未配置则回退本地 Lite 文件。

        本地文件路径会统一转成绝对路径并确保父目录存在，因为 Milvus Lite
        对相对路径的解析依赖当前工作目录，容易在换目录执行时连到别的库。
        """
        if self.milvus_uri:
            if "://" in self.milvus_uri:
                return self.milvus_uri
            path = Path(self.milvus_uri).expanduser()
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            path.parent.mkdir(parents=True, exist_ok=True)
            return str(path)
        if self.milvus_host:
            return f"http://{self.milvus_host}:{self.milvus_port}"
        path = PROJECT_ROOT / "storage" / "milvus.db"
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)

    def require_api_key(self) -> str:
        if not self.api_key:
            raise RuntimeError(
                "未检测到 DASHSCOPE_API_KEY。请在项目根目录创建 .env 文件：\n"
                '  echo "DASHSCOPE_API_KEY=sk-xxxxxx" > .env\n'
                "或在执行命令前导出环境变量。"
            )
        return self.api_key


settings = Settings()
