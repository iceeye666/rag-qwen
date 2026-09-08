"""RAG 主流程：文档解析 → 向量化 → 检索 → 生成。

所有外部依赖（embedding / 向量库 / LLM）都支持注入，便于离线测试与替换。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .chunker import clean_text, sliding_window_chunks
from .config import Settings, settings as default_settings
from .loaders import load_document
from .prompts import REFUSAL, build_messages
from .vectorstore import Hit


@dataclass
class Source:
    chunk_id: str
    source: str
    score: float
    text: str


@dataclass
class Answer:
    question: str
    answer: str
    sources: list[Source] = field(default_factory=list)
    refused: bool = False
    top_score: float = 0.0


class RAGPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        embedder=None,
        store=None,
        chat=None,
    ):
        self.settings = settings or default_settings
        self.embedder = embedder
        self.store = store
        self.chat = chat
        if embedder is None or store is None or chat is None:
            from .embeddings import QwenEmbeddings
            from .llm import QwenChat
            from .vectorstore import ChromaStore

            self.embedder = embedder or QwenEmbeddings(self.settings)
            self.store = store or ChromaStore(self.settings)
            self.chat = chat or QwenChat(self.settings)

    # ---------- 写入 ----------
    def ingest(self, path: str) -> int:
        # ① 解析：PDF/TXT/MD → 纯文本
        doc = load_document(path)
        # ② 清洗：去页眉页脚、压缩多余空白
        text = clean_text(doc.text)
        if not text:
            raise ValueError(f"未能从 {doc.source} 提取到有效文本")

        # ③ 切分：句子对齐的滑动窗口，得到若干 Chunk
        chunks = sliding_window_chunks(
            text,
            source=doc.source,
            chunk_size=self.settings.chunk_size,
            stride=self.settings.chunk_stride,
            min_chunk_chars=self.settings.min_chunk_chars,
        )
        if not chunks:
            raise ValueError(f"切分结果为空：{doc.source}")

        # ④ 向量化：每个 chunk 的文本 → 浮点向量
        vectors = self.embedder.embed_documents([c.text for c in chunks])
        # ⑤ 入库：把文本、向量、元数据（chunk_id/source/start）一并写入 Chroma。
        # 元数据在检索时回带，用于向上层展示“答案来自哪个文件的第几块”。
        self.store.add(
            ids=[c.chunk_id for c in chunks],
            texts=[c.text for c in chunks],
            vectors=vectors,
            metadatas=[
                {"chunk_id": c.chunk_id, "source": c.source, "start": c.start}
                for c in chunks
            ],
        )
        return len(chunks)

    # ---------- 检索 ----------
    def retrieve(self, question: str, top_k: int | None = None) -> list[Hit]:
        # 把问题向量化，再去向量库里按余弦相似度取最相近的 top_k 个 chunk
        vector = self.embedder.embed_query(question)
        return self.store.query(vector, top_k or self.settings.top_k)

    # ---------- 问答 ----------
    def ask(self, question: str, top_k: int | None = None) -> Answer:
        question = question.strip()
        if not question:
            raise ValueError("问题不能为空")

        # ① 检索：拿到和问题最相关的若干 chunk
        hits = self.retrieve(question, top_k)
        # 把内部 Hit 转成对外暴露的 Source（多一层封装，方便上层使用）
        sources = [
            Source(h.chunk_id, h.source, h.score, h.text) for h in hits
        ]

        # ② 硬门槛（抗幻觉第一道闸）：连最相关块的相似度都低于阈值，
        #    说明文档里大概率没有答案，直接返回固定拒答话术，根本不调用 LLM。
        if not hits or hits[0].score < self.settings.score_threshold:
            return Answer(
                question=question,
                answer=REFUSAL,
                sources=sources,
                refused=True,
                top_score=hits[0].score if hits else 0.0,
            )

        # ③ 构造 grounded Prompt：把 top 命中片段编号拼成上下文，连同问题交给 LLM
        snippets = [(i + 1, h.text) for i, h in enumerate(hits)]
        messages = build_messages(question, snippets)
        text = self.chat.chat(messages)

        # ④ 判定是否拒答：模型若返回了固定拒答话术（忽略空格/全角空格差异），
        #    标记为 refused。这样上层既能拿到文字，也有结构化标志位做分支处理。
        norm = text.replace(" ", "").replace("\u3000", "")
        refused = REFUSAL.replace(" ", "") in norm
        return Answer(
            question=question,
            answer=text,
            sources=sources,
            refused=refused,
            top_score=hits[0].score,
        )
