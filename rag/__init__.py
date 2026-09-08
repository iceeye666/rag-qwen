"""RAG 文档智能问答系统（基于通义千问）。"""

from .chunker import Chunk, clean_text, sliding_window_chunks
from .config import Settings, settings
from .embeddings import QwenEmbeddings
from .loaders import Document, load_document
from .llm import QwenChat
from .pipeline import Answer, RAGPipeline, Source
from .vectorstore import ChromaStore, Hit

__all__ = [
    "Answer",
    "Chunk",
    "ChromaStore",
    "Document",
    "Hit",
    "QwenChat",
    "QwenEmbeddings",
    "RAGPipeline",
    "Settings",
    "clean_text",
    "load_document",
    "settings",
    "sliding_window_chunks",
    "Source",
]
