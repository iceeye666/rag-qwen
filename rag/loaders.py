"""文档加载：PDF 走 pdfminer，纯文本直接读取。

统一返回 Document(source=..., text=...)，让上层不用关心文件类型。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pdfminer.high_level import extract_text

SUPPORTED_SUFFIX = {".pdf", ".txt", ".md", ".markdown"}


@dataclass
class Document:
    source: str
    text: str


def load_pdf(path: Path) -> Document:
    """用 pdfminer 提取 PDF 全文。"""
    raw = extract_text(str(path)) or ""
    return Document(source=path.name, text=raw)


def load_text(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    return Document(source=path.name, text=raw)


def load_document(path: str | Path) -> Document:
    """按后缀自动选择加载器。"""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"文档不存在: {p}")
    if p.suffix.lower() not in SUPPORTED_SUFFIX:
        raise ValueError(
            f"暂不支持的文件类型: {p.suffix}，仅支持 {sorted(SUPPORTED_SUFFIX)}"
        )
    loader = load_pdf if p.suffix.lower() == ".pdf" else load_text
    return loader(p)
