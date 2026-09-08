"""文本清洗 + 滑动窗口切分。

设计要点（面向中文）：
1. 先做噪声清洗——去掉页眉页脚式的孤立行、压缩多余空白；
2. 再按标点断句，保证窗口边界永远落在句子边界上，不在句子中间下刀；
3. 最后用滑动窗口（chunk_size=250, stride=100）合成块：
   窗口每次前进 stride 个字符，相邻块重叠约 chunk_size - stride 个字符，
   使跨窗口边界的上下文不会断裂。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 中文/英文句末标点 + 换行
_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;\n])")
_WS = re.compile(r"[ \t\u3000]+")
_MULTI_NL = re.compile(r"\n{2,}")


@dataclass
class Chunk:
    text: str
    chunk_id: str
    source: str
    start: int  # 在清洗后全文中的起始字符位置（近似）


def clean_text(raw: str) -> str:
    """清洗 PDF 抽取噪声：统一空白、丢弃页眉页脚行、合并空行。"""
    lines: list[str] = []
    for line in raw.splitlines():
        line = _WS.sub(" ", line).strip()
        if not line:
            continue
        # 丢弃形如 “12” / “— 3 —” / “第 3 页” 的页眉页脚
        if re.fullmatch(r"[\d\-—·.、第页\s]{1,12}", line):
            continue
        lines.append(line)
    return _MULTI_NL.sub("\n", "\n".join(lines)).strip()


def split_sentences(text: str, hard_limit: int = 600) -> list[str]:
    """按标点断句（保留标点）；超长句再按 hard_limit 硬切，避免单句撑爆窗口。"""
    parts: list[str] = []
    for s in _SENT_SPLIT.split(text):
        s = s.strip()
        if not s:
            continue
        while len(s) > hard_limit:
            parts.append(s[:hard_limit])
            s = s[hard_limit:]
        if s:
            parts.append(s)
    return parts


def sliding_window_chunks(
    text: str,
    source: str = "",
    chunk_size: int = 250,
    stride: int = 100,
    min_chunk_chars: int = 20,
) -> list[Chunk]:
    """句子对齐的滑动窗口切分。

    stride = 窗口每次前进的字符数（默认 100），
    因此相邻两块的字符重叠量约为 chunk_size - stride（默认 150）。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须为正整数")
    if not 0 < stride <= chunk_size:
        raise ValueError("stride 必须满足 0 < stride <= chunk_size")

    # 断句后，切分的基本单位从“字符”升级为“句子”。
    # 好处：窗口边界一定落在句末标点上，不会把一句话从中间劈开，语义更完整。
    sents = split_sentences(text)
    if not sents:
        return []

    lens = [len(s) for s in sents]
    n = len(sents)
    # 每句在拼接文本中的起始偏移，用于记录 chunk.start
    offsets = [0] * n
    for i in range(1, n):
        offsets[i] = offsets[i - 1] + lens[i - 1]

    chunks: list[Chunk] = []
    start = 0          # 当前窗口的起始句索引
    while start < n:
        # 从 start 起向后“贪心”累加句子，直到字数累计达到 chunk_size（或句子用完）
        acc = 0
        end = start
        while end < n and acc < chunk_size:
            acc += lens[end]
            end += 1

        body = "".join(sents[start:end])
        if len(body) >= min_chunk_chars or not chunks:
            chunks.append(
                Chunk(
                    text=body,
                    chunk_id=f"{source}#{len(chunks)}",
                    source=source,
                    start=offsets[start],
                )
            )

        if end >= n:
            break

        # 窗口前进：不是简单的 start += 1，而是按“字符数 stride”前进。
        # 因为句子长短不一，必须用字符累计来对齐步长，目标让下一个窗口起点
        # 比当前起点前进约 stride 个字符（即相邻块重叠约 chunk_size - stride 字）。
        advanced = 0
        new_start = start
        while new_start < end and advanced + lens[new_start] < stride:
            advanced += lens[new_start]
            new_start += 1
        # 边界情况：若首句本身长度就 >= stride，new_start 没动，强制前进一句以免死循环
        start = new_start + 1 if new_start == start else new_start

    return _dedup(chunks)


def _dedup(chunks: list[Chunk]) -> list[Chunk]:
    """去掉内容完全重复的块（PDF 重复抽取时常见）。"""
    seen: set[str] = set()
    out: list[Chunk] = []
    for c in chunks:
        key = re.sub(r"\s+", "", c.text)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out
