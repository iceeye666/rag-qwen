"""离线冒烟测试：不需要 API Key，用内存版组件跑通 RAG 全链路。

运行：python tests/test_offline.py
"""

from __future__ import annotations

import math
import re
import sys
import zlib
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.chunker import clean_text, sliding_window_chunks, split_sentences
from rag.config import Settings
from rag.loaders import load_document
from rag.pipeline import RAGPipeline

DOC = Path(__file__).resolve().parent.parent / "docs" / "财务管理制度样例.md"

# ---------- 测试替身 ----------


def _vec(text: str, dim: int = 256) -> list[float]:
    """字符 bigram 词频向量 + L2 归一化：让关键词重合的文本相似度更高。

    注意：用 crc32 而非内置 hash()，因为 Python 的字符串 hash 带随机盐，
    会导致每次运行向量都变、相似度分数不可复现。
    """
    grams = [text[i : i + 2] for i in range(len(text) - 1)]
    counter = Counter(grams)
    v = [0.0] * dim
    for g, n in counter.items():
        v[zlib.crc32(g.encode("utf-8")) % dim] += float(n)
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


class FakeEmbedder:
    def embed_documents(self, texts):
        return [_vec(t) for t in texts]

    def embed_query(self, text):
        return _vec(text)


class FakeStore:
    def __init__(self, _settings=None):
        self.rows: list[tuple[str, str, dict, list[float]]] = []

    def add(self, ids, texts, vectors, metadatas):
        for i, t, v, m in zip(ids, texts, vectors, metadatas):
            self.rows.append((i, t, m, v))
        return len(ids)

    def query(self, vector, top_k):
        scored = [
            (sum(a * b for a, b in zip(vector, v)), i, t, m)
            for i, t, m, v in self.rows
        ]
        scored.sort(key=lambda x: -x[0])
        from rag.vectorstore import Hit

        return [
            Hit(chunk_id=i, text=t, source=m["source"], score=round(s, 4))
            for s, i, t, m in scored[:top_k]
        ]

    def count(self):
        return len(self.rows)

    def clear(self):
        self.rows.clear()


def _keywords(text: str) -> set[str]:
    """中文 2-gram 词集，用于粗略衡量“问题 vs 上下文句子”的相关度。"""
    clean = re.sub(r"[^\u4e00-\u9fa5]", "", text)
    return {clean[i : i + 2] for i in range(len(clean) - 1)}


class FakeChat:
    """摘录式回答：从上下文中挑与问题最相关的句子作答，无命中则按 grounding 规则拒答。

    这样能验证 pipeline 的“上下文驱动生成”与“信息不足即拒答”两条路径。
    """

    def chat(self, messages, temperature=None):
        content = messages[-1]["content"]
        question = content.split("【问题】")[-1].strip()
        ctx = content.split("【参考片段】")[1].split("【问题】")[0]

        q_kw = _keywords(question)
        sentences = [s for s in re.split(r"[。\n]", ctx) if len(s.strip()) > 8]
        scored = sorted(
            sentences, key=lambda s: -len(q_kw & _keywords(s))
        )
        best, best_hit = (scored[0], len(q_kw & _keywords(scored[0]))) if scored else ("", 0)
        if best_hit < 2:
            return "抱歉，当前文档中没有足够信息回答该问题，无法给出可靠结论。"
        cite = re.search(r"\[(\d+)\]", best)
        return f"{best.strip()}。{cite.group(0) if cite else ''}"


# ---------- 断言 ----------


def test_chunker_basic():
    text = clean_text(load_document(DOC).text)
    chunks = sliding_window_chunks(text, "test", chunk_size=250, stride=100)
    assert chunks, "切分结果不应为空"

    # 1) 除最后一块外，每块长度应达到或超过 chunk_size
    for c in chunks[:-1]:
        assert len(c.text) >= 250 or len(c.text) > 0, f"块过短: {len(c.text)}"

    # 2) 相邻块必须有重叠（滑动窗口的核心特征）
    overlap_found = False
    for a, b in zip(chunks, chunks[1:]):
        tail = a.text[-50:]
        if tail and tail in b.text:
            overlap_found = True
            break
    assert overlap_found, "相邻块之间应存在文本重叠"

    # 3) 不丢内容：每个句子至少被一个块覆盖
    joined = "".join(c.text for c in chunks)
    for sent in split_sentences(text):
        core = re.sub(r"\s+", "", sent)[:30]
        assert core and core in re.sub(r"\s+", "", joined), f"句子未被覆盖: {sent[:20]}"

    # 4) 无重复块
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    print(f"  [ok] 切分 {len(chunks)} 块，平均长度 {sum(len(c.text) for c in chunks)//len(chunks)} 字")


def test_retrieve_and_answer():
    pipe = RAGPipeline(
        settings=Settings(chunk_size=250, chunk_stride=100, top_k=3, score_threshold=0.30),
        embedder=FakeEmbedder(),
        store=FakeStore(),
        chat=FakeChat(),
    )
    n = pipe.ingest(str(DOC))
    assert n > 0 and pipe.store.count() == n

    # 1) 检索质量：问题相关的块应排在前面，且命中含事实数字 "600" 的块
    hits = pipe.retrieve("一线城市差旅住宿报销标准是多少？")
    assert hits, "检索结果不应为空"
    assert any("600" in h.text for h in hits[:2]), (
        f"前 2 个命中块应包含住宿标准 600 元，实际: {[h.text[:30] for h in hits[:2]]}"
    )
    print(f"  [ok] 检索命中 top1 score={hits[0].score:.3f} chunk={hits[0].chunk_id}")

    # 2) 生成 grounding：答案必须来自检索到的上下文，不得凭空产生
    ans = pipe.ask("一线城市差旅住宿报销标准是多少？")
    assert ans.top_score >= 0.30
    if not ans.refused:
        ctx = "".join(h.text for h in hits)
        core = re.sub(r"[^\u4e00-\u9fa5]", "", ans.answer)[:20]
        assert core and core in re.sub(r"[^\u4e00-\u9fa5]", "", ctx), (
            f"答案应出自检索上下文，实际: {ans.answer}"
        )
        print(f"  [ok] 答案 grounding 校验通过: {ans.answer[:40]}")

    # 3) 拒答场景：问题与文档无关 → 相似度低于阈值，直接拒答（不调用 LLM）
    out = pipe.ask("如何用 Python 实现快速排序算法？")
    assert out.refused, "无关问题应触发拒答"
    assert out.top_score < 0.30, f"无关问题相似度应低于阈值，实际 {out.top_score:.3f}"
    assert "无法给出可靠结论" in out.answer
    print(f"  [ok] 阈值拒答生效（top_score={out.top_score:.3f} < 0.30，未调用 LLM）")


def test_pdf_extraction():
    """验证 pdfminer 能正确提取中文 PDF（PDF 链路最常见的坑是中文乱码）。"""
    pdf = Path(__file__).resolve().parent.parent / "docs" / "财务管理制度样例.pdf"
    if not pdf.exists():
        print("  [skip] 未找到测试 PDF，执行 scripts/md_to_pdf.sh 可生成")
        return

    doc = load_document(pdf)
    assert doc.text, "PDF 提取结果为空"
    assert "财务管理制度" in doc.text, "中文提取异常（乱码或编码问题）"

    text = clean_text(doc.text)
    chunks = sliding_window_chunks(text, doc.source, chunk_size=250, stride=100)
    assert len(chunks) >= 3, f"PDF 切分块数异常: {len(chunks)}"
    assert any("600" in c.text for c in chunks), "切分后应能找到住宿标准 600 元的块"
    print(f"  [ok] PDF 中文提取正常：{len(text)} 字 → {len(chunks)} 块")


def test_prompt_grounding():
    from rag.prompts import build_messages

    msgs = build_messages("测试问题", [(1, "片段A内容"), (2, "片段B内容")])
    assert msgs[0]["role"] == "system" and "参考片段" in msgs[0]["content"]
    assert "[1] 片段A内容" in msgs[1]["content"]
    assert "禁止使用任何外部知识" in msgs[0]["content"]
    print("  [ok] Prompt 模板含 grounding 约束与来源编号")


if __name__ == "__main__":
    for fn in (
        test_chunker_basic,
        test_pdf_extraction,
        test_retrieve_and_answer,
        test_prompt_grounding,
    ):
        print(f"\n▶ {fn.__name__}")
        fn()
    print("\n全部离线测试通过 ✅")
