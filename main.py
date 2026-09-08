"""命令行入口：ingest / ask / chat / info。

用法：
    python main.py ingest docs/财务管理制度.pdf [--reset]
    python main.py ask "差旅住宿报销标准是多少？"
    python main.py chat
    python main.py info
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag.config import settings
from rag.pipeline import RAGPipeline

REFUSAL_MARK = "抱歉，当前文档中没有足够信息"


def _print_answer(ans) -> None:
    print(f"\n问：{ans.question}")
    print(f"答：{ans.answer}")
    if ans.sources:
        print(f"\n引用来源（最高相似度 {ans.top_score:.3f}）：")
        for i, s in enumerate(ans.sources, 1):
            preview = s.text[:80].replace("\n", " ")
            print(f"  [{i}] {s.source} · {s.chunk_id} · score={s.score:.3f}")
            print(f"      {preview}...")
    print()


def cmd_ingest(pipe: RAGPipeline, path: str, reset: bool) -> int:
    if reset:
        pipe.store.clear()
        print("已清空向量库。")
    n = pipe.ingest(path)
    print(f"入库完成：{Path(path).name} → {n} 个文本块，向量库现有 {pipe.store.count()} 条。")
    return 0


def cmd_ask(pipe: RAGPipeline, question: str, top_k: int | None) -> int:
    ans = pipe.ask(question, top_k=top_k)
    _print_answer(ans)
    return 0


def cmd_chat(pipe: RAGPipeline, top_k: int | None) -> int:
    print("进入交互式问答（输入 exit / quit 退出）")
    while True:
        try:
            q = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            return 0
        if q.lower() in {"exit", "quit", "q"}:
            print("再见。")
            return 0
        if not q:
            continue
        try:
            _print_answer(pipe.ask(q, top_k=top_k))
        except Exception as e:  # noqa: BLE001
            print(f"[出错] {e}")


def cmd_info(pipe: RAGPipeline) -> int:
    print(f"向量库路径 : {settings.persist_dir}")
    print(f"集合名称   : {settings.collection_name}")
    print(f"已入库块数 : {pipe.store.count()}")
    print(f"Embedding  : {settings.embed_model}")
    print(f"生成模型   : {settings.chat_model}")
    print(f"切分参数   : chunk_size={settings.chunk_size}, stride={settings.chunk_stride}")
    print(f"检索参数   : top_k={settings.top_k}, 拒答阈值={settings.score_threshold}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RAG 文档智能问答系统（通义千问）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_in = sub.add_parser("ingest", help="解析并入库文档（pdf/txt/md）")
    p_in.add_argument("path")
    p_in.add_argument("--reset", action="store_true", help="入库前清空向量库")

    p_ask = sub.add_parser("ask", help="单轮问答")
    p_ask.add_argument("question")
    p_ask.add_argument("-k", "--top-k", type=int, default=None)

    sub.add_parser("chat", help="交互式问答")
    sub.add_parser("info", help="查看配置与库状态")

    args = parser.parse_args(argv)
    try:
        pipe = RAGPipeline()
    except RuntimeError as e:  # 缺少 API Key 等配置问题
        print(f"[配置错误] {e}")
        return 1

    if args.cmd == "ingest":
        return cmd_ingest(pipe, args.path, args.reset)
    if args.cmd == "ask":
        return cmd_ask(pipe, args.question, args.top_k)
    if args.cmd == "chat":
        return cmd_chat(pipe, args.top_k)
    return cmd_info(pipe)


if __name__ == "__main__":
    sys.exit(main())
