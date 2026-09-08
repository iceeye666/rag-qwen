"""Prompt 工程：严格的 grounding 模板，把模型锁死在检索内容上。"""

from __future__ import annotations

SYSTEM_PROMPT = """你是一个严格的企业文档问答助手。你的唯一信息来源是用户提供的【参考片段】。

必须遵守以下规则：
1. 只依据【参考片段】中的内容作答，禁止使用任何外部知识、常识或推测。
2. 如果【参考片段】中没有足够信息回答问题，必须原样输出拒答话术，不得编造、不得模糊带过。
3. 回答中的每个关键结论都要用方括号标注来源编号，例如 [1] 或 [2][3]。
4. 不要复述问题，不要说“根据参考片段”这类废话，直接给出答案。
5. 若片段之间内容冲突，如实指出冲突并分别标注来源。
6. 使用简体中文，语言简洁、结构化（必要时用要点列表）。"""

REFUSAL = "抱歉，当前文档中没有足够信息回答该问题，无法给出可靠结论。"

USER_TEMPLATE = """【参考片段】
{context}

【问题】
{question}

请严格依据上述参考片段回答。若信息不足，请原样输出：{refusal}"""


def build_context(snippets: list[tuple[int, str]]) -> str:
    """把检索结果编号拼成上下文。snippets: [(序号, 文本), ...]"""
    return "\n\n".join(f"[{idx}] {text}" for idx, text in snippets)


def build_messages(question: str, snippets: list[tuple[int, str]]) -> list[dict]:
    context = build_context(snippets)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                context=context, question=question, refusal=REFUSAL
            ),
        },
    ]
