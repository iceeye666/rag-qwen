# RAG 文档智能问答系统（基于通义千问）

面向私有文档（财务制度、内部手册、知识库等）的轻量级检索增强生成问答系统。
支持 PDF 上传入库，用**通义千问 text-embedding-v2** 做向量化、**Chroma** 做本地持久化检索、
**qwen-plus** 做 grounded 生成，答案严格限定在文档事实内，命中不足时明确拒答。

## 架构

```
        PDF/TXT/MD
            │
            ▼
   ┌────────────────┐   pdfminer 解析 + 清洗（去页眉页脚/压缩空白）
   │   文档加载      │
   └────────┬───────┘
            ▼
   ┌────────────────┐   滑动窗口切分 chunk_size=250 / stride=100
   │   智能切分      │   窗口按句子边界对齐，相邻块重叠 150 字
   └────────┬───────┘
            ▼
   ┌────────────────┐   text-embedding-v2（自动分批，单批 ≤10 条）
   │    向量化       │
   └────────┬───────┘
            ▼
   ┌────────────────┐   Chroma 持久化，cosine 空间
   │   向量库入库    │
   └────────┬───────┘
            │
   问题 ────┴───► 向量化 ──► 相似度检索 top_k ──► 相似度 < 0.30 ?
                                                    │         │
                                                  是│         │否
                                                    ▼         ▼
                                              直接拒答   构造 grounding Prompt
                                              （省调用）        │
                                                               ▼
                                                        qwen-plus 生成
                                                               │
                                                               ▼
                                                    答案 + 来源引用 [1][2]
```

## 快速开始

```bash
# 1) 安装依赖
pip install -r requirements.txt

# 2) 配置 API Key（阿里云百炼控制台获取）
cp .env.example .env
#    编辑 .env 填入 DASHSCOPE_API_KEY=sk-xxxxxx

# 3) 自检接口连通性（可选但推荐）
python scripts/check_api.py

# 4) 文档入库
python main.py ingest docs/财务管理制度样例.pdf --reset

# 5) 提问
python main.py ask "一线城市差旅住宿报销标准是多少？"

# 或者进入交互式
python main.py chat

# 查看配置与库状态
python main.py info
```

## 离线自检（无需 API Key）

```bash
python tests/test_offline.py
```

用内存版组件（字符 bigram 假向量 + 摘录式 LLM）跑通全链路，覆盖：

- 切分：块长度、相邻块重叠、不丢句子、无重复块
- PDF：pdfminer 中文提取是否正常（乱码是最常见坑）
- 检索：问题相关块能否排到 top2
- 生成：答案是否严格来自检索上下文（grounding 校验）
- 拒答：无关问题相似度低于阈值时直接拒答，不调用 LLM

## 目录结构

```
rag-qwen/
├── rag/
│   ├── config.py        # 全部可调参数（.env 覆盖）
│   ├── loaders.py       # PDF(pdfminer) / TXT / MD 加载
│   ├── chunker.py       # 清洗 + 断句 + 滑动窗口切分
│   ├── embeddings.py    # text-embedding-v2，自动分批
│   ├── vectorstore.py   # Chroma 持久化封装
│   ├── llm.py           # qwen-plus 生成
│   ├── prompts.py       # grounding Prompt 模板
│   └── pipeline.py      # RAG 主流程（解析→向量化→检索→生成）
├── main.py              # CLI：ingest / ask / chat / info
├── tests/               # 离线冒烟测试
├── scripts/             # md_to_pdf.sh（造测试 PDF）、check_api.py（连通性自检）
└── docs/                # 示例文档（md + pdf）
```

## 关键参数与调优

| 参数 | 默认 | 说明 |
|---|---|---|
| `CHUNK_SIZE` / `CHUNK_STRIDE` | 250 / 100 | 窗口大小与步长，重叠量 = 250-100 = 150 字。中文语义密度高，250 字足够承载一条制度条款 |
| `TOP_K` | 4 | 送入 Prompt 的片段数。太多会稀释注意力并增加 token 成本 |
| `SCORE_THRESHOLD` | 0.30 | 余弦相似度硬门槛，低于此值直接拒答。**这是抗幻觉的第一道闸** |
| `TEMPERATURE` | 0.1 | 低温度保证答案稳定、少发挥 |
| `EMBED_BATCH_SIZE` | 10 | DashScope 单次 embedding 条数上限 |

调优建议：

- 答非所问/漏答案 → 调低 `SCORE_THRESHOLD`（0.20~0.25）或调大 `TOP_K`
- 答案混入无关内容 → 调高 `SCORE_THRESHOLD`，或减小 `CHUNK_SIZE` 提升片段纯度
- 长条款被截断 → 调大 `CHUNK_SIZE`，同时按比例调大 `CHUNK_STRIDE` 保持重叠比

## 抗幻觉设计

1. **硬门槛**：检索相似度低于阈值直接拒答，模型根本没有"发挥"的机会
2. **Prompt 约束**：system prompt 明确禁止外部知识，要求逐条结论标注来源编号
3. **低温度**：`temperature=0.1`，抑制发散
4. **冲突披露**：片段冲突时要求如实指出并分别标注，而非自行取舍
5. **拒答话术固定**：信息不足时原样输出固定话术，便于上层程序判定

## 可扩展方向

- 多文件/目录批量入库，按 source 元数据做过滤检索
- 接入 rerank（如 `gte-rerank`）做二次精排，提升 top1 准确率
- FastAPI 封装 HTTP 接口 + 前端对话页
- 换用 `text-embedding-v3` / `qwen-max`，或接入本地 embedding 实现完全离线

## 已知限制

- 不支持扫描件 PDF（无文字层），需先接 OCR
- 表格内容按行抽取后语义可能断裂，复杂表格建议单独处理
- 单文档维度去重，未做跨文档的同源合并
