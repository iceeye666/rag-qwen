# RAG 文档智能问答系统（基于通义千问）

面向私有文档（财务制度、内部手册、知识库等）的轻量级检索增强生成问答系统。
支持 PDF 上传入库，用**通义千问 text-embedding-v2** 做向量化、**Chroma / Milvus** 做向量持久化检索、
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
   ┌────────────────┐   Chroma / Milvus 持久化，cosine 空间
   │   向量库入库    │   VECTOR_BACKEND 切换后端
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
#    如需 Milvus Lite 本地模式，再补装可选依赖：
#    pip install "pymilvus[milvus_lite]"

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

## 向量后端切换（Chroma / Milvus）

默认使用 Chroma，无需任何额外部署。若要改用 Milvus，只需在 `.env` 中把
`VECTOR_BACKEND` 设为 `milvus` 即可，其余命令（`ingest` / `ask` / `chat` / `info`）完全不变。

```bash
# .env
VECTOR_BACKEND=milvus
```

### 方式一：Milvus Lite（本地文件，零部署，推荐先跑通）

无需启动任何服务，`pymilvus` 会直接以本地文件形式运行。先补装 Lite 可选依赖：

```bash
pip install -r requirements.txt
pip install "pymilvus[milvus_lite]"
```

不配置连接信息时，默认落盘到 `./storage/milvus.db`（Lite 会在该路径生成数据库目录）：

```bash
# .env 中不必写 MILVUS_URI，留空即用本地 Lite 文件
VECTOR_BACKEND=milvus
# MILVUS_URI=./storage/milvus.db   # 也可显式指定路径，相对路径按项目根目录解析
```

```bash
python main.py ingest docs/财务管理制度样例.pdf --reset
python main.py ask "一线城市差旅住宿报销标准是多少？"
python main.py info   # 应显示「向量后端: milvus / 部署形态: 本地 Milvus Lite」
```

> 说明：Milvus Lite 支持 macOS 与 Linux，暂不支持 Windows（Windows 下请使用方式二）。

### 方式二：远程 Milvus 服务

用 Docker 快速起一个 standalone 实例：

```bash
docker run -d --name milvus-standalone \
  -p 19530:19530 -p 9091:9091 \
  milvusdb/milvus:latest milvus run standalone
```

然后在 `.env` 中配置连接信息（两种写法二选一，`MILVUS_URI` 优先级更高）：

```bash
VECTOR_BACKEND=milvus
MILVUS_HOST=127.0.0.1
MILVUS_PORT=19530
# 或直接：
# MILVUS_URI=http://127.0.0.1:19530
```

连接 Zilliz Cloud 等需要鉴权的托管服务时，额外填 `MILVUS_TOKEN`：
`MILVUS_URI=https://in01-xxxx.aws-us-west-2.vectordb.zillizcloud.com:19530` + `MILVUS_TOKEN=<api-key>`。

### 连接目标解析顺序

`MILVUS_URI` → `MILVUS_HOST:MILVUS_PORT` → 本地 Lite 文件 `./storage/milvus.db`。

### 注意事项

- **相似度口径一致**：Milvus 集合以 `COSINE` 度量创建，检索返回的分数即余弦相似度，
  与 Chroma 的 `1 - 距离` 口径一致，因此 `SCORE_THRESHOLD`（默认 0.30）在两个后端下含义相同，无需调整。
- **维度必须匹配**：`MILVUS_DIM` 需与 embedding 模型输出维度一致（`text-embedding-v2` → 1536，
  `text-embedding-v3` → 1024）。维度不符时入库会直接报错并提示正确取值。
- **切换后端后需重新入库**：两种后端的存储互不通用，切换 `VECTOR_BACKEND` 后请重新执行
  `python main.py ingest <文档> --reset`。
- **回退**：把 `VECTOR_BACKEND` 改回 `chroma` 即可回到原 Chroma 库，历史数据与集合名不受影响。

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
│   ├── vectorstore.py   # Chroma / Milvus 向量库封装 + create_store 工厂
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
| `VECTOR_BACKEND` | chroma | 向量库后端：`chroma` 或 `milvus` |
| `MILVUS_DIM` | 1536 | Milvus 集合向量维度，须与 `EMBED_MODEL` 输出一致（v2→1536，v3→1024） |

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
