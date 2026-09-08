# RAG 文档问答系统 · 学习指南（通义千问版）

> 这是一个**最小但完整**的 RAG（检索增强生成）案例，适合作为入门学习材料。
> 代码已加详细中文注释，建议配合本指南与源码对照阅读。

---

## 0. 你从这个项目能学到什么

- RAG 的**标准链路**：文档解析 → 清洗切分 → 向量化 → 向量检索 → LLM 生成
- **中文友好的切分算法**：句子对齐的滑动窗口（重叠保证上下文不丢）
- **如何调用通义千问**（embedding + 生成），并复用 OpenAI 客户端
- **抗幻觉设计**：相似度硬门槛拒答 + grounding Prompt 约束
- **可测试性**：用"依赖注入 + 假组件"实现无需 API Key 的离线全链路测试

---

## 1. 项目一句话概括

把私有文档（PDF / TXT / MD）向量化存进本地 Chroma 向量库；用户提问时，检索最相关片段，交给 `qwen-plus` 生成**只依据文档内容**的答案；当证据不足时明确拒答，而不是编造。

---

## 2. 数据流全景

**入库（ingest）**
```
PDF/TXT/MD
   │  load_document        (rag/loaders.py)
   ▼
清洗后纯文本
   │  clean_text           (rag/chunker.py)
   ▼
句子列表 ──► 滑动窗口切分 ──► Chunk[]   (rag/chunker.py)
   │                                  │
   │  embed_documents                 │
   ▼                                  ▼
向量[] ──────────►  Chroma 持久化入库   (rag/vectorstore.py)
```

**问答（ask）**
```
问题
   │  embed_query
   ▼
问题向量 ──► Chroma 检索 top_k ──► Hit[]  (rag/vectorstore.py)
   │
   │  相似度 < SCORE_THRESHOLD ?
   ├─ 是 ──► 直接返回拒答话术（不调用 LLM，省钱且稳）
   └─ 否 ──► 构造 grounding Prompt ──► qwen-plus 生成 ──► Answer + 来源引用
```

---

## 3. 模块地图与推荐阅读顺序

```
rag/
├── config.py        # ① 全部参数集中处，.env 可覆盖        —— 先读，建立全局认知
├── loaders.py       # ② 文档加载（PDF 走 pdfminer）        —— 最简单，建立信心
├── chunker.py       # ③ 清洗 + 断句 + 滑动窗口切分          —— 算法核心，重点看
├── embeddings.py    # ④ 通义千问 embedding（自动分批）     —— 看批处理与零向量
├── vectorstore.py   # ⑤ Chroma 封装（cosine 距离→相似度）   —— 看距离换算
├── prompts.py       # ⑥ grounding Prompt 模板              —— 看抗幻觉约束
├── llm.py           # ⑦ qwen-plus 生成                     —— 看 OpenAI 兼容
└── pipeline.py      # ⑧ RAG 主流程，串起上面所有模块        —— 最后读，水到渠成
main.py              # CLI 入口：ingest / ask / chat / info
tests/test_offline.py # 离线测试：内存版假组件跑通全链路
scripts/            # check_api.py（连通自检）、md_to_pdf.sh（造测试 PDF）
```

**阅读顺序建议**：①②③④⑤⑥⑦⑧ 再回看 `main.py` 与 `tests/`。

---

## 4. 逐模块精读

### 4.1 `config.py` —— 配置中心
- 所有可调参数集中在一个 `frozen` 的 `Settings` dataclass。
- 用 `os.getenv(..., default)` 读取环境变量，`.env` 文件会被 `load_dotenv()` 自动加载。
- 设计成 `frozen`：运行期不可变，避免被意外篡改。
- `require_api_key()`：缺 Key 时抛出清晰错误，引导用户创建 `.env`。

### 4.2 `loaders.py` —— 文档加载
- 统一返回 `Document(source, text)`，上层不关心文件类型。
- 按后缀分派：`pdf` → `pdfminer` 提取；其余纯文本直接读。
- 不支持的类型显式报错（fail-fast）。

### 4.3 `chunker.py` —— 切分算法（重点）
三个步骤：
1. **`clean_text`**：去页眉页脚（形如 "12"、"第 3 页" 的行）、压缩空白。
2. **`split_sentences`**：按句末标点（。！？；等）断句，超长句再硬切。
3. **`sliding_window_chunks`**：句子对齐的滑动窗口。
   - `chunk_size=250`：每个窗口目标字数。
   - `stride=100`：窗口每次前进的字符数。
   - **相邻块重叠 ≈ chunk_size − stride = 150 字**，避免跨边界的语义断裂。
   - 窗口按"句子"边界对齐，不会把一句话从中间劈开。
   - `_dedup`：去掉 PDF 重复抽取产生的完全重复块。

### 4.4 `embeddings.py` —— 向量化
- 复用 `openai.OpenAI` 客户端，把 `base_url` 指向 DashScope 兼容地址。
- **自动分批**：DashScope 单次 embedding 条数有上限，按 `EMBED_BATCH_SIZE` 分批，按原始下标回填，保证顺序一致。
- **零向量占位**：空白文本补零向量而非报错，避免单块失败拖垮整批入库。

### 4.5 `vectorstore.py` —— 向量库
- 封装 `chromadb.PersistentClient`，数据持久化到磁盘（`storage/chroma`）。
- 集合用 `hnsw:space="cosine"`（余弦空间）。
- `query` 返回的 `distances` 是**距离**，换算 `score = 1 − dist` 得到相似度。
- `clear()`：Chroma 无"清空集合"API，故删集合重建，重建时**必须重新指定 cosine 空间**，否则相似度口径不一致。

### 4.6 `prompts.py` —— grounding 约束
- `SYSTEM_PROMPT` 明确要求：只依据【参考片段】作答、禁止外部知识、每条结论标注来源编号 `[1]`、冲突须披露。
- `REFUSAL` 是固定拒答话术，便于程序判定。

### 4.7 `llm.py` —— 生成
- 同样复用 OpenAI 客户端调用 `qwen-plus`。
- `temperature` 默认 0.1，抑制发散、保证答案稳定。

### 4.8 `pipeline.py` —— 主流程
`RAGPipeline` 把上面所有模块串起来，且**支持依赖注入**（构造时传 `embedder/store/chat`），这正是离线测试能跑通的关键：
- `ingest`：解析→清洗→切分→向量化→入库（5 步）。
- `retrieve`：问题向量化后去 Chroma 取 top_k。
- `ask`：检索 → 硬门槛判定 → 构造 Prompt → 生成 → 判定是否拒答。

### 4.9 `main.py` —— CLI
四个子命令：`ingest`（入库）、`ask`（单轮）、`chat`（交互）、`info`（看配置）。用 `argparse` 实现。

---

## 5. 三个最值得理解的核心设计

### 5.1 滑动窗口为何要"重叠"
如果不重叠，一条跨窗口的条款会被切成两半，检索命中一半时语义不全。
重叠 150 字让边界处的上下文被两个块共同覆盖，检索召回更稳。
**思考题**：重叠越大越安全，但存储和检索成本也越高，如何权衡？

### 5.2 embedding 为何要"分批 + 零向量"
- 分批：绕开云服务的单次条数上限。
- 零向量：PDF 偶尔抽出的空块不应让整批入库失败。
**思考题**：如果直接对空文本调用 embedding 会怎样？

### 5.3 余弦相似度硬门槛（抗幻觉第一道闸）
`ask` 在调用 LLM **之前**就判断：`hits[0].score < SCORE_THRESHOLD` 直接拒答。
省了一次昂贵的 LLM 调用，也彻底堵死了"模型强行编答案"的可能。

---

## 6. 抗幻觉的四道防线（对应代码）

| 防线 | 位置 | 作用 |
|---|---|---|
| 硬门槛 | `pipeline.ask` 的 `score_threshold` | 相似度不足直接拒答，不调用 LLM |
| Prompt 约束 | `prompts.SYSTEM_PROMPT` | 禁止外部知识、要求标注来源 |
| 低温度 | `config.temperature=0.1` | 抑制发散发挥 |
| 固定拒答话术 | `prompts.REFUSAL` | 信息不足时输出可程序判定的固定文本 |

---

## 7. 如何运行

### 7.1 环境安装
```bash
# 方式 A（uv，推荐，本项目用 uv 管理）
uv venv
uv pip install -r requirements.txt        # 或 uv sync（见下方说明）

# 方式 B（pip）
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 7.2 离线测试（无需 API Key，先跑这个理解流程）
```bash
python tests/test_offline.py
```
它用内存版假组件（字符 bigram 假向量 + 摘录式假 LLM）跑通全链路，验证：
切分质量、PDF 中文提取、检索排序、答案 grounding、阈值拒答。

### 7.3 在线运行（需通义千问 API Key）
```bash
cp .env.example .env          # 编辑填入 DASHSCOPE_API_KEY=sk-xxxxxx
python scripts/check_api.py   # 自检 embedding / chat 两接口是否连通
python main.py ingest docs/财务管理制度样例.pdf --reset
python main.py ask "一线城市差旅住宿报销标准是多少？"
python main.py chat           # 交互式
python main.py info           # 看配置与库状态
```

---

## 8. 参数调优速查

| 参数 | 默认 | 调大 / 调小的影响 |
|---|---|---|
| `CHUNK_SIZE` | 250 | 大→片段信息更全但更杂；小→更纯但易截断 |
| `CHUNK_STRIDE` | 100 | 小→重叠多、召回稳但成本高 |
| `TOP_K` | 4 | 大→上下文多但稀释注意力、费 token |
| `SCORE_THRESHOLD` | 0.30 | **抗幻觉总闸**：低→更宽松；高→更易拒答 |
| `TEMPERATURE` | 0.1 | 低→稳定少发挥 |

调参经验：
- 答非所问/漏答案 → 调低阈值（0.20~0.25）或调大 `TOP_K`
- 答案混入无关内容 → 调高阈值或减小 `CHUNK_SIZE`
- 长条款被截断 → 调大 `CHUNK_SIZE`，并按比例调大 `CHUNK_STRIDE` 保持重叠比

---

## 9. 自测思考题（检验是否真懂了）

1. 为什么窗口按"句子"而不是"字符"对齐？
2. `stride=100`、`chunk_size=250`，相邻块重叠多少字？
3. 为什么用余弦距离，且 `score = 1 − dist`？
4. 把 `SCORE_THRESHOLD` 调到 `0.0` 会怎样？调到 `1.0` 呢？
5. `RAGPipeline` 的 `embedder/store/chat` 可注入，这给测试带来什么好处？
6. 离线测试里 `FakeEmbedder` 为什么用 `zlib.crc32` 而不是内置 `hash()`？
7. 若要做"多文档按文件名过滤检索"，应在哪一层加逻辑？（提示：元数据 `source`）

---

## 10. 扩展路线（学完后可动手）

- **多文件批量入库**：遍历目录，按 `source` 元数据过滤检索
- **二次精排**：接入 rerank 模型（如 gte-rerank）提升 top1 准确率
- **HTTP 服务**：用 FastAPI 封装接口 + 前端对话页
- **完全离线**：换本地 embedding 模型，去掉 API 依赖
- **可观测性**：把每次问答的 chunks/scores 落日志，便于调参

---

> 学习建议：先跑通 `tests/test_offline.py`，用断点/打印观察 `sliding_window_chunks` 与
> `RAGPipeline.ask` 的中间产物，再接上真实 API 看端到端效果。理解"拒答"比理解"答对"更重要。
