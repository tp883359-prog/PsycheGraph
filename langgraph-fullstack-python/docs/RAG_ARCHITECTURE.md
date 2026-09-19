# RAG Architecture — 本地理论知识库与证据契约

> Phase 6 文档。目标不是"再增加一个 LLM Agent"，而是让已有 Agent 的读解**有据可依**：
> 引入本地理论知识库检索（RAG）、证据契约（Evidence Contract）与可追溯引用。

> Phase 7 跟进：本文档描述的是**检索层**。审核层（确定性校验 + Critic + 有界修订 +
> 安全终止）见 `docs/CRITIC_ARCHITECTURE.md`；加入审核层后一轮调用为 6（一次通过）或 8
> （一次修订），检索本身仍然只在 `evidence` 节点发生一次。

---

## 1. 目标与非目标

**目标**

1. 三个学派 Specialist 在作答前拿到该学派知识库中的真实片段。
2. 模型只能引用系统给它的 `evidence_id`；引用必须可追溯到某个真实 chunk。
3. 最终回答底部列出**真实**来源（作者 / 书名 / 年份 / 页码或章节），页码只来自真实解析。
4. 索引在离线建立；服务运行时只读索引，不重新 embedding。
5. 没有知识库、没有索引、没有 embedding 模型时，系统必须**降级而不是崩溃**。

**非目标（本阶段明确不做）**

- 不做 hybrid search、BM25、reranker、查询改写用的额外 LLM 调用。
- 不做长期记忆、不做评估框架、不做前端 UI、不做 Docker。
- 不引入任何云端 embedding 服务；不使用网络下载的来源不明文本。

本阶段（Phase 6）没有加入 Critic；审核层是 Phase 7 单独引入的（见
`docs/CRITIC_ARCHITECTURE.md`），因为"引用真实存在"与"推断成立"是两个问题。

---

## 2. 图拓扑

```mermaid
flowchart LR
    START((START)) --> S[supervisor<br/>1 次 DeepSeek]
    S --> E[evidence<br/>0 次 DeepSeek<br/>本地检索]
    E --> F[freudian]
    E --> O[object_relations]
    E --> L[lacanian]
    F --> Y[synthesizer<br/>1 次 DeepSeek]
    O --> Y
    L --> Y
    Y --> R[deterministic_validator<br/>0 次 · 纯代码]
    R --> C[critic<br/>1 次 DeepSeek]
    C --> FIN[finalize | safe_finalize<br/>0 次 · 只写一条 AI 消息]
    FIN --> END((END))
```

- `evidence` 节点位于 Supervisor 与三个 Specialist 之间，**不调用任何 LLM**。
- 三个 Specialist 仍在同一 superstep 并发执行；Synthesizer 等三者全部写回后才启动。
- 一轮对话的 DeepSeek 调用次数（Phase 7）：
  `1 (Supervisor) + 3 (Specialists) + 1 (Synthesizer) + 1 (Critic) = 6`；
  触发一次修订时 `+2`（修订合成 + 再次审核）= 8，这是本阶段的上界。
- 检索仍只在 `evidence` 节点发生一次；修订不会重新检索、不会重跑 Specialist。

---

## 3. 代码与目录结构

```
knowledge/                          # 你的知识库（随仓库提交，内容由你负责）
├── README.md                       # 收录规则、版权原则、命令
├── freudian/                       # 学派目录 → 该目录下文件的 school="freudian"
├── object_relations/
└── lacanian/

src/react_agent/rag/                # 检索层（与 agent 逻辑解耦）
├── settings.py                     # 环境变量 → RagSettings（含路径、top_k、chunk 参数）
├── embeddings.py                   # 懒加载 BAAI/bge-m3（HuggingFaceEmbeddings）
├── ingestion.py                    # 文件 → 文档 → chunk（含 metadata 契约）
├── vectorstore.py                  # Chroma 持久化封装（打开 / 计数 / 重建）
├── retriever.py                    # 三个学派的查询、过滤、EvidenceItem 生成、降级
└── citations.py                    # 引用与来源列表渲染

src/react_agent/agents/evidence.py  # evidence 节点（无 LLM）
scripts/index_knowledge.py          # 离线建索引 CLI（--rebuild / --dry-run / --knowledge-dir）
scripts/verify_rag.py               # 真实运行验证（检索 + 4 个真实用例 + 性能）
data/vectorstore/                   # 向量库（git-ignored，运行时不重建）
tests/fixtures/knowledge/           # 自编测试语料（test_fixture: true）
```

---

## 4. Embedding

| 项目 | 取值 |
|---|---|
| 模型 | `BAAI/bge-m3`（本地运行，`EMBEDDING_MODEL` 可覆盖） |
| 设备 | `cpu`（`EMBEDDING_DEVICE`；显式配置后才用 `cuda`） |
| 归一化 | `encode_kwargs={"normalize_embeddings": True}`，因此余弦相似度可直接比较 |
| 加载方式 | `langchain_huggingface.HuggingFaceEmbeddings`，进程内 `lru_cache` 复用 |

关键实现细节：`HuggingFaceEmbeddings` 在**构造时**就会加载模型权重。因此
`retriever.retrieve_evidence()` 先检查索引目录是否存在，**再**构造 embedding 模型：
没有索引的部署（例如还没建库的新环境）不会因为一次提问而加载 2GB 权重。

---

## 5. 向量库

- Chroma，持久化到 `RAG_VECTORSTORE_PATH`（默认 `data/vectorstore`，已 gitignore）。
- collection：`psychegraph_theory`，`collection_metadata={"hnsw:space": "cosine"}`。
- 检索使用 `similarity_search_with_score`，`score = 1 - distance` 即为余弦相似度。
- 只在离线索引时写入；运行时只读。索引目录不存在、collection 为空、索引损坏
  （打开抛异常）时，`open_vectorstore()` 返回 `None`，图继续运行。

---

## 6. 元数据契约（Ingestion）

每个 chunk 都携带以下字段，取值原则是**只记录真实读到的信息，未知一律 `None`**：

| 字段 | 来源 | 说明 |
|---|---|---|
| `source_id` | 相对路径 slug | 例如 `freudian_freudian_concepts` |
| `school` | 目录名（或 front matter 覆盖） | `freudian` / `object_relations` / `lacanian` / `general` |
| `title` | front matter `title` 或文件名 | 永远有值 |
| `author` | front matter `author` | 不知道就是 `None` |
| `work_title` | front matter `work_title` | 不知道就是 `None` |
| `year` | front matter `year`（纯数字才接受） | 不知道就是 `None` |
| `page` | PDF 的真实物理页码（1-based） | Markdown/TXT 恒为 `None` |
| `section` | Markdown 标题文本 | 无标题即 `None` |
| `source_path` | 相对 `knowledge/` 的路径 | 例如 `freudian/note.md` |
| `language` | front matter `language` 或 CJK 检测 | `zh` / `en` |
| `test_fixture` | front matter `test_fixture` | 测试语料标记，防止被当成真实文献 |
| `chunk_id` | `f"{source_id}_{index:06d}"` | **稳定 id**，重复索引不会变化 |

分片：`RecursiveCharacterTextSplitter`，`chunk_size=1000`、`chunk_overlap=150`（可配置）。
同一 `source_id` 的计数在所有 section 之间连续，因此 id 稳定且可复现。

---

## 7. 检索策略

每个学派一次查询，共三次：

```
query = supervisor_plan[<school>_focus] + "\n" + 最新一条用户消息     # 截断到 1200 字符
filter = {"school": {"$in": [<school>, "general"]}}
k = RAG_TOP_K (默认 5)
```

- **不做 LLM 查询改写**：证据节点 0 次模型调用，也不会让延迟翻倍。
- **学派隔离由向量库过滤保证**：弗洛伊德 Specialist 的提示词里不会出现拉康材料。
- `general/` 目录的文件（方法论、术语表）三个学派都能检索到。
- 排序按余弦相似度降序，分数写入 `evidence_items[h]["retrieval_score"]` 供审计。

---

## 8. 证据契约（Evidence Contract）

`EvidenceItem` **只能由检索器产生**，模型任何时候都不能"发明文献"。落地为四道闸门：

1. **输入约束**：Specialist / Synthesizer 的提示词中列出本次真实可用的 `evidence_id`，
   并附带规则：只能从其中选择；没有对应文献时 `evidence_ids` 留空并在 `uncertainty` 里说明；
   `textual_basis` 只能写用户提供的文本；文献内容不是用户的事实；不得大段抄录原文。
2. **输出校验**：节点在写回 state 前把模型写出的 id **解析**到本次真正提供的 id 上
   （完全匹配 / 大小写与标点差异 / 唯一后缀，例如模型把 `freudian_freudian_concepts_000002`
   写成 `freudian_concepts_000002`）；解析成功后把结果里的 id 就地改写为真实 id。
   解析不到的 id（包括有歧义的缩写）抛 `InvalidEvidenceReferenceError`，
   宁可让这一轮失败，也不把编造的文献送到用户面前。
   解析只会命中"已提供的 id"，因此不会凭空产生引用。
3. **空证据规则**：检索为空时，提示词给出 `NO_EVIDENCE_NOTICE`，明确要求
   `evidence_ids` 必须为空、并在 `uncertainty`/`limitations` 中说明缺少本地文献支持。
4. **渲染隔离**：引用文本由 `citations.py` 用**索引里的元数据**渲染；
   模型即使在自己的 `final_response` 里写页码，系统也会在来源列表里给出真实信息，
   且`render_sources_section` 只渲染 `used_evidence_ids` 命中的条目。

状态字段（均为纯 JSON，可被 checkpoint 序列化）：

| 字段 | 类型 | 说明 |
|---|---|---|
| `evidence_by_school` | `dict[str, list[dict]]` | 三个学派各自的 `EvidenceItem` 序列化结果 |
| `evidence_meta` | `dict` | `available` / `reason` / `elapsed_seconds` / `counts` |

---

## 9. 引用渲染规则

`describe_source(item)` 严格按"有则显示、无则跳过"拼装：

```
[作者, ]《书名》[, 年份][, p. 页码 | , 章节「section」]
```

- 有真实页码（PDF）→ 显示 `p. N`。
- 没有页码但有标题小标题 → 显示 `章节「...」`。
- 都没有 → 显示 `未标注来源（source_id）` 并附 `evidence_id`。
- `final_response` 之后追加：

```
理论依据（本地知识库）
[E1] 某作者, 《某书》, 章节「防御机制」
```

没有使用任何证据时**不追加**该区块——不制造"有文献支持"的假象。

---

## 10. 失败与降级

| 情况 | 行为 | `evidence_meta.reason` |
|---|---|---|
| `RAG_ENABLED=false` | 直接返回空证据 | `disabled` |
| 索引目录不存在 | 不加载 embedding 模型，返回空证据 | `no-index` |
| 索引存在但为空 | 视为无索引 | `no-index` |
| 未安装 embedding 依赖 | 捕获异常，返回空证据 | `embedding-unavailable: ...` |
| 检索抛异常（索引损坏等） | 记 warning，该学派返回空列表，运行继续 | `retrieval-error: <类型>` |
| 模型结构化输出缺字段/被截断 | 重试一次；仍失败则抛具名的 `RuntimeError`（如 `limitations: missing`），不泄露裸 Pydantic 错误 | — |

无论哪种情况，图都会跑完，`evidence_ids` 为空，回答中说明"没有本地文献支持"。
结构化输出重试一次会让该轮多出 1 次调用（原本 6 次 → 7 次）。

---

## 11. 建索引与运行命令

```powershell
$env:PYTHONUTF8="1"

# 离线建库（幂等：chunk_id 稳定，重复运行覆盖同名 chunk）
uv run python scripts/index_knowledge.py --rebuild

# 只做加载与分片统计，不写向量库
uv run python scripts/index_knowledge.py --dry-run

# 用测试语料单独建一个库（与本机真实知识库隔离）
$env:RAG_VECTORSTORE_PATH="data/vectorstore_fixture"
uv run python scripts/index_knowledge.py --knowledge-dir tests/fixtures/knowledge --rebuild

# 真实验证（检索 + 4 个用例 + 性能）
uv run --env-file .env python scripts/verify_rag.py
```

**服务启动时不建索引、不做 embedding**。LangGraph Server 只读取已有 collection。

---

## 12. 测试策略

单元测试不下载 BGE-M3，而是用两个替身：

- `tests/unit_tests/rag_helpers.py::HashingEmbeddings`：字符 n-gram 哈希到 96 维并归一化，
  确定性、无网络，足以验证过滤 / top_k / 排序 / 元数据。
- Chroma 建在 `tmp_path` 下（fixture `rag_index`），并在 `react_agent.rag.retriever`
  命名空间里替换 `get_embeddings` / `get_rag_settings`，其余代码走真实路径。
- 默认 autouse fixture 把 `RAG_ENABLED=false`，保证任何单元测试都不会加载真实模型。

覆盖点：分片 id 稳定性、元数据真实性（页码 / 作者 / 年份不编造）、front matter、
PDF 真实页码、学派过滤隔离、top_k、分数区间、空索引 / 空查询 / 禁用 / 检索异常降级、
证据节点不调用模型、Specialist 只看自己的证据、引用非法 id 抛错、来源列表渲染规则。

真实（联网 + 真模型）验证见 `scripts/verify_rag.py` 与 `scripts/verify_multi_agent.py`。

### 12.1 真实验证发现的问题与修复

跑真实模型时暴露了两个单测覆盖不到的问题，已修复并补了回归测试：

1. **`knowledge/README.md` 被当成理论材料**。首次 `--dry-run` 显示 `knowledge/` 下唯一的
   README 被索引成 7 个 `general` chunk，会被当作"文献"喂给三个学派。
   修复：`is_indexable_name()` 跳过 `README*`、`.`/`_` 开头与不支持的扩展名
   （`tests/unit_tests/test_rag_ingestion.py::test_documentation_and_hidden_files_are_not_indexed`）。
2. **模型缩写 evidence id 导致整轮失败**。真实运行中弗洛伊德 Specialist 把
   `freudian_freudian_concepts_000002` 写成 `freudian_concepts_000002`，
   严格校验直接让这次对话失败。修复：先做**唯一解析**（完全匹配 → 大小写/标点归一 →
   唯一后缀），把引用改写回真实 id；解析不到（含歧义缩写）仍然抛
   `InvalidEvidenceReferenceError`。安全性不变，鲁棒性提高。
3. **结构化输出缺字段时抛出裸 Pydantic 错误**。《哈姆雷特》用例的 Synthesizer 返回缺少
   `limitations`，用户会看到 `pydantic_core.ValidationError` 栈。修复：把这类失败视为
   一次可重试的提供方问题，重试一次；仍失败则抛具名 `RuntimeError`
   （例如 `synthesizer: ... (schema violation: limitations: missing; ...)`）。
   重试成功时该轮调用数为 6（已写入验证脚本与本文档）。

第四个发现不是缺陷而是设计取舍的记录：检索总会返回 top-k（即使相似度不高），
因此提示词里明确写了"用不上就留空、并在 uncertainty 里说明"，验证脚本也允许
`used_evidence_ids` 为空。

### 12.2 Phase 7 服务器路径验证发现的问题与修复

Phase 7 首次把检索层放回 `langgraph dev` 里跑（之前的真实验证都是进程内脚本），
发现了一个只在服务器路径出现的缺陷：

- **`project_root()` 里的阻塞调用**。`rag/settings.py` 原本写的是
  `Path(__file__).resolve().parents`，而 `resolve()` 在 Windows 上会调用
  `os.getcwd()`；`langgraph dev` 的 blocking 检测器（blockbuster）会把它当作阻塞
  调用直接报错，`evidence` 节点因此让整轮运行失败（`BlockingError: Blocking call to
  os.getcwd`）。修复：改为从 `__file__` 直接派生（`Path(os.path.abspath(__file__))`），
  相对路径只做拼接，并把日志里的 `Path.cwd()` 回退改成模块目录；回归测试
  `tests/unit_tests/test_rag_retrieval.py::test_settings_resolution_never_calls_the_working_directory`
  把 `os.getcwd` 换成抛异常的函数，确保以后不会再引入同类调用。

修复后同一轮验证（服务器）跑通：`evidence` 正常降级（未配置索引时不报错），
state 里 `evidence_meta` 与审查字段同时存在，`messages` 仍然只有 `human/ai`。

---

## 13. 性能实测

> 环境：Windows / CPU（`EMBEDDING_DEVICE=cpu`）/ BGE-M3 / 16 个测试 chunk。
> 数据来自 `scripts/verify_rag.py` 的一次完整真实运行（4 个用例全部通过，`EXIT=0`）。

| 指标 | 数值 |
|---|---|
| 首次下载 + 首次建索引（含下载 2.27GB 权重） | 模型加载 331.05s，embed+write 2.55s |
| 索引构建（权重已缓存） | 分片 9.47s，embed+write 2.55s，共 16 chunk |
| embedding 模型热加载（每进程一次） | 22.5–23.9s |
| 每轮检索（3 个学派，top_k=5） | 0.35–0.68s |
| 单轮端到端（4 个用例） | 36.2s / 42.1s / 44.7s / 44.8s |
| 每轮 DeepSeek 调用次数 | Phase 6 实测为 5（`SupervisorPlan×1 + SchoolAnalysis×3 + SynthesisResult×1`）；加入审核层后正常为 6，修订一次为 8（详见 `CRITIC_ARCHITECTURE.md`） |

由于 `evidence` 节点不调用 LLM，加入 RAG 后**模型调用次数不变**，
额外成本主要是每轮 3 次向量检索（毫秒级）与首次的模型加载（一次性，进程内复用）。

---

## 14. 已知限制与技术债

1. **语料规模有限**：仓库自带的 `tests/fixtures/knowledge` 是自编测试材料，
   只用于验证机制；真实分析质量取决于你放入 `knowledge/` 的材料。
2. **无重排**：仅向量相似度排序，未做 reranker / hybrid；这是本阶段的有意取舍。
3. **PDF 无 front matter**：PDF 的 `author`/`year` 无法从文本可靠获得，保持 `None`。
4. **全量重建**：`--rebuild` 会清空 collection 重建；没有按文件增删的增量同步。
5. **首次下载**：BGE-M3 约 2.3GB，需要一次网络下载（`HF_TOKEN` 可提高限额）；
   Windows 未开启开发者模式时 huggingface 会提示 symlink 警告，缓存仍可用。
6. **`app.py` 未改动**：SSE 仍只转发 `messages` 通道的 `content`（每轮 1 个内容块），
   因此没有 token 级流式输出；这是 Phase 4/5 遗留债务，不在本阶段范围内。
7. **模型仍可能少用证据**：契约保证"不能引用不存在的证据"，但不能强制"必须引用"；
   因此 `used_evidence_ids` 为空是合法结果，且此时不会渲染来源区块。
8. **检索层不负责推理正确性**：真实引用但推断错误的问题由 Phase 7 的审核层处理
   （`docs/CRITIC_ARCHITECTURE.md`），检索层仍然只保证"证据存在且可追溯"。
