# Phase 9：Web UI、Workflow Streaming 与 Evidence 呈现

本文件说明 Phase 9 的网页层为什么这样设计。它覆盖：为什么用 workflow
streaming 而不是 token streaming、SSE 事件契约、`WorkflowEvent`、Graph 原始
事件到 UI 事件的转换、证据与引用、安全清洗、修订路径、错误处理、响应式与
可访问性、Evaluation 页面，以及已知限制。

代码位置：

```
src/react_agent/app.py          薄入口：from react_agent.web.app import app
src/react_agent/web/
  app.py          FastHTML 实例与静态资源（CSS/JS 内联、htmx-ext-sse）
  routes.py       页面、send-message、SSE stream、Evaluation 页面
  streaming.py    LangGraph 原始流 -> 清洗后的 WorkflowEvent
  events.py       UI 事件契约（WorkflowEvent / EventMetadata / 节点规格）
  fragments.py    事件 -> HTML 片段（事件到 DOM 的唯一映射点）
  components.py   页面骨架、聊天气泡、输入区、证据面板、引用块
  workflow.py     Workflow 面板：节点行、状态、事件日志
  citations.py    证据清洗、引用标签、来源描述
  render.py       回答的安全渲染（转义、段落/列表/强调、引用按钮）
  evaluation.py   评估摘要的读取与页面
  pending.py      线程级「一条消息 + 一个运行中」注册表
```

Phase 9 **没有改动生产 Graph**：`graph.py`、三个 specialist 的 prompt、RAG、
Critic、MAX_REVISION、finalizer 全部保持 Phase 8 的状态；网页层只读事件。

---

## 1. 为什么是 Workflow Streaming，不是 token streaming

Phase 7 已经确认：Structured Output 走 function calling，模型在生成期间会发出
大量 tool-call 参数 chunk，而**真正可见的回答只在 `finalize` 之后存在**。Phase 9
的实测（`scripts/inspect_stream.py`，stream mode `messages-tuple`）：

```
一次完整运行：metadata 1 个事件、tasks 18 个、updates 9 个、messages 6304 个
其中 messages 事件几乎全部是参数分片，可见回答一个都没有
```

把 6304 个参数分片伪装成"正在输入"是没有信息量的动画。因此本阶段：

- 向服务器请求的 stream mode 只有 `tasks` 与 `updates`（不请求 `messages`）；
- 等待期间展示**真实的节点状态**（谁在跑、跑了多久）；
- 最终回答在 `finalize` / `safe_finalize` 之后**一次性**出现。

如果将来真的需要逐 token 的最终回答，需要重做 Synthesizer / Critic /
Finalizer 的架构，本阶段不做。

## 2. SSE 事件契约

浏览器只认识五个命名事件（`react_agent.web.events.EventType`）：

| 事件 | 何时发出 | 载荷 | 目标 |
|---|---|---|---|
| `workflow` | 节点开始 / 结束（来自 `tasks` 与 `updates`） | 日志行 + 该节点的 OOB 行替换 + 状态栏文字 | `#workflow-log`（append）+ OOB |
| `sources` | `evidence` 节点完成 | 按学派分组的证据卡片 | `#evidence-body`（innerHTML） |
| `answer` | **只有** `finalize` / `safe_finalize` 的 update | 聊天气泡：正文 + 引用列表 + 视角标签 + 质量行 | `#chatlist`（append）+ OOB 删除占位气泡 |
| `error` | 运行失败（异常或 `error` 流事件） | 一句安全提示 | `#chatlist`（append） |
| `close` | 流结束（永远最后一个） | 结束状态栏 + OOB 恢复输入区 + OOB 清空 `#stream-controls` | `#run-status`（innerHTML） |

实现要点：

- FastHTML 的 `sse_message(element, event=...)` 生成命名帧；一个帧里可以包含
  多个元素，htmx 会把带 `hx-swap-oob` 的片段提取出来做带外替换，其余部分插入
  主目标。
- **一个隐藏的 `#sse-hub`** 拥有 EventSource；它内部是五个"转发 div"
  （`sse-swap="workflow"` + `hx-target=...`），因为 htmx SSE 扩展要求
  `sse-swap` 元素必须是 `sse-connect` 元素的后代，而目标可以在文档任何位置。
- `#sse-hub` 带 `sse-close="close"`：收到 `close` 时浏览器主动关闭连接，因此
  服务器不需要依赖连接断开，也不会触发自动重连风暴。
- 每一次 run 都会用**新的** `#sse-hub` 替换 `#stream-controls` 的内容，`close`
  再把该容器清空，旧 EventSource 随元素移除被扩展自动关闭。

### 为什么先 POST 再 GET

EventSource 只能发 GET，而消息需要 POST。流程是：

1. `POST /conversations/{id}/send-message`：把消息放进进程内注册表
   （`pending.RunRegistry`），返回用户气泡 + 占位气泡 + OOB 片段（禁用输入区、
   插入 hub、重置流程面板与证据面板）。
2. 浏览器拿到响应后立刻打开 `GET /conversations/{id}/stream`。
3. stream 处理器取走消息，用 `runs.stream(...)` **同时创建并流式读取**这次 run。

这样不存在"创建 run 之后、浏览器 join 之前事件已经跑完"的经典竞态；代价是
注册表是进程内的（见第 10 节）。当线程已有 pending 或运行中的 run 时，第二次
POST 会返回"上一条还在跑"，避免并发 run。

## 3. WorkflowEvent 契约

```python
WorkflowEvent(
    event_type: "workflow" | "answer" | "sources" | "error" | "close",
    node: str | None,            # 例如 "critic"
    status: "waiting"|"running"|"completed"|"revising"|"failed"|"skipped" | None,
    label: str,                  # 用户可读节点名
    message: str,                # 一句安全状态文本
    run_id: str | None,
    timestamp: str,              # ISO-8601 UTC
    elapsed_seconds: float,      # 自 run 开始
    metadata: EventMetadata,     # 封闭 dataclass（唯一可加字段的地方）
)
```

`EventMetadata` 是**封闭的 dataclass**：`duration_seconds`、`evidence_count`、
`evidence_counts`、`verdict`、`issues`、`revision_count`、
`finalization_status`、`specialists`、`attempt`、`answer`、`sources`、
`error_code`。想给页面加字段，只能在这里加，并且必须是预先算好的小值。这就是
"白名单"的落地方式：不是运行时过滤一个字典，而是根本没有别的地方能塞东西。

`status` 是固定字面量集合，UI 从**文字**（`已完成`、`运行中`、`未触发` …）与
符号（`✓`、`◐`、`–`）读取状态，颜色只是辅助——不从颜色推断业务状态。

## 4. 原始事件 → UI 事件

`streaming.StreamTranslator` 是唯一读原始流的地方，它同时是**唯一可能泄漏**的
地方，因此规则写死在代码里：

| 原始事件 | 处理 |
|---|---|
| `metadata` | 只取 `run_id`（供日志与事件字段），其余丢弃 |
| `tasks`（start） | 记录开始时间，发出 `running`；`revise_synthesis` 用 `revising`；`critic` 第二次进入时消息变成"第二次审核" |
| `tasks`（result） | 只在 `error` 非空时发出 `failed`（不伪装成功） |
| `updates` | 按节点白名单取字段：证据计数、`verdict`、问题数量、`revision_count`、`finalization_status`、最终回答正文与 `used_evidence_ids`；**其余一律不读** |
| `error` | 只取异常类型名，映射到固定安全码（`run_failed` 等），正文只写日志 |
| 其他（`values`、`messages`、`debug`…） | 直接忽略 |

时长是真实的：`tasks` 的 start/update 到达时间差（毫秒级），只在测得到时显示
（`<0.1s` 表示确实很快）；没有 per-node instrumentation 的地方不编造数字。

运行修复：`translator.finish()` 会把本次没有触发的节点标成 `未触发`（例如
审核通过时的 `revise_synthesis`），并发出最后一个 `close`。如果 `finalize` 的
update 因为流中断而缺失，`translator.recover()` 会从线程 checkpoint 的真实
`final_result` 补一条 `answer`——回答仍然只出现一次（`answer_emitted` 保证）。

## 5. 证据呈现（Theory Evidence）

- 数据来自 `evidence` 节点 update 的 `evidence_by_school`，按学派分组
  （弗洛伊德 / 客体关系 / 拉康）；同一 chunk 被多个学派检索到时按 id 去重。
- 每张卡片只包含：`evidence_id`、学派、作者 / 作品 / 年份 / 页码 / 章节
  （索引里真的有时才有）、一段 220 字符的 excerpt、以及可展开的更长片段
  （≤900 字符）。
- `source_path` **从不进入卡片**：`citations.safe_source_label()` 只保留最后一个
  路径段，且当该段本身是 `C:\Users\<name>` 这类目录名时返回 None（否则会泄漏
  账号名）。`Windows 用户目录`、`Chroma` 内部字段、embedding、`retrieval_score`
  都不会出现。
- 面板顶部固定说明："演示库 = 项目自编语料，专供「检索 → 引用」链路彩排。"

来源描述行与生产代码完全一致：`citations.describe_card()` 复刻
`react_agent.rag.citations.describe_source()` 的规则（作者、`《作品》`、年份、
`p. N` / `章节「…」`），并有单测直接比对两者输出。

## 6. 引用交互（点击 `[E1]`）

- 映射来自**本轮真实数据**：`final_result.used_evidence_ids` 的顺序就是
  `[E1]…[En]` 的顺序（finalizer 用同一顺序渲染来源列表）。前端不解析回答文本，
  也不猜来源。
- 回答里的引用标签被渲染成 `role="button"`、带 `data-evidence="<id>"` 的元素；
  点击或按 Enter / Space 时：打开右侧抽屉（窄屏）、滚动到对应卡片、展开
  `<details>`、加 2 秒高亮。**没有任何后端请求**，因此不会重新 retrieval。
- 找不到对应 id 的标签（例如模型自己编的 `[E9]`）保持纯文本，不生成链接。
- 页面刷新后，最新一条回答的引用仍然可点：`fragments.citations_for_state()` 用
  checkpoint 里的 `final_result` + `evidence_by_school` 重建映射，同时证据面板
  也从同一份 state 渲染。更早轮次的引用保持纯文本（旧轮次的 evidence 已被覆盖，
  不做假映射）。

## 7. 最终回答与安全渲染

- 气泡只展示 `final_result.final_response` 的正文，不展示任何原始 JSON。
- 渲染在 `render.py`：先按块拆分（段落、`-`/`*` 列表、数字列表、`##` 标题），
  再到行内（`**强调**`、引用标签）。所有文本都作为 FastHTML 节点的子节点输出，
  **由 FT 负责转义**；网页层从不拼接 HTML 字符串，也不使用 `NotStr`。
  单测覆盖 `<script>` / `<img onerror=...>` 注入。
- 回答旁展示三个理论视角标签；某个 specialist 没有完成时显示
  `拉康（未完成）`，不伪装成功。
- 质量行来自 `finalization_status`：`质量审核：通过` / `修订一次后通过` /
  `未通过，采用保守回答`。Critic 的问题全文**不展示**。

## 8. 修订路径与安全兜底

- `revise_synthesis` 行初始就是 `未触发`（可选节点），触发时变成 `修订中`，
  完成后 `已完成`；Critic 第二次运行时消息是"第二次审核：审核通过"。
- 事件日志按发生顺序追加，因此动态路径看得见：
  `质量审核 → 发现需要修订的问题 → 修订综合回答 → 第二次审核 → 完成`。
- 进入 `safe_finalize` 时页面显示："质量检查未能通过，系统采用了更保守的回答。"
  不显示内部失败理由、不显示 stack trace、不声称"心理学错误"。

## 9. 错误处理与重试

- 失败路径只有一句话："本次分析未完成，请重试。"加一行说明（可以修改后重新发送，
  系统不会自动重放本轮）。异常类型只用于选择安全错误码，正文写到服务器日志。
- `close` 事件总会到达（异常路径也会），它负责恢复输入区。如果浏览器在 run
  结束前重连，服务器发现没有 pending 消息会立刻回一个 `close`，输入区不会被
  永久禁用。
- 自动 Retry **没有实现**：重放一条消息会在 checkpoint 里留下重复的用户轮次，
  而"不要重写历史"优先于"少按一次键"。这是刻意的取舍，写进 README 的已知限制。

## 10. 布局、响应式与可访问性

- 桌面三栏：会话（260px）/ 聊天（自适应）/ 流程 + 依据（380px）。
- ≤1180px：右栏变成抽屉（顶栏"流程 / 依据"按钮，Esc 关闭，运行开始时自动打开、
  结束时自动收起，避免遮挡输入框）。
- ≤860px：会话栏折叠为抽屉，聊天优先。
- 可访问性：所有按钮有 `aria-label`；状态同时有文字和符号；`#workflow-log` 是
  `role="log" aria-live="polite"`，`#chatlist` 与 `#run-status` 也是 live region；
  `:focus-visible` 有可见轮廓；`prefers-reduced-motion` 时关闭动画。
- 输入区：Enter 发送 / Shift+Enter 换行；运行期间 textarea 与按钮都 `disabled`，
  完成后由 `close` 事件替换为可用版本。

## 11. Evaluation 页面

- 只读 `docs/evaluation_summary.json`（由
  `scripts/export_evaluation_summary.py` 从 Phase 8 的
  `data/evals/full/report_rescored.json` 生成，只含聚合数字与固定说明）。
- 展示 Phase 8 实际存在的指标：Variant、平均 LLM 调用、平均延迟、平均 token、
  Theory Differentiation、Observation Fidelity、Overinterpretation Control、
  Clinical Boundary、Citation Validity、Citation Support、临床拒绝率、Critic
  通过率 / 修订率 / 兜底率。**N/A 不打折成 0**。
- 页面固定声明这些指标测的是 Agent 工程行为，不代表心理诊断有效性、精神分析
  理论的医学有效性或临床治疗效果；并声明 RAG 评测使用项目自编测试语料。
- 文件不存在时页面说明如何生成，而不是显示数字。

## 12. 安全清单（都有单测）

| 项目 | 处理 |
|---|---|
| `DEEPSEEK_API_KEY` / 任何 API key | 从不进入事件或页面；错误文本只含固定中文句子 |
| 绝对路径（`C:\Users\...`） | `source_path` 不进入卡片；来源标签只保留安全的文件名 |
| `draft_result` / `critique` / `supervisor_plan` / `specialist_results` | 适配器不读取这些字段的正文；测试用带这些字段的载荷断言输出不含其内容 |
| 原始 tool-call 参数 / prompt | 不请求 `messages` 事件；日志里也没有 |
| 异常 repr / traceback | 只写服务器日志；浏览器只有安全码对应的一句话 |
| 用户输入 | 作为 FT 文本节点输出，自动转义 |

## 13. 已知限制与技术债

1. ~~检索阻塞事件循环~~（Phase 10A 已修复）：evidence 节点改为
   `await asyncio.to_thread(retrieve_evidence, ...)`，dev server 不再需要
   `--allow-blocking`；真实运行已验证（见第 14 节）。
2. **进程内 pending 注册表**：多进程/多副本部署时需要换成共享存储；Phase 10A
   明确写入 `WEB_REPLICAS=1` 约束，并在 `docs/SECURITY_DEPLOYMENT.md` 说明原因
   （注册表、速率限制器都是进程内存）。
3. **流程面板不回溯**：刷新后看到的是当前会话与最新回答（含引用与证据），
   运行中的步骤日志不重建。
4. **无取消按钮**：运行中不可中断；输入区只是被禁用。
5. **一次一条消息**：同一会话同时只允许一个 run，第二次提交在服务端被 429 拒绝
   （Phase 10A 起带中文提示气泡）。
6. ~~CDN 资源~~（Phase 10A 已修复）：htmx 2.0.7 与 htmx-ext-sse 2.2.1 已
   用 `scripts/fetch_static_assets.py` 下载到 `src/react_agent/static/`，由
   `/static/{filename}` 白名单路由提供；Google Fonts 已移除。离线环境下
   UI 仍完整可用（只有 htmx 仍在用 Tailwind 之外的自定义 CSS）。
7. **excerpt 保留原文**：测试语料里本身带 `**强调**` 标记，卡片按原文展示，
   不做 Markdown 解析（保持“引自索引”的诚实性）。
8. **历史轮次引用不可点**：只有最新一轮能映射到证据面板（面板展示的也是最新一轮
   的 evidence）。
9. **会话 cookie 不是账号**：`user_id` 是 `HttpOnly; SameSite=lax` 的随机 uuid，
   服务端用它做会话归属与限流；清空 cookie 会失去自己会话的访问权（不会删除数据），
   同 token 的 API 调用仍可列出全部线程。

## 14. 验证记录（Phase 9 与 Phase 10A）

- 单测：`tests/unit_tests/test_web_streaming.py`（适配器、泄漏、修订、兜底、
  失败、恢复）、`test_web_presentation.py`（转义、引用、面板、评估读取）、
  `test_web_routes.py`（路由 + SSE 集成，含正常 / 修订 / 兜底 / 失败序列）、
  `test_web_guardrails.py`（/health、静态白名单、413/429、跨会话 404）、
  `test_web_operations.py`（健康报告、输入/速率/并发限制、归属、auth、日志）。
- 真实服务器：`uv run langgraph dev --no-reload --allow-blocking`，三个用例
  （`我梦见水。` / `从三个精神分析视角分析《哈姆雷特》。` / 诊断请求）在浏览器里
  逐步核对：流程逐节点更新（含三个并行 specialist 各自的状态与真实耗时）、
  Evidence 面板按学派出现、回答只出现一次、引用可点击、SSE 自动关闭、输入框恢复。
- 该轮真实验证还发现并修复了两个只会在浏览器里暴露的缺陷：发送响应同时把第二个
  输入区/第二个 SSE hub 追加进 DOM（重复 id），以及窄屏下抽屉遮挡输入区。
- Phase 10A：去掉 `--allow-blocking` 后完整跑通一次真实回答（9 个节点、6-8 次
  DeepSeek 调用、73.1 s，`blocking errors: 0`），证据节点日志
  `available=True elapsed=24.152s`（首次硬盘冷加载）。

## 15. Phase 10A 新增的运行时行为

| 变化 | 位置 | 效果 |
|---|---|---|
| 事件循环不再被检索阻塞 | `agents/evidence.py` | 去掉 `--allow-blocking` 依赖 |
| `/health` | `web/routes.py` + `web/health.py` | 4 个安全字段，供容器 healthcheck 与负载均衡使用 |
| 输入上限 4000 字符 | `web/limits.py` + 路由 | `413` + 中文提示气泡 |
| 速率限制（每分钟 6 / 每小时 40） | `web/limits.py` | `429`，会话维度滑动窗口 |
| 并发上限（每会话 1 条） | `web/limits.py` + `pending.py` | 第二次提交 `429`，队列不被覆盖 |
| 会话归属 | `web/access.py` + `ensure_thread` | 他人会话返回 404，不确认存在性 |
| 会话 cookie | `web/routes.py:session_cookie` | 直接打开会话页也会拿到 cookie |
| 共享令牌 | `auth.py` + `LANGGRAPH_DEMO_API_TOKEN` | 设置后 API 需 `Bearer`；Web 内部调用自动转发 |
| 本地静态资源 | `static/` + `/static/{filename}` | 白名单两个文件，无 CDN |
| 结构化日志 | `logging_config.py` | `LOG_LEVEL`，线程 id 只打前 8 位，第三方库降噪 |
