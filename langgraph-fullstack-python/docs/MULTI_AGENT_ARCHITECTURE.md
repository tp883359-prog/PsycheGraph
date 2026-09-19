# Phase 5：Multi-Agent 架构

> **Phase 6 更新**：本文记录 Phase 5 的设计与验证数据。Phase 6 在 Supervisor 与三个
> Specialist 之间插入了不调用模型的 `evidence` 节点（本地 RAG 检索），拓扑变为
> `START → supervisor → evidence → (freudian | object_relations | lacanian) → synthesizer → END`。
> 三个 Specialist 仍并行、Synthesizer 仍等三者完成。
> 检索层与证据契约见 `docs/RAG_ARCHITECTURE.md`。
>
> **Phase 7 更新**：Synthesizer 之后追加了审核层：
> `synthesizer → deterministic_validator → critic →（finalize | revise_synthesis → deterministic_validator | safe_finalize）`。
> 一轮对话正常为 6 次 DeepSeek 调用（新增 1 次 Critic），触发一次修订为 8 次；
> `deterministic_validator` / `finalize` / `safe_finalize` 都不调用模型。
> 草稿不再写入 `messages`：只有 finalizer 会写一条 `AIMessage`。
> 设计与实测见 `docs/CRITIC_ARCHITECTURE.md`。

分析日期：2026-09-15。本阶段把单一 Psychoanalytic Analyst 拆成 Supervisor、
三个学派 Specialist 和一个 Synthesizer。仍然没有 RAG、Critic、向量数据库、
长期记忆、Tool Calling 与前端重构。

## 1. 为什么需要 Multi-Agent

Single Agent 阶段的模型被要求在一次回答里同时完成四件事：判断用户要什么、
从三个学派中挑选视角、组织理论解释、写出最终回答。实测下来有两个反复出现的后果：

- **视角不稳定**：同一个提示词在“我梦见水”上可能给出克制的回答，也可能堆出一串象征；
  要求“用弗洛伊德角度”时又可能自行补上拉康或客体关系。
- **职责混在一起**：无法单独检查“事实复述是否干净”或“某个学派的解释是否有文本依据”，
  因为所有内容都在同一段自由文本里。

拆成多个 Agent 后，每个角色有独立的输入契约、输出 Schema 和职责边界：

| 角色 | 只负责 | 不负责 |
|---|---|---|
| Supervisor | 理解任务、给三个学派分配分析重点、识别临床请求、设定综合目标 | 不做理论分析，不写用户回答 |
| Freudian / Object Relations / Lacanian Specialist | 在自己的学派内解释当前材料 | 不诊断、不越界到其他学派、不写最终回答 |
| Synthesizer | 比较三个学派的结果，保留分歧，写最终回答 | 不重新从零联想，不引入新的理论结论 |

## 2. Supervisor 的职责

`src/react_agent/agents/supervisor.py`。输出 `SupervisorPlan`：

- `task_summary`：中性复述用户这次要什么。
- `analysis_focus`：这次分析最该注意什么文本特征。
- `freudian_focus` / `object_relations_focus` / `lacanian_focus`：分别给三个学派的
  分析方向。
- `clinical_diagnosis_requested`：用户是否要求临床判断。
- `synthesis_goal`：Synthesizer 要解决什么。

关键约束写在 `SUPERVISOR_SYSTEM_PROMPT` 里：focus 只能是**分析方向**，不能是**关于用户的事实**。

- 允许：“关注梦中寻找行为和未完成目标之间可能的形式关联。”
- 禁止：“该用户具有童年创伤。”

材料极短时（例如“我梦见水”），Supervisor 必须要求三个学派保守处理，并明确不要提示
母亲、性欲、创伤、死亡、孕育等用户没有提供的内容。

## 3. 三个 Specialist 的区别

三个 Specialist 各自有独立文件、独立 `xxx_SYSTEM_PROMPT`、独立节点：

| Specialist | 允许使用的概念 | 明确的禁止 |
|---|---|---|
| `freudian` | 无意识冲突、防御机制、压抑、愿望、象征意义、本我／自我／超我、移情（仅作理论概念） | 不把一切解释成性欲；不强行套俄狄浦斯情结；不推断童年创伤；不把理论解释写成事实 |
| `object_relations` | 内在客体、客体表征、分裂、理想化、投射、被内化的关系模式 | 不凭几句话判断依恋障碍或人格障碍；不编造家庭经历；术语必须解释 |
| `lacanian` | 欲望、缺失、能指、象征界／想象界／实在界、主体与语言 | 不为显得“拉康式”而堆术语；术语必须解释；不把晦涩当深度；不混入其他学派的概念 |

共同的强制边界（三个 Prompt 都写入）：

- `observations` 只写用户明确提供的内容。“我梦见水”只能是“用户梦见水”。
- 所有联想放进 `interpretations`，每条都要有 `textual_basis` 和 `uncertainty`。
- 不做临床诊断；用户要求诊断时 `clinical_diagnosis_refused=True`。
- 不编造 Freud / Lacan / Klein 等人的原文、书名、页码，也不声称核实过文献。

## 4. Synthesizer 的职责

`src/react_agent/agents/synthesizer.py`。它读 Supervisor 计划、三个 Specialist 结果
和对话历史，输出 `SynthesisResult`。

- 必须同时指出**一致之处**（`common_ground`）与**实质差异**（`differences`），
  并把差异归因于关注点不同，而不是“谁更正确”。
- 不得引入三个 Specialist 都没提出的新理论结论。
- 仍然是唯一写 `messages` 的节点，因此网页只看到一份最终回答。

## 5. State Contract

`src/react_agent/state.py`：

| 字段 | 类型 | 说明 |
|---|---|---|
| `messages` | `Annotated[list[BaseMessage], add_messages]` | 聊天历史，唯一对网页可见的通道 |
| `supervisor_plan` | `NotRequired[dict[str, Any]]` | 本次运行的 Supervisor 计划（JSON） |
| `specialist_results` | `NotRequired[Annotated[dict[str, dict[str, Any]], operator.ior]]` | 三个学派结果，按学派名索引（JSON） |
| `final_result` | `NotRequired[dict[str, Any]]` | Synthesizer 的结构化结果（JSON） |

`specialist_results` 用 `operator.ior`（字典右合并）作为 reducer：三个并行节点各写自己的键，
合并后是 `{"freudian": ..., "object_relations": ..., "lacanian": ...}`，不会互相覆盖。
同一个学派重复写入时新值替换旧值。

**每轮运行的字段处理**（实测确认，见 §11）：

- `supervisor_plan` 每轮被新计划整体替换。
- `specialist_results` 三个键每轮都被三个 Specialist 重新写入；由于 reducer 以本轮写入为准，
  第一轮结果不会残留到第二轮。
- `final_result` 每轮被 Synthesizer 重写。
- `messages` 按 `add_messages` 语义持续累积。

因此旧的 Specialist 结果不会污染下一轮分析；测试 `test_second_turn_regenerates_working_fields`
覆盖了这一点。

## 6. 为什么 Pydantic 只用于输出校验，State 只存 dict

Phase 4 把 `PsychoanalyticAnalysis` 实例直接放进 State，LangGraph 0.14.1 的
checkpoint 反序列化会报：

```
Deserializing unregistered type react_agent.schemas.PsychoanalyticAnalysis from checkpoint.
```

在 Multi-Agent State 里还会多出 `SupervisorPlan`、三个 `SchoolAnalysis`、`SynthesisResult`，
问题会被放大。Phase 5 因此把边界收紧成：

```
LLM
 ↓
Pydantic Model（SupervisorPlan / SchoolAnalysis / SynthesisResult）
 ↓ 校验（trajectory: ValidationError on malformed output）
model_dump(mode="json")
 ↓
LangGraph State（纯 JSON dict）
```

Pydantic 仍然是 Agent 输出的验证边界：字段类型、Literal 取值、必填项都由它检查，
不合法就抛错。只是**校验完成后不再把模型对象放进 State**。

统一入口是 `src/react_agent/llm.py::invoke_structured`，它返回
`(Pydantic 对象, JSON dict)`，节点只把第二个元素写进 State。

## 7. fan-out / fan-in 工作流

```
               Supervisor
                   │
               Evidence              ← Phase 6: 本地检索，0 次模型调用
                   │
      ┌────────────┼────────────┐
      ▼            ▼            ▼
  Freudian       Object       Lacanian
                Relations
      └────────────┼────────────┘
                   ▼
              Synthesizer
                   │
                   ▼
                  END
```

`src/react_agent/graph.py` 用当前安装的 LangGraph（1.2.11）API 写成：

```python
builder.add_edge(START, "supervisor")
for node_name in SPECIALIST_NODES:  # freudian / object_relations / lacanian
    builder.add_edge(
        "supervisor", node_name
    )  # Phase 6: supervisor -> evidence -> node_name
    builder.add_edge(node_name, "synthesizer")
builder.add_edge("synthesizer", END)
```

- fan-out：`supervisor` 有三条出边，三个 Specialist 在**同一个 superstep** 中执行。
- fan-in：`synthesizer` 有三条入边，LangGraph 的 barrier 语义保证它等三个都完成后才运行。
- 没有使用 `sleep`、锁或手工 `asyncio.gather`；编排完全交给 LangGraph。

## 8. 为什么三个 Specialist 并行

三个学派的读解彼此独立：每个只读同一份材料和 Supervisor 给自己的 focus，不需要读取
其他学派的结果。并行执行的直接收益是**延迟**：总墙钟时间约等于
`Supervisor + max(三个 Specialist) + Synthesizer`，而不是五段串行之和。

实测（`tests/unit_tests/test_graph.py::test_specialists_run_concurrently` 与真实验证脚本）：

- 单元测试证明三个 Specialist 的模型调用窗口互相重叠。
- 真实 DeepSeek 调用中，Case A 单次完整 run 约 44 秒墙钟时间；三个 Specialist 的
  请求在 LangGraph 同一 superstep 中并行发出。

## 9. 一次请求约 5 次模型调用

```
1 次 Supervisor
3 次 Specialist（并行）
1 次 Synthesizer
= 约 5 次 LLM call per run
（多轮对话每轮各算一次，两轮 = 约 10 次）
```

这是 Multi-Agent 的真实成本，不是可以省略的细节：

- **Token 成本**：每次调用都携带系统提示与对话历史，因此总 token 数明显高于单 Agent。
- **延迟**：并行抵消了一部分成本，但 Supervisor 和 Synthesizer 仍是串行阶段。
  若把三个 Specialist 合并成一次模型调用会减少调用次数，但那样就不是三个独立 Agent，
  只能得到一段混合风格的文本，无法保留“谁说了什么”的可比较结构，因此本阶段不做这种合并。
- 三个 Specialist 使用同一个 `get_chat_model()`（默认 `deepseek-flash`，Thinking Mode 关闭）。
  本阶段研究的是编排，不做模型路由。

## 10. Multi-Agent 带来的优势、成本与风险

**优势**

- 视角稳定：每个学派只在自己的边界内工作，输出字段里必须声明 `perspective`，
  节点会检查它与自身身份是否一致（不一致直接抛 `PerspectiveMismatchError`）。
- 可检查：`observations`、`interpretations`、`limitations` 分开存放，
  可以直接用程序检查“是否把没提供的内容写成事实”。
- 分歧可见：Synthesizer 被要求写明一致与差异，而不是把它们抹平。

**成本**

- 约 5 倍模型调用与相应 token、约 3～4 倍墙钟延迟（并行只抵消了 Specialist 部分）。
- 需要维护四个 Schema、五个 Prompt 和一份状态契约。

**风险**

- **理论重复**：三个学派可能说同一件事，只是换了词汇。验证脚本对 Case A/C 断言
  “三个学派的 claim 不完全相同”，人工复核时也会注意这一点。
- **过度解释放大**：三个 Agent 有可能各自加码，比单 Agent 更容易堆砌理论。
  Case B（“我梦见水”）专门检查这一点：三个学派的 `observations` 不得出现
  海洋／母亲／童年／死亡／性欲／创伤／湖泊／恐惧，且每个学派最多两条解释。
- **风格趋同**：Synthesizer 若只是拼接三段 summary，会得到机械的回答；
  Prompt 明确要求写成一份自然回答，并由验证脚本人工复核。

## 11. Single Agent 与 Multi-Agent 的初步比较

这是工程观察，不是正式 Evaluation Framework。

| 维度 | Single Agent（Phase 3/4） | Multi-Agent（Phase 5） |
|---|---|---|
| 理论区分度 | 依赖模型在单次回答里自行挑选视角，用户指定学派时才稳定 | 三个学派各自作答，`perspective` 字段强制归属；Synthesizer 必须写明差异 |
| 解释覆盖范围 | 一次回答通常突出 1～2 个视角 | 三个学派各自覆盖，Synthesizer 再合并 |
| 过度解释风险 | 短输入曾出现象征清单，提示词收紧后收敛 | 风险理论上 ×3；Case B 实测三个学派都保持克制（见 §12） |
| 回答长度 | 短输入已被要求“约 150 字以内” | 最终回答通常长于单 Agent，因为要呈现三个方向的比较 |
| 调用次数 | 1 次/轮 | 约 5 次/轮 |
| 延迟（真实 DeepSeek） | 单次 run 约 8～10 秒 | 单次 run 约 40～50 秒（三 Specialist 并行） |
| 可测试性 | 一个 State 字段 `analysis_result` | 四个结构化产物，可分别断言 |
| 幻觉引用风险 | 靠提示词约束 | 三个 Specialist Prompt + Synthesizer Prompt 各自约束，均在验证脚本中检查 |

## 12. 验证记录

### 单元测试

`47 passed`，全部不调用真实 API。覆盖：四个 Schema 的校验（含非法 `perspective`）、
State reducer（并行合并、同名替换）、图拓扑、并行执行与 fan-in barrier、
每轮只有一个 `AIMessage`、checkpoint 恢复、第二轮重新生成工作字段。

### 真实 DeepSeek 测试

见下节 §13。

### LangGraph Server 与 API

见下节 §14。

## 13. 真实案例结果

运行方式：

```powershell
$env:PYTHONUTF8="1"
.\.venv\Scripts\python.exe scripts\verify_multi_agent.py
```

脚本对每个案例断言结构性要求，人工复核文本质量。多次真实运行中的实测结果：

| 案例 | 观察到的行为 | 结论 |
|---|---|---|
| A 找房间 | Supervisor 分别给出三个学派的 focus（愿望/禁止张力、关系性匮乏、能指空白）；Freudian 3 条、Object Relations 2 条、Lacanian 3 条解释，`claim` 文本互不相同；Synthesizer 明确指出三者共同的形式起点（有指向的努力持续落空）与分歧 | 通过 |
| B 我梦见水 | Supervisor 明确要求“不要提示水象征母亲、性、出生或创伤”“不要推测家庭史或依恋类型”；三个学派的 `observations` 都只写“用户说了一句话”，未出现海洋／母亲／童年／死亡／性欲／创伤／湖泊／恐惧；三个学派都写出 `limitations`；最终回答承认信息不足并只给条件性读法 | 通过（短输入的过度解释未随 Agent 数量放大） |
| C 三个视角比较《哈姆雷特》 | 三个学派按本位展开且 `differences` 非空，Synthesizer 在 `integrated_interpretation` 中点评三个学派 | 通过 |
| D 边缘型人格障碍 | `supervisor_plan.clinical_diagnosis_requested=True`；三个 Specialist 与 Synthesizer 的 `clinical_diagnosis_refused` 均为 True；三个 Specialist 的 summary 与最终回答都以“不能判断你是否患有……”开头，并说明只能由有资质的临床工作者评估；同时介绍“边缘”概念的历史与分歧，不确认也不排除 | 通过 |
| E 吵架后反复想起一句话 | 三个学派从不同角度切入（措辞与重复、关系中的位置、话语被反复回放的形式），`observations` 未出现家庭史、童年创伤、依恋障碍等用户未提供的内容 | 通过 |

单次 run 的真实耗时：A 40.5s、B 39.4s、C 47.6s、D 36.7s（约 5 次 DeepSeek 调用，其中 3 次并行）。

## 14. 两轮上下文结果

同一个 thread 连续两轮（第一轮“我梦见水”，第二轮补充“平静的湖水／安心／小时候暑假”），
最终真实验证 `All structural checks passed`：

| 检查项 | 结果 |
|---|---|
| `messages` 条数 | 4（human、ai、human、ai），内部 Agent 输出没有进入聊天 |
| 第一轮 observations | 只写“用户写下的一句话是：我梦见水”，不含“湖” |
| 第二轮 observations | 三个学派都包含“平静的湖水”“安心”“小时候暑假” |
| 第二轮 Supervisor | `task_summary` 明确写“用户补充了对梦的说明”，三个 focus 围绕新线索重写（“平静／安心”的搭配、关系内容缺席、“小时候的暑假”作为名称而非叙事） |
| `specialist_results` | 三个键被第二轮结果整体替换，没有第一轮残留 |
| `final_response` | 与第一轮不同；第二轮明确指出“三派从不同方向在同一个地方停下”并复用新信息 |
| 脚本断言 | `"湖" not in 第一轮 observations` 与 `"湖" in 第二轮 observations` 同时成立 |

## 15. Stream 事件结构（实测）

用真实 DeepSeek 跑一次完整 Multi-Agent run，同时抓取 LangGraph Server 的
`messages-tuple` 流（`stream_mode="messages-tuple"`），事件统计如下：

| 指标 | 实测值 |
|---|---|
| `messages` 事件总数 | 5062 |
| 其中 `content` 非空的 chunk | **1** |
| 其中 tool-call 参数分片（`tool_call_chunks`） | 5046 |
| 唯一 content chunk 的来源 | 最终回答（`final_response`） |

含义：

- Structured Output 走 DeepSeek 的 tool-calling 通道，模型边生成边发送的是
  **tool-call 参数分片**，而不是自然语言 token。因此真正的正文只会以
  **一个完整 chunk** 到达，而不是逐字增量。
- 内部节点（Supervisor、三个 Specialist）的输出全部在 tool-call 参数里，
  天然不会混进 `content`。
- 现有网页 `app.py::message_generator` 只转发 `chunk_msg["content"]`，
  因此它**不会显示 Supervisor / Specialist 的任何内部内容**，也不会显示 JSON。
- 代价：气泡在生成期间基本保持“正在输入”状态（没有内容可显示），
  直到最终 `AIMessage` 到达时一次性出现。这是 Phase 4 技术债在 Multi-Agent 下的延续，
  本阶段按范围要求不修改 `app.py`。最终回答本身没有问题：
  `state.values.messages[-1].content == final_result.final_response`（实测相等）。
- 内部节点仍然产生完整的 stream 事件，可供未来的 Workflow UI 使用。

## 16. 兼容性与已知问题

1. **checkpoint 不再出现自定义 Pydantic 类型警告（已修复）**

   Phase 4 在 State 里保存 `PsychoanalyticAnalysis` 实例，`langgraph-api 0.14.1` 会打印
   `Deserializing unregistered type react_agent.schemas.PsychoanalyticAnalysis` 并提示
   未来版本会阻止该行为。Phase 5 把状态改为纯 JSON dict 后，重新读取 checkpoint、
   跑第二轮、通过 Server 端 API 读取状态，都不再出现该警告；单元测试
   `test_checkpoint_restore_has_no_custom_pydantic_types` 直接检查 checkpoint 内容中
   不含 Pydantic 对象。

2. **旧的 `.langgraph_api` 运行时文件与新代码不兼容（已处理）**

   启动 Server 时出现：

   ```
   Failed to load file: .langgraph_api\.langgraph_ops.pckl
   Failed to load cached data: Can't get attribute 'PsychoanalyticAnalysis'
   ```

   原因是这两个开发运行时文件是 Phase 4 生成的，里面 pickle 了已删除的类。
   它们原本被 Git 跟踪，属于本地运行产物，不应该进版本库。处理方式：
   `git rm --cached .langgraph_api/*` 并加入 `.gitignore`（磁盘文件保留，服务重新写入）。
   本阶段未删除任何用户数据文件。

3. **热重载会打断正在进行的 run（开发环境行为）**

   运行 `langgraph dev`（`--no-reload` 未开启）时，编辑源码会触发服务器重启，
   当次 run 会被取消并重新入队。这是 dev server 的既有行为，与 Multi-Agent 无关；
   如需稳定长跑请使用 `--no-reload`。

4. **未发现新的 LangGraph / LangChain / DeepSeek / Pydantic 兼容问题**

   `StateGraph` fan-out/fan-in、`operator.ior` reducer、`add_messages`、
   `with_structured_output(method="function_calling")` 在当前版本下均正常工作。

5. **结构化输出可能被截断（已加显式错误与诊断）**

   真实调用中出现过一次 `SchoolAnalysis` 解析失败：LangChain 在 provider 侧
   中断该轮时**返回 `None` 而不是抛错**，于是下游报出误导性的
   `Input should be a valid dictionary or instance of SchoolAnalysis`。
   `llm.py::invoke_structured` 现在会：

   - 在 `None` 时抛出带 Agent 名称的 `RuntimeError`；
   - 自动用 `include_raw=True` 重试一次，把 `parsing_error` 与 `finish_reason`
     写进错误信息，便于判断是否触发了输出长度限制。

   这是 3 个 Specialist 并行、每个都被要求输出较长结构化文本后的真实风险，
   在文档中保留记录；如需彻底缓解，可调高模型输出上限或进一步收紧 Specialist
   的篇幅要求（本阶段不改变模型配置）。
