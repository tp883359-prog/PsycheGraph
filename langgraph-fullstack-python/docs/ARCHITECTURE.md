# 当前项目架构与 PsycheGraph 改造准备

## 1. 阅读范围与结论

分析日期：2026-09-15。实际项目根目录是 `C:/Users/DJW16/Desktop/PsycheGraph/langgraph-fullstack-python`；上层 `PsycheGraph/启动项.txt` 也指向这里。因此本文位于该项目的 `docs/ARCHITECTURE.md`。

当前系统是 **FastHTML 网页 + LangGraph API 服务 + 预构建 ReAct 工厂生成的单节点 StateGraph + Anthropic 聊天模型**。Graph 配置了 `tools=[]`，实际执行是一次模型响应，没有工具循环，也没有多个智能体。

本文以项目业务源码、配置、测试和本地 `.venv/Lib/site-packages/` 中已安装框架源码为依据。下文以 `依赖/` 代指该 site-packages 目录，便于区分项目代码和框架内部代码。没有安装依赖、调用模型、修改业务代码或启动额外服务。用户已确认服务可以启动；本次未进行浏览器和模型请求的端到端验证。

## 2. 当前目录结构

```text
PsycheGraph/
├── 启动项.txt                    # 进入实际项目、设置 UTF-8、启动开发服务
└── langgraph-fullstack-python/   # 实际项目根目录
    ├── langgraph.json           # Graph、HTTP app、auth 和环境文件注册
    ├── pyproject.toml           # Python 项目、依赖、构建、检查配置
    ├── uv.lock                  # 依赖解析锁文件
    ├── README.md                # 模板介绍与启动说明，含旧配置注释
    ├── LICENSE                  # 许可文件
    ├── Makefile                 # 测试、格式化、静态检查命令
    ├── .env                     # 本地环境变量；本文不记录密钥值
    ├── .env.example             # Provider、Tavily、LangSmith 配置示例
    ├── .gitignore
    ├── .codespellignore
    ├── .sesskey                 # 本地会话相关文件；生成来源未确认
    ├── .github/workflows/
    │   ├── unit-tests.yml       # CI：检查、拼写、单元测试
    │   └── integration-tests.yml # 定时/手动集成测试
    ├── src/react_agent/
    │   ├── __init__.py
    │   ├── graph.py
    │   ├── app.py
    │   └── auth.py
    ├── tests/
    │   ├── unit_tests/{__init__.py,test_configuration.py}
    │   ├── integration_tests/{__init__.py,test_graph.py}
    │   └── cassettes/103fe67e-a040-4e4e-aadb-b20a7057f904.yaml
    ├── react_agent.egg-info/    # 安装产生的包元数据、依赖和文件清单
    ├── .venv/                  # 已安装 Python 环境，非业务模块
    ├── .langgraph_api/          # 开发运行时数据
    │   ├── .langgraph_checkpoint.{1,2,3}.pckl
    │   ├── .langgraph_ops.pckl
    │   ├── .langgraph_retry_counter.pckl
    │   ├── store.pckl
    │   └── store.vectors.pckl
    └── docs/ARCHITECTURE.md     # 本次新增
```

目录清点包含隐藏配置和运行时目录；依赖环境、缓存和二进制保存文件不按业务源码逐文件解读。未读取密钥值或反序列化用户会话。源码目录没有 `state.py`、`tools.py`、`configuration.py`、`prompts.py` 或单独的前端工程。

### 配置和辅助文件

- `langgraph.json`：`dependencies: ["."]` 指向当前 Python 包；`graphs.agent` 注册 `./src/react_agent/graph.py:graph`；`env` 指向 `.env`；`auth.path` 注册 `auth.py:auth`；`http.app` 注册 `app.py:app`。`$schema` 用于配置结构描述。这里没有模型名称配置。
- `pyproject.toml`：项目名 `react-agent`、版本 `0.0.1`、Python `>=3.11`；声明 LangGraph、LangChain、Anthropic/OpenAI/Fireworks 适配器、FastHTML、SDK 等依赖；使用 setuptools 构建，同一源码目录映射到 `react_agent` 和 `langgraph.templates.react_agent`；配置 Ruff，开发依赖包含 API、CLI 和 pytest。安装某个 Provider 或 Tavily 包不等于业务已经使用它。
- `uv.lock`：锁定具体解析版本；本次只读。记录包括 LangGraph `1.2.11`、prebuilt `1.1.0`、API `0.14.1`、SDK `0.4.4`、LangChain `1.4.0`、Anthropic 适配器 `1.7.2`。这些是锁文件记录，不作为正在运行进程版本的证明。
- `README.md`：启动和单部署架构介绍有参考价值，但底部注释中的 `model` schema、Sonnet 默认值及工具示例不代表当前执行配置。
- `tests/unit_tests/test_configuration.py::test_graph` 只有 `pass`；文件名不表示配置模块已经存在。
- `tests/integration_tests/test_graph.py::test_react_agent_simple_passthrough` 直接 `await graph.ainvoke({"messages": [("user", "Hi there!")]})`，没有结果断言，也不覆盖 HTTP、SSE 或会话隔离。本次未运行会调用外部模型的测试。
- cassette 记录了历史 Sonnet/search 请求，与当前 Haiku/空工具源码不同，不能据此认定当前存在搜索工具。

## 3. src/react_agent/ 主要文件

| 文件 | 主要符号与职责 |
|---|---|
| `__init__.py` | 导入并导出 `graph`，`__all__ = ["graph"]`。导入该包会触发 `graph.py` 初始化。文档字符串中的工具循环描述是模板说明。 |
| `graph.py:5` | 模块变量 `graph = create_react_agent(...)`，同时完成模型初始化和 Graph 构建。模型字符串、工具注册参数、系统提示均在这里。 |
| `app.py:36` | `langgraph_client = get_client()` 创建异步 SDK 客户端。 |
| `app.py:277` | `app = FastHTML(...)` 创建网页应用；同文件嵌入 CSS/CDN 引用、JavaScript 和 HTML 组件。这里的前端不是 React JavaScript 应用。 |
| `auth.py` | `auth = Auth()`；`authenticate(authorization: str)` 无条件返回 `"default_user"`，提供放行式认证钩子。 |

### app.py 函数索引

| 函数 | 位置 | 作用 |
|---|---|---|
| `get_user_id` | 280 | 读取 `user_id` cookie，没有则生成 UUID。 |
| `ChatMessage` | 291 | 根据 `msg["type"]` 和 `msg["content"]` 渲染消息气泡。 |
| `ChatInputBubble` | 334 | 生成可编辑输入区、隐藏 `msg` 字段和 HTMX 提交表单。 |
| `ConversationList` | 391 | `threads.search(metadata={"user_id": user_id}, limit=50, offset=0)` 列出侧栏会话。 |
| `root` | 431 | `/` 生成 thread UUID，设置 cookie，302 跳转到会话页。 |
| `conversation` | 441 | 创建/复用 thread，读取 State 历史并生成网页。 |
| `new_thread` | 509 | `/new-thread` 生成新 UUID、设置 cookie、跳转。 |
| `AssistantMessagePlaceholder` | 521 | 创建助手占位气泡，并声明 SSE 订阅地址及 DOM 更新目标。 |
| `send_message` | 564 | 解析表单，创建 LangGraph run，返回用户气泡和助手占位 HTML。 |
| `message_generator` | 589 | 订阅 SDK run 事件，转换为浏览器 SSE 字符串。 |
| `get_message` | 608 | 返回包装异步生成器的 `StreamingResponse`。 |

前端服务与 Agent 通过 `thread_id`、`run_id`、`assistant_id="agent"` 和 `messages` 数据契约连接。`app.py` 不直接导入或调用业务 `graph`，也不创建 LLM。

## 4. Graph、Agent、State、LLM、Tool 与 Prompt 入口

### Graph 和 Agent

`langgraph.json → graphs.agent → src/react_agent/graph.py:graph` 是服务加载入口。`graph.py` 的 `create_react_agent` 是 Agent 创建工厂，返回编译后的 Graph；项目没有自定义 Agent 类。

本地 `依赖/langgraph/prebuilt/chat_agent_executor.py::create_react_agent` 在 787 行附近检查 `tool_calling_enabled`。当前为空工具，因此创建 `StateGraph`，增加 `agent` 节点，节点包装 `RunnableCallable(call_model, acall_model)`，设置入口后 `workflow.compile(...)`。语义拓扑为 `START → agent → END`；本分支靠没有后继节点结束，不是项目手写了 `add_edge("agent", END)`。

### LLM 初始化、调用与模型配置

唯一业务模型入口是 `graph.py:6` 的字符串：

```python
"anthropic:claude-3-5-haiku-latest"
```

初始化链：

```text
create_react_agent(model=上述字符串)
→ langchain.chat_models.init_chat_model(model)
→ langchain/chat_models/base.py::_parse_model
→ _init_chat_model_helper / Provider 创建器
→ langchain_anthropic.chat_models.ChatAnthropic
```

本地 `base.py` 的 Provider 注册表将 `anthropic` 映射到 `ChatAnthropic`，`_parse_model` 拆分冒号前的 Provider 和冒号后的模型名。`ChatAnthropic` 支持从 `ANTHROPIC_API_KEY` 读取凭据，`.env.example` 和本地 `.env` 的键名也包含它。密钥是否有效、运行进程是否有额外环境覆盖、实际外发端点及 `latest` 对应的服务端具体版本均**未确认**：本次没有读取进程配置或实际调用模型。

模型名称目前通过 Python 工厂的位置参数传入，属于源码硬编码；网页 `runs.create` 没有传入模型配置，也没有 `configurable.model` 读取逻辑。OpenAI、Fireworks 的依赖和示例环境变量不改变当前 Anthropic 选择。

调用链在 `chat_agent_executor.py`：`_get_prompt_runnable(prompt) | model` 构成 `static_model`；`acall_model(state, runtime, config)` 调用 `await static_model.ainvoke(model_input, config)`，同步路径是 `call_model` 内的 `.invoke(...)`。结果为 `AIMessage`，返回 `{"messages": [response]}`。Provider 的外部请求由 `ChatAnthropic` 内部 `_agenerate` / `_astream` 等方法处理，业务文件没有手写 HTTP 模型请求。

### State

业务没有自定义 State。`create_react_agent` 未收到 `state_schema` 和 `response_format`，默认选择 `依赖/langgraph/prebuilt/chat_agent_executor.py:57::AgentState`：

```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    remaining_steps: NotRequired[RemainingSteps]
```

`messages` 由 `add_messages` 合并新旧消息（同 ID 可更新），不是每次请求简单覆盖；`remaining_steps` 是框架管理的剩余执行步数。HTTP 输入的 human 消息转换为消息对象参与执行；模型输出的 `AIMessage` 合并回同一 State。网页 `conversation()` 中的 `state = await threads.get_state(...)` 是 API 返回对象，实际图数据从 `state["values"]` 中读取。

### Tool

当前没有定义业务 Tool。唯一注册位置是 `graph.py:7` 的 `tools=[]`。框架工厂具备 `ToolNode`、`model.bind_tools(...)`、`should_continue` 等能力，但当前没有实际工具节点和调用。

将来注册工具后，预构建分支可按 `AIMessage.tool_calls → ToolNode → ToolMessage → agent` 循环；相关框架源码位于 `chat_agent_executor.py` 和 `langgraph/prebuilt/tool_node.py`。这是扩展路径，不是当前发生的数据流。`TAVILY_API_KEY`、`tavily-python` 和 cassette 都不能作为业务注册工具的证据。

### Prompt

系统提示直接位于 `graph.py:8`：`You are a friendly, curious, geeky AI.`。框架 `_get_prompt_runnable` 将字符串包装成 `SystemMessage`，在每次模型输入前加到 State 消息列表前面；它并非项目显式保存到历史 State 的普通聊天消息。当前没有精神分析理论提示词、角色模板或 Prompt 配置模块。

## 5. 一次完整网页请求的数据流

### 5.1 会话准备

1. 浏览器访问 `/`，进入 `app.py::root(request)`：生成 `thread_id`、调用 `get_user_id`、设置 `user_id` cookie，跳转 `/conversations/{thread_id}`。
2. `conversation(thread_id, request)` 调用 `langgraph_client.threads.create(thread_id=thread_id, if_exists="do_nothing", metadata={"user_id": user_id})`。
3. `threads.get_state(thread_id)` 获取历史；`values = state["values"]`，再读 `messages`。读取异常统一退回 `messages=[]`，因此空页面不一定证明没有历史。
4. `ChatMessage` 展示历史，`ChatInputBubble` 创建输入表单，`ConversationList` 渲染侧栏。

### 5.2 提交、运行与回显

| 步骤 | 文件 / 函数 / 类 | 关键变量与动作 |
|---|---|---|
| 1. 用户输入 | `app.py::ChatInputBubble`、内嵌 `setupChatInput` 与提交事件 | 文本在 `#msg-input-div`，同步到隐藏 `Input(name="msg", id="msg-input")`。 |
| 2. 浏览器 HTTP | 表单 `hx_post` | POST `/conversations/{thread_id}/send-message`，表单字段 `msg`。 |
| 3. 网页后端 | `app.py::send_message` | `form_data = await request.form()`；取 `msg`，拒绝空白内容，构造用户气泡 `user_msg_div`。 |
| 4. 创建运行 | `langgraph_client.runs.create` | 传入 `thread_id`、`assistant_id="agent"`、`input={"messages": [{"type": "human", "content": msg}]}`、`stream_mode="messages-tuple"`。 |
| 5. SDK/API | `依赖/langgraph_sdk/_async/runs.py::RunsClient.create` | HTTP 语义为 POST `/threads/{thread_id}/runs`；返回 run 元数据而非最终回答。 |
| 6. 服务接收 | `依赖/langgraph_api/api/runs.py::create_run` | 从 `request.path_params` 取 thread，从 `RunCreateStateful` 解析 payload，调用 `create_valid_run` 创建运行记录。 |
| 7. 后台执行 | `依赖/langgraph_api/worker.py::worker`、`stream.py::astream_state` | worker 消费运行；`astream_state` 通过 `get_graph` 取得目标 Graph，并提供服务 checkpoint，驱动图的流式执行。 |
| 8. State 输入 | `AgentState`、`add_messages` | 同一 thread 已保存状态与本次 human 消息合并，交给 `agent` 节点。 |
| 9. Agent/LLM | `chat_agent_executor.py::acall_model` | 构造 `model_input`，系统提示加到消息前，`static_model.ainvoke(...)` 调用 `ChatAnthropic`。 |
| 10. Tool | `graph.py::tools=[]` | 当前跳过：没有 Tool 调用、ToolMessage 或二次返回 Agent 的环节。 |
| 11. 返回 Graph | `acall_model` | 返回 `{"messages": [response]}`；响应合并回 State，本次图运行完成；服务保存运行状态/checkpoint。 |
| 12. 返回占位界面 | `send_message`、`AssistantMessagePlaceholder` | `run_id = run["run_id"]`；返回用户气泡和 SSE 占位 HTML，HTMX 追加到 `#chatlist`。此步骤与后台图执行可交叠，不等待最终回答。 |
| 13. 浏览器订阅 | 占位元素 `sse_connect` | GET `/conversations/{thread_id}/get-message?run_id={run_id}`。 |
| 14. 转发流 | `app.py::get_message`、`message_generator` | `StreamingResponse(..., media_type="text/event-stream")`；`runs.join_stream(thread_id, run_id, stream_mode="messages-tuple")` 订阅运行。 |
| 15. API 流订阅 | `依赖/langgraph_sdk/_async/runs.py::RunsClient.join_stream`、`langgraph_api/api/runs.py::join_run_stream` | GET `/threads/{thread_id}/runs/{run_id}/stream`，SDK 提供 `chunk.event` / `chunk.data`。 |
| 16. 显示 | `message_generator` 与 HTMX SSE | 仅处理 `chunk.event == "messages"`，逐项取 `chunk_msg.get("content", "")`，输出 `event: message`；浏览器按 `sse_swap="message"`、`hx_swap="innerHTML"` 替换占位内容。最后生成 `event: close`。 |

### 5.3 HTTP 如何进入 LangGraph

服务从 `langgraph.json` 同时加载 Graph 和 FastHTML app。本地 `依赖/langgraph_api/server.py` 将自定义 `user_router` 路由与内置 API 路由合并，并调用 `configure_loopback_transports(app)`。

`app.py` 使用无 URL 的 `get_client()`；本地 `依赖/langgraph_sdk/_async/client.py::get_client` 为该分支配置 `httpx.AsyncClient` 和 ASGI 进程内 transport，逻辑地址是 `http://api`，`root_path="/noauth"`。因此浏览器到网页是真实 HTTP，而网页到同部署 LangGraph API 是具有 HTTP 路由语义的进程内调用，不必再次经过 localhost TCP。内部认证路径也不能当作浏览器身份验证。

```text
浏览器 ──POST 表单──> FastHTML send_message
                         │ SDK / ASGI loopback
                         ▼
                   LangGraph create_run ──> worker
                                             │
                          checkpoint <──> State → agent → ChatAnthropic
                                             │                │
                                             └── AIMessage <──┘

浏览器 <──SSE── get_message/message_generator <── join_stream <── run 事件
```

流式事件可以在模型生成过程中到达，不是先完成整个 Graph 才返回所有文本。最终 State 与逐段事件是两种不同输出。

### 5.4 当前流式处理需要验证的地方

源码已经接好 SSE，但没有按消息 ID 累积 token、解析内容块、区分节点、处理工具消息或显示错误的逻辑。`innerHTML` 每次替换内容；如果事件是文本增量，界面可能只显示最后一段。若 content 为 Anthropic 内容块列表，直接字符串插值也可能不能正常展示。实际事件形状和浏览器表现**未确认**，原因是本次未提交真实模型请求。

代码发出 `event: close`，占位元素没有显式 `sse_close` 配置；连接结束/重连表现也未实测。后续多节点输出需要先统一事件解析，再区分角色进度和最终综合结果。

## 6. Agent 架构分类

| 分类 | 判断 | 源码依据 |
|---|---|---|
| 普通 LLM Chain | 单次行为类似聊天链，但整体载体是 Graph | `static_model = prompt_runnable | model`；外层工厂编译 StateGraph。 |
| ReAct Agent | 使用预构建 ReAct 工厂；当前没有 action 循环 | `from langgraph.prebuilt import create_react_agent`，`tools=[]`。 |
| Tool Calling Agent | 框架有支持，当前没有启用工具调用 | 工厂有 `bind_tools` / `ToolNode` 分支，空工具时走单节点分支。 |
| LangGraph StateGraph | 是，且已编译 | 工厂内部 `StateGraph(...)`、`add_node("agent", ...)`、`workflow.compile(...)`。 |
| 多智能体/Supervisor/RAG | 当前未实现 | 业务源码没有对应节点、共享领域 State、检索或路由逻辑。 |

这些名称分属不同层级：StateGraph 是执行与状态组织方式，ReAct/tool calling 是智能体行为模式，FastHTML 是展示层。

## 7. 能力检查

以当前项目实际接线为口径；“框架可扩展”不等于业务功能已完成。

| 能力 | 状态 | 依据与边界 |
|---|---|---|
| streaming | 已支持 | `app.py::send_message` 的 stream_mode、`message_generator` 的 `join_stream`、`get_message` 的 SSE 响应、占位元素 SSE 属性。文本增量和内容块显示正确性未确认，见 5.4。 |
| tool calling | 部分支持 | 预构建工厂支持注册和循环，但 `graph.py:7` 是空列表；当前实际工具功能未启用。 |
| structured output | 未支持 | 未传 `response_format`、没有业务结果 schema 或 `with_structured_output`。框架可扩展，不代表当前回答是结构化结果；`messages` JSON 也不是领域结构化输出。 |
| persistence / checkpoint | 部分支持 | 网页通过 thread/run 使用服务状态；`依赖/langgraph_api/stream.py::astream_state` 向 `get_graph` 注入 checkpointer；`依赖/langgraph_runtime_inmem/checkpoint.py::InMemorySaver` 用 `PersistentDict` 加载本地保存文件；`.langgraph_api` 已存在 checkpoint 文件。业务 Graph 未显式配置 saver，独立 `graph.ainvoke` 不自动获得服务持久化；生产数据库、备份和重启恢复验证未确认。 |
| conversation memory | 已支持 | `conversation` 创建/复用 thread、读取 `values.messages`；后续 run 使用同一 `thread_id`，`AgentState.messages` 经 `add_messages` 累积。范围是线程内消息历史，没有摘要、跨会话用户长期记忆或领域证据记忆。 |
| async | 已支持 | 网页路由 `async def`、SDK `await` / `async for`、框架 `acall_model`、集成测试 `graph.ainvoke`。 |
| authentication | 部分支持 | `langgraph.json::auth.path` 接入 `auth.py::auth`；`authenticate` 不验证 authorization，固定返回 default_user。具备钩子但没有有效用户认证和会话访问授权。 |

### auth.py 与 cookie 的关系

`auth.py` 的说明就是 permissive auth，用来替代原 API key 认证要求；它不解析 token，不查用户，也没有资源授权处理器。`get_user_id` 的随机 cookie 用于 thread metadata 和侧栏筛选，和 `authenticate` 返回的固定身份不是同一个体系。

`conversation`、`send_message`、`get_message` 接收路径中的 thread ID，未检查其所有者。侧栏按 metadata 搜索不能保证已知 thread ID 的访问隔离。`langgraph.json` 没有显式启用自定义路由认证；本地服务源码通过 `enable_auth_on_custom_routes` 控制该部分。当前网页自己也标明 unauthenticated demo。实际运行环境的额外中间件策略未确认。

## 8. 改造成 PsycheGraph：保留、重构、新增

以下均为未来建议，本次只创建本文档。

### 可以保留

- 单部署方式、LangGraph API、FastHTML 基础、thread/run 标识和异步 SDK 调用流程。
- 聊天页面、侧栏、新建会话、历史消息展示组件，可作为原型 UI 基础。
- `langgraph.json` 的 Graph/HTTP/auth 注册机制，以及现有 Python 打包和开发工具体系；未来包重命名再统一调整路径。
- 开发期 checkpoint 机制和 `messages` 的消息合并思路。

### 需要重构

| 当前模块 | 建议改动 | 原因 |
|---|---|---|
| `graph.py` 单工厂调用 | 显式构建外层 StateGraph，注册 Supervisor、理论分析、Evidence、Critic、Synthesizer 节点/子图 | 当前只有单节点，无法表达任务分派、结果汇合、批评修订和终止。 |
| 隐式 `AgentState` | 建立领域 State 与结果 schema | 单一 messages 无法稳定存放文本版本、出处、各学派结果、批评和最终综合。 |
| 硬编码模型和 prompt | 分离模型工厂、运行配置、角色 prompt | 让不同角色使用明确配置并可测试，避免把角色逻辑塞进网页。 |
| `app.py` | 将页面、路由和流事件转换分开；保留 thread/run API 契约 | 多节点流需要消息累积、来源标识、角色进度、证据和最终报告展示。 |
| `auth.py` | 验证身份，将用户身份传递到资源访问检查 | 当前固定身份和 metadata 筛选无法提供会话隔离。 |
| 测试 | 增加确定性路由、schema、证据引用、循环终止、SSE 和权限测试 | 当前占位单测和无断言模型调用无法验证多智能体流程。 |

### 应该新增

- `Supervisor`：生成任务计划、选择理论角色、判断是否需要证据、控制修订次数和完成条件。
- `Freudian Agent`、`Object Relations Agent`、`Lacanian Agent`：接受同一分析对象和证据契约，各自输出解释、引用及解释边界。每个角色可以是节点，也可以在确实需要工具循环时做子图。
- `Evidence / RAG`：文档加载、分段、索引、检索、出处定位；保存文献版本、页码/段落、文本片段 ID。当前没有语料库、embedding 或检索实现。
- `Critic`：核对文本证据、引用是否匹配、理论归属是否混淆、解释是否超出材料；输出可执行修订意见。
- `Synthesizer`：汇总各理论解释，保留分歧，形成有出处的文本分析报告。
- 结构化模型：例如未来的 `AnalysisTask`、`EvidenceItem`、`SchoolAnalysis`、`Critique`、`SynthesisResult`；这里是建议命名，当前源码不存在。

### 推荐编排和共享状态

建议第一版采用可预测的工作流：`START → Supervisor → Evidence → 理论分析节点 → 汇合 → Critic → Synthesizer → END`。证据节点先建立共同材料；三个理论节点在任务相互独立时可并行。Critic 需要修订时将任务交回 Supervisor，并由明确的轮次上限结束循环。检索也可作为角色子图的工具，但需要统一证据编号。

共享 State 建议包含：`messages`、`source_text`、`source_metadata`、`analysis_request`、`plan`、`evidence`、按角色存放的 `analyses`、`critique`、`revision_count`、`final_report`、`errors`。这些是待设计字段。并行节点对共享字段的写入必须定义合并规则，避免覆盖其他角色结果；领域产物和聊天历史分别管理。

## 9. 推荐新目录结构（尚未创建）

```text
langgraph-fullstack-python/
├── langgraph.json
├── pyproject.toml
├── uv.lock
├── .env.example
├── src/psychegraph/
│   ├── __init__.py
│   ├── graph.py                # 外层图构建与导出
│   ├── state.py                # 共享状态及合并规则
│   ├── schemas.py              # 任务、证据、理论分析、批评、综合结果
│   ├── configuration.py        # 模型与运行参数
│   ├── models.py               # 模型初始化工厂
│   ├── auth.py                 # 认证与资源授权
│   ├── agents/
│   │   ├── supervisor.py
│   │   ├── freudian.py
│   │   ├── object_relations.py
│   │   ├── lacanian.py
│   │   ├── critic.py
│   │   └── synthesizer.py
│   ├── prompts/               # 各角色系统提示和输出约定
│   ├── evidence/
│   │   ├── node.py             # 图中的证据节点
│   │   ├── ingestion.py        # 加载与分段
│   │   ├── retrieval.py        # 检索接口
│   │   └── citations.py        # 出处定位和引用校验
│   ├── tools/
│   │   ├── __init__.py         # 工具注册
│   │   └── retrieval.py        # 需要时包装检索为 Tool
│   └── web/
│       ├── app.py             # FastHTML 应用
│       ├── routes.py          # HTTP 路由
│       ├── components.py      # 页面组件
│       └── streaming.py       # 事件解析、聚合与 SSE 输出
├── tests/
│   ├── unit_tests/             # 状态合并、路由、schema、引用
│   ├── integration_tests/      # Graph、API、存储、访问控制
│   └── fixtures/              # 小型理论文本和预期分析结果
└── docs/
    ├── ARCHITECTURE.md
    ├── STATE_CONTRACT.md
    └── THEORY_AND_EVIDENCE.md
```

未来迁移时需要同时更新 Graph/HTTP/auth 注册路径、打包配置和测试导入。语料和索引存储位置需按数据规模及存储方案确定，当前未确认，不先假定某种向量数据库。

## 10. 优先级与核验边界

最关键的三个改动点：

1. 将单节点预构建 Graph 改为显式的 Supervisor—理论角色—Critic—Synthesizer 工作流，设计路由、汇合和终止条件。
2. 建立领域 State、结构化角色结果和 Evidence/RAG 引用契约，使理论解释可追踪到材料出处。
3. 抽离角色模型与 prompt，并将网页流处理改为理解多节点事件的展示层；随后落实身份、会话授权和可验证的持久化方案。

未确认事项集中包括：当前运行进程与锁文件版本是否完全一致、Provider 凭据与真实请求端点、流式内容的实际浏览器表现、SSE 重连行为、开发数据重启恢复结果、生产存储方案及外部认证策略。上述结论来自静态源码与目录检查，没有将“服务能启动”当作这些能力的端到端证明。

本次交付只新增本文；未修改现有业务代码、`pyproject.toml` 或 `uv.lock`，未安装依赖、接入 DeepSeek 或实现多智能体。
