# 智能客服 Agent 提示词、上下文、记忆与业务闭环工程设计

## 1. 项目定位

本设计用于补强当前电商客服多 Agent 项目的决策质量、上下文管理、记忆机制、工具权限、业务规则、人工接管和评估闭环。

当前项目已经具备以下基础：

- `LangGraph StateGraph` 编排 Router、Supervisor、Specialists、Guard 和确认中断。
- `SqliteSaver` checkpoint 支持 `interrupt` / `Command(resume=...)` 的待确认动作恢复。
- `AuthorizedToolExecutor` 控制工具调用、租约、幂等和审计。
- `PendingAction`、`ToolCall`、`AgentRun` 等审计和业务状态模型。
- `KnowledgeAgent` 使用 Milvus/RAG 返回带引用的知识回答。
- 售后图片链路已有 `VisualEvidence`、`VisionAgent` 和图片证据可用性判断。

当前主要缺口是：

- `RouterAgent` 和 `SupervisorAgent` 主要只基于当前 `message` 做判断，缺少工程化上下文投影。
- `RouteAnalysis` 和 `SupervisorDecision` schema 过粗，无法表达置信度、轮次类型、缺失槽位、升级信号和记忆引用。
- 缺少结构化的 `ConversationSlots`，多轮省略、纠正和补充信息容易依赖 LLM 猜测。
- 缺少分阶段的短期消息裁剪、会话摘要、长期记忆、记忆写入准则和记忆淘汰策略。
- 工具权限虽然已有执行器保护，但缺少声明式 `ToolPolicy` / `ToolRegistry`。
- 售后资格、转人工和风险判断没有独立的业务规则引擎。
- 评估体系主要覆盖 RAG，缺少 Agent 工作流、工具、安全和端到端评估。
- 缺少从 trace 到 badcase 再到候选修复的离线闭环。

本设计目标不是让 LLM 获得更多执行权限，而是让系统在正确上下文中安全、可审计地完成客服业务闭环。

## 2. 设计原则

### 2.1 记忆不等于 Prompt

不能把所有历史消息直接拼到 prompt。记忆必须先经过读取、过滤、排序、截断和投影，再分别提供给 Router、Supervisor、Specialists 或 Synthesis。

```text
MemoryStore / Transcript / Checkpoint
  -> MemoryManager
  -> ContextProjector
  -> RouterContext / SupervisorContext / SpecialistContext / SynthesisContext
```

### 2.2 对齐 LangChain / LangGraph Context 模型

本设计按 LangChain / LangGraph 推荐的 context 分层落地：

- Static runtime context：prompt、agent 能力表、工具规格、业务规则。
- Dynamic runtime context/state：当前 turn 的 `RuntimeState`、route、decision、tool result、pending action。
- Dynamic cross-conversation context/store：跨会话长期记忆、用户偏好、历史事件摘要。

LangGraph state/checkpoint 管工作记忆和短期运行状态；长期记忆放入 SQL MemoryStore 或 LangGraph Store 风格的 namespace/key/value 存储；RAG 只负责平台语义知识。

参考：

- https://docs.langchain.com/oss/python/concepts/context
- https://docs.langchain.com/oss/python/langgraph/add-memory
- https://docs.langchain.com/oss/python/langgraph/persistence
- https://docs.langchain.com/oss/python/langgraph/interrupts
- https://docs.langchain.com/oss/python/langgraph/graph-api
- https://docs.langchain.com/oss/python/langchain/structured-output
- https://reference.langchain.com/python/langchain-core/messages/utils/trim_messages

### 2.3 LLM 只做决策，不做授权

LLM 可以输出结构化分析和规划，但不能授权工具、不能直接提交售后、不能直接退款、不能绕过用户确认。

保持以下边界：

```text
RouterAnalysis 不包含 authorized_tools。
SupervisorDecision 不包含 execute_now / submit_now / refund_now。
写动作仍然只能是 draft_after_sales / draft_handoff。
提交或取消 pending action 只能走 confirm()。
工具权限由 AuthorizedToolExecutor + ToolRegistry + PolicyEngine 强制执行。
```

### 2.4 LangGraph / LangChain 原语优先

实现时优先使用 LangGraph / LangChain 已提供的 state、message、store、trimming、summarization、interrupt、structured output 等原语，不重复造基础设施。

必须遵守：

- message history 优先使用 `MessagesState` 或 `messages: Annotated[list[AnyMessage], add_messages]`。
- 消息追加、覆盖和反序列化交给 `add_messages` reducer。
- 短期记忆优先使用 LangGraph checkpointer。
- 长期记忆优先使用 LangGraph Store 风格的 namespace/key/value 接口。
- P0 长对话输入裁剪使用 `trim_messages`；checkpoint 体积控制使用 `RemoveMessage`；长对话摘要优先使用 LangGraph summarization pattern 或 `langmem.short_term.SummarizationNode`。
- 只有官方包不满足业务约束时，才实现薄封装或适配器。
- 每个 P0/P1 功能实现前必须查当前 LangGraph/LangChain 官方文档，并在实现计划或 PR 说明中记录采用的包原语和文档链接。

包优先替代原则：

| 模块 | 当前/潜在自研点 | 用现成包替代 | 具体要求 |
|---|---|---|---|
| 消息状态 | 自己维护 recent messages | `messages: Annotated[list[AnyMessage], add_messages]` | P0 加入 `RuntimeState` |
| 消息追加 | 手动 `state["messages"] + [...]` | `add_messages` reducer | 节点通过 `return {"messages": [...]}` 更新 |
| LLM 输入裁剪 | 手写最近 N 条 | `trim_messages` | `ContextProjector` 调用 |
| Prompt 拼接 | `model_factory.py` 内联字符串 | `ChatPromptTemplate` / `SystemMessage` / `HumanMessage` | prompt 放 `prompts.py` |
| 结构化输出 | 自写 JSON 解析 | `with_structured_output` | 保留现有方式 |
| 工具参数 schema | 自定义 ToolSpec args | LangChain `@tool` / `BaseTool.args_schema` 或现有 `CUSTOMER_TOOL_SCHEMAS` | `ToolPolicy` 只管权限 |
| 工具权限 | prompt 约束 allowed agents | `caller_agent + ToolPolicy` 代码强校验 | P0 必做 |
| workflow 测试 | 手工交互测试 | `pytest + JSONL golden cases` | P0 写 20 条 |

P0 必须优先采用的现成能力：

| 模块 | 现成包 | 说明 |
|---|---|---|
| 永久消息删除 | `RemoveMessage` | 控制 checkpoint 体积 |
| 摘要 | LangGraph summarization pattern / `langmem.short_term.SummarizationNode` | 长对话摘要 |
| 长期记忆 | LangGraph `InMemoryStore` / `PostgresStore` | 用户偏好、跨会话记忆 |
| Evaluation 平台 | `pytest + JSONL golden cases` | P0 主评估入口 |
| tracing | LangSmith | LLM 调用链路观测补充；没有配置时不阻塞本地测试 |

P1/P2 可增强的现成能力：

| 模块 | 现成包 | 说明 |
|---|---|---|
| RAG reranker | `FlagEmbedding` / `sentence-transformers` | 检索质量稳定后加 |
| Evaluation 平台扩展 | LangSmith eval / ragas | P0 pytest 稳定后扩展 |

推荐导入形态：

```python
from typing import Annotated

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, RemoveMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langgraph.graph.message import add_messages


class RuntimeState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    # 其他业务状态字段继续保留
```

### 2.5 当前事实优先于记忆

记忆用于辅助判断，不能覆盖当前消息、工具结果或数据库事实。

冲突优先级：

```text
当前用户消息
> 当前 pending action
> 当前工具结果 / 数据库事实
> ConversationSlots
> 最近消息
> ConversationSummary
> LongTermMemory
> RAG 平台知识
```

## 3. 总体架构

```text
API Layer
  FastAPI / Gradio / Debug UI

Conversation Layer
  ConversationService
  Message persistence
  Image asset storage
  AgentRun / ToolCall audit

Graph Runtime Layer
  LangGraph StateGraph
  RuntimeState
  messages: Annotated[list[AnyMessage], add_messages]
  Checkpointer
  Interrupt / Resume
  Turn Lease

Decision Layer
  RouterAgent
  SupervisorAgent
  Structured Output
  Prompt Versioning
  RouteAnalysis / SupervisorDecision

Context & Memory Layer
  ContextProjector
  MessagesState / add_messages
  trim_messages
  RemoveMessage / SummarizationNode
  ConversationSummary
  ConversationSlots
  PendingActionContext
  LangGraph Store / LongTermMemoryStore
  EpisodicMemory
  SemanticMemory / RAG

Tool & Policy Layer
  ToolRegistry
  ToolPolicy
  AuthorizedToolExecutor
  ToolPermission
  PolicyEngine
  RiskEngine
  ConfirmationGate
  Idempotency
  Audit

Specialist Layer
  ProductAgent
  OrderAgent
  KnowledgeAgent
  VisionAgent
  AfterSalesAgent
  HandoffAgent

Evaluation & Learning Layer
  Golden test set
  Workflow evaluation
  RAG evaluation
  Safety evaluation
  Badcase mining
  Memory candidate review
  Regression test
```

## 4. Prompt 工程

### 4.1 Prompt 存放位置

新增：

```text
python-impl/src/smart_cs/infrastructure/prompts.py
```

包含：

```python
PROMPT_VERSION = "decision-memory-v1"
ROUTER_SYSTEM_PROMPT = "..."
SUPERVISOR_SYSTEM_PROMPT = "..."
```

`model_factory.py` 只负责装配模型和调用 structured output，不再内联大段 prompt。

### 4.2 Router Prompt 职责

Router system prompt 必须明确：

- 只输出 `RouteAnalysis`。
- 只分析意图、实体、风险、置信度、轮次类型和缺失槽位。
- 不选择 specialist agent。
- 不授权工具。
- 不创建售后、退款、换货或人工接入动作。
- 不生成最终客服回复。
- 当前消息优先于历史记忆。
- 如果使用记忆辅助判断，必须填写 `referenced_memory_ids`。

Router 使用的动态输入是 `RouterContext` JSON。

调用形态：

```python
self._routing_model.invoke(
    [
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=router_context.model_dump_json()),
    ]
)
```

### 4.3 Supervisor Prompt 职责

Supervisor system prompt 必须明确：

- 只输出 `SupervisorDecision`。
- 只规划 specialist 执行顺序，不执行工具。
- 可选 agents 只能来自声明列表。
- 可选 action 只能是 `read`、`draft_after_sales`、`draft_handoff`。
- 写动作必须 `requires_confirmation=True`。
- `draft_after_sales` 必须先有 `OrderAgent`，最后是 `AfterSalesAgent`。
- `draft_handoff` 最后必须是 `HandoffAgent`。
- 高风险、低置信度、规则冲突、多次失败或明确要求人工时应规划 handoff。
- 如果缺少必要实体，应填写 `missing_entities`，不得编造。

Supervisor 使用的动态输入是 `SupervisorContext` JSON。

调用形态：

```python
self._planning_model.invoke(
    [
        SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
        HumanMessage(content=supervisor_context.model_dump_json()),
    ]
)
```

## 5. Decision Schema

### 5.1 RouteAnalysis

`RouteAnalysis` 从粗粒度 intent 识别升级为可评估的路由分析。

```python
class RouteAnalysis(BaseModel):
    intent: Literal["product", "order", "knowledge", "after_sales", "handoff"] = Field(
        description="本轮客户消息的主意图，只能选择一个。"
    )
    entities: dict[str, str] = Field(
        default_factory=dict,
        description="从当前消息和允许使用的上下文中明确得到的业务实体，不得臆造。",
    )
    risk: Literal["low", "medium", "high"] = Field(
        default="low",
        description="本轮请求的业务和安全风险等级。",
    )
    confidence: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="模型对 intent / entities / risk 判断的置信度。",
    )
    turn_type: Literal[
        "new_request",
        "follow_up",
        "correction",
        "confirmation_like",
        "rejection_like",
        "information_update",
    ] = Field(
        default="new_request",
        description="本轮消息在多轮对话中的作用。",
    )
    missing_entities: list[str] = Field(
        default_factory=list,
        description="继续处理当前意图所需但仍缺失的实体。",
    )
    escalation_signals: list[str] = Field(
        default_factory=list,
        description="投诉、法律威胁、隐私、支付安全等升级信号。",
    )
    referenced_memory_ids: list[str] = Field(
        default_factory=list,
        description="本次判断实际使用过的记忆 ID。",
    )
```

### 5.2 SupervisorDecision

`SupervisorDecision` 增加规划解释和缺失信息，但不扩大执行权限。

```python
class SupervisorDecision(BaseModel):
    agents: list[Literal[
        "ProductAgent",
        "OrderAgent",
        "KnowledgeAgent",
        "VisionAgent",
        "AfterSalesAgent",
        "HandoffAgent",
    ]]
    action: Literal["read", "draft_after_sales", "draft_handoff"]
    requires_confirmation: bool = False
    missing_entities: list[str] = Field(default_factory=list)
    planning_flags: list[str] = Field(default_factory=list)
    handoff_reason: str | None = None
    referenced_memory_ids: list[str] = Field(default_factory=list)
```

`validate_decision()` 继续作为 deterministic guard：

- 空 agent 列表拒绝。
- 未声明 agent 拒绝。
- 写动作 agent 必须在最后。
- 写动作必须强制 `requires_confirmation=True`。
- `draft_after_sales` 必须包含 `OrderAgent` 且在 `AfterSalesAgent` 前。
- 带图售后可由 deterministic rule 补齐 `VisionAgent` / `KnowledgeAgent`。

## 6. Context & Memory 工程

### 6.1 RuntimeState 作为工作记忆

当前 `RuntimeState` 保留现有业务字段，并新增 LangGraph 原生 message channel。message history 不再优先通过自定义 `recent_messages` 列表维护，而是通过 `add_messages` reducer 进入图状态并由 checkpointer 持久化。

```python
from typing import Annotated

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.graph.message import add_messages


messages: Annotated[list[AnyMessage], add_messages]
decision_context: dict[str, Any]
working_memory: dict[str, Any]
conversation_slots: dict[str, Any]
```

工作记忆生命周期是当前 thread / conversation，随 LangGraph checkpoint 保存。当前 `Message` 数据表可继续作为 API transcript 和审计持久化，但图运行时的短期消息状态以 `messages` channel 为准。

用户消息进入图时直接在 `graph.invoke()` 输入里提供，不新增会重复追加消息的 `message_ingest` 节点：

```python
graph.invoke(
    {
        "messages": [HumanMessage(content=message)],
        "message": message,
        # 其他业务状态字段
    },
    config=config,
)
```

助手回复只在最终合成节点写回：
助手回复只能写一次：`AIMessage` 只允许在 `synthesize` 节点写入 `messages` channel。图外 HTTP response 整理只读取 `state["reply"]` 生成 API 响应，不再写入 `messages`，避免 assistant reply 在 checkpoint 中重复出现。

```python
return {"messages": [AIMessage(content=reply)]}
```

不要手工拼接 `state["messages"] + [...]` 来更新状态；追加、覆盖、反序列化和按 ID 更新交给 `add_messages`。

### 6.2 MessagesState 与消息删减策略

短期消息管理优先使用 LangGraph / LangChain 官方能力：

- `MessagesState`：适合只有 messages 的图。
- 自定义 `RuntimeState` + `messages: Annotated[list[AnyMessage], add_messages]`：适合本项目这种同时有业务状态和消息状态的图。
- P0 使用 `trim_messages`：在调用 LLM 前裁剪输入消息，不改变 checkpoint，便于调试。
- P0 使用 `RemoveMessage`：从 LangGraph state 永久删除已摘要覆盖的旧消息，必须配合 `add_messages` reducer。
- P0 使用 LangGraph summarization pattern 或 `SummarizationNode`：如果引入 `langmem`，优先使用 `langmem.short_term.SummarizationNode` 做滚动摘要。

推荐热路径裁剪：

```python
from langchain_core.messages.utils import count_tokens_approximately, trim_messages


llm_messages = trim_messages(
    state["messages"],
    strategy="last",
    token_counter=count_tokens_approximately,
    max_tokens=2048,
    start_on="human",
    end_on=("human", "tool"),
)
```

裁剪原则：

- `trim_messages` 用于 LLM 输入预算控制。
- P0 使用 `RemoveMessage` 控制 checkpoint 体积，但必须先摘要旧消息，再删除已被摘要覆盖的旧消息。
- 删除消息后必须保证消息序列仍符合模型约束，例如 tool result 必须跟随对应 tool call。
- 对长对话优先摘要旧消息，再保留最近少量原文消息。

### 6.3 ConversationSlots

新增结构化业务槽位，避免让 LLM 从自然语言历史中猜当前业务对象。

```python
class ConversationSlots(BaseModel):
    active_order_id: str | None = None
    active_product_id: str | None = None
    active_after_sales_id: str | None = None
    active_ticket_id: str | None = None
    last_intent: str | None = None
    last_entities: dict[str, str] = Field(default_factory=dict)
    unresolved_question: str | None = None
    last_tool_results: dict[str, Any] = Field(default_factory=dict)
```

典型场景：

```text
用户：帮我查一下 O1001
系统：订单 O1001 已签收
用户：那我要退货
```

第二轮售后应由 `ConversationSlots.active_order_id` 确定性继承 `O1001`。

### 6.4 SlotCarry 节点

在 LangGraph 中新增轻量节点：

```text
router -> slot_carry -> supervisor
```

`slot_carry` 只做确定性补槽，不调用 LLM。

规则：

- 如果 `route.entities` 缺 `order_id`。
- 且 `route.turn_type` 是 `follow_up` / `correction` / `information_update`。
- 且 `conversation_slots.active_order_id` 存在。
- 则补入 `route.entities["order_id"]`。

`slot_carry` 不允许补入跨客户、无来源或低可信长期记忆中的订单号。

### 6.5 StateUpdater 节点

`slot_carry` 只负责从旧 slots 补当前 route；还需要独立的 `state_update` 节点把本轮新事实写回 `ConversationSlots`。该节点不调用 LLM，不执行工具，只根据 route、specialist results 和 business result 做确定性更新。

建议规则：

- `route.entities["order_id"]` 存在时，更新 `active_order_id`。
- `business_result["order_id"]` 存在时，更新 `active_order_id`。
- `business_result["ticket_id"]` 存在且售后已提交时，更新 `active_ticket_id`。
- `business_result.status == "pending_confirmation"` 时，记录当前 pending action 摘要和 action status。
- `confirm_action` resume 后再次运行 `state_update`，根据 submitted / cancelled 结果更新 action status。
- confirmed submitted 后，如果返回 `ticket_id`，更新 `active_ticket_id`。
- confirmed cancelled 后，清理对应 pending action slot，但保留可审计的最后 action status。
- `route.turn_type == "correction"` 时，当前显式实体覆盖旧 slot。
- `business_result.status == "cancelled"` 时，清理对应 pending/action slot。
- 每轮结束更新 `last_intent` 和 `last_entities`。
- `specialist_results` 中可审计的最后结果写入 `last_tool_results` 的摘要视图。

`state_update` 应位于写工具或 handoff 完成之后、guard 之前：

```text
write_specialists_or_handoff
-> validate_evidence
-> state_update
-> guard
```

确认路径也必须经过 `state_update`：

```text
confirm_action
-> state_update
-> guard
-> synthesize
```

pending 阶段的 `state_update` 负责记录“当前有待确认动作”；confirm 后的 `state_update` 负责记录“动作已提交或已取消”以及 `active_ticket_id` 等终态事实。

### 6.6 RouterContext

Router 只看路由必需上下文。

```python
class RouterContext(BaseModel):
    current_message: str
    recent_messages: list[MessageContext] = Field(default_factory=list)
    conversation_summary: str | None = None
    conversation_slots: ConversationSlots
    pending_action: PendingActionContext | None = None
    customer_memories: list[MemoryView] = Field(default_factory=list)
    has_image: bool = False
    visual_evidence: dict[str, Any] | None = None
```

`recent_messages` 由 `ContextProjector` 从 `state["messages"]` 中投影出来，并先经过 `trim_messages` 或消息数量限制。Router 不看完整 tool audit，不看无关历史 conversation，不看完整 RAG 文档。

### 6.7 SupervisorContext

Supervisor 需要比 Router 更多约束和能力表。

```python
class SupervisorContext(BaseModel):
    current_message: str
    route: RouteAnalysis
    recent_messages: list[MessageContext] = Field(default_factory=list)
    conversation_summary: str | None = None
    conversation_slots: ConversationSlots
    pending_action: PendingActionContext | None = None
    customer_memories: list[MemoryView] = Field(default_factory=list)
    visual_context: VisualContext
    agent_capabilities: dict[str, str]
    tool_policies: list[ToolPolicy]
    policy_hints: list[str] = Field(default_factory=list)
    planning_constraints: list[str] = Field(default_factory=list)
```

### 6.8 SynthesisContext

最终回复合成不自由读取长期记忆。

Synthesis 只允许使用：

- `specialist_results`
- `guarded_contents`
- `citations`
- `pending_action`

这保留当前 `ResponseGuard` 的安全边界，避免最终回复阶段使用记忆编造业务事实。

## 7. P0 MemoryStore 与记忆策略

本节纳入 P0。P0 必须实现最小可用的 `ConversationSummary`、`RemoveMessage`、长期记忆接口、`MemoryExtractor` 和 `MemoryPolicy`，但实现原则仍然是优先调用 LangGraph / LangChain / langmem 原语，只有业务审计和持久化需要时才写薄适配层。

### 7.1 ConversationSummary

会话级滚动摘要优先使用 LangGraph 官方推荐的 summarization pattern：扩展 state 中的 `summary` / `context` 字段，并在消息过长时运行 summarization node。

如果项目引入 `langmem`，优先使用：

```python
from langmem.short_term import SummarizationNode
```

如果不引入 `langmem`，才实现项目内 `summarize_conversation` node。该节点必须：

- 读取已有 summary。
- 结合旧 summary 和新增消息生成新 summary。
- 用 `RemoveMessage` 删除已摘要的旧消息。
- 保留最近 N 条原始消息。

数据库层可新增 `ConversationSummary` 作为跨重启、跨检查点清理后的审计和查询副本：

```python
class ConversationSummary(Base):
    conversation_id: str
    customer_id: str
    summary: str
    open_items: dict[str, Any]
    last_intent: str | None
    last_entities: dict[str, str]
    updated_at: datetime
```

用途：

- 压缩长对话。
- 支持多轮追问。
- 保存未解决事项。
- 不替代原始 Message transcript。
- 不替代 `messages` channel，只作为摘要副本和 ContextProjector 输入之一。

### 7.2 MemoryRecord

长期记忆优先使用 LangGraph Store 风格 namespace/key/value 结构保存。实现选择：

```text
开发 / 测试：InMemoryStore
生产建议：PostgresStore / RedisStore / 项目 SQL 适配器
```

如果继续使用项目现有 SQLAlchemy 持久化，需要实现一个薄 `MemoryStoreAdapter`，对外暴露接近 LangGraph Store 的能力：

```text
put(namespace, key, value)
get(namespace, key)
search(namespace, query, limit)
delete(namespace, key)
```

图节点访问长期记忆时优先走 LangGraph runtime 注入的 store：

```python
from langgraph.runtime import Runtime


def node(state: RuntimeState, runtime: Runtime[RuntimeContext]):
    namespace = ("customer", runtime.context.customer_id, "memories")
    memories = runtime.store.search(namespace, query=state["messages"][-1].content, limit=3)
```

`MemoryRecord` 是 SQL 适配器或审计层的实体结构：

```python
class MemoryRecord(Base):
    id: str
    scope: Literal["customer", "conversation", "global"]
    owner_id: str
    memory_type: Literal[
        "preference",
        "stable_fact",
        "service_event",
        "issue_pattern",
        "badcase_candidate",
    ]
    key: str
    value_json: dict[str, Any]
    source: Literal[
        "user_message",
        "tool_result",
        "confirmed_action",
        "conversation_summary",
        "human_review",
    ]
    confidence: Literal["low", "medium", "high"]
    risk_level: Literal["low", "medium", "high"]
    created_by: Literal["system", "llm_candidate", "human"]
    approved_by: str | None
    expires_at: datetime | None
    last_used_at: datetime | None
    usage_count: int
    created_at: datetime
    updated_at: datetime
```

### 7.3 记忆写入准则

```text
客服事件摘要：自动写，不需要确认。
稳定用户偏好：生成候选，高价值或可能影响服务结果时需要确认。
敏感标签：默认不自动写，需要规则审核或人工审核。
法务/平台政策知识：Agent 不允许写入，只能由人工维护知识库。
badcase 经验：生成候选，必须人工审核并通过回归测试。
```

写入链路：

```text
MemoryExtractor
  -> MemoryPolicy
  -> runtime.store / MemoryStoreAdapter
```

`MemoryExtractor` 只生成候选，不能直接持久化高风险长期记忆。

### 7.4 记忆淘汰和冲突

长期记忆不是越多越好。检索排序应考虑：

```text
相关性 score
+ 置信度 confidence
+ 新鲜度 recency
+ 使用次数 usage_count
- 风险惩罚 risk_penalty
- 过期惩罚 expiry_penalty
```

必须支持：

- `expires_at`
- `last_used_at`
- `usage_count`
- `manual_delete`
- 冲突记忆降权
- 新事实覆盖旧偏好

## 8. Tool & Policy 工程

本节纳入 P0。工具权限和业务规则不能只靠 prompt 约束，必须在执行器和规则引擎中用代码强校验。

### 8.1 ToolPolicy / ToolRegistry

当前 `AuthorizedToolExecutor` 已经执行权限保护，但工具业务权限应显式声明。不要重复维护 LangChain tool 参数 schema；工具参数 schema 继续来自 LangChain `@tool` / `BaseTool.args_schema` 或现有 `CUSTOMER_TOOL_SCHEMAS`，`ToolPolicy` 只承载业务权限元数据。

```python
@dataclass(frozen=True)
class ToolPolicy:
    name: str
    risk_level: Literal["low", "medium", "high"]
    allowed_agents: frozenset[str]
    requires_confirmation: bool
    idempotent: bool
```

示例：

| 工具 | 类型 | 风险 | 是否确认 | 允许 Agent |
|---|---|---|---:|---|
| `search_products` | read | low | 否 | `ProductAgent` |
| `lookup_order` | read | medium | 否 | `OrderAgent` |
| `draft_after_sales` | write | medium | 是 | `AfterSalesAgent` |
| `draft_handoff` | write | medium | 是 | `HandoffAgent` |
| `submit_confirmed_action` | write | high | 是 | `ConfirmActionNode` |

`AuthorizedToolExecutor` 必须校验：

- tool 是否已声明。
- 调用 agent 是否被允许。
- 写动作是否具备 turn fence。
- 写动作是否满足确认策略。
- 调用是否记录审计。

为使 `allowed_agents` 在执行时真实生效，`AuthorizedToolExecutor.invoke()` 必须增加 `caller_agent` 参数：

```python
def invoke(
    self,
    tool_name: str,
    arguments: dict[str, Any],
    *,
    caller_agent: str,
    turn_fence: TurnFence | None = None,
) -> dict[str, Any]:
    ...
```

Specialist 调用工具时必须传入自己的 agent 名称：

```python
self.executor.invoke(
    "lookup_order",
    arguments,
    caller_agent="OrderAgent",
)
```

执行器使用 `ToolRegistry` 查询 policy：

```python
policy = self.tool_registry.get(tool_name)
if caller_agent not in policy.allowed_agents:
    raise ToolPermissionError("Tool is not allowed for this agent")
```

### 8.2 PolicyEngine / EligibilityEngine

RAG 只能提供政策依据，不能替代业务资格判断。

新增：

```python
class PolicyDecision(BaseModel):
    eligible: bool
    reason_code: str
    explanation: str
    next_action: Literal["allow_draft", "explain", "handoff"]
    requires_human_review: bool = False
```

售后链路应调整为：

```text
OrderAgent 查询订单
KnowledgeAgent 检索政策依据
PolicyEngine 判断 eligibility
AfterSalesAgent 仅在 allow_draft 时创建售后草稿
否则 ResponseGuard 解释原因或 HandoffAgent 创建人工接入草稿
```

PolicyEngine 负责：

- 是否可退。
- 是否可换。
- 是否可取消。
- 是否可改地址。
- 是否超过售后期。
- 是否需要人工审核。
- 是否存在大额、敏感、争议风险。

## 9. Handoff 策略

`HandoffAgent` 不只是工具调用，而是人工接管策略。

P0 只保留最小 handoff：在 `PolicyEngine` 或 Supervisor 判定需要人工时创建 `draft_handoff` 草稿，并继续走 pending confirmation。完整 `HandoffPayload`、人工接管上下文包和人工处理结果回流放到 P1/P2。

触发条件：

- 用户明确要求人工。
- `risk=high`。
- `confidence=low` 且涉及售后或订单争议。
- 多次工具失败。
- 规则冲突。
- 高金额或敏感售后。
- 法律威胁、投诉升级、曝光威胁。
- 图片证据不可用或不确定。

转人工 payload：

```python
class HandoffPayload(BaseModel):
    conversation_id: str
    customer_id: str
    summary: str
    last_intent: str | None
    slots: ConversationSlots
    tool_results: dict[str, Any]
    risk_signals: list[str]
    retrieved_policy_refs: list[str]
    suggested_resolution: str | None
```

目标：

- 用户不重复描述。
- 人工能快速接手。
- 人工处理结果能回流 `MemoryStore` 和 `ConversationSummary`。

## 10. Graph Runtime 调整

本节纳入 P0。Graph 调整必须和 `messages` channel、memory writeback、pending confirmation、confirm 后状态更新一起落地。

当前主线：

```text
router
-> supervisor
-> specialists
-> validate_evidence
-> guard
-> synthesize
-> confirm_action
```

建议调整为：

```text
graph.invoke 输入直接携带 HumanMessage
-> context_project
-> router
-> slot_carry
-> supervisor
-> read_specialists
-> policy_check
-> write_specialists_or_handoff
-> validate_evidence
-> state_update
-> memory_writeback
-> guard
-> synthesize
-> confirm_action
-> state_update
-> memory_writeback
-> guard
-> synthesize
```

说明：

- API 层继续写 `Message` 表作为审计 transcript；`graph.invoke()` 输入同时携带 `HumanMessage` 进入 `messages` channel；不新增会再次追加消息的 `message_ingest` 节点。
- `context_project` 构建 RouterContext 所需的动态上下文。
- `slot_carry` 做确定性槽位继承。
- `read_specialists` 只执行读取事实的 agent，例如 `ProductAgent`、`OrderAgent`、`KnowledgeAgent`、`VisionAgent`。
- `policy_check` 在读工具结果之后做业务资格和风险判断，不在缺少订单状态、商品类目、政策依据或图片证据时提前判断。
- `write_specialists_or_handoff` 只在 policy 允许或要求 handoff 时创建 `draft_after_sales` / `draft_handoff` 草稿。
- pending 前的 `state_update` 记录当前待确认动作和相关业务槽位。
- `confirm_action` 之后的 `state_update` 记录 submitted / cancelled 终态，提交成功时更新 `active_ticket_id`。
- `memory_writeback` 只写 `ConversationSummary`、长期记忆候选和已确认业务事件，不写 `AIMessage`。
- `synthesize` 是唯一写入 `AIMessage` 的节点；HTTP response 整理不再写入 `messages`。
- `confirm_action` 仍然通过 LangGraph interrupt 暂停和恢复。

Graph 构建优先使用 LangGraph 原生能力：

```python
builder = StateGraph(RuntimeState, context_schema=RuntimeContext)
graph = builder.compile(checkpointer=checkpointer)
```

P0 引入长期记忆后注入 store；开发和测试可先用 `InMemoryStore`，需要持久化时使用 `PostgresStore` 或项目内 SQL 薄适配器：

```python
graph = builder.compile(checkpointer=checkpointer, store=store)
```

`RuntimeContext` 放 invocation 级静态上下文，例如：

```python
class RuntimeContext(BaseModel):
    conversation_id: str
    customer_id: str
    prompt_version: str
```

仍可在 `RuntimeState` 中保留 `conversation_id` / `customer_id` 以兼容现有代码，但新实现应逐步把只读 invocation context 放入 `context_schema`。

## 11. Evaluation & Learning

本节纳入 P0。P0 不要求接入完整评估平台，但必须用 `pytest + JSONL golden cases` 建立可重复运行的工作流评估和 badcase 候选闭环。

### 11.1 Workflow Evaluation

新增 Agent 工作流评估集，覆盖：

Router：

- Intent Accuracy
- Macro-F1
- Slot F1
- Risk Recall
- Turn-type Accuracy

Supervisor：

- Agent Plan Accuracy
- Invalid Plan Rate
- Write-action Confirmation Recall
- Missing Entity Recall

Tool：

- Tool Selection Accuracy
- Argument Accuracy
- Tool Success Rate
- Permission Violation Rate
- Idempotency Correctness

Policy：

- Eligibility Accuracy
- Reason Code Accuracy
- Human Review Recall

Safety：

- 未确认写操作率
- 错误承诺率
- 高风险漏拦率
- 隐私泄露率

End-to-End：

- Task Success Rate
- Average Turns
- Handoff Accuracy
- Cost / Latency

RAG 继续保留已有：

- Faithfulness
- Answer Relevancy
- Context Recall
- Context Precision
- Citation Accuracy

### 11.2 Badcase 闭环

新增离线闭环：

```text
AgentRun / ToolCall / UserFeedback
-> BadcaseDetector
-> CaseLabeling
-> RootCauseAnalysis
-> CandidateFix
-> HumanReview
-> RegressionEval
-> VersionedRelease
```

badcase 修复映射：

| 类型 | 修复方向 |
|---|---|
| 意图识别错 | Router prompt / schema / few-shot eval case |
| 工具选错 | Supervisor rule / ToolPolicy |
| 参数抽错 | Slot extractor / SlotCarry |
| RAG 错 | chunk / metadata / reranker |
| 规则判断错 | PolicyEngine |
| 应转人工没转 | HandoffPolicy / RiskEngine |
| 回复乱承诺 | ResponseGuard |

禁止 Agent 自动修改生产 prompt。正确流程是：

```text
自动生成候选
-> 人工审核
-> 离线评估
-> 回归测试
-> 灰度发布
```

## 12. 代码可读性约束

本节纳入 P0 验收。实现必须保持节点薄、规则集中、边界清晰：

1. 每个 LangGraph node 原则上控制在 30 行以内。
2. node 只做编排和状态读写，不写复杂业务规则。
3. 复杂规则放到独立 service，例如 `SlotCarryService`、`StateUpdater`、`PolicyEngine`。
4. `ContextProjector` 只负责构造 context，不调用 LLM，不执行工具。
5. `ToolRegistry` 只存工具业务策略，不执行工具，不重复维护 LangChain tool 参数 schema。
6. `AuthorizedToolExecutor` 只负责鉴权、执行、幂等、租约和审计。
7. Pydantic schema 集中放在 `agents/state.py` 或新增 `agents/context.py`，避免散落。
8. Prompt 放在 `infrastructure/prompts.py`，不得内联在 `model_factory.py`。
9. 测试按 router、context projector、slot carry、state updater、policy、tool permission、workflow 分层。
10. 不允许在 graph node 中直接拼接长 prompt。

## 13. 分阶段交付

### P0：必须完成

1. `RouteAnalysis` / `SupervisorDecision` schema 扩展。
2. `prompts.py` 与 SystemMessage / HumanMessage 分层提示词。
3. `RuntimeState.messages: Annotated[list[AnyMessage], add_messages]`。
4. `graph.invoke()` 输入直接携带 `HumanMessage`，不新增重复追加消息的 `message_ingest` 节点。
5. `RouterContext` / `SupervisorContext` / `ContextProjector`。
6. `trim_messages` 输入裁剪策略。
7. `ConversationSlots` 与 `slot_carry` 节点。
8. `state_update` 节点，pending 阶段和 confirm 后都确定性写回 slots/action status。
9. `ToolPolicy` / `ToolRegistry` 最小版，只做业务权限元数据。
10. `AuthorizedToolExecutor.invoke(..., caller_agent=...)`，强制校验 `allowed_agents`。
11. `PolicyEngine` 最小版，只覆盖售后 `allow_draft` / `explain` / `handoff`。
12. `ConversationSummary`，优先采用 LangGraph summarization pattern 或 `langmem.short_term.SummarizationNode`。
13. `RemoveMessage` 永久删除策略，用于控制 checkpoint 体积；删除前必须先摘要。
14. LangGraph Store、`InMemoryStore`、`PostgresStore` 或项目 SQL 薄适配器形式的长期 `MemoryStore`。
15. `MemoryExtractor` / `MemoryPolicy`，支持客服事件摘要自动写、用户偏好候选写、敏感记忆审核。
16. `memory_writeback` 节点，pending 阶段记录 pending action，confirm 后记录 confirmed/cancelled 业务事件。
17. `Evaluation & Learning` 最小闭环：20 条 Agent workflow JSONL golden tests、pytest runner、badcase candidate 输出。
18. 第 12 节代码可读性约束作为 P0 验收门槛。

P0 明确不做：

- 完整 `HandoffPayload` 和人工工作台上下文包。
- RAG reranker。
- LangSmith / ragas 平台化评估。
- 多模型路由。
- 自动修改生产 prompt。
- 自动批准长期敏感记忆。

### P1：增强质量

1. 长期记忆检索排序、过期淘汰和冲突降权增强。
2. RAG reranker。
3. 风险分层和高风险 recall 测试扩展。
4. Prompt version 管理和灰度记录。
5. Badcase 自动归因增强。
6. LangSmith eval / ragas 平台化评估。
7. 完整 `HandoffPayload`、人工处理结果回流和客服工作台上下文包。

### P2：接近生产

1. 多租户权限边界。
2. PII 脱敏。
3. 数据保留和删除策略。
4. 工具限流、超时、重试。
5. 成本和延迟监控。
6. LLM fallback。
7. 多模型路由。
8. 灰度发布和 A/B 测试。
9. 线上观测 dashboard。

## 14. 非目标

本设计不要求立即实现：

- 真实支付、退款或物流外部 API。
- 自动修改生产 prompt。
- 自动批准长期敏感记忆。
- 让 LLM 直接提交售后、退款、赔付或人工接管。
- 全量生产级客服工作台。
- 完整多租户 SaaS 权限系统。

这些可以作为后续产品化方向，但不能写成当前已完成能力。

## 15. 验收标准

P0 完成后应满足：

- Router 和 Supervisor 使用结构化 context，而不是裸 `message` 字符串。
- Prompt 使用 SystemMessage / HumanMessage 分层。
- LangGraph state 使用 `messages: Annotated[list[AnyMessage], add_messages]` 或等价 `MessagesState` 机制保存短期消息历史。
- `graph.invoke()` 输入携带 `HumanMessage`，没有重复追加消息的 `message_ingest` 节点。
- 调用 LLM 前使用 `trim_messages` 控制上下文窗口。
- 长对话使用 `ConversationSummary` 和 `RemoveMessage` 控制 checkpoint 体积，且删除前已有摘要覆盖。
- `AIMessage` 只由 `synthesize` 节点写入一次，图外 HTTP response 整理不写入 `messages`。
- `RouteAnalysis` 能表达 `confidence`、`turn_type`、`missing_entities`、`escalation_signals`。
- `SupervisorDecision` 能表达 `missing_entities`、`planning_flags`、`handoff_reason`。
- 多轮省略场景通过 `ConversationSlots` 和 `slot_carry` 确定性补槽。
- `state_update` 能在 pending 阶段记录 pending action，并在 confirm 后记录 submitted / cancelled 与 `active_ticket_id`。
- `memory_writeback` 能写入会话摘要、已确认业务事件和符合策略的长期记忆候选。
- 工具调用由 `ToolPolicy` / `ToolRegistry` 声明和校验。
- `AuthorizedToolExecutor.invoke()` 通过 `caller_agent` 强制校验 `allowed_agents`。
- 售后资格判断在 read specialists 之后走 `PolicyEngine`，不由 LLM 直接决定。
- 写动作仍然必须 pending confirmation。
- 20 条 Agent workflow golden tests 覆盖 router、supervisor、tool、policy、safety 和 end-to-end 的最小关键路径。
- badcase candidate 能从失败测试或运行 trace 中离线输出，不自动修改生产 prompt。
- 第 12 节代码可读性约束通过代码 review 或静态检查抽样验证。

## 16. 一句话总结

本方案以 LangGraph state/checkpoint/store 和 LangChain message utilities 为基础，P0 必须同时落地 `add_messages`、`trim_messages`、`ConversationSummary`、`RemoveMessage`、长期记忆薄接口、ContextProjector、ConversationSlots、state_update、memory_writeback、ToolPolicy、caller_agent、PolicyEngine、pytest golden evaluation 和代码可读性约束；完整 handoff payload、RAG reranker、多模型路由和平台化评估放到 P1/P2。
