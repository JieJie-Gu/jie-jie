# Prompt Context Memory Agent Engineering P0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the P0 prompt engineering, context projection, message state, memory, tool policy, policy engine, graph runtime, and workflow evaluation design for the smart customer-service agent.

**Architecture:** Keep the existing FastAPI, SQLAlchemy repository, `AuthorizedToolExecutor`, and LangGraph checkpoint foundation. Add LangGraph-native `messages` state, structured context projection, deterministic slot/state update services, tool policy enforcement, minimal policy engine, memory writeback, and pytest JSONL workflow evaluation without granting the LLM extra authority.

**Tech Stack:** Python 3.11, LangGraph, LangChain Core messages, Pydantic v2, SQLAlchemy, pytest, SQLite

---

## Scope

This plan implements the P0 scope from `docs/superpowers/specs/2026-05-28-prompt-context-memory-agent-engineering-design.md`.

P0 includes:

- `RouteAnalysis` / `SupervisorDecision` schema expansion.
- `prompts.py` with System/Human prompt separation.
- `RuntimeState.messages: Annotated[list[AnyMessage], add_messages]`.
- `graph.invoke()` directly passing `HumanMessage`.
- `ContextProjector` with `trim_messages`.
- `ConversationSlots`, `slot_carry`, and `state_update`.
- `ToolPolicy`, `ToolRegistry`, and `caller_agent` enforcement.
- Minimal `PolicyEngine`.
- `ConversationSummary`, `RemoveMessage`, minimal long-term memory store, `MemoryExtractor`, `MemoryPolicy`, and `memory_writeback`.
- Runtime graph rewire.
- `pytest + JSONL golden cases` workflow evaluation and badcase candidate output.

P0 excludes:

- Full human-workbench `HandoffPayload`.
- RAG reranker.
- LangSmith / ragas platform evaluation.
- Multi-model routing.
- Automatic production prompt modification.
- Automatic approval of sensitive long-term memories.

## Preflight

Run all commands from the worktree root:

```powershell
cd .worktrees/gradio-demo-frontend
```

Before implementation, verify the package imports currently available in the local environment:

```powershell
cd python-impl
python -c "from langgraph.graph.message import add_messages; from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage; from langchain_core.messages.utils import trim_messages, count_tokens_approximately; from langgraph.store.memory import InMemoryStore; from langgraph.types import interrupt, Command; print('langgraph primitives ok')"
```

Expected: prints `langgraph primitives ok`.

Before implementing Task 1, Task 3, Task 4, Task 7, Task 8, and Task 9, check the current official docs for the package primitive used by that task and record the adopted import/API **and doc link** in the commit message or execution summary:

```text
Task 1 / 8: LangGraph add_messages, StateGraph, context_schema, Runtime, store
  Doc: https://langgraph-ai.github.io/langgraph/concepts/low_level/#messagesstate
Task 3: LangChain tool schema boundaries; ToolPolicy remains business metadata only
  Doc: https://python.langchain.com/docs/concepts/tools/
Task 4: trim_messages and count_tokens_approximately
  Doc: https://python.langchain.com/docs/how_to/trim_messages/
Task 7: RemoveMessage, LangGraph Store put/search, summarization pattern
  Doc: https://langgraph-ai.github.io/langgraph/concepts/persistence/#memory-store
Task 8: LangGraph interrupt, Command(resume=...) for confirm_action pause/resume
  Doc: https://langgraph-ai.github.io/langgraph/concepts/human_in_the_loop/
Task 9: pytest parametrization and JSONL golden case runner
  Doc: https://docs.pytest.org/en/stable/how-to/parametrize.html
```

If an import path fails, do not invent a local substitute first. Check the docs and adjust the import/API in the task before writing implementation code.

## File Map

Create:

```text
python-impl/src/smart_cs/infrastructure/prompts.py
python-impl/src/smart_cs/application/context_projector.py
python-impl/src/smart_cs/application/state_update.py
python-impl/src/smart_cs/application/policy.py
python-impl/src/smart_cs/application/memory.py
python-impl/src/smart_cs/tools/policy.py
python-impl/tests/unit/test_context_projector.py
python-impl/tests/unit/test_state_update.py
python-impl/tests/unit/test_tool_policy.py
python-impl/tests/unit/test_policy_engine.py
python-impl/tests/unit/test_memory.py
python-impl/tests/evaluation/golden_cases.jsonl
python-impl/tests/evaluation/test_workflow_golden.py
```

Modify:

```text
python-impl/src/smart_cs/agents/state.py
python-impl/src/smart_cs/agents/router.py
python-impl/src/smart_cs/agents/supervisor.py
python-impl/src/smart_cs/agents/specialists.py
python-impl/src/smart_cs/application/agent_runtime.py
python-impl/src/smart_cs/tools/executor.py
python-impl/src/smart_cs/infrastructure/model_factory.py
python-impl/src/smart_cs/infrastructure/repositories.py
python-impl/src/smart_cs/domain/models.py
python-impl/src/smart_cs/main.py
python-impl/tests/unit/test_router_supervisor.py
python-impl/tests/unit/test_tools.py
python-impl/tests/integration/test_action_confirmation.py
```

Do not modify:

```text
python-impl/src/smart_cs/rag/vector_store.py
python-impl/src/smart_cs/rag/evaluation.py
python-impl/src/smart_cs/api/routers/conversations.py
```

The API layer should keep its current response shape. Message state changes stay inside LangGraph runtime.

## Task 1: Expand Decision, Context, And Runtime State Schemas

**Files:**
- Modify: `python-impl/src/smart_cs/agents/state.py`
- Modify: `python-impl/tests/unit/test_router_supervisor.py`

- [ ] **Step 1: Write failing schema tests**

Append to `python-impl/tests/unit/test_router_supervisor.py`:

```python
from typing import Annotated, get_args, get_origin, get_type_hints

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


def test_route_analysis_contains_context_quality_fields() -> None:
    route = RouteAnalysis(
        intent="after_sales",
        entities={"order_id": "O1001"},
        risk="medium",
        confidence="high",
        turn_type="follow_up",
        missing_entities=[],
        escalation_signals=["complaint"],
        referenced_memory_ids=["mem-1"],
    )

    assert route.confidence == "high"
    assert route.turn_type == "follow_up"
    assert route.escalation_signals == ["complaint"]
    assert route.referenced_memory_ids == ["mem-1"]


def test_supervisor_decision_contains_planning_metadata_without_execution_authority() -> None:
    decision = SupervisorDecision(
        agents=["OrderAgent", "KnowledgeAgent", "AfterSalesAgent"],
        action="draft_after_sales",
        requires_confirmation=True,
        missing_entities=[],
        planning_flags=["needs_policy_check"],
        handoff_reason=None,
        referenced_memory_ids=["mem-1"],
    )

    dumped = decision.model_dump()

    assert dumped["planning_flags"] == ["needs_policy_check"]
    assert "execute_now" not in dumped
    assert "submit_now" not in dumped
    assert "refund_now" not in dumped


def test_runtime_state_messages_uses_add_messages_reducer() -> None:
    hints = get_type_hints(RuntimeState, include_extras=True)
    messages_hint = hints["messages"]

    assert get_origin(messages_hint) is Annotated
    args = get_args(messages_hint)
    assert args[0] == list[AnyMessage]
    assert add_messages in args[1:]
```

- [ ] **Step 2: Run schema tests to verify red**

Run:

```powershell
cd python-impl
pytest tests/unit/test_router_supervisor.py::test_route_analysis_contains_context_quality_fields tests/unit/test_router_supervisor.py::test_supervisor_decision_contains_planning_metadata_without_execution_authority tests/unit/test_router_supervisor.py::test_runtime_state_messages_uses_add_messages_reducer -q
```

Expected: FAIL because the new fields and `messages` reducer do not exist.

- [ ] **Step 3: Add schema fields and message channel**

Modify `python-impl/src/smart_cs/agents/state.py`:

```python
from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class RouteAnalysis(BaseModel):
    intent: Literal["product", "order", "knowledge", "after_sales", "handoff"]
    entities: dict[str, str] = Field(default_factory=dict)
    risk: Literal["low", "medium", "high"] = "low"
    confidence: Literal["low", "medium", "high"] = "medium"
    turn_type: Literal[
        "new_request",
        "follow_up",
        "correction",
        "confirmation_like",
        "rejection_like",
        "information_update",
    ] = "new_request"
    missing_entities: list[str] = Field(default_factory=list)
    escalation_signals: list[str] = Field(default_factory=list)
    referenced_memory_ids: list[str] = Field(default_factory=list)


class SupervisorDecision(BaseModel):
    agents: list[
        Literal[
            "ProductAgent",
            "OrderAgent",
            "KnowledgeAgent",
            "VisionAgent",
            "AfterSalesAgent",
            "HandoffAgent",
        ]
    ]
    action: Literal["read", "draft_after_sales", "draft_handoff"]
    requires_confirmation: bool = False
    missing_entities: list[str] = Field(default_factory=list)
    planning_flags: list[str] = Field(default_factory=list)
    handoff_reason: str | None = None
    referenced_memory_ids: list[str] = Field(default_factory=list)


class ConversationSlots(BaseModel):
    active_order_id: str | None = None
    active_product_id: str | None = None
    active_after_sales_id: str | None = None
    active_ticket_id: str | None = None
    last_intent: str | None = None
    last_entities: dict[str, str] = Field(default_factory=dict)
    unresolved_question: str | None = None
    last_tool_results: dict[str, Any] = Field(default_factory=dict)
    pending_action: dict[str, Any] | None = None
    action_status: str | None = None


class MemoryView(BaseModel):
    memory_id: str
    memory_type: str
    value: dict[str, Any]
    confidence: Literal["low", "medium", "high"] = "medium"
    source: str


class RouterContext(BaseModel):
    current_message: str
    recent_messages: list[dict[str, str]] = Field(default_factory=list)
    conversation_summary: str | None = None
    conversation_slots: ConversationSlots = Field(default_factory=ConversationSlots)
    pending_action: dict[str, Any] | None = None
    customer_memories: list[MemoryView] = Field(default_factory=list)
    has_image: bool = False
    visual_evidence: dict[str, Any] | None = None


class SupervisorContext(BaseModel):
    current_message: str
    route: RouteAnalysis
    recent_messages: list[dict[str, str]] = Field(default_factory=list)
    conversation_summary: str | None = None
    conversation_slots: ConversationSlots = Field(default_factory=ConversationSlots)
    pending_action: dict[str, Any] | None = None
    customer_memories: list[MemoryView] = Field(default_factory=list)
    has_image: bool = False
    visual_evidence: dict[str, Any] | None = None
    agent_capabilities: dict[str, str] = Field(default_factory=dict)
    tool_policies: list[dict[str, Any]] = Field(default_factory=list)
    policy_hints: list[str] = Field(default_factory=list)
    planning_constraints: list[str] = Field(default_factory=list)


class RuntimeContext(BaseModel):
    conversation_id: str
    customer_id: str
    prompt_version: str


class RuntimeState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    conversation_id: str
    customer_id: str
    request_id: str
    message: str
    has_image: bool
    visual_evidence: dict[str, Any] | None
    asset_key: str | None
    route: dict[str, Any]
    decision: dict[str, Any]
    decision_context: dict[str, Any]
    conversation_slots: dict[str, Any]
    conversation_summary: str | None
    customer_memories: list[dict[str, Any]]
    agents_invoked: list[str]
    specialist_results: list[dict[str, Any]]
    read_results: list[dict[str, Any]]
    policy_decision: dict[str, Any] | None
    business_result: dict[str, Any] | None
    pending_confirmation: dict[str, Any] | None
    guarded_contents: list[str]
    reply: str | None
    status: str
```

- [ ] **Step 4: Run schema tests to verify green**

Run:

```powershell
cd python-impl
pytest tests/unit/test_router_supervisor.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

Run:

```powershell
git add python-impl/src/smart_cs/agents/state.py python-impl/tests/unit/test_router_supervisor.py
git commit -m "feat: expand decision and runtime state schemas"
```

## Task 2: Move Prompts Out Of Model Factory And Use Structured Context

**Files:**
- Create: `python-impl/src/smart_cs/infrastructure/prompts.py`
- Modify: `python-impl/src/smart_cs/agents/router.py`
- Modify: `python-impl/src/smart_cs/agents/supervisor.py`
- Modify: `python-impl/src/smart_cs/infrastructure/model_factory.py`
- Modify: `python-impl/tests/unit/test_router_supervisor.py`

- [ ] **Step 1: Write failing prompt/context adapter tests**

Append to `python-impl/tests/unit/test_router_supervisor.py`:

```python
from smart_cs.agents.state import ConversationSlots, RouterContext, SupervisorContext
from smart_cs.infrastructure.model_factory import LangChainDecisionModel
from smart_cs.infrastructure.prompts import (
    PROMPT_VERSION,
    ROUTER_SYSTEM_PROMPT,
    SUPERVISOR_SYSTEM_PROMPT,
)


class RecordingStructuredModel:
    def __init__(self, result: dict):
        self.result = result
        self.calls: list[object] = []

    def invoke(self, messages):
        self.calls.append(messages)
        return self.result


class RecordingChatModel:
    def __init__(self) -> None:
        self.models: dict[object, RecordingStructuredModel] = {}

    def with_structured_output(self, schema):
        if schema is RouteAnalysis:
            model = RecordingStructuredModel({"intent": "order", "entities": {"order_id": "O1001"}})
        elif schema is SupervisorDecision:
            model = RecordingStructuredModel({"agents": ["OrderAgent"], "action": "read"})
        else:
            raise AssertionError(f"unexpected schema {schema}")
        self.models[schema] = model
        return model


def test_prompts_have_version_and_role_boundaries() -> None:
    assert PROMPT_VERSION == "decision-memory-v1"
    assert "只输出 RouteAnalysis" in ROUTER_SYSTEM_PROMPT
    assert "不授权工具" in ROUTER_SYSTEM_PROMPT
    assert "只输出 SupervisorDecision" in SUPERVISOR_SYSTEM_PROMPT
    assert "写动作必须" in SUPERVISOR_SYSTEM_PROMPT


def test_langchain_decision_model_invokes_structured_models_with_message_list() -> None:
    chat = RecordingChatModel()
    model = LangChainDecisionModel(chat)
    router_context = RouterContext(
        current_message="查询 O1001",
        conversation_slots=ConversationSlots(active_order_id=None),
    )
    route = model.route(router_context)
    supervisor_context = SupervisorContext(current_message="查询 O1001", route=route)

    decision = model.plan(supervisor_context)

    routing_messages = chat.models[RouteAnalysis].calls[0]
    planning_messages = chat.models[SupervisorDecision].calls[0]
    assert routing_messages[0].content == ROUTER_SYSTEM_PROMPT
    assert "current_message" in routing_messages[1].content
    assert planning_messages[0].content == SUPERVISOR_SYSTEM_PROMPT
    assert "route" in planning_messages[1].content
    assert decision.agents == ["OrderAgent"]


def test_rules_model_marks_follow_up_when_slots_exist() -> None:
    context = RouterContext(
        current_message="那我要退货",
        conversation_slots=ConversationSlots(active_order_id="O1001"),
    )

    route = RulesDecisionModel().route(context)

    assert route.intent == "after_sales"
    assert route.turn_type == "follow_up"


def test_rules_after_sales_plan_includes_policy_knowledge_agent() -> None:
    route = RouteAnalysis(
        intent="after_sales",
        entities={"order_id": "O1001"},
        risk="medium",
    )

    decision = RulesDecisionModel().plan(
        SupervisorContext(current_message="订单 O1001 鞋底开胶，申请退款", route=route)
    )

    assert decision.agents == ["OrderAgent", "KnowledgeAgent", "AfterSalesAgent"]
    assert "requires_policy_check" in decision.planning_flags


def test_after_sales_plan_without_knowledge_agent_is_rejected() -> None:
    with pytest.raises(ValueError, match="KnowledgeAgent"):
        validate_decision(
            SupervisorDecision(
                agents=["OrderAgent", "AfterSalesAgent"],
                action="draft_after_sales",
                requires_confirmation=True,
            )
        )
```

- [ ] **Step 2: Run prompt tests to verify red**

Run:

```powershell
cd python-impl
pytest tests/unit/test_router_supervisor.py::test_prompts_have_version_and_role_boundaries tests/unit/test_router_supervisor.py::test_langchain_decision_model_invokes_structured_models_with_message_list -q
```

Expected: FAIL because `prompts.py` and context-based model methods do not exist.

- [ ] **Step 3: Add prompt constants**

Create `python-impl/src/smart_cs/infrastructure/prompts.py`:

```python
from __future__ import annotations

PROMPT_VERSION = "decision-memory-v1"

ROUTER_SYSTEM_PROMPT = """你是电商客服 RouterAgent。
只输出 RouteAnalysis。
你只分析客户本轮消息的 intent、entities、risk、confidence、turn_type、missing_entities、escalation_signals、referenced_memory_ids。
不选择 specialist agent。
不授权工具。
不创建售后、退款、换货或人工接入动作。
不生成最终客服回复。
当前用户消息优先于历史消息、摘要和长期记忆。
如果使用记忆辅助判断，必须填写 referenced_memory_ids。
实体必须来自当前消息或明确给定的上下文，不得臆造。
"""

SUPERVISOR_SYSTEM_PROMPT = """你是电商客服 SupervisorAgent。
只输出 SupervisorDecision。
你只规划 specialist 执行顺序和 action，不执行工具。
agents 只能来自声明列表。
action 只能是 read、draft_after_sales、draft_handoff。
写动作必须 requires_confirmation=True。
draft_after_sales 必须先有 OrderAgent 和 KnowledgeAgent，最后是 AfterSalesAgent。
draft_handoff 最后必须是 HandoffAgent。
高风险、低置信度、规则冲突、多次失败或用户明确要求人工时应规划 handoff。
如果缺少必要实体，填写 missing_entities，不得编造。
"""
```

- [ ] **Step 4: Change Router/Supervisor protocols to context input**

Modify `python-impl/src/smart_cs/agents/router.py`:

```python
from __future__ import annotations

from typing import Protocol

from smart_cs.agents.state import RouteAnalysis, RouterContext


class RoutingDecisionModel(Protocol):
    def route(self, context: RouterContext) -> RouteAnalysis:
        raise NotImplementedError


class RouterAgent:
    """Analyze customer intent without choosing or authorizing tools."""

    def __init__(self, decision_model: RoutingDecisionModel) -> None:
        self.decision_model = decision_model

    def analyze(self, context: RouterContext) -> RouteAnalysis:
        return self.decision_model.route(context)
```

Modify `python-impl/src/smart_cs/agents/supervisor.py` protocol and `plan()` signature:

```python
class PlanningDecisionModel(Protocol):
    def plan(self, context: SupervisorContext) -> SupervisorDecision:
        raise NotImplementedError
```

and:

```python
    def plan(self, context: SupervisorContext, *, has_image: bool = False) -> SupervisorDecision:
        return validate_decision(self.decision_model.plan(context), has_image=has_image)
```

Update `validate_decision()` so ordinary after-sales plans require policy retrieval before drafting:

```python
    if decision.action == "draft_after_sales":
        if "OrderAgent" not in decision.agents:
            raise ValueError("Action draft_after_sales requires OrderAgent")
        if "KnowledgeAgent" not in decision.agents:
            raise ValueError("Action draft_after_sales requires KnowledgeAgent")
        if decision.agents.index("OrderAgent") > decision.agents.index("AfterSalesAgent"):
            raise ValueError("Action draft_after_sales requires OrderAgent before AfterSalesAgent")
        if decision.agents.index("KnowledgeAgent") > decision.agents.index("AfterSalesAgent"):
            raise ValueError("Action draft_after_sales requires KnowledgeAgent before AfterSalesAgent")
```

Keep `synthesize()` unchanged.

- [ ] **Step 5: Update model factory to use SystemMessage/HumanMessage**

Modify `python-impl/src/smart_cs/infrastructure/model_factory.py`:

```python
from langchain_core.messages import HumanMessage, SystemMessage

from smart_cs.agents.state import RouteAnalysis, RouterContext, SupervisorContext, SupervisorDecision
from smart_cs.infrastructure.prompts import ROUTER_SYSTEM_PROMPT, SUPERVISOR_SYSTEM_PROMPT
```

Update `RulesDecisionModel`:

```python
    def route(self, context: RouterContext) -> RouteAnalysis:
        message = context.current_message
        entities: dict[str, str] = {}
        order_match = ORDER_ID_PATTERN.search(message)
        if order_match is not None:
            entities["order_id"] = order_match.group(1).upper()
        turn_type = self._infer_turn_type(message, context)

        if self._contains(message, self._handoff_keywords):
            return RouteAnalysis(
                intent="handoff",
                entities=entities,
                risk="high",
                confidence="high",
                turn_type=turn_type,
                escalation_signals=["handoff_requested"],
            )
        if self._contains(message, self._knowledge_domain_keywords) and self._contains(
            message, self._knowledge_question_keywords
        ):
            return RouteAnalysis(
                intent="knowledge",
                entities=entities,
                confidence="high",
                turn_type=turn_type,
            )
        if self._contains(message, self._after_sales_keywords):
            return RouteAnalysis(
                intent="after_sales",
                entities=entities,
                risk="medium",
                confidence="high",
                turn_type=turn_type,
            )
        if self._contains(message, self._product_keywords):
            return RouteAnalysis(intent="product", entities=entities, confidence="high", turn_type=turn_type)
        if entities or self._contains(message, self._order_keywords):
            return RouteAnalysis(intent="order", entities=entities, confidence="high", turn_type=turn_type)
        return RouteAnalysis(intent="knowledge", entities=entities, confidence="medium", turn_type=turn_type)

    def plan(self, context: SupervisorContext) -> SupervisorDecision:
        route = context.route
        if route.intent == "after_sales":
            return SupervisorDecision(
                agents=["OrderAgent", "KnowledgeAgent", "AfterSalesAgent"],
                action="draft_after_sales",
                planning_flags=["requires_order_fact", "requires_policy_check"],
            )
        if route.intent == "handoff":
            return SupervisorDecision(
                agents=["HandoffAgent"],
                action="draft_handoff",
                handoff_reason="customer_requested_or_high_risk",
            )
        if route.intent == "product":
            return SupervisorDecision(agents=["ProductAgent"], action="read")
        if route.intent == "order":
            return SupervisorDecision(agents=["OrderAgent"], action="read")
        return SupervisorDecision(agents=["KnowledgeAgent"], action="read")

    @staticmethod
    def _infer_turn_type(message: str, context: RouterContext) -> str:
        stripped = message.strip()
        if stripped in {"确认", "可以", "提交", "提交吧", "确认提交"}:
            return "confirmation_like"
        if stripped in {"取消", "不用了", "先不要", "不提交"}:
            return "rejection_like"
        if any(marker in message for marker in ("不是", "说错", "改成", "换成")):
            return "correction"
        if context.conversation_slots.active_order_id and any(
            marker in message for marker in ("那", "这个", "刚才", "它", "这单")
        ):
            return "follow_up"
        return "new_request"
```

Update `LangChainDecisionModel`:

```python
    def route(self, context: RouterContext) -> RouteAnalysis:
        result = self._routing_model.invoke(
            [
                SystemMessage(content=ROUTER_SYSTEM_PROMPT),
                HumanMessage(content=context.model_dump_json()),
            ]
        )
        return RouteAnalysis.model_validate(result)

    def plan(self, context: SupervisorContext) -> SupervisorDecision:
        result = self._planning_model.invoke(
            [
                SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
                HumanMessage(content=context.model_dump_json()),
            ]
        )
        return SupervisorDecision.model_validate(result)
```

- [ ] **Step 6: Update existing tests to instantiate context**

In `test_rules_agents_plan_after_sales_in_business_order`, replace direct message calls and update the expected after-sales plan:

```python
router_context = RouterContext(current_message=message)
route = RouterAgent(model).analyze(router_context)
decision = SupervisorAgent(model).plan(SupervisorContext(current_message=message, route=route))

assert decision.agents == ["OrderAgent", "KnowledgeAgent", "AfterSalesAgent"]
```

- [ ] **Step 7: Run router/supervisor tests**

Run:

```powershell
cd python-impl
pytest tests/unit/test_router_supervisor.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit Task 2**

Run:

```powershell
git add python-impl/src/smart_cs/infrastructure/prompts.py python-impl/src/smart_cs/agents/router.py python-impl/src/smart_cs/agents/supervisor.py python-impl/src/smart_cs/infrastructure/model_factory.py python-impl/tests/unit/test_router_supervisor.py
git commit -m "feat: use prompt files and structured decision contexts"
```

## Task 3: Add ToolPolicy And caller_agent Enforcement

**Files:**
- Create: `python-impl/src/smart_cs/tools/policy.py`
- Modify: `python-impl/src/smart_cs/tools/executor.py`
- Modify: `python-impl/src/smart_cs/agents/specialists.py`
- Modify: `python-impl/tests/unit/test_tools.py`
- Create: `python-impl/tests/unit/test_tool_policy.py`

- [ ] **Step 1: Write failing tool policy tests**

Create `python-impl/tests/unit/test_tool_policy.py`:

```python
from __future__ import annotations

import pytest

from smart_cs.domain.errors import ToolPermissionError
from smart_cs.infrastructure.database import Database
from smart_cs.infrastructure.repositories import SqlRepository
from smart_cs.tools.executor import AuthorizedToolExecutor
from smart_cs.tools.policy import default_tool_registry


def repository(tmp_path):
    repo = SqlRepository(Database(f"sqlite:///{tmp_path / 'policy.db'}"))
    repo.create_schema()
    repo.seed_demo_data()
    return repo


def test_default_tool_registry_exposes_policy_view() -> None:
    registry = default_tool_registry()
    view = registry.as_view()

    assert any(item["name"] == "lookup_order" for item in view)
    assert registry.get("draft_after_sales").requires_confirmation is True
    assert registry.get("lookup_order").allowed_agents == frozenset({"OrderAgent"})


def test_executor_rejects_tool_for_wrong_agent(tmp_path) -> None:
    tools = AuthorizedToolExecutor(repository(tmp_path))

    with pytest.raises(ToolPermissionError, match="not allowed"):
        tools.invoke(
            "lookup_order",
            {"customer_id": "C001", "order_id": "O1001"},
            caller_agent="ProductAgent",
        )


def test_executor_allows_declared_agent(tmp_path) -> None:
    tools = AuthorizedToolExecutor(repository(tmp_path))

    result = tools.invoke(
        "lookup_order",
        {"customer_id": "C001", "order_id": "O1001"},
        caller_agent="OrderAgent",
    )

    assert result["order_id"] == "O1001"


def test_confirm_action_tools_require_confirm_action_node(tmp_path) -> None:
    repo = repository(tmp_path)
    action = repo.create_pending_action(
        "C001",
        "after_sales",
        "鞋底开胶",
        order_id="O1001",
        conversation_id="conv-confirm-policy",
    )
    tools = AuthorizedToolExecutor(repo)

    with pytest.raises(ToolPermissionError, match="not allowed"):
        tools.submit_confirmed_action(
            action.id,
            "C001",
            caller_agent="AfterSalesAgent",
        )

    result = tools.submit_confirmed_action(
        action.id,
        "C001",
        caller_agent="ConfirmActionNode",
    )

    assert result["status"] == "submitted"


def test_cancel_action_tool_requires_confirm_action_node(tmp_path) -> None:
    repo = repository(tmp_path)
    action = repo.create_pending_action(
        "C001",
        "after_sales",
        "鞋底开胶",
        order_id="O1001",
        conversation_id="conv-cancel-policy",
    )
    tools = AuthorizedToolExecutor(repo)

    with pytest.raises(ToolPermissionError, match="not allowed"):
        tools.cancel_pending_action(
            action.id,
            "C001",
            caller_agent="AfterSalesAgent",
        )

    result = tools.cancel_pending_action(
        action.id,
        "C001",
        caller_agent="ConfirmActionNode",
    )

    assert result["status"] == "cancelled"
```

- [ ] **Step 2: Run tool policy tests to verify red**

Run:

```powershell
cd python-impl
pytest tests/unit/test_tool_policy.py -q
```

Expected: FAIL because `smart_cs.tools.policy` and `caller_agent` are not implemented.

- [ ] **Step 3: Implement ToolPolicy and ToolRegistry**

Create `python-impl/src/smart_cs/tools/policy.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ToolPolicy:
    name: str
    risk_level: Literal["low", "medium", "high"]
    allowed_agents: frozenset[str]
    requires_confirmation: bool
    idempotent: bool


class ToolRegistry:
    def __init__(self, policies: list[ToolPolicy]) -> None:
        self._policies = {policy.name: policy for policy in policies}

    def get(self, name: str) -> ToolPolicy:
        try:
            return self._policies[name]
        except KeyError as error:
            raise ValueError(f"Unknown customer tool: {name}") from error

    def as_view(self) -> list[dict[str, object]]:
        return [
            {
                "name": policy.name,
                "risk_level": policy.risk_level,
                "allowed_agents": sorted(policy.allowed_agents),
                "requires_confirmation": policy.requires_confirmation,
                "idempotent": policy.idempotent,
            }
            for policy in self._policies.values()
        ]


def default_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ToolPolicy("search_products", "low", frozenset({"ProductAgent"}), False, True),
            ToolPolicy("lookup_order", "medium", frozenset({"OrderAgent"}), False, True),
            ToolPolicy("draft_after_sales", "medium", frozenset({"AfterSalesAgent"}), True, True),
            ToolPolicy("draft_handoff", "medium", frozenset({"HandoffAgent"}), True, True),
            ToolPolicy("submit_confirmed_action", "high", frozenset({"ConfirmActionNode"}), True, True),
            ToolPolicy("cancel_pending_action", "medium", frozenset({"ConfirmActionNode"}), False, True),
        ]
    )


def default_tool_policy_view() -> list[dict[str, object]]:
    return default_tool_registry().as_view()
```

- [ ] **Step 4: Enforce caller_agent in executor and specialists**

Modify `AuthorizedToolExecutor.__init__`, `_authorize_tool()`, `invoke()`, `submit_confirmed_action()`, and `cancel_pending_action()`:

```python
from smart_cs.tools.policy import ToolPolicy, ToolRegistry, default_tool_registry


def __init__(
    self,
    repository: CustomerFactsRepository,
    tool_registry: ToolRegistry | None = None,
) -> None:
    self.repository = repository
    self.tool_registry = tool_registry or default_tool_registry()


def _authorize_tool(self, tool_name: str, caller_agent: str) -> ToolPolicy:
    policy = self.tool_registry.get(tool_name)
    if caller_agent not in policy.allowed_agents:
        raise ToolPermissionError(f"Tool {tool_name} is not allowed for {caller_agent}")
    return policy


def invoke(
    self,
    tool_name: str,
    arguments: dict[str, Any],
    *,
    caller_agent: str,
    turn_fence: TurnFence | None = None,
) -> dict[str, Any]:
    provided_arguments = dict(arguments)
    policy = self._authorize_tool(tool_name, caller_agent)
    if tool_name in self._write_handlers and not policy.requires_confirmation:
        raise ToolPermissionError(f"Write tool {tool_name} must require confirmation")
    # Keep the existing read/write dispatch and audit logic after these checks.


def submit_confirmed_action(
    self,
    action_id: str,
    customer_id: str,
    *,
    caller_agent: str,
    turn_fence: TurnFence | None = None,
) -> dict[str, Any]:
    self._authorize_tool("submit_confirmed_action", caller_agent)
    arguments = {"action_id": action_id, "customer_id": customer_id}

    def operation(session: Any) -> dict[str, Any]:
        self._require_write_fence(turn_fence, customer_id, session)
        action, ticket = self.repository.submit_pending_action(action_id, customer_id, session=session)
        return self._action_result(action, ticket)

    return self._audited_write_call("submit_confirmed_action", arguments, operation)


def cancel_pending_action(
    self,
    action_id: str,
    customer_id: str,
    *,
    caller_agent: str,
    turn_fence: TurnFence | None = None,
) -> dict[str, Any]:
    self._authorize_tool("cancel_pending_action", caller_agent)
    arguments = {"action_id": action_id, "customer_id": customer_id}

    def operation(session: Any) -> dict[str, Any]:
        self._require_write_fence(turn_fence, customer_id, session)
        action = self.repository.cancel_pending_action(action_id, customer_id, session=session)
        return self._action_result(action)

    return self._audited_write_call("cancel_pending_action", arguments, operation)
```

Modify `SpecialistDispatcher` calls:

```python
return self.executor.invoke("search_products", arguments, caller_agent="ProductAgent")
return self.executor.invoke("lookup_order", arguments, caller_agent="OrderAgent")
return self.executor.invoke(
    "draft_after_sales",
    arguments,
    caller_agent="AfterSalesAgent",
    turn_fence=turn_fence,
)
return self.executor.invoke(
    "draft_handoff",
    arguments,
    caller_agent="HandoffAgent",
    turn_fence=turn_fence,
)
```

Update existing direct executor tests in `test_tools.py` to pass the correct `caller_agent`:

```python
tools.invoke(
    "lookup_order",
    {"customer_id": "C002", "order_id": "O1001"},
    caller_agent="OrderAgent",
)
tools.invoke(
    "draft_after_sales",
    {"customer_id": "C001", "order_id": "O1001", "reason": "鞋底开胶"},
    caller_agent="AfterSalesAgent",
)
tools.invoke(
    "draft_handoff",
    {"customer_id": "C001", "reason": "需要人工沟通"},
    caller_agent="HandoffAgent",
)
tools.invoke("search_products", {"query": "跑鞋"}, caller_agent="ProductAgent")
```

- [ ] **Step 5: Run tool tests**

Run:

```powershell
cd python-impl
pytest tests/unit/test_tool_policy.py tests/unit/test_tools.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 3**

Run:

```powershell
git add python-impl/src/smart_cs/tools/policy.py python-impl/src/smart_cs/tools/executor.py python-impl/src/smart_cs/agents/specialists.py python-impl/tests/unit/test_tool_policy.py python-impl/tests/unit/test_tools.py
git commit -m "feat: enforce tool policy by caller agent"
```

## Task 4: Add ContextProjector With trim_messages

**Files:**
- Create: `python-impl/src/smart_cs/application/context_projector.py`
- Create: `python-impl/tests/unit/test_context_projector.py`

- [ ] **Step 1: Write failing ContextProjector tests**

Create `python-impl/tests/unit/test_context_projector.py`:

```python
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from smart_cs.agents.state import ConversationSlots, RouteAnalysis
from smart_cs.application.context_projector import ContextProjector


def test_router_context_projects_trimmed_recent_messages_and_slots() -> None:
    projector = ContextProjector(max_context_tokens=200)
    state = {
        "message": "那我要退货",
        "messages": [
            HumanMessage(content="查询订单 O1001"),
            AIMessage(content="订单 O1001 当前状态为 delivered。"),
            HumanMessage(content="那我要退货"),
        ],
        "conversation_slots": {"active_order_id": "O1001"},
        "conversation_summary": "用户正在围绕订单 O1001 咨询。",
        "customer_memories": [
            {
                "memory_id": "mem-1",
                "memory_type": "preference",
                "value": {"tone": "concise"},
                "source": "human_review",
            }
        ],
        "has_image": False,
        "visual_evidence": None,
        "pending_confirmation": None,
    }

    context = projector.build_router_context(state)

    assert context.current_message == "那我要退货"
    assert context.conversation_slots.active_order_id == "O1001"
    assert context.conversation_summary == "用户正在围绕订单 O1001 咨询。"
    assert context.recent_messages[-1] == {"role": "human", "content": "那我要退货"}
    assert context.customer_memories[0].memory_id == "mem-1"


def test_supervisor_context_includes_agent_capabilities_and_tool_policy_view() -> None:
    projector = ContextProjector(max_context_tokens=200)
    state = {
        "message": "订单 O1001 鞋底开胶，申请退款",
        "messages": [HumanMessage(content="订单 O1001 鞋底开胶，申请退款")],
        "conversation_slots": {},
        "conversation_summary": None,
        "customer_memories": [],
        "has_image": True,
        "visual_evidence": {"summary": "鞋底疑似开胶"},
        "pending_confirmation": None,
    }
    route = RouteAnalysis(intent="after_sales", entities={"order_id": "O1001"}, risk="medium")

    context = projector.build_supervisor_context(state, route)

    assert context.route.intent == "after_sales"
    assert context.has_image is True
    assert "OrderAgent" in context.agent_capabilities
    assert any(policy["name"] == "draft_after_sales" for policy in context.tool_policies)
    assert "写动作必须 requires_confirmation=True" in context.planning_constraints
```

- [ ] **Step 2: Run ContextProjector tests to verify red**

Run:

```powershell
cd python-impl
pytest tests/unit/test_context_projector.py -q
```

Expected: FAIL because `ContextProjector` does not exist.

- [ ] **Step 3: Implement ContextProjector**

Create `python-impl/src/smart_cs/application/context_projector.py`:

```python
from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages

from smart_cs.agents.state import (
    ConversationSlots,
    MemoryView,
    RouteAnalysis,
    RouterContext,
    SupervisorContext,
)
from smart_cs.tools.policy import default_tool_policy_view


AGENT_CAPABILITIES = {
    "ProductAgent": "Read-only product search and product facts.",
    "OrderAgent": "Read-only order and logistics lookup for the current customer.",
    "KnowledgeAgent": "Read-only policy retrieval with citations.",
    "VisionAgent": "Read-only visual evidence projection for after-sales images.",
    "AfterSalesAgent": "Creates after-sales draft only after policy allows it.",
    "HandoffAgent": "Creates human handoff draft when handoff policy requires it.",
}


class ContextProjector:
    def __init__(self, *, max_context_tokens: int = 2048) -> None:
        self.max_context_tokens = max_context_tokens

    def build_router_context(self, state: dict[str, Any]) -> RouterContext:
        return RouterContext(
            current_message=str(state["message"]),
            recent_messages=self._recent_messages(state),
            conversation_summary=state.get("conversation_summary"),
            conversation_slots=self._slots(state),
            pending_action=state.get("pending_confirmation"),
            customer_memories=self._memory_views(state),
            has_image=bool(state.get("has_image")),
            visual_evidence=state.get("visual_evidence"),
        )

    def build_supervisor_context(
        self, state: dict[str, Any], route: RouteAnalysis
    ) -> SupervisorContext:
        return SupervisorContext(
            current_message=str(state["message"]),
            route=route,
            recent_messages=self._recent_messages(state),
            conversation_summary=state.get("conversation_summary"),
            conversation_slots=self._slots(state),
            pending_action=state.get("pending_confirmation"),
            customer_memories=self._memory_views(state),
            has_image=bool(state.get("has_image")),
            visual_evidence=state.get("visual_evidence"),
            agent_capabilities=AGENT_CAPABILITIES,
            tool_policies=default_tool_policy_view(),
            planning_constraints=[
                "写动作必须 requires_confirmation=True",
                "draft_after_sales 必须包含 OrderAgent 和 KnowledgeAgent，且 AfterSalesAgent 位于最后",
                "draft_handoff 必须由 HandoffAgent 位于最后",
            ],
        )

    def _recent_messages(self, state: dict[str, Any]) -> list[dict[str, str]]:
        messages = list(state.get("messages") or [])
        trimmed = trim_messages(
            messages,
            strategy="last",
            token_counter=count_tokens_approximately,
            max_tokens=self.max_context_tokens,
            start_on="human",
            end_on=("human", "tool"),
        )
        return [self._message_view(message) for message in trimmed]

    @staticmethod
    def _message_view(message: AnyMessage) -> dict[str, str]:
        if isinstance(message, HumanMessage):
            role = "human"
        elif isinstance(message, AIMessage):
            role = "ai"
        elif isinstance(message, SystemMessage):
            role = "system"
        else:
            role = getattr(message, "type", "message")
        return {"role": role, "content": str(message.content)}

    @staticmethod
    def _slots(state: dict[str, Any]) -> ConversationSlots:
        return ConversationSlots.model_validate(state.get("conversation_slots") or {})

    @staticmethod
    def _memory_views(state: dict[str, Any]) -> list[MemoryView]:
        return [MemoryView.model_validate(item) for item in state.get("customer_memories") or []]
```

- [ ] **Step 4: Run ContextProjector tests to verify green**

Run:

```powershell
cd python-impl
pytest tests/unit/test_context_projector.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 4**

Run:

```powershell
git add python-impl/src/smart_cs/application/context_projector.py python-impl/tests/unit/test_context_projector.py
git commit -m "feat: add structured context projector"
```

## Task 5: Add ConversationSlots, SlotCarry, And StateUpdater

**Files:**
- Create: `python-impl/src/smart_cs/application/state_update.py`
- Create: `python-impl/tests/unit/test_state_update.py`

- [ ] **Step 1: Write failing slot/state update tests**

Create `python-impl/tests/unit/test_state_update.py`:

```python
from __future__ import annotations

from smart_cs.agents.state import ConversationSlots, RouteAnalysis
from smart_cs.application.state_update import StateUpdater, carry_slots


def test_carry_slots_inherits_active_order_for_follow_up_after_sales() -> None:
    route = RouteAnalysis(intent="after_sales", turn_type="follow_up", entities={})
    slots = ConversationSlots(active_order_id="O1001")

    updated = carry_slots(route, slots)

    assert updated.entities["order_id"] == "O1001"


def test_carry_slots_does_not_override_explicit_correction() -> None:
    route = RouteAnalysis(
        intent="order",
        turn_type="correction",
        entities={"order_id": "O1002"},
    )
    slots = ConversationSlots(active_order_id="O1001")

    updated = carry_slots(route, slots)

    assert updated.entities["order_id"] == "O1002"


def test_state_updater_records_pending_and_confirmed_ticket() -> None:
    updater = StateUpdater()
    state = {
        "route": {"intent": "after_sales", "entities": {"order_id": "O1001"}},
        "conversation_slots": {},
        "business_result": {
            "action_id": "A1",
            "action_type": "after_sales",
            "order_id": "O1001",
            "status": "pending_confirmation",
        },
        "specialist_results": [{"order_id": "O1001", "status": "delivered"}],
    }

    pending = updater.update(state)["conversation_slots"]

    assert pending["active_order_id"] == "O1001"
    assert pending["pending_action"]["action_id"] == "A1"
    assert pending["action_status"] == "pending_confirmation"

    confirmed_state = {
        **state,
        "conversation_slots": pending,
        "business_result": {
            "action_id": "A1",
            "action_type": "after_sales",
            "order_id": "O1001",
            "status": "submitted",
            "ticket_id": "T1",
        },
    }
    confirmed = updater.update(confirmed_state)["conversation_slots"]

    assert confirmed["active_ticket_id"] == "T1"
    assert confirmed["action_status"] == "submitted"
    assert confirmed["pending_action"] is None


def test_state_updater_records_cancelled_action_status() -> None:
    updater = StateUpdater()
    state = {
        "route": {"intent": "after_sales", "entities": {"order_id": "O1001"}},
        "conversation_slots": {
            "active_order_id": "O1001",
            "pending_action": {"action_id": "A1"},
            "action_status": "pending_confirmation",
        },
        "business_result": {
            "action_id": "A1",
            "action_type": "after_sales",
            "order_id": "O1001",
            "status": "cancelled",
        },
        "specialist_results": [],
    }

    cancelled = updater.update(state)["conversation_slots"]

    assert cancelled["active_order_id"] == "O1001"
    assert cancelled["pending_action"] is None
    assert cancelled["action_status"] == "cancelled"
```

- [ ] **Step 2: Run state update tests to verify red**

Run:

```powershell
cd python-impl
pytest tests/unit/test_state_update.py -q
```

Expected: FAIL because `state_update.py` does not exist.

- [ ] **Step 3: Implement carry_slots and StateUpdater**

Create `python-impl/src/smart_cs/application/state_update.py`:

```python
from __future__ import annotations

from typing import Any

from smart_cs.agents.state import ConversationSlots, RouteAnalysis
from smart_cs.domain.enums import ActionStatus


FOLLOW_UP_TURN_TYPES = {"follow_up", "correction", "information_update"}


def carry_slots(route: RouteAnalysis, slots: ConversationSlots) -> RouteAnalysis:
    if (
        "order_id" not in route.entities
        and route.turn_type in FOLLOW_UP_TURN_TYPES
        and slots.active_order_id is not None
    ):
        entities = dict(route.entities)
        entities["order_id"] = slots.active_order_id
        return route.model_copy(update={"entities": entities})
    return route


class StateUpdater:
    def update(self, state: dict[str, Any]) -> dict[str, Any]:
        slots = ConversationSlots.model_validate(state.get("conversation_slots") or {})
        route = RouteAnalysis.model_validate(state.get("route") or {"intent": "knowledge"})
        business_result = state.get("business_result") or {}
        specialist_results = list(state.get("specialist_results") or [])

        route_order_id = route.entities.get("order_id")
        if route_order_id is not None:
            slots.active_order_id = route_order_id
        result_order_id = business_result.get("order_id")
        if result_order_id is not None:
            slots.active_order_id = str(result_order_id)

        slots.last_intent = route.intent
        slots.last_entities = dict(route.entities)
        if specialist_results:
            slots.last_tool_results = self._compact_results(specialist_results)

        status = business_result.get("status")
        if status == ActionStatus.PENDING_CONFIRMATION.value:
            slots.pending_action = dict(business_result)
            slots.action_status = ActionStatus.PENDING_CONFIRMATION.value
        elif status in {ActionStatus.SUBMITTED.value, ActionStatus.CANCELLED.value}:
            slots.pending_action = None
            slots.action_status = str(status)
            ticket_id = business_result.get("ticket_id")
            if ticket_id is not None:
                slots.active_ticket_id = str(ticket_id)

        return {"conversation_slots": slots.model_dump()}

    @staticmethod
    def _compact_results(results: list[dict[str, Any]]) -> dict[str, Any]:
        compact: dict[str, Any] = {}
        for index, result in enumerate(results):
            key = result.get("action_type") or result.get("status") or f"result_{index}"
            compact[str(key)] = {
                item_key: item_value
                for item_key, item_value in result.items()
                if item_key in {"order_id", "product_id", "status", "action_type", "ticket_id"}
            }
        return compact
```

- [ ] **Step 4: Run state update tests to verify green**

Run:

```powershell
cd python-impl
pytest tests/unit/test_state_update.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 5**

Run:

```powershell
git add python-impl/src/smart_cs/application/state_update.py python-impl/tests/unit/test_state_update.py
git commit -m "feat: add slot carry and state updater"
```

## Task 6: Add Minimal PolicyEngine And Split Read/Write Specialists

**Files:**
- Create: `python-impl/src/smart_cs/application/policy.py`
- Modify: `python-impl/src/smart_cs/agents/specialists.py`
- Modify: `python-impl/src/smart_cs/application/agent_runtime.py`
- Create: `python-impl/tests/unit/test_policy_engine.py`

- [ ] **Step 1: Write failing policy tests**

Create `python-impl/tests/unit/test_policy_engine.py`:

```python
from __future__ import annotations

from smart_cs.application.policy import PolicyDecision, PolicyEngine


def test_policy_engine_allows_draft_for_delivered_after_sales_order() -> None:
    decision = PolicyEngine().evaluate_after_sales(
        order_result={"order_id": "O1001", "status": "delivered"},
        knowledge_result={"citations": [{"source": "policy.md"}]},
        visual_evidence=None,
    )

    assert decision == PolicyDecision(
        eligible=True,
        reason_code="AFTER_SALES_DRAFT_ALLOWED",
        explanation="订单已签收，可创建售后草稿并等待用户确认。",
        next_action="allow_draft",
        requires_human_review=False,
    )


def test_policy_engine_handoff_when_visual_evidence_is_uncertain() -> None:
    decision = PolicyEngine().evaluate_after_sales(
        order_result={"order_id": "O1001", "status": "delivered"},
        knowledge_result={"citations": [{"source": "policy.md"}]},
        visual_evidence={"usable_for_draft": False},
    )

    assert decision.next_action == "handoff"
    assert decision.requires_human_review is True


def test_policy_engine_explains_when_order_missing() -> None:
    decision = PolicyEngine().evaluate_after_sales(
        order_result={"status": "information_required"},
        knowledge_result={},
        visual_evidence=None,
    )

    assert decision.eligible is False
    assert decision.next_action == "explain"
    assert decision.reason_code == "ORDER_REQUIRED"


def test_policy_engine_requires_policy_context_for_after_sales() -> None:
    decision = PolicyEngine().evaluate_after_sales(
        order_result={"order_id": "O1001", "status": "delivered"},
        knowledge_result={},
        visual_evidence=None,
    )

    assert decision.eligible is False
    assert decision.next_action == "handoff"
    assert decision.reason_code == "POLICY_CONTEXT_REQUIRED"
    assert decision.requires_human_review is True
```

- [ ] **Step 2: Run policy tests to verify red**

Run:

```powershell
cd python-impl
pytest tests/unit/test_policy_engine.py -q
```

Expected: FAIL because `PolicyEngine` does not exist.

- [ ] **Step 3: Implement PolicyEngine**

Create `python-impl/src/smart_cs/application/policy.py`:

```python
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class PolicyDecision(BaseModel):
    eligible: bool
    reason_code: str
    explanation: str
    next_action: Literal["allow_draft", "explain", "handoff"]
    requires_human_review: bool = False


class PolicyEngine:
    def evaluate_after_sales(
        self,
        *,
        order_result: dict[str, Any],
        knowledge_result: dict[str, Any],
        visual_evidence: dict[str, Any] | None,
    ) -> PolicyDecision:
        if order_result.get("order_id") is None:
            return PolicyDecision(
                eligible=False,
                reason_code="ORDER_REQUIRED",
                explanation="需要先提供订单编号，才能判断售后资格。",
                next_action="explain",
            )
        if not knowledge_result.get("citations"):
            return PolicyDecision(
                eligible=False,
                reason_code="POLICY_CONTEXT_REQUIRED",
                explanation="缺少可引用的售后政策依据，建议转人工审核。",
                next_action="handoff",
                requires_human_review=True,
            )
        if visual_evidence is not None and visual_evidence.get("usable_for_draft") is False:
            return PolicyDecision(
                eligible=False,
                reason_code="VISUAL_EVIDENCE_UNCERTAIN",
                explanation="图片证据暂不能确认问题，建议转人工审核。",
                next_action="handoff",
                requires_human_review=True,
            )
        if order_result.get("status") in {"delivered", "shipped"}:
            return PolicyDecision(
                eligible=True,
                reason_code="AFTER_SALES_DRAFT_ALLOWED",
                explanation="订单已签收，可创建售后草稿并等待用户确认。",
                next_action="allow_draft",
            )
        return PolicyDecision(
            eligible=False,
            reason_code="ORDER_STATUS_NOT_ELIGIBLE",
            explanation="当前订单状态暂不满足创建售后草稿条件。",
            next_action="explain",
        )
```

- [ ] **Step 4: Split SpecialistDispatcher read/write execution**

Modify `python-impl/src/smart_cs/agents/specialists.py`:

```python
READ_AGENTS = {"ProductAgent", "OrderAgent", "KnowledgeAgent", "VisionAgent"}
WRITE_AGENTS = {"AfterSalesAgent", "HandoffAgent"}
```

Add methods:

```python
def execute_read_agents(self, **kwargs) -> SpecialistExecution:
    decision = kwargs["decision"]
    read_agents = [agent for agent in decision.agents if agent in READ_AGENTS]
    if not read_agents:
        return SpecialistExecution(
            agents_invoked=[],
            results=[],
            result={"status": "no_read"},
            pending_confirmation=None,
        )
    read_decision = decision.model_copy(update={"agents": read_agents, "action": "read"})
    return self.execute(**{**kwargs, "decision": read_decision})


def execute_write_agents(self, **kwargs) -> SpecialistExecution:
    decision = kwargs["decision"]
    write_agents = [agent for agent in decision.agents if agent in WRITE_AGENTS]
    if not write_agents:
        return SpecialistExecution(agents_invoked=[], results=[], result={"status": "no_write"})
    write_decision = decision.model_copy(update={"agents": write_agents})
    return self.execute(**{**kwargs, "decision": write_decision})
```

- [ ] **Step 5: Run policy tests**

Run:

```powershell
cd python-impl
pytest tests/unit/test_policy_engine.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 6**

Run:

```powershell
git add python-impl/src/smart_cs/application/policy.py python-impl/src/smart_cs/agents/specialists.py python-impl/tests/unit/test_policy_engine.py
git commit -m "feat: add policy engine and read write specialist split"
```

## Task 7: Add Memory Store, Summary, RemoveMessage, And Memory Writeback

**Files:**
- Create: `python-impl/src/smart_cs/application/memory.py`
- Modify: `python-impl/src/smart_cs/domain/models.py`
- Modify: `python-impl/src/smart_cs/infrastructure/repositories.py`
- Create: `python-impl/tests/unit/test_memory.py`

- [ ] **Step 1: Write failing memory tests**

Create `python-impl/tests/unit/test_memory.py`:

```python
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

from smart_cs.application.memory import (
    ConversationSummarizer,
    MemoryCandidate,
    MemoryDecision,
    MemoryExtractor,
    MemoryPolicy,
    MemoryWriter,
    MemoryWriteback,
    SqlMemoryStoreAdapter,
)
from smart_cs.infrastructure.database import Database
from smart_cs.infrastructure.repositories import SqlRepository


class RecordingStore:
    def __init__(self) -> None:
        self.puts: list[tuple[tuple[str, str, str], str, dict]] = []

    def put(self, namespace, key, value) -> None:
        self.puts.append((namespace, key, value))

    def search(self, namespace, query: str, limit: int):
        return [item for item in self.puts if item[0] == namespace][:limit]


def repo(tmp_path):
    repository = SqlRepository(Database(f"sqlite:///{tmp_path / 'memory.db'}"))
    repository.create_schema()
    repository.seed_demo_data()
    repository.claim_conversation("conv-memory", "C001")
    return repository


def test_memory_candidate_requires_retrieval_and_audit_fields() -> None:
    candidate = MemoryCandidate(
        scope="customer",
        owner_id="C001",
        memory_type="preference",
        key="shoe_size",
        title="Shoe size preference: 42",
        description="User explicitly said they usually wear size 42 shoes.",
        value={"shoe_size": "42"},
        evidence=[{"text": "我一般穿42码", "conversation_id": "conv-memory"}],
        source="user_message",
        confidence="high",
        risk_level="low",
    )

    dumped = candidate.model_dump()

    assert dumped["title"] == "Shoe size preference: 42"
    assert dumped["description"]
    assert dumped["evidence"][0]["text"] == "我一般穿42码"
    assert dumped["review_status"] == "pending"


def test_memory_policy_returns_write_candidate_review_or_discard() -> None:
    policy = MemoryPolicy()

    assert policy.decide({"memory_type": "service_event", "risk_level": "low"}) == MemoryDecision(
        action="write",
        reason="low_risk_service_event",
    )
    assert policy.decide({"memory_type": "preference", "risk_level": "medium"}) == MemoryDecision(
        action="candidate",
        reason="user_preference_candidate",
    )
    assert policy.decide({"memory_type": "sensitive_label", "risk_level": "high"}) == MemoryDecision(
        action="human_review",
        reason="sensitive_memory",
    )
    assert policy.decide({"memory_type": "badcase_candidate", "risk_level": "medium"}) == MemoryDecision(
        action="human_review",
        reason="badcase_requires_review",
    )
    assert policy.decide({"memory_type": "unsupported", "risk_level": "low"}) == MemoryDecision(
        action="discard",
        reason="unsupported_memory",
    )


def test_memory_extractor_creates_pending_submitted_and_cancelled_candidates() -> None:
    extractor = MemoryExtractor()

    pending = extractor.extract(
        {
            "conversation_id": "conv-memory",
            "customer_id": "C001",
            "business_result": {
                "action_id": "A1",
                "action_type": "after_sales",
                "status": "pending_confirmation",
            },
        }
    )
    submitted = extractor.extract(
        {
            "conversation_id": "conv-memory",
            "customer_id": "C001",
            "business_result": {
                "action_id": "A1",
                "action_type": "after_sales",
                "status": "submitted",
                "ticket_id": "T1",
            },
        }
    )
    cancelled = extractor.extract(
        {
            "conversation_id": "conv-memory",
            "customer_id": "C001",
            "business_result": {
                "action_id": "A1",
                "action_type": "after_sales",
                "status": "cancelled",
            },
        }
    )

    assert pending[0]["key"] == "after_sales:A1:pending_confirmation"
    assert pending[0]["source"] == "pending_action"
    assert submitted[0]["key"] == "after_sales:A1:submitted"
    assert submitted[0]["source"] == "confirmed_action"
    assert cancelled[0]["key"] == "after_sales:A1:cancelled"
    assert cancelled[0]["source"] == "confirmed_action"


def test_memory_extractor_creates_preference_candidate_from_user_message() -> None:
    candidates = MemoryExtractor().extract(
        {
            "conversation_id": "conv-memory",
            "customer_id": "C001",
            "message": "我一般穿42码",
            "business_result": {},
        }
    )

    preference = next(candidate for candidate in candidates if candidate["memory_type"] == "preference")
    assert preference["scope"] == "customer"
    assert preference["owner_id"] == "C001"
    assert preference["key"] == "shoe_size"
    assert preference["value"] == {"shoe_size": "42"}
    assert preference["title"] == "Shoe size preference: 42"
    assert preference["evidence"][0]["text"] == "我一般穿42码"


def test_memory_writeback_summarizes_removed_messages_and_writes_to_store(tmp_path) -> None:
    repository = repo(tmp_path)
    store = RecordingStore()
    writeback = MemoryWriteback(
        repository=repository,
        summarizer=ConversationSummarizer(summary_keep_last=1),
    )
    state = {
        "conversation_id": "conv-memory",
        "customer_id": "C001",
        "messages": [
            HumanMessage(id="m1", content="查询订单 O1001"),
            AIMessage(id="m2", content="订单 O1001 当前状态为 delivered。"),
            HumanMessage(id="m3", content="那我要退货"),
        ],
        "business_result": {
            "action_id": "A1",
            "action_type": "after_sales",
            "status": "submitted",
            "ticket_id": "T1",
        },
    }

    update = writeback.update(state, store=store)
    summary = repository.get_conversation_summary("conv-memory", "C001")

    assert summary is not None
    assert "查询订单 O1001" in summary.summary
    assert "订单 O1001 当前状态为 delivered。" in summary.summary
    assert store.puts[0][0] == ("conversation", "conv-memory", "events")
    assert store.puts[0][1] == "after_sales:A1:submitted"
    assert all(isinstance(message, RemoveMessage) for message in update["messages"])
    assert [message.id for message in update["messages"]] == ["m1", "m2"]


def test_memory_writeback_records_pending_and_cancelled_events(tmp_path) -> None:
    repository = repo(tmp_path)
    store = RecordingStore()
    writeback = MemoryWriteback(
        repository=repository,
        summarizer=ConversationSummarizer(summary_keep_last=10),
    )

    for status in ("pending_confirmation", "cancelled"):
        writeback.update(
            {
                "conversation_id": "conv-memory",
                "customer_id": "C001",
                "messages": [HumanMessage(id=f"{status}:m1", content="售后状态变化")],
                "business_result": {
                    "action_id": "A2",
                    "action_type": "after_sales",
                    "status": status,
                },
            },
            store=store,
        )

    keys = [key for _namespace, key, _value in store.puts]
    assert "after_sales:A2:pending_confirmation" in keys
    assert "after_sales:A2:cancelled" in keys


def test_memory_writer_splits_active_memories_and_candidates() -> None:
    store = RecordingStore()
    writer = MemoryWriter()
    service_event = MemoryCandidate(
        scope="conversation",
        owner_id="conv-memory",
        memory_type="service_event",
        key="after_sales:A1:submitted",
        title="After-sales submitted",
        description="Confirmed after-sales action was submitted.",
        value={"status": "submitted"},
        evidence=[{"conversation_id": "conv-memory"}],
        source="confirmed_action",
        confidence="high",
        risk_level="low",
    )
    preference = MemoryCandidate(
        scope="customer",
        owner_id="C001",
        memory_type="preference",
        key="shoe_size",
        title="Shoe size preference: 42",
        description="User explicitly said they usually wear size 42 shoes.",
        value={"shoe_size": "42"},
        evidence=[{"text": "我一般穿42码", "conversation_id": "conv-memory"}],
        source="user_message",
        confidence="high",
        risk_level="low",
    )

    writer.write(service_event, MemoryDecision(action="write", reason="low_risk_service_event"), store)
    writer.write(preference, MemoryDecision(action="candidate", reason="user_preference_candidate"), store)

    assert store.puts[0][0] == ("conversation", "conv-memory", "events")
    assert store.puts[1][0] == ("customer", "C001", "memory_candidates")
    assert all(namespace != ("customer", "C001", "memories") for namespace, _key, _value in store.puts)


def test_sql_memory_store_adapter_uses_repository_namespace_methods(tmp_path) -> None:
    repository = repo(tmp_path)
    store = SqlMemoryStoreAdapter(repository)

    namespace = ("conversation", "conv-memory", "memories")
    store.put(namespace, "k1", {"value": "v1"})

    assert store.search(namespace, query="v1", limit=5)[0].key == "k1"
```

- [ ] **Step 2: Run memory tests to verify red**

Run:

```powershell
cd python-impl
pytest tests/unit/test_memory.py -q
```

Expected: FAIL because memory models, store adapter, policy decisions, and repository methods do not exist.

- [ ] **Step 3: Add persistence models**

Modify `python-impl/src/smart_cs/domain/models.py` by adding:

```python
class ConversationSummary(Base):
    __tablename__ = "conversation_summaries"

    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id"), primary_key=True
    )
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.id"), nullable=False, index=True)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    open_items: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    last_intent: Mapped[str | None] = mapped_column(String(64), nullable=True)
    last_entities: Mapped[dict[str, str]] = mapped_column(JSON, default=dict, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )


class MemoryRecord(Base):
    __tablename__ = "memory_records"
    __table_args__ = (
        Index("ix_memory_records_namespace", "namespace"),
        Index("ix_memory_records_owner_type", "scope", "owner_id", "memory_type"),
    )

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    namespace: Mapped[str] = mapped_column(String(255), nullable=False)
    scope: Mapped[str] = mapped_column(String(32), nullable=False)
    owner_id: Mapped[str] = mapped_column(String(64), nullable=False)
    memory_type: Mapped[str] = mapped_column(String(64), nullable=False)
    key: Mapped[str] = mapped_column(String(255), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    value_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    evidence_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list, nullable=False)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    confidence: Mapped[str] = mapped_column(String(16), nullable=False)
    risk_level: Mapped[str] = mapped_column(String(16), nullable=False)
    review_status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    created_by: Mapped[str] = mapped_column(String(32), nullable=False)
    approved_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    usage_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)
```

- [ ] **Step 4: Add repository methods**

Modify imports in `repositories.py` to include `ConversationSummary` and `MemoryRecord`.

Add methods:

```python
def upsert_conversation_summary(
    self,
    conversation_id: str,
    customer_id: str,
    summary: str,
    *,
    open_items: dict[str, Any] | None = None,
    last_intent: str | None = None,
    last_entities: dict[str, str] | None = None,
) -> ConversationSummary:
    with self.transaction() as session:
        self._require_conversation_owner(session, conversation_id, customer_id)
        row = session.get(ConversationSummary, conversation_id)
        if row is None:
            row = ConversationSummary(
                conversation_id=conversation_id,
                customer_id=customer_id,
                summary=summary,
                open_items=open_items or {},
                last_intent=last_intent,
                last_entities=last_entities or {},
            )
            session.add(row)
        else:
            row.summary = summary
            row.open_items = open_items or {}
            row.last_intent = last_intent
            row.last_entities = last_entities or {}
        session.flush()
        return row


def get_conversation_summary(self, conversation_id: str, customer_id: str) -> ConversationSummary | None:
    with self.transaction() as session:
        self._require_conversation_owner(session, conversation_id, customer_id)
        return session.get(ConversationSummary, conversation_id)


def put_memory(
    self,
    namespace: tuple[str, str, str],
    key: str,
    value: dict[str, Any],
    *,
    scope: str,
    owner_id: str,
    memory_type: str,
    source: str,
    confidence: str,
    risk_level: str,
    created_by: str,
) -> MemoryRecord:
    namespace_text = "/".join(namespace)
    memory_id = f"{namespace_text}:{key}"
    with self.transaction() as session:
        row = session.get(MemoryRecord, memory_id)
        if row is None:
            row = MemoryRecord(
                id=memory_id,
                namespace=namespace_text,
                scope=scope,
                owner_id=owner_id,
                memory_type=memory_type,
                key=key,
                title=str(value.get("title", key)),
                description=str(value.get("description", "")),
                value_json=value,
                evidence_json=list(value.get("evidence", [])),
                source=source,
                confidence=confidence,
                risk_level=risk_level,
                review_status=str(value.get("review_status", "pending")),
                created_by=created_by,
            )
            session.add(row)
        else:
            row.title = str(value.get("title", key))
            row.description = str(value.get("description", ""))
            row.value_json = value
            row.evidence_json = list(value.get("evidence", []))
            row.confidence = confidence
            row.risk_level = risk_level
            row.source = source
            row.review_status = str(value.get("review_status", "pending"))
        session.flush()
        return row


def search_memories(self, namespace: tuple[str, str, str], query: str, limit: int) -> list[MemoryRecord]:
    namespace_text = "/".join(namespace)
    with self.transaction() as session:
        statement = (
            select(MemoryRecord)
            .where(MemoryRecord.namespace == namespace_text)
            .order_by(MemoryRecord.updated_at.desc(), MemoryRecord.id.desc())
            .limit(limit)
        )
        return list(session.scalars(statement))
```

- [ ] **Step 5: Implement memory writeback**

Create `python-impl/src/smart_cs/application/memory.py`:

```python
from __future__ import annotations

import re
from typing import Any, Literal, Protocol

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, RemoveMessage
from pydantic import BaseModel, Field

from smart_cs.domain.enums import ActionStatus


class MemoryStoreProtocol(Protocol):
    def put(self, namespace: tuple[str, str, str], key: str, value: dict[str, Any]) -> None:
        raise NotImplementedError

    def search(self, namespace: tuple[str, str, str], query: str, limit: int) -> list[Any]:
        raise NotImplementedError


class MemoryDecision(BaseModel):
    action: Literal["write", "candidate", "human_review", "discard"]
    reason: str


class MemoryCandidate(BaseModel):
    scope: Literal["customer", "conversation", "tenant"]
    owner_id: str
    memory_type: Literal["preference", "service_event", "risk_event", "sensitive_label", "badcase_candidate"]
    key: str
    title: str
    description: str
    value: dict[str, Any]
    evidence: list[dict[str, Any]]
    source: str
    confidence: Literal["low", "medium", "high"]
    risk_level: Literal["low", "medium", "high"]
    review_status: Literal["pending", "approved", "rejected"] = "pending"


class MemoryPolicy:
    def decide(self, candidate: dict[str, Any]) -> MemoryDecision:
        memory_type = candidate.get("memory_type")
        risk_level = candidate.get("risk_level")
        if memory_type == "service_event" and risk_level == "low":
            return MemoryDecision(action="write", reason="low_risk_service_event")
        if memory_type == "preference":
            return MemoryDecision(action="candidate", reason="user_preference_candidate")
        if memory_type in {"sensitive_label", "risk_label", "risk_event"}:
            return MemoryDecision(action="human_review", reason="sensitive_memory")
        if memory_type == "badcase_candidate":
            return MemoryDecision(action="human_review", reason="badcase_requires_review")
        return MemoryDecision(action="discard", reason="unsupported_memory")


PREFERENCE_PATTERNS = [
    ("shoe_size", re.compile(r"我(?:一般|通常)?穿(\d{2})码")),
    ("color_preference", re.compile(r"我(?:喜欢|偏好)(黑色|白色|灰色|蓝色)")),
    ("contact_preference", re.compile(r"(?:以后|之后).*(?:别|不要).*打电话")),
]


class MemoryExtractor:
    def extract(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        candidates: list[MemoryCandidate] = []
        candidates.extend(self._extract_service_events(state))
        candidates.extend(self._extract_explicit_preferences(state))
        return [candidate.model_dump() for candidate in candidates]

    def _extract_service_events(self, state: dict[str, Any]) -> list[MemoryCandidate]:
        result = state.get("business_result") or {}
        action_id = result.get("action_id")
        action_type = result.get("action_type")
        status = result.get("status")
        if action_id is None or action_type is None or status not in {
            ActionStatus.PENDING_CONFIRMATION.value,
            ActionStatus.SUBMITTED.value,
            ActionStatus.CANCELLED.value,
        }:
            return []
        source = "pending_action" if status == ActionStatus.PENDING_CONFIRMATION.value else "confirmed_action"
        return [
            MemoryCandidate(
                scope="conversation",
                owner_id=state["conversation_id"],
                memory_type="service_event",
                key=f"{action_type}:{action_id}:{status}",
                title=f"{action_type} {status}",
                description=f"Conversation action {action_id} reached {status}.",
                value={
                    key: value
                    for key, value in result.items()
                    if key in {"action_id", "action_type", "status", "ticket_id", "order_id"}
                },
                evidence=[
                    {
                        "conversation_id": state["conversation_id"],
                        "customer_id": state["customer_id"],
                        "business_result": result,
                    }
                ],
                source=source,
                confidence="high",
                risk_level="low",
            )
        ]

    def _extract_explicit_preferences(self, state: dict[str, Any]) -> list[MemoryCandidate]:
        message = str(state.get("message") or "")
        if not message:
            return []
        candidates: list[MemoryCandidate] = []
        for key, pattern in PREFERENCE_PATTERNS:
            match = pattern.search(message)
            if match is None:
                continue
            raw_value = match.group(1) if match.groups() else "no_phone_call"
            value = {key: raw_value}
            candidates.append(
                MemoryCandidate(
                    scope="customer",
                    owner_id=state["customer_id"],
                    memory_type="preference",
                    key=key,
                    title=self._preference_title(key, raw_value),
                    description=f"User explicitly stated preference {key}={raw_value}.",
                    value=value,
                    evidence=[{"text": message, "conversation_id": state["conversation_id"]}],
                    source="user_message",
                    confidence="high",
                    risk_level="low",
                )
            )
        return candidates

    @staticmethod
    def _preference_title(key: str, raw_value: str) -> str:
        if key == "shoe_size":
            return f"Shoe size preference: {raw_value}"
        if key == "color_preference":
            return f"Color preference: {raw_value}"
        if key == "contact_preference":
            return "Contact preference: no phone call"
        return f"Preference: {key}"


class SqlMemoryStoreAdapter:
    def __init__(self, repository: Any) -> None:
        self.repository = repository

    def put(self, namespace: tuple[str, str, str], key: str, value: dict[str, Any]) -> None:
        scope, owner_id, _bucket = namespace
        self.repository.put_memory(
            namespace,
            key,
            value,
            scope=scope,
            owner_id=owner_id,
            memory_type=str(value.get("memory_type", "service_event")),
            source=str(value.get("source", "system")),
            confidence=str(value.get("confidence", "medium")),
            risk_level=str(value.get("risk_level", "low")),
            created_by="system",
        )

    def search(self, namespace: tuple[str, str, str], query: str, limit: int) -> list[Any]:
        return self.repository.search_memories(namespace, query=query, limit=limit)


class MemoryWriter:
    def write(
        self,
        candidate: MemoryCandidate,
        decision: MemoryDecision,
        store: MemoryStoreProtocol,
    ) -> None:
        if decision.action == "discard":
            return
        namespace = self._namespace_for(candidate, decision)
        value = candidate.model_dump()
        value["memory_decision"] = decision.model_dump()
        store.put(namespace, candidate.key, value)

    @staticmethod
    def _namespace_for(
        candidate: MemoryCandidate,
        decision: MemoryDecision,
    ) -> tuple[str, str, str]:
        if candidate.memory_type == "service_event" and decision.action == "write":
            return ("conversation", candidate.owner_id, "events")
        if candidate.memory_type == "badcase_candidate":
            return ("tenant", candidate.owner_id or "default", "badcase_candidates")
        if decision.action == "write":
            return (candidate.scope, candidate.owner_id, "memories")
        return (candidate.scope, candidate.owner_id, "memory_candidates")


class ConversationSummarizer:
    def __init__(
        self,
        *,
        summary_keep_last: int = 6,
        max_summary_chars: int = 2000,
        summarizer: Any | None = None,
    ) -> None:
        self.summary_keep_last = summary_keep_last
        self.max_summary_chars = max_summary_chars
        self.summarizer = summarizer

    def removable_messages(self, messages: list[AnyMessage]) -> list[AnyMessage]:
        if len(messages) <= self.summary_keep_last:
            return []
        return [
            message
            for message in messages[: -self.summary_keep_last]
            if isinstance(message, (HumanMessage, AIMessage)) and getattr(message, "id", None) is not None
        ]

    def summarize(self, state: dict[str, Any], removable: list[AnyMessage]) -> str:
        current = str(state.get("conversation_summary") or "").strip()
        removed_text = "\n".join(str(message.content) for message in removable)
        business_result = state.get("business_result") or {}
        if self.summarizer is not None and removed_text:
            response = self.summarizer.invoke(
                {
                    "existing_summary": current,
                    "new_messages": removed_text,
                    "business_result": business_result,
                }
            )
            text = getattr(response, "content", str(response))
        else:
            # P0 deterministic fallback for tests/local runs. Production should pass
            # a LangGraph summarization-pattern runnable or langmem SummarizationNode.
            parts = [part for part in [current, removed_text, str(business_result) if business_result else ""] if part]
            text = "\n".join(parts)
        return text[-self.max_summary_chars:] if text else current


class MemoryWriteback:
    def __init__(
        self,
        *,
        repository: Any,
        summarizer: ConversationSummarizer | None = None,
        extractor: MemoryExtractor | None = None,
        policy: MemoryPolicy | None = None,
        writer: MemoryWriter | None = None,
    ) -> None:
        self.repository = repository
        self.summarizer = summarizer or ConversationSummarizer()
        self.extractor = extractor or MemoryExtractor()
        self.policy = policy or MemoryPolicy()
        self.writer = writer or MemoryWriter()

    def update(self, state: dict[str, Any], *, store: MemoryStoreProtocol) -> dict[str, Any]:
        conversation_id = state["conversation_id"]
        customer_id = state["customer_id"]
        messages = list(state.get("messages") or [])
        removable = self.summarizer.removable_messages(messages)
        remove_messages = [RemoveMessage(id=str(message.id)) for message in removable]
        summary = self.summarizer.summarize(state, removable)
        route = state.get("route") or {}
        if summary:
            self.repository.upsert_conversation_summary(
                conversation_id,
                customer_id,
                summary,
                open_items={},
                last_intent=route.get("intent"),
                last_entities=route.get("entities") or {},
            )

        for raw_candidate in self.extractor.extract(state):
            candidate = MemoryCandidate.model_validate(raw_candidate)
            decision = self.policy.decide(candidate.model_dump())
            self.writer.write(candidate, decision, store)

        return {"conversation_summary": summary, "messages": remove_messages}
```

- [ ] **Step 6: Run memory tests**

Run:

```powershell
cd python-impl
pytest tests/unit/test_memory.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit Task 7**

Run:

```powershell
git add python-impl/src/smart_cs/application/memory.py python-impl/src/smart_cs/domain/models.py python-impl/src/smart_cs/infrastructure/repositories.py python-impl/tests/unit/test_memory.py
git commit -m "feat: add conversation memory writeback"
```

## Task 8: Rewire LangGraph Runtime

**Files:**
- Modify: `python-impl/src/smart_cs/application/agent_runtime.py`
- Modify: `python-impl/src/smart_cs/main.py`
- Modify: `python-impl/tests/integration/test_action_confirmation.py`

- [ ] **Step 1: Write failing runtime graph tests**

Append to `python-impl/tests/integration/test_action_confirmation.py`:

```python
import pytest

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command


def test_runtime_checkpoint_contains_human_message_and_single_ai_message(runtime_and_repo) -> None:
    runtime, _repository = runtime_and_repo

    result = runtime.invoke("conv-messages", "C001", "查询订单 O1001")
    state = runtime.graph.get_state({"configurable": {"thread_id": "conv-messages"}}).values

    assert result["status"] == "completed"
    assert sum(isinstance(message, HumanMessage) for message in state["messages"]) == 1
    assert sum(isinstance(message, AIMessage) for message in state["messages"]) == 1
    assert state["messages"][-1].content == result["reply"]


def test_follow_up_after_order_uses_conversation_slot(runtime_and_repo) -> None:
    runtime, _repository = runtime_and_repo

    first = runtime.invoke("conv-slot-follow-up", "C001", "查询订单 O1001")
    second = runtime.invoke("conv-slot-follow-up", "C001", "那我要退货")

    assert first["status"] == "completed"
    assert second["status"] == "pending_confirmation"
    assert second["pending_confirmation"]["order_id"] == "O1001"


def test_confirm_updates_slots_and_memory(runtime_and_repo) -> None:
    runtime, repository = runtime_and_repo
    pending = runtime.invoke("conv-memory-runtime", "C001", "订单 O1001 鞋底开胶，申请退款")

    completed = runtime.confirm(
        "conv-memory-runtime",
        "C001",
        pending["pending_confirmation"]["action_id"],
        approved=True,
    )
    state = runtime.graph.get_state({"configurable": {"thread_id": "conv-memory-runtime"}}).values
    summary = repository.get_conversation_summary("conv-memory-runtime", "C001")
    action = repository.get_action(
        "conv-memory-runtime",
        "C001",
        pending["pending_confirmation"]["action_id"],
    )

    assert completed["status"] == "completed"
    assert action.status == "submitted"
    assert state["conversation_slots"]["active_ticket_id"] == completed["result"]["ticket_id"]
    assert state["conversation_slots"]["action_status"] == "submitted"
    assert state.get("pending_confirmation") is None
    assert summary is not None


def test_confirm_clears_pending_confirmation(runtime_and_repo) -> None:
    runtime, repository = runtime_and_repo
    pending = runtime.invoke("conv-clear-pending", "C001", "订单 O1001 鞋底开胶，申请退款")

    runtime.confirm(
        "conv-clear-pending",
        "C001",
        pending["pending_confirmation"]["action_id"],
        approved=False,
    )
    state = runtime.graph.get_state({"configurable": {"thread_id": "conv-clear-pending"}}).values
    action = repository.get_action(
        "conv-clear-pending",
        "C001",
        pending["pending_confirmation"]["action_id"],
    )

    assert action.status == "cancelled"
    assert state.get("pending_confirmation") is None
    assert state["conversation_slots"]["action_status"] == "cancelled"


def test_natural_language_confirmation_does_not_submit_pending_action(runtime_and_repo) -> None:
    runtime, _repository = runtime_and_repo
    pending = runtime.invoke("conv-natural-confirm", "C001", "订单 O1001 鞋底开胶，申请退款")

    result = runtime.invoke("conv-natural-confirm", "C001", "确认")
    state = runtime.graph.get_state({"configurable": {"thread_id": "conv-natural-confirm"}}).values

    assert pending["status"] == "pending_confirmation"
    assert result["status"] == "pending_confirmation"
    assert state["conversation_slots"]["action_status"] == "pending_confirmation"
    assert state.get("pending_confirmation") is not None


def test_invalid_confirmation_resume_is_rejected(runtime_and_repo) -> None:
    runtime, _repository = runtime_and_repo
    runtime.invoke("conv-invalid-resume", "C001", "订单 O1001 鞋底开胶，申请退款")

    with pytest.raises(ValueError, match="Confirmation requires boolean approval"):
        runtime.graph.invoke(
            Command(resume={}),
            config=runtime._config("conv-invalid-resume"),
        )

    state = runtime.graph.get_state({"configurable": {"thread_id": "conv-invalid-resume"}}).values
    assert state["conversation_slots"]["action_status"] == "pending_confirmation"
    assert state.get("pending_confirmation") is not None


def test_context_project_reads_only_active_customer_memories(runtime_and_repo) -> None:
    runtime, _repository = runtime_and_repo
    runtime.store.put(
        ("customer", "C001", "memories"),
        "active-pref",
        {
            "memory_type": "preference",
            "value": {"tone": "concise"},
            "confidence": "high",
            "source": "approved_memory",
        },
    )
    runtime.store.put(
        ("customer", "C001", "memory_candidates"),
        "candidate-pref",
        {
            "memory_type": "preference",
            "value": {"shoe_size": "42"},
            "confidence": "high",
            "source": "user_message",
        },
    )

    runtime.invoke("conv-active-memory", "C001", "查询订单 O1001")
    state = runtime.graph.get_state({"configurable": {"thread_id": "conv-active-memory"}}).values
    memory_ids = [
        memory["memory_id"]
        for memory in state["decision_context"]["router_context"]["customer_memories"]
    ]

    assert memory_ids == ["active-pref"]
```

In the same file, update the imports and `runtime_and_repo` fixture so the runtime test exercises memory writeback:

```python
from smart_cs.application.memory import MemoryWriteback


class StubKnowledgeAgent:
    def answer(self, _message: str):
        class Answer:
            def as_result(self) -> dict:
                return {
                    "status": "knowledge_answer",
                    "answer": "售后政策：订单签收后可申请售后草稿，需用户确认后提交。",
                    "citations": [{"source": "stub-policy.md"}],
                }

        return Answer()


runtime = AgentRuntime(
    executor=AuthorizedToolExecutor(repository),
    decision_model=RulesDecisionModel(),
    checkpoint_path=tmp_path / "checkpoints.db",
    knowledge_agent=StubKnowledgeAgent(),
    memory_writeback=MemoryWriteback(repository=repository),
)
```

- [ ] **Step 2: Run runtime tests to verify red**

Run:

```powershell
cd python-impl
pytest tests/integration/test_action_confirmation.py::test_runtime_checkpoint_contains_human_message_and_single_ai_message tests/integration/test_action_confirmation.py::test_follow_up_after_order_uses_conversation_slot tests/integration/test_action_confirmation.py::test_confirm_updates_slots_and_memory tests/integration/test_action_confirmation.py::test_confirm_clears_pending_confirmation tests/integration/test_action_confirmation.py::test_natural_language_confirmation_does_not_submit_pending_action tests/integration/test_action_confirmation.py::test_invalid_confirmation_resume_is_rejected tests/integration/test_action_confirmation.py::test_context_project_reads_only_active_customer_memories -q
```

Expected: FAIL because runtime has no message channel, context projector, slot carry, or memory writeback.

- [ ] **Step 3: Inject new services into AgentRuntime**

Modify `AgentRuntime.__init__`:

```python
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore
from langgraph.types import interrupt, Command

from smart_cs.application.context_projector import ContextProjector
from smart_cs.application.memory import MemoryWriteback
from smart_cs.application.policy import PolicyEngine
from smart_cs.application.state_update import StateUpdater, carry_slots
from smart_cs.agents.state import RuntimeContext
from smart_cs.infrastructure.prompts import PROMPT_VERSION
```

Add constructor parameters:

```python
        context_projector: ContextProjector | None = None,
        policy_engine: PolicyEngine | None = None,
        memory_writeback: MemoryWriteback | None = None,
        store: Any | None = None,
```

Set:

```python
self.context_projector = context_projector or ContextProjector()
self.policy_engine = policy_engine or PolicyEngine()
self.state_updater = StateUpdater()
self.memory_writeback = memory_writeback
self.store = store or InMemoryStore()
```

- [ ] **Step 4: Add HumanMessage to graph input**

Modify `invoke()` to assign the request id once and pass only current-turn fields. Do not pass `conversation_slots`, `conversation_summary`, or `customer_memories` here because plain state keys without reducers would overwrite checkpoint state on every turn.

```python
request_id = f"{conversation_id}:{uuid4()}"

result = self.graph.invoke(
    {
        "messages": [HumanMessage(id=f"{request_id}:human", content=message)],
        "conversation_id": conversation_id,
        "customer_id": customer_id,
        "request_id": request_id,
        "message": message,
        "has_image": visual_evidence is not None,
        "visual_evidence": visual_evidence,
        "asset_key": asset_key,
        "route": {},
        "decision": {},
        "agents_invoked": [],
        "specialist_results": [],
        "business_result": None,
        "pending_confirmation": None,
        "guarded_contents": [],
        "reply": None,
        "status": "running",
        "read_results": [],
        "policy_decision": None,
    },
    config=self._config(conversation_id),
    context=RuntimeContext(
        conversation_id=conversation_id,
        customer_id=customer_id,
        prompt_version=PROMPT_VERSION,
    ).model_dump(),
)
```

Do not add any `message_ingest` node.

- [ ] **Step 5: Rebuild graph topology**

Modify `_build_graph()`:

```python
workflow = StateGraph(RuntimeState, context_schema=RuntimeContext)
workflow.add_node("context_project", self._context_project_node)
workflow.add_node("router", self._router_node)
workflow.add_node("slot_carry", self._slot_carry_node)
workflow.add_node("supervisor", self._supervisor_node)
workflow.add_node("read_specialists", self._read_specialists_node)
workflow.add_node("policy_check", self._policy_check_node)
workflow.add_node("write_specialists_or_handoff", self._write_specialists_or_handoff_node)
workflow.add_node("validate_evidence", self._validate_evidence_node)
workflow.add_node("state_update", self._state_update_node)
workflow.add_node("memory_writeback", self._memory_writeback_node)
workflow.add_node("confirm_action", self._confirm_action_node)
workflow.add_node("guard", self._guard_node)
workflow.add_node("synthesize", self._synthesize_node)
workflow.add_edge(START, "context_project")
workflow.add_edge("context_project", "router")
workflow.add_edge("router", "slot_carry")
workflow.add_edge("slot_carry", "supervisor")
workflow.add_edge("supervisor", "read_specialists")
workflow.add_edge("read_specialists", "policy_check")
workflow.add_edge("policy_check", "write_specialists_or_handoff")
workflow.add_edge("write_specialists_or_handoff", "validate_evidence")
workflow.add_edge("validate_evidence", "state_update")
workflow.add_edge("state_update", "memory_writeback")
workflow.add_edge("memory_writeback", "guard")
workflow.add_edge("confirm_action", "state_update")
workflow.add_edge("guard", "synthesize")
workflow.add_conditional_edges(
    "synthesize",
    self._next_after_synthesis,
    {"confirm_action": "confirm_action", "end": END},
)
return workflow.compile(checkpointer=self._checkpointer, store=self.store)
```

- [ ] **Step 6: Implement new runtime nodes**

Add these methods to `AgentRuntime`:

```python
def _context_project_node(
    self, state: RuntimeState, runtime: Runtime[RuntimeContext]
) -> dict[str, Any]:
    self._assert_turn_lease()
    memories = runtime.store.search(
        ("customer", state["customer_id"], "memories"),
        query=state["message"],
        limit=5,
    )
    # Deliberately do not read ("customer", customer_id, "memory_candidates").
    # Unreviewed preferences, sensitive labels, and badcase candidates must not
    # enter RouterContext or SupervisorContext.
    customer_memories = [
        {
            "memory_id": getattr(item, "key", f"memory-{index}"),
            "memory_type": item.value.get("memory_type", "service_event"),
            "value": item.value,
            "confidence": item.value.get("confidence", "medium"),
            "source": item.value.get("source", "runtime_store"),
        }
        for index, item in enumerate(memories)
    ]
    projected_state = {**state, "customer_memories": customer_memories}
    router_context = self.context_projector.build_router_context(projected_state)
    return {
        "customer_memories": customer_memories,
        "decision_context": {"router_context": router_context.model_dump()},
    }


def _router_node(self, state: RuntimeState) -> dict[str, Any]:
    self._assert_turn_lease()
    context = self.context_projector.build_router_context(state)
    route = self.router.analyze(context)
    self._assert_turn_lease()
    return {"route": route.model_dump(), "decision_context": {"router_context": context.model_dump()}}


def _slot_carry_node(self, state: RuntimeState) -> dict[str, Any]:
    route = RouteAnalysis.model_validate(state["route"])
    slots = ConversationSlots.model_validate(state.get("conversation_slots") or {})
    updated = carry_slots(route, slots)
    return {"route": updated.model_dump()}


def _supervisor_node(self, state: RuntimeState) -> dict[str, Any]:
    self._assert_turn_lease()
    route = RouteAnalysis.model_validate(state["route"])
    context = self.context_projector.build_supervisor_context(state, route)
    decision = self.supervisor.plan(context, has_image=bool(state.get("has_image")))
    self._assert_turn_lease()
    return {
        "decision": decision.model_dump(),
        "decision_context": {
            **state.get("decision_context", {}),
            "supervisor_context": context.model_dump(),
        },
    }


def _read_specialists_node(self, state: RuntimeState) -> dict[str, Any]:
    execution = self.specialists.execute_read_agents(
        message=state["message"],
        customer_id=state["customer_id"],
        route=RouteAnalysis.model_validate(state["route"]),
        decision=SupervisorDecision.model_validate(state["decision"]),
        conversation_id=state["conversation_id"],
        idempotency_key=state.get("request_id"),
        turn_fence=self._current_turn_fence(),
        visual_evidence=state.get("visual_evidence"),
        asset_key=state.get("asset_key"),
    )
    return {
        "agents_invoked": [*state.get("agents_invoked", []), *execution.agents_invoked],
        "read_results": execution.results,
        "specialist_results": execution.results,
    }


def _policy_check_node(self, state: RuntimeState) -> dict[str, Any]:
    route = RouteAnalysis.model_validate(state["route"])
    if route.intent != "after_sales":
        return {"policy_decision": None}
    order_result = next((result for result in state.get("read_results", []) if "order_id" in result or result.get("status") == "information_required"), {})
    knowledge_result = next((result for result in state.get("read_results", []) if "citations" in result), {})
    decision = self.policy_engine.evaluate_after_sales(
        order_result=order_result,
        knowledge_result=knowledge_result,
        visual_evidence=state.get("visual_evidence"),
    )
    return {"policy_decision": decision.model_dump()}


def _write_specialists_or_handoff_node(self, state: RuntimeState) -> dict[str, Any]:
    decision = SupervisorDecision.model_validate(state["decision"])
    policy_decision = state.get("policy_decision")
    if policy_decision is not None and policy_decision.get("next_action") == "explain":
        result = {"status": "policy_explained", "message": policy_decision["explanation"]}
        return {"business_result": result, "specialist_results": list(state.get("specialist_results", [])) + [result]}
    if policy_decision is not None and policy_decision.get("next_action") == "handoff":
        decision = decision.model_copy(update={"agents": ["HandoffAgent"], "action": "draft_handoff", "requires_confirmation": True})
    execution = self.specialists.execute_write_agents(
        message=state["message"],
        customer_id=state["customer_id"],
        route=RouteAnalysis.model_validate(state["route"]),
        decision=decision,
        conversation_id=state["conversation_id"],
        idempotency_key=state.get("request_id"),
        turn_fence=self._current_turn_fence(),
        visual_evidence=state.get("visual_evidence"),
        asset_key=state.get("asset_key"),
    )
    if not execution.results:
        terminal = (state.get("read_results") or [{"status": "completed"}])[-1]
        return {"business_result": terminal}
    results = list(state.get("specialist_results", [])) + execution.results
    return {
        "agents_invoked": [*state.get("agents_invoked", []), *execution.agents_invoked],
        "specialist_results": results,
        "business_result": execution.result,
        "pending_confirmation": execution.pending_confirmation,
    }


def _state_update_node(self, state: RuntimeState) -> dict[str, Any]:
    return self.state_updater.update(state)


def _memory_writeback_node(
    self, state: RuntimeState, runtime: Runtime[RuntimeContext]
) -> dict[str, Any]:
    if self.memory_writeback is None:
        return {}
    return self.memory_writeback.update(state, store=runtime.store)


def _transition_action(
    self,
    action_id: str,
    customer_id: str,
    approved: bool,
) -> dict[str, Any]:
    if approved:
        return self.executor.submit_confirmed_action(
            action_id,
            customer_id,
            caller_agent="ConfirmActionNode",
            turn_fence=self._current_turn_fence(),
        )
    return self.executor.cancel_pending_action(
        action_id,
        customer_id,
        caller_agent="ConfirmActionNode",
        turn_fence=self._current_turn_fence(),
    )


def _confirm_action_node(self, state: RuntimeState) -> dict[str, Any]:
    pending = state.get("pending_confirmation") or {}
    action_id = pending.get("action_id")
    if action_id is None:
        raise ValueError("Missing pending action for confirmation")
    # interrupt() pauses the graph; resumed by Command(resume=...) from the confirm() API.
    approval = interrupt({
        "status": ActionStatus.PENDING_CONFIRMATION.value,
        "pending_confirmation": pending,
        "reply": state.get("reply") or self.guard.render(pending),
        "agents_invoked": list(state.get("agents_invoked", [])),
    })
    if not isinstance(approval, dict) or type(approval.get("approved")) is not bool:
        raise ValueError("Confirmation requires boolean approval")
    approved = approval["approved"]
    if not approved:
        result = self.executor.cancel_pending_action(
            action_id,
            state["customer_id"],
            caller_agent="ConfirmActionNode",
            turn_fence=self._current_turn_fence(),
        )
        return {"business_result": result, "pending_confirmation": None}

    result = self.executor.submit_confirmed_action(
        action_id,
        state["customer_id"],
        caller_agent="ConfirmActionNode",
        turn_fence=self._current_turn_fence(),
    )
    return {"business_result": result, "pending_confirmation": None}
```

- [ ] **Step 7: Make synthesize write AIMessage exactly once**

Modify `_synthesize_node()`:

```python
if self._next_after_synthesis(state) == "confirm_action":
    return {
        "status": ActionStatus.PENDING_CONFIRMATION.value,
        "reply": reply,
        "messages": [AIMessage(id=f"{state['request_id']}:pending:assistant", content=reply)],
    }
return {
    "status": "completed",
    "reply": reply,
    "pending_confirmation": None,
    "messages": [AIMessage(id=f"{state['request_id']}:completed:assistant", content=reply)],
}
```

Do not write `AIMessage` in `_public_result()`, `_pending_result()`, `_completed_result()`, or API response code.

`confirmation_like` and `rejection_like` are only router turn-type signals in P0. They must not call `_confirm_action_node`; write actions are submitted only through `confirm()` / explicit UI confirmation.

- [ ] **Step 8: Wire memory in main.py**

Modify `build_runtime()`:

```python
from smart_cs.application.memory import MemoryWriteback
```

and:

```python
memory_writeback = MemoryWriteback(repository=repository)
runtime = AgentRuntime(
    executor=AuthorizedToolExecutor(repository),
    decision_model=decision_model,
    checkpoint_path=settings.checkpoint_path,
    knowledge_agent=knowledge_agent,
    memory_writeback=memory_writeback,
)
```

- [ ] **Step 9: Run runtime tests**

Run:

```powershell
cd python-impl
pytest tests/integration/test_action_confirmation.py -q
```

Expected: PASS.

- [ ] **Step 10: Commit Task 8**

Run:

```powershell
git add python-impl/src/smart_cs/application/agent_runtime.py python-impl/src/smart_cs/main.py python-impl/tests/integration/test_action_confirmation.py
git commit -m "feat: rewire runtime for context memory workflow"
```

## Task 9: Add JSONL Golden Workflow Evaluation

**Files:**
- Create: `python-impl/tests/evaluation/golden_cases.jsonl`
- Create: `python-impl/tests/evaluation/test_workflow_golden.py`

- [ ] **Step 1: Add golden cases**

Create `python-impl/tests/evaluation/golden_cases.jsonl` with exactly 20 lines:

```jsonl
{"id":"order_lookup","customer_id":"C001","message":"查询订单 O1001","expected_status":"completed","expected_agents":["OrderAgent"],"expected_intent":"order","expected_action":"read","expected_contains":"订单 O1001"}
{"id":"product_search","customer_id":"C001","message":"推荐跑鞋","expected_status":"completed","expected_agents":["ProductAgent"],"expected_intent":"product","expected_action":"read","expected_contains":"轻量跑鞋"}
{"id":"knowledge_policy","customer_id":"C001","message":"退货政策多久","expected_status":"completed","expected_agents":["KnowledgeAgent"],"expected_intent":"knowledge","expected_action":"read","expected_contains":"政策"}
{"id":"after_sales_with_order","customer_id":"C001","message":"订单 O1001 鞋底开胶，申请退款","expected_status":"pending_confirmation","expected_agents":["OrderAgent","KnowledgeAgent","AfterSalesAgent"],"expected_intent":"after_sales","expected_action":"draft_after_sales","expected_contains":"售后申请草稿"}
{"id":"handoff_direct","customer_id":"C001","message":"我要转人工","expected_status":"pending_confirmation","expected_agents":["HandoffAgent"],"expected_intent":"handoff","expected_action":"draft_handoff","expected_contains":"人工"}
{"id":"after_sales_missing_order","customer_id":"C001","message":"我要退货","expected_status":"completed","expected_agents":["OrderAgent"],"expected_intent":"after_sales","expected_action":"draft_after_sales","expected_contains":"订单编号"}
{"id":"logistics_lookup","customer_id":"C001","message":"O1001 物流到哪了","expected_status":"completed","expected_agents":["OrderAgent"],"expected_intent":"order","expected_action":"read","expected_contains":"O1001"}
{"id":"price_question","customer_id":"C001","message":"轻量跑鞋多少钱","expected_status":"completed","expected_agents":["ProductAgent"],"expected_intent":"product","expected_action":"read","expected_contains":"轻量跑鞋"}
{"id":"refund_rule_question","customer_id":"C001","message":"退款规则是什么","expected_status":"completed","expected_agents":["KnowledgeAgent"],"expected_intent":"knowledge","expected_action":"read","expected_contains":"政策"}
{"id":"complaint_handoff","customer_id":"C001","message":"我要投诉并找人工客服","expected_status":"pending_confirmation","expected_agents":["HandoffAgent"],"expected_intent":"handoff","expected_action":"draft_handoff","expected_contains":"人工"}
{"id":"follow_up_after_order","customer_id":"C001","messages":["查询订单 O1001","那我要退货"],"expected_status":"pending_confirmation","expected_agents":["OrderAgent","KnowledgeAgent","AfterSalesAgent"],"expected_intent":"after_sales","expected_action":"draft_after_sales","expected_contains":"售后申请草稿"}
{"id":"correction_order","customer_id":"C001","messages":["查询订单 O1001","不是这个订单，是 O1001"],"expected_status":"completed","expected_agents":["OrderAgent"],"expected_intent":"order","expected_action":"read","expected_contains":"O1001"}
{"id":"confirm_after_sales","customer_id":"C001","message":"订单 O1001 鞋底开胶，申请退款","confirm":true,"expected_status":"completed","expected_agents":["OrderAgent","KnowledgeAgent","AfterSalesAgent"],"expected_intent":"after_sales","expected_action":"draft_after_sales","expected_contains":"售后申请已受理"}
{"id":"reject_after_sales","customer_id":"C001","message":"订单 O1001 鞋底开胶，申请退款","confirm":false,"expected_status":"completed","expected_agents":["OrderAgent","KnowledgeAgent","AfterSalesAgent"],"expected_intent":"after_sales","expected_action":"draft_after_sales","expected_contains":"已取消"}
{"id":"wrong_customer_order","customer_id":"C002","message":"查询订单 O1001","expected_error":"ToolPermissionError","expected_intent":"order","expected_action":"read","expected_contains":"not available"}
{"id":"missing_order_after_sales","customer_id":"C001","message":"鞋底开胶，帮我售后","expected_status":"completed","expected_agents":["OrderAgent"],"expected_intent":"after_sales","expected_action":"draft_after_sales","expected_contains":"订单编号"}
{"id":"shipping_policy","customer_id":"C001","message":"发货多久能到","expected_status":"completed","expected_agents":["KnowledgeAgent"],"expected_intent":"knowledge","expected_action":"read","expected_contains":"政策"}
{"id":"product_generic","customer_id":"C001","message":"有什么商品推荐","expected_status":"completed","expected_agents":["ProductAgent"],"expected_intent":"product","expected_action":"read","expected_contains":"产品"}
{"id":"manual_sensitive","customer_id":"C001","message":"我要曝光你们，马上人工处理","expected_status":"pending_confirmation","expected_agents":["HandoffAgent"],"expected_intent":"handoff","expected_action":"draft_handoff","expected_contains":"人工"}
{"id":"order_status_only","customer_id":"C001","message":"O1001","expected_status":"completed","expected_agents":["OrderAgent"],"expected_intent":"order","expected_action":"read","expected_contains":"O1001"}
```

- [ ] **Step 2: Write evaluation runner**

Create `python-impl/tests/evaluation/test_workflow_golden.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from smart_cs.application.memory import MemoryWriteback
from smart_cs.application.agent_runtime import AgentRuntime
from smart_cs.domain.errors import ToolPermissionError
from smart_cs.infrastructure.database import Database
from smart_cs.infrastructure.model_factory import RulesDecisionModel
from smart_cs.infrastructure.repositories import SqlRepository
from smart_cs.tools.executor import AuthorizedToolExecutor


GOLDEN_CASES = Path(__file__).with_name("golden_cases.jsonl")
BADCASE_OUTPUT = Path(__file__).with_name("badcase_candidates.jsonl")


class StubKnowledgeAgent:
    def answer(self, _message: str):
        class Answer:
            def as_result(self) -> dict:
                return {
                    "status": "knowledge_answer",
                    "answer": "售后政策：订单签收后可申请售后草稿，需用户确认后提交。",
                    "citations": [{"source": "stub-policy.md"}],
                }

        return Answer()


@pytest.fixture
def runtime(tmp_path):
    repository = SqlRepository(Database(f"sqlite:///{tmp_path / 'golden.db'}"))
    repository.create_schema()
    repository.seed_demo_data()
    runtime = AgentRuntime(
        executor=AuthorizedToolExecutor(repository),
        decision_model=RulesDecisionModel(),
        checkpoint_path=tmp_path / "golden-checkpoints.db",
        knowledge_agent=StubKnowledgeAgent(),
        memory_writeback=MemoryWriteback(repository=repository),
    )
    try:
        yield runtime
    finally:
        runtime.close()


def load_cases() -> list[dict]:
    return [json.loads(line) for line in GOLDEN_CASES.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="session", autouse=True)
def clean_badcase_output():
    BADCASE_OUTPUT.unlink(missing_ok=True)
    yield


def test_golden_cases_count_is_20() -> None:
    assert len(load_cases()) == 20


@pytest.mark.parametrize("case", load_cases(), ids=lambda case: case["id"])
def test_workflow_golden_case(runtime, case) -> None:
    conversation_id = f"golden-{case['id']}-{uuid4().hex}"
    customer_id = case["customer_id"]
    messages = case.get("messages") or [case["message"]]
    result = None
    try:
        for message in messages:
            result = runtime.invoke(conversation_id, customer_id, message)
        assert result is not None
        if "confirm" in case:
            result = runtime.confirm(
                conversation_id,
                customer_id,
                result["pending_confirmation"]["action_id"],
                approved=case["confirm"],
            )
    except ToolPermissionError as error:
        if case.get("expected_error") == "ToolPermissionError":
            assert case["expected_contains"] in str(error)
            return
        record_badcase(case, {"error": type(error).__name__, "message": str(error)})
        raise

    state = runtime.graph.get_state({"configurable": {"thread_id": conversation_id}}).values
    route = state["route"]
    decision = state["decision"]
    failures = []
    if result["status"] != case["expected_status"]:
        failures.append(f"status {result['status']} != {case['expected_status']}")
    if route["intent"] != case["expected_intent"]:
        failures.append(f"intent {route['intent']} != {case['expected_intent']}")
    if decision["action"] != case["expected_action"]:
        failures.append(f"action {decision['action']} != {case['expected_action']}")
    for expected_agent in case.get("expected_agents", []):
        if expected_agent not in result.get("agents_invoked", []):
            failures.append(f"missing agent {expected_agent}")
    if case["expected_contains"] not in result["reply"]:
        failures.append(f"reply does not contain {case['expected_contains']}")
    if failures:
        record_badcase(case, {"failures": failures, "result": result, "state": state})
    assert failures == []


def record_badcase(case: dict, payload: dict) -> None:
    with BADCASE_OUTPUT.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"case_id": case["id"], "payload": payload}, ensure_ascii=False) + "\n"
        )
```

- [ ] **Step 3: Run golden evaluation**

Run:

```powershell
cd python-impl
pytest tests/evaluation/test_workflow_golden.py -q
```

Expected: PASS. If a case fails, `tests/evaluation/badcase_candidates.jsonl` contains the failing case payload.

- [ ] **Step 4: Commit Task 9**

Run:

```powershell
git add python-impl/tests/evaluation/golden_cases.jsonl python-impl/tests/evaluation/test_workflow_golden.py
git commit -m "test: add workflow golden evaluation"
```

## Task 10: Full Regression And P0 Acceptance

**Files:**
- Modify only files needed for regression fixes discovered by tests.

- [ ] **Step 1: Run focused unit tests**

Run:

```powershell
cd python-impl
pytest tests/unit/test_router_supervisor.py tests/unit/test_context_projector.py tests/unit/test_state_update.py tests/unit/test_tool_policy.py tests/unit/test_policy_engine.py tests/unit/test_memory.py tests/unit/test_tools.py -q
```

Expected: PASS.

- [ ] **Step 2: Run integration tests**

Run:

```powershell
cd python-impl
pytest tests/integration/test_action_confirmation.py tests/integration/test_image_after_sales.py -q
```

Expected: PASS.

- [ ] **Step 3: Run API tests**

Run:

```powershell
cd python-impl
pytest tests/api/test_conversations.py tests/api/test_image_message.py tests/api/test_knowledge_reply.py tests/api/test_health.py -q
```

Expected: PASS.

- [ ] **Step 4: Run golden evaluation**

Run:

```powershell
cd python-impl
pytest tests/evaluation/test_workflow_golden.py -q
```

Expected: PASS and no new `tests/evaluation/badcase_candidates.jsonl` changes for passing cases.

- [ ] **Step 5: Run complete test suite**

Run:

```powershell
cd python-impl
pytest -q
```

Expected: PASS.

- [ ] **Step 6: Manual code-readability acceptance**

Check these constraints before final commit:

```text
Each graph node is approximately 30 lines or fewer; over-long nodes must be refactored.
agent_runtime graph nodes are orchestration-only; no complex business rules in nodes.
Complex business rules go in independent services (SlotCarryService, StateUpdater, PolicyEngine).
No graph node contains a long inline prompt.
Prompt constants live in infrastructure/prompts.py, not inlined in model_factory.py.
Pydantic schemas (RouteAnalysis, SupervisorDecision, ConversationSlots, etc.) are centralized in agents/state.py.
ToolPolicy does not duplicate LangChain/customer tool argument schema.
ToolRegistry stores business policy only; no tool execution, no duplicate LangChain tool param schemas.
AuthorizedToolExecutor only handles auth, execution, idempotency, lease, and audit.
ContextProjector only constructs context; does not execute tools or call LLM.
ContextProjector does not read memory_candidates.
memory_writeback does not write AIMessage.
memory_writeback does not write candidate/human_review records into active memories.
RemoveMessage is generated only for HumanMessage and AIMessage.
synthesize is the only node that returns AIMessage in messages.
interrupt payload keeps the existing pending API shape.
invalid confirmation resume payloads raise ValueError instead of cancelling.
No long prompt string concatenation inside graph nodes.
Tests are layered: router, context projector, slot carry, state updater, policy, tool permission, workflow.
```

- [ ] **Step 7: Commit regression fixes**

Run:

```powershell
git status --short
git add python-impl/src/smart_cs python-impl/tests
git commit -m "test: verify p0 agent engineering workflow"
```

If there are no regression fixes after Task 9, skip this commit and record the passing commands in the execution summary.

## Self-Review Checklist

Spec coverage:

- Schema extension: Task 1.
- Prompt engineering: Task 2.
- LangGraph `messages` and `add_messages`: Task 1 and Task 8.
- `trim_messages`: Task 4.
- `AIMessage` single write: Task 8.
- `ConversationSlots`, `slot_carry`, `state_update`: Task 5 and Task 8.
- `ToolPolicy`, `ToolRegistry`, `caller_agent`: Task 3.
- `PolicyEngine`: Task 6.
- `ConversationSummary`, `RemoveMessage`, `MemoryExtractor`, `MemoryPolicy`, memory writeback: Task 7 and Task 8.
- `confirm_action` via LangGraph `interrupt`/`Command(resume=...)`: Task 8.
- Runtime graph order: Task 8.
- Evaluation and badcase candidate: Task 9.
- Code readability constraints: Task 10.

Placeholder scan:

- The plan intentionally avoids open-ended implementation placeholders.
- Each task includes exact file paths, concrete test commands, expected failures, and expected passes.

Type consistency:

- `RouterAgent.analyze()` accepts `RouterContext`.
- `SupervisorAgent.plan()` accepts `SupervisorContext`.
- `RulesDecisionModel.route()` and `LangChainDecisionModel.route()` accept `RouterContext`.
- `RulesDecisionModel.plan()` and `LangChainDecisionModel.plan()` accept `SupervisorContext`.
- `RuntimeState.messages` uses `Annotated[list[AnyMessage], add_messages]`.
- `AuthorizedToolExecutor.invoke()` requires `caller_agent`.

Review correction coverage:

- Task order creates `ToolPolicy` before `ContextProjector`.
- `graph.invoke()` does not pass persistent `conversation_slots`, `conversation_summary`, or `customer_memories`.
- `RulesDecisionModel` infers `follow_up`, `correction`, `confirmation_like`, and `rejection_like`.
- Ordinary after-sales plans include `OrderAgent`, `KnowledgeAgent`, and `AfterSalesAgent`.
- `_read_specialists_node()` appends read agents to `agents_invoked`.
- `execute_read_agents()` handles empty read-agent lists.
- `MemoryWriteback` depends on a store protocol and receives `runtime.store`.
- `MemoryPolicy` returns `write`, `candidate`, `human_review`, or `discard`.
- Active memories and memory candidates are split into separate namespaces.
- `ContextProjector` reads only `("customer", customer_id, "memories")`, limits active memories to 5, and never reads `memory_candidates`.
- Conversation service events write to `("conversation", conversation_id, "events")`.
- `MemoryCandidate` includes `title`, `description`, `evidence`, and `review_status`.
- `MemoryExtractor` handles pending, submitted, cancelled action events, and explicit user preference candidates.
- `ConversationSummarizer` is separate from extractor/policy/writer; `RemoveMessage` is returned only for `HumanMessage` / `AIMessage` after the summary covers removed content.
- `MemoryRecord.id` uses `String(255)`.
- `_confirm_action_node()` uses `interrupt()` with the existing pending response shape: `status`, `pending_confirmation`, `reply`, and `agents_invoked`.
- `_confirm_action_node()` rejects invalid resume payloads unless `approved` is explicitly boolean.
- `_confirm_action_node()` does not call `SpecialistDispatcher`; it submits or cancels through `AuthorizedToolExecutor`.
- `submit_confirmed_action` and `cancel_pending_action` require `caller_agent="ConfirmActionNode"` and are covered by tool-policy tests.
- Rejected confirmations persist cancellation through `cancel_pending_action` instead of constructing a fake cancelled result.
- `_confirm_action_node()` clears `pending_confirmation` after approved and rejected confirmations.
- Natural-language `confirmation_like` does not submit write actions in P0.
- Golden evaluation enforces exactly 20 cases and appends badcase candidates.
