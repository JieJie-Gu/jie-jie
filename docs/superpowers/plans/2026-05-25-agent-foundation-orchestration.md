# Agent Foundation And Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立可运行的 Python 电商客服多 Agent 后端，完成文本会话、业务工具、独立 Router、Supervisor 编排，以及需确认的售后动作。

**Architecture:** 新实现放在 `src/smart_cs` 中。`RouterAgent` 只输出意图、实体和风险分析，`SupervisorAgent` 通过结构化输出决定子 Agent 调用顺序，随后由确定性授权节点限制工具和副作用。LangGraph `StateGraph` 负责执行流程，`interrupt`、`Command` 与 SQLite checkpointer 负责确认后继续执行。无 API 学习模式使用明确标注的规则模型；配置模型时使用 LangChain chat model，而不是绕过框架直接调用 HTTP SDK。

**Tech Stack:** Python 3.11, FastAPI, Pydantic v2, SQLAlchemy 2, LangChain, `langchain-openai`, LangGraph, `langgraph-checkpoint-sqlite`, pytest

---

## Boundary

本计划覆盖第 1 至 3 天，只交付文本业务主链路：

- 商品咨询、订单查询、售后草稿和人工转接草稿。
- Router 与 Supervisor 是两个独立 Agent 节点。
- 所有写操作先产生草稿，再由 `/actions/confirm` 恢复被中断的图执行。
- 售后提交所需的是用户本人确认，不是人工客服接管；人工转接仅用于争议、系统失败、证据不足或用户明确要求。
- 订单和工单事实来自 SQLite 工具，不来自知识库。
- API key 不是启动和学习项目的必要条件：`SMART_CS_MODEL_MODE=rules` 可运行确定性的学习演示；真实 LLM 模式单独配置。

此阶段不实现 Milvus、图片理解、前端、生产认证、外部工单系统或传统 ReAct 循环。

## Official Component Decisions

| Need | Adopted official component | Project-specific code retained |
| --- | --- | --- |
| 有状态编排 | `langgraph.graph.StateGraph`, `START`, `END` | 业务状态类型与节点映射 |
| 路由/规划输出 | chat model `.with_structured_output(PydanticModel)` | 路由 schema、安全 allow-list |
| 工具 schema | `langchain.tools.tool` | 客户归属校验和写操作授权器 |
| 操作确认 | `langgraph.types.interrupt`, `Command(resume=...)` | API 的确认端点和动作审计 |
| 暂停状态保存 | `langgraph.checkpoint.sqlite.SqliteSaver` | SQLite 文件配置 |

官方参考：

- <https://docs.langchain.com/oss/python/langgraph/workflows-agents>
- <https://docs.langchain.com/oss/python/langgraph/interrupts>
- <https://docs.langchain.com/oss/python/langchain/multi-agent>
- <https://docs.langchain.com/oss/python/integrations/chat/openai>

## File Map

Create:

```text
python-impl/pyproject.toml
python-impl/src/smart_cs/__init__.py
python-impl/src/smart_cs/config.py
python-impl/src/smart_cs/main.py
python-impl/src/smart_cs/api/dependencies.py
python-impl/src/smart_cs/api/schemas.py
python-impl/src/smart_cs/api/routers/conversations.py
python-impl/src/smart_cs/domain/enums.py
python-impl/src/smart_cs/domain/models.py
python-impl/src/smart_cs/domain/repositories.py
python-impl/src/smart_cs/infrastructure/database.py
python-impl/src/smart_cs/infrastructure/repositories.py
python-impl/src/smart_cs/infrastructure/model_factory.py
python-impl/src/smart_cs/tools/customer_tools.py
python-impl/src/smart_cs/tools/executor.py
python-impl/src/smart_cs/agents/state.py
python-impl/src/smart_cs/agents/router.py
python-impl/src/smart_cs/agents/supervisor.py
python-impl/src/smart_cs/agents/specialists.py
python-impl/src/smart_cs/agents/guardrails.py
python-impl/src/smart_cs/application/agent_runtime.py
python-impl/src/smart_cs/application/conversation_service.py
python-impl/scripts/seed_demo_data.py
python-impl/tests/conftest.py
python-impl/tests/api/test_health.py
python-impl/tests/api/test_conversations.py
python-impl/tests/unit/test_tools.py
python-impl/tests/unit/test_router_supervisor.py
python-impl/tests/integration/test_action_confirmation.py
```

Modify:

```text
python-impl/.env.example
python-impl/Dockerfile
```

保留旧 `agents/`、`api/`、`memory/`、`mcp/`、`tracing/` 到最终交付计划再清理，避免重建阶段混用 import path。

### Task 1: Build The Package And Health Endpoint

**Files:**
- Create: `python-impl/pyproject.toml`
- Create: `python-impl/src/smart_cs/__init__.py`
- Create: `python-impl/src/smart_cs/config.py`
- Create: `python-impl/src/smart_cs/main.py`
- Create: `python-impl/tests/conftest.py`
- Create: `python-impl/tests/api/test_health.py`
- Modify: `python-impl/.env.example`

- [ ] **Step 1: Write the failing API smoke test**

```python
# python-impl/tests/api/test_health.py
from fastapi.testclient import TestClient

from smart_cs.main import app


def test_health_reports_foundation_phase() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "service": "smart-cs-agent",
        "phase": "foundation",
    }
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
cd python-impl
pytest tests/api/test_health.py -q
```

Expected: collection fails because `smart_cs` does not exist.

- [ ] **Step 3: Create packaging and settings**

Use this dependency set in `python-impl/pyproject.toml`:

```toml
[project]
name = "smart-cs-agent"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "fastapi>=0.115,<1",
  "uvicorn[standard]>=0.32,<1",
  "pydantic>=2.10,<3",
  "pydantic-settings>=2.7,<3",
  "sqlalchemy>=2.0,<3",
  "langchain>=1.0,<2",
  "langchain-openai>=1.0,<2",
  "langgraph>=1.0,<2",
  "langgraph-checkpoint-sqlite>=2.0",
  "python-multipart>=0.0.20,<1",
]

[project.optional-dependencies]
test = ["httpx>=0.28,<1", "pytest>=8.3,<9"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

Generate and commit the environment lock file used by the chosen installer after these packages resolve together; the code examples intentionally follow the current LangChain/LangGraph documentation namespace.

```python
# python-impl/src/smart_cs/config.py
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SMART_CS_", env_file=".env")
    database_url: str = "sqlite:///./data/smart_cs.db"
    checkpoint_path: Path = Path("data/checkpoints.db")
    model_mode: str = "rules"
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str | None = None
    llm_api_key: str | None = None
```

Set these values in `python-impl/.env.example`:

```dotenv
SMART_CS_DATABASE_URL=sqlite:///./data/smart_cs.db
SMART_CS_CHECKPOINT_PATH=data/checkpoints.db
SMART_CS_MODEL_MODE=rules
SMART_CS_LLM_MODEL=gpt-4o-mini
SMART_CS_LLM_BASE_URL=
SMART_CS_LLM_API_KEY=
```

- [ ] **Step 4: Implement the health application and verify**

```python
# python-impl/src/smart_cs/main.py
from fastapi import FastAPI

app = FastAPI(title="Smart CS Multi-Agent")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy", "service": "smart-cs-agent", "phase": "foundation"}
```

Run:

```bash
cd python-impl
pytest tests/api/test_health.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the package skeleton**

```bash
git add python-impl/pyproject.toml python-impl/.env.example python-impl/src/smart_cs python-impl/tests
git commit -m "feat: create smart cs python application skeleton"
```

### Task 2: Persist Customer Facts And Authorised Tools

**Files:**
- Create: `python-impl/src/smart_cs/domain/enums.py`
- Create: `python-impl/src/smart_cs/domain/models.py`
- Create: `python-impl/src/smart_cs/domain/repositories.py`
- Create: `python-impl/src/smart_cs/infrastructure/database.py`
- Create: `python-impl/src/smart_cs/infrastructure/repositories.py`
- Create: `python-impl/src/smart_cs/tools/customer_tools.py`
- Create: `python-impl/src/smart_cs/tools/executor.py`
- Create: `python-impl/tests/unit/test_tools.py`
- Create: `python-impl/scripts/seed_demo_data.py`

- [ ] **Step 1: Write failing authorization tests**

```python
# python-impl/tests/unit/test_tools.py
import pytest

from smart_cs.tools.executor import AuthorizedToolExecutor, ToolPermissionError


def test_order_lookup_rejects_another_customer(repo) -> None:
    tools = AuthorizedToolExecutor(repo)
    with pytest.raises(ToolPermissionError):
        tools.invoke("lookup_order", {"customer_id": "C002", "order_id": "O1001"})


def test_after_sales_only_creates_draft_before_confirmation(repo) -> None:
    tools = AuthorizedToolExecutor(repo)
    result = tools.invoke(
        "draft_after_sales",
        {"customer_id": "C001", "order_id": "O1001", "reason": "鞋底开胶"},
    )
    assert result["status"] == "pending_confirmation"
    assert repo.list_tickets("C001") == []
```

- [ ] **Step 2: Run and observe missing tool layer**

```bash
cd python-impl
pytest tests/unit/test_tools.py -q
```

Expected: FAIL importing the new tool layer.

- [ ] **Step 3: Implement repository entities and LangChain tool schemas**

Persist `Customer`, `Product`, `Order`, `PendingAction`, `Ticket`, `Message`, and `ToolCall` in SQLAlchemy. Define read and draft tools with official tool decorators:

```python
# python-impl/src/smart_cs/tools/customer_tools.py
from langchain.tools import tool


@tool
def search_products(query: str) -> dict:
    """Search product facts by a customer-visible query."""
    raise RuntimeError("Executed only through AuthorizedToolExecutor")


@tool
def lookup_order(customer_id: str, order_id: str) -> dict:
    """Read an order owned by the current customer."""
    raise RuntimeError("Executed only through AuthorizedToolExecutor")


@tool
def draft_after_sales(customer_id: str, order_id: str, reason: str) -> dict:
    """Prepare a return or refund request that still needs confirmation."""
    raise RuntimeError("Executed only through AuthorizedToolExecutor")


@tool
def draft_handoff(customer_id: str, reason: str) -> dict:
    """Prepare an escalation request that still needs confirmation."""
    raise RuntimeError("Executed only through AuthorizedToolExecutor")
```

The executor owns authorization and records every operation. Its public contract is:

```python
class AuthorizedToolExecutor:
    def invoke(self, tool_name: str, arguments: dict) -> dict: ...
    def submit_confirmed_action(self, action_id: str, customer_id: str) -> dict: ...
    def cancel_pending_action(self, action_id: str, customer_id: str) -> dict: ...
```

Rules in `invoke`:

1. Validate the customer owns the requested order.
2. Execute read operations immediately.
3. Store after-sales and handoff operations as `PendingAction(status="pending_confirmation")`.
4. Record success and rejected attempts in `ToolCall`.

Rules in `submit_confirmed_action`:

1. Load the pending action for the same customer.
2. Idempotently create one `Ticket`.
3. Mark the pending action `submitted`.
4. Record the submission in `ToolCall`.

Rules in `cancel_pending_action`:

1. Verify the draft belongs to the same customer.
2. Mark it `cancelled` without creating a `Ticket`.
3. Record cancellation in `ToolCall`.

- [ ] **Step 4: Seed a demonstrable customer case and run tests**

`seed_demo_data.py` must insert customer `C001`, order `O1001`, a delivered shoe product, and customer `C002` so the ownership test is meaningful.

```bash
cd python-impl
pytest tests/unit/test_tools.py -q
python scripts/seed_demo_data.py
```

Expected: tests PASS and the script prints inserted demo identifiers `C001` and `O1001`.

- [ ] **Step 5: Commit tools and persistence**

```bash
git add python-impl/src/smart_cs/domain python-impl/src/smart_cs/infrastructure python-impl/src/smart_cs/tools python-impl/tests/unit python-impl/scripts
git commit -m "feat: add authorised ecommerce business tools"
```

### Task 3: Implement Router, Supervisor And Pause/Resume Workflow

**Files:**
- Create: `python-impl/src/smart_cs/infrastructure/model_factory.py`
- Create: `python-impl/src/smart_cs/agents/state.py`
- Create: `python-impl/src/smart_cs/agents/router.py`
- Create: `python-impl/src/smart_cs/agents/supervisor.py`
- Create: `python-impl/src/smart_cs/agents/specialists.py`
- Create: `python-impl/src/smart_cs/agents/guardrails.py`
- Create: `python-impl/src/smart_cs/application/agent_runtime.py`
- Create: `python-impl/tests/unit/test_router_supervisor.py`
- Create: `python-impl/tests/integration/test_action_confirmation.py`

- [ ] **Step 1: Write tests showing responsibilities and suspension**

```python
# python-impl/tests/unit/test_router_supervisor.py
from smart_cs.agents.state import RouteAnalysis, SupervisorDecision
from smart_cs.agents.supervisor import validate_decision


def test_router_result_does_not_authorize_a_write() -> None:
    route = RouteAnalysis(intent="after_sales", entities={"order_id": "O1001"}, risk="medium")
    assert not hasattr(route, "authorized_tools")


def test_supervisor_validation_forces_confirmation_for_refund() -> None:
    decision = SupervisorDecision(agents=["OrderAgent", "AfterSalesAgent"], action="draft_after_sales")
    checked = validate_decision(decision)
    assert checked.requires_confirmation is True
```

```python
# python-impl/tests/integration/test_action_confirmation.py
def test_after_sales_graph_interrupts_then_submits_on_resume(runtime, repo) -> None:
    first = runtime.invoke("conv-1", "C001", "订单 O1001 鞋底开胶，申请退款")
    assert first["pending_confirmation"]["action_type"] == "after_sales"
    assert repo.list_tickets("C001") == []

    completed = runtime.confirm("conv-1", "C001", approved=True)
    assert completed["reply"].startswith("售后申请已受理")
    assert len(repo.list_tickets("C001")) == 1


def test_rejected_confirmation_cancels_draft_without_ticket(runtime, repo) -> None:
    runtime.invoke("conv-2", "C001", "订单 O1001 鞋底开胶，申请退款")
    completed = runtime.confirm("conv-2", "C001", approved=False)
    assert completed["reply"] == "已取消本次申请。"
    assert repo.list_tickets("C001") == []
```

- [ ] **Step 2: Define structured agent contracts**

```python
# python-impl/src/smart_cs/agents/state.py
from typing import Literal, TypedDict
from pydantic import BaseModel, Field


class RouteAnalysis(BaseModel):
    intent: Literal["product", "order", "knowledge", "after_sales", "handoff"]
    entities: dict[str, str] = Field(default_factory=dict)
    risk: Literal["low", "medium", "high"] = "low"


class SupervisorDecision(BaseModel):
    agents: list[Literal["ProductAgent", "OrderAgent", "KnowledgeAgent", "AfterSalesAgent", "HandoffAgent"]]
    action: Literal["read", "draft_after_sales", "draft_handoff"]
    requires_confirmation: bool = False


class RuntimeState(TypedDict, total=False):
    conversation_id: str
    customer_id: str
    message: str
    route: RouteAnalysis
    decision: SupervisorDecision
    result: dict
    reply: str
```

- [ ] **Step 3: Use LangChain structured output for configured models**

`RulesDecisionModel` is the no-key learning mode and must be named in logs. `LangChainDecisionModel` is the configured model path:

```python
# python-impl/src/smart_cs/infrastructure/model_factory.py
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from smart_cs.agents.state import RouteAnalysis, SupervisorDecision
from smart_cs.config import Settings


class LangChainDecisionModel:
    def __init__(self, model: BaseChatModel) -> None:
        self.route_model = model.with_structured_output(RouteAnalysis)
        self.plan_model = model.with_structured_output(SupervisorDecision)

    def route(self, message: str) -> RouteAnalysis:
        return self.route_model.invoke(
            [("system", "Classify an ecommerce support request without taking actions."), ("human", message)]
        )

    def plan(self, message: str, route: RouteAnalysis) -> SupervisorDecision:
        prompt = f"Message: {message}\nRoute: {route.model_dump_json()}\nSelect specialist agents and one action."
        return self.plan_model.invoke([("system", "You supervise ecommerce support specialists."), ("human", prompt)])


def configured_chat_model(settings: Settings) -> BaseChatModel:
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=0,
    )
```

Implement `RulesDecisionModel` with explicit keyword-to-route and intent-to-plan dictionaries for tests and local study runs. It must not be used for RAG metric claims.

- [ ] **Step 4: Keep Supervisor authority deterministic after model planning**

```python
# python-impl/src/smart_cs/agents/router.py
from smart_cs.agents.state import RouteAnalysis


class RouterAgent:
    """Analyse intent and entities without executing tools."""

    def __init__(self, decision_model) -> None:
        self.decision_model = decision_model

    def analyze(self, message: str) -> RouteAnalysis:
        return self.decision_model.route(message)
```

```python
# python-impl/src/smart_cs/agents/supervisor.py
from smart_cs.agents.state import SupervisorDecision

WRITING_ACTIONS = {"draft_after_sales", "draft_handoff"}
ALLOWED_AGENTS = {"ProductAgent", "OrderAgent", "KnowledgeAgent", "AfterSalesAgent", "HandoffAgent"}


def validate_decision(decision: SupervisorDecision) -> SupervisorDecision:
    if not decision.agents or any(agent not in ALLOWED_AGENTS for agent in decision.agents):
        raise ValueError("Supervisor proposed an invalid specialist plan")
    if decision.action in WRITING_ACTIONS:
        return decision.model_copy(update={"requires_confirmation": True})
    return decision.model_copy(update={"requires_confirmation": False})


class SupervisorAgent:
    """Plan specialist work; authorization is applied after planning."""

    def __init__(self, decision_model) -> None:
        self.decision_model = decision_model

    def plan(self, message: str, route) -> SupervisorDecision:
        proposed = self.decision_model.plan(message, route)
        return validate_decision(proposed)
```

The distinction is intentional: the LLM selects the useful specialists and order; application policy owns permissions and confirmation.

- [ ] **Step 5: Compile the LangGraph workflow with durable confirmation**

```python
# python-impl/src/smart_cs/application/agent_runtime.py
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from smart_cs.agents.router import RouterAgent
from smart_cs.agents.state import RuntimeState
from smart_cs.agents.supervisor import SupervisorAgent


class AgentRuntime:
    def __init__(self, settings, decision_model, specialists, guard, executor) -> None:
        self.executor = executor
        self.router_agent = RouterAgent(decision_model)
        self.supervisor_agent = SupervisorAgent(decision_model)
        self.specialists = specialists
        self.guard = guard
        saver = SqliteSaver(sqlite3.connect(settings.checkpoint_path, check_same_thread=False))
        builder = StateGraph(RuntimeState)
        builder.add_node("router", self._route)
        builder.add_node("supervisor", self._plan)
        builder.add_node("specialists", self._run_specialists)
        builder.add_node("confirm_action", self._confirm_action)
        builder.add_node("guard", self._guard)
        builder.add_edge(START, "router")
        builder.add_edge("router", "supervisor")
        builder.add_edge("supervisor", "specialists")
        builder.add_conditional_edges("specialists", self._next, {"confirm_action": "confirm_action", "guard": "guard"})
        builder.add_edge("confirm_action", "guard")
        builder.add_edge("guard", END)
        self.graph = builder.compile(checkpointer=saver)

    def _route(self, state: RuntimeState) -> dict:
        return {"route": self.router_agent.analyze(state["message"])}

    def _plan(self, state: RuntimeState) -> dict:
        return {"decision": self.supervisor_agent.plan(state["message"], state["route"])}

    def _run_specialists(self, state: RuntimeState) -> dict:
        return {"result": self.specialists.execute(state, self.executor)}

    def _next(self, state: RuntimeState) -> str:
        return "confirm_action" if state["decision"].requires_confirmation else "guard"

    def _confirm_action(self, state: RuntimeState) -> dict:
        approval = interrupt({"type": "confirm_action", "draft": state["result"]})
        if approval.get("approved") is not True:
            self.executor.cancel_pending_action(state["result"]["action_id"], state["customer_id"])
            return {"reply": "已取消本次申请。"}
        return {"result": self.executor.submit_confirmed_action(state["result"]["action_id"], state["customer_id"])}

    def _guard(self, state: RuntimeState) -> dict:
        return {"reply": state.get("reply") or self.guard.render(state["result"])}

    def invoke(self, conversation_id: str, customer_id: str, message: str) -> dict:
        config = {"configurable": {"thread_id": conversation_id}}
        return self.graph.invoke({"conversation_id": conversation_id, "customer_id": customer_id, "message": message}, config)

    def confirm(self, conversation_id: str, customer_id: str, approved: bool) -> dict:
        config = {"configurable": {"thread_id": conversation_id}}
        return self.graph.invoke(Command(resume={"approved": approved, "customer_id": customer_id}), config)
```

Ensure the interrupt node has no database write before `interrupt`; LangGraph re-enters the node after resume.

- [ ] **Step 6: Run orchestration tests and commit**

```bash
cd python-impl
pytest tests/unit/test_router_supervisor.py tests/integration/test_action_confirmation.py -q
git add src/smart_cs tests
git commit -m "feat: orchestrate customer agents with langgraph confirmation"
```

Expected: tests PASS; a ticket exists only after `confirm`.

### Task 4: Expose Conversation And Confirmation APIs

**Files:**
- Create: `python-impl/src/smart_cs/api/dependencies.py`
- Create: `python-impl/src/smart_cs/api/schemas.py`
- Create: `python-impl/src/smart_cs/api/routers/conversations.py`
- Create: `python-impl/src/smart_cs/application/conversation_service.py`
- Modify: `python-impl/src/smart_cs/main.py`
- Create: `python-impl/tests/api/test_conversations.py`
- Modify: `python-impl/Dockerfile`

- [ ] **Step 1: Test that HTTP follows graph suspension and resumption**

```python
def test_http_after_sales_requires_confirm_then_returns_ticket(client) -> None:
    conversation = client.post("/api/conversations", json={"customer_id": "C001"}).json()
    reply = client.post(
        f"/api/conversations/{conversation['id']}/messages",
        json={"content": "订单 O1001 鞋底开胶，申请售后"},
    ).json()
    assert reply["status"] == "pending_confirmation"
    assert reply["pending_action"]["action_type"] == "after_sales"

    completed = client.post(
        f"/api/conversations/{conversation['id']}/actions/confirm",
        json={"approved": True},
    ).json()
    assert completed["status"] == "completed"
    assert completed["reply"].startswith("售后申请已受理")
```

- [ ] **Step 2: Build dependencies without requiring an API key**

`build_runtime(settings)` must select `RulesDecisionModel()` when `settings.model_mode == "rules"` and otherwise wrap `configured_chat_model(settings)` in `LangChainDecisionModel`. Startup must create the business and checkpoint directories before constructing SQLite connections.

`ConversationService.confirm` first loads the conversation and verifies its stored `customer_id` matches the authenticated/demo caller before invoking `runtime.confirm`. It maps the graph interrupt payload into the HTTP `pending_action` response; endpoints do not inspect checkpoint storage directly.

- [ ] **Step 3: Implement thin endpoints**

Provide exactly:

```text
GET  /health
POST /api/conversations
POST /api/conversations/{conversation_id}/messages
POST /api/conversations/{conversation_id}/actions/confirm
GET  /api/conversations/{conversation_id}/tool-calls
```

Endpoints delegate to `ConversationService`; they do not reproduce routing or authorization logic.

- [ ] **Step 4: Verify API and runnable service**

```bash
cd python-impl
pytest tests -q
uvicorn smart_cs.main:app --app-dir src --host 127.0.0.1 --port 8000
```

Expected: all tests PASS; `/health` responds with the foundation phase JSON and a local rule-mode after-sales request pauses before submission.

- [ ] **Step 5: Commit API delivery**

```bash
git add python-impl/src/smart_cs python-impl/tests python-impl/Dockerfile
git commit -m "feat: expose safe conversation workflow api"
```

## Acceptance Checklist

- [ ] `RouterAgent` and `SupervisorAgent` are distinct nodes and distinct testable responsibilities.
- [ ] A configured chat model uses LangChain `.with_structured_output`; rule mode is visibly labelled as non-evaluation development mode.
- [ ] Tool schemas use `@tool`, while authorization remains application code.
- [ ] A side-effecting action is suspended with `interrupt` and resumed with `Command`.
- [ ] SQLite contains both business data and LangGraph checkpoints.
- [ ] No RAG or image claims appear in foundation test output.
