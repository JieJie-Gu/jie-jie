# Spec Exact Alignment Supplement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align the completed implementation plans exactly to `2026-05-25-python-ecommerce-agent-core-design.md`, adding missing required behavior and removing public behavior that is outside the spec.

**Architecture:** The public API, tool set, RAG metadata, conversation states, and handoff rules must match the spec exactly. Text and optional image messages go through `POST /api/conversations/{id}/messages`; non-spec routes and tool aliases are removed from the public contract. Low-confidence image evidence keeps the conversation active and asks for recapture or user confirmation; handoff is only for explicit human requests, uncovered or sensitive disputes, unresolved evidence conflicts after supplementation, or dependency failures.

**Tech Stack:** Python 3.11, FastAPI, Pydantic v2, SQLAlchemy 2, LangGraph, LangChain, Milvus, pytest, Markdown

---

## Scope Boundary

This supplement is not a new feature set. It is a corrective implementation plan for exact spec conformance:

- Keep the required feature set from the spec.
- Remove public capabilities that the spec does not list.
- Do not add frontend pages, production authentication, image vector search, file conversion, external ecommerce APIs, advanced RAG strategies, or autonomous side-effect loops.

## Required Spec Surface

The implementation must expose exactly these API routes:

```text
POST /api/conversations
POST /api/conversations/{id}/messages
GET  /api/conversations/{id}/messages
GET  /api/conversations/{id}/runs
POST /api/conversations/{id}/actions/confirm
GET  /health
```

The implementation must expose exactly these business and knowledge tools:

```text
search_products
get_product
get_order
get_shipment
retrieve_knowledge
get_ticket
draft_after_sales
draft_handoff
submit_after_sales
confirm_handoff
```

The implementation must use exactly these conversation states:

```text
active
pending_confirmation
pending_handoff
closed
```

## File Map

Create:

```text
python-impl/tests/api/test_spec_api_contract.py
python-impl/tests/unit/test_spec_tool_contract.py
python-impl/tests/unit/test_rag_spec_metadata.py
python-impl/tests/integration/test_retrieval_log.py
python-impl/tests/integration/test_image_clarification_flow.py
```

Modify:

```text
python-impl/src/smart_cs/api/routers/conversations.py
python-impl/src/smart_cs/api/schemas.py
python-impl/src/smart_cs/application/conversation_service.py
python-impl/src/smart_cs/application/agent_runtime.py
python-impl/src/smart_cs/agents/guardrails.py
python-impl/src/smart_cs/agents/knowledge.py
python-impl/src/smart_cs/agents/specialists.py
python-impl/src/smart_cs/domain/enums.py
python-impl/src/smart_cs/domain/models.py
python-impl/src/smart_cs/domain/repositories.py
python-impl/src/smart_cs/infrastructure/repositories.py
python-impl/src/smart_cs/rag/indexing.py
python-impl/src/smart_cs/rag/retrieval.py
python-impl/src/smart_cs/tools/customer_tools.py
python-impl/src/smart_cs/tools/executor.py
python-impl/tests/api/test_conversations.py
python-impl/tests/api/test_image_message.py
python-impl/tests/unit/test_markdown_windows.py
python-impl/tests/unit/test_tools.py
README.md
docs/architecture.md
docs/code-walkthrough.md
docs/project-plan.md
docs/interview/agent-project-qa.md
```

### Task 1: Match The Spec API Surface Exactly

**Files:**
- Create: `python-impl/tests/api/test_spec_api_contract.py`
- Modify: `python-impl/src/smart_cs/api/routers/conversations.py`
- Modify: `python-impl/src/smart_cs/api/schemas.py`
- Modify: `python-impl/src/smart_cs/application/conversation_service.py`
- Modify: `python-impl/tests/api/test_conversations.py`
- Modify: `python-impl/tests/api/test_image_message.py`

- [ ] **Step 1: Write failing API surface tests**

```python
# python-impl/tests/api/test_spec_api_contract.py
def test_spec_api_surface_contains_only_declared_conversation_routes(client) -> None:
    paths = set(client.app.openapi()["paths"])
    assert "/api/conversations" in paths
    assert "/api/conversations/{conversation_id}/messages" in paths
    assert "/api/conversations/{conversation_id}/runs" in paths
    assert "/api/conversations/{conversation_id}/actions/confirm" in paths
    assert "/health" in paths
    assert "/api/conversations/{conversation_id}/messages-with-image" not in paths
    assert "/api/conversations/{conversation_id}/tool-calls" not in paths


def test_messages_endpoint_accepts_text_json_and_lists_history(client) -> None:
    conversation = client.post("/api/conversations", json={"customer_id": "C001"}).json()

    sent = client.post(
        f"/api/conversations/{conversation['id']}/messages",
        json={"customer_id": "C001", "content": "订单 O1001 到哪里了？"},
    )
    assert sent.status_code == 200

    listed = client.get(
        f"/api/conversations/{conversation['id']}/messages",
        params={"customer_id": "C001"},
    )
    assert listed.status_code == 200
    body = listed.json()
    assert body["messages"][0]["content"] == "订单 O1001 到哪里了？"
    assert body["messages"][0]["content_type"] == "text"


def test_messages_endpoint_accepts_single_optional_image(client, clear_damage_jpeg) -> None:
    conversation = client.post("/api/conversations", json={"customer_id": "C001"}).json()

    response = client.post(
        f"/api/conversations/{conversation['id']}/messages",
        data={"customer_id": "C001", "content": "订单 O1001 鞋底开胶，申请退货"},
        files={"image": ("damage.jpg", clear_damage_jpeg, "image/jpeg")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"active", "pending_confirmation", "pending_handoff"}
    assert body["visual_evidence"]["evidence_summary"]


def test_runs_endpoint_returns_agent_runs_and_tool_calls(client) -> None:
    conversation = client.post("/api/conversations", json={"customer_id": "C001"}).json()
    client.post(
        f"/api/conversations/{conversation['id']}/messages",
        json={"customer_id": "C001", "content": "订单 O1001 鞋底开胶，申请售后"},
    )

    response = client.get(
        f"/api/conversations/{conversation['id']}/runs",
        params={"customer_id": "C001"},
    )

    assert response.status_code == 200
    body = response.json()
    assert "runs" in body
    assert "tool_calls" in body
```

- [ ] **Step 2: Run the API tests to verify red**

Run:

```bash
cd python-impl
pytest tests/api/test_spec_api_contract.py -q
```

Expected: FAIL because the current API still exposes non-spec routes or does not fully support message history and optional image input on `/messages`.

- [ ] **Step 3: Define exact response schemas**

Update `python-impl/src/smart_cs/api/schemas.py`:

```python
from pydantic import BaseModel, Field


class ConversationMessageResponse(BaseModel):
    id: str
    conversation_id: str
    customer_id: str
    role: str
    content: str
    content_type: str
    asset_key: str | None = None
    visual_evidence: dict | None = None


class ConversationMessagesResponse(BaseModel):
    messages: list[ConversationMessageResponse] = Field(default_factory=list)


class AgentRunsResponse(BaseModel):
    runs: list[dict] = Field(default_factory=list)
    tool_calls: list[dict] = Field(default_factory=list)
```

Ensure `ConversationWorkflowResponse.status` only returns:

```python
Literal["active", "pending_confirmation", "pending_handoff", "closed"]
```

- [ ] **Step 4: Replace non-spec routes with the exact route set**

Update `python-impl/src/smart_cs/api/routers/conversations.py` so it contains exactly these conversation routes:

```python
from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile, status


@router.post("", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
def create_conversation(
    request: ConversationCreateRequest,
    service: ConversationService = Depends(get_service),
) -> dict[str, str]:
    return service.create_conversation(request.customer_id)


@router.post("/{conversation_id}/messages", response_model=ConversationWorkflowResponse)
async def send_message(
    conversation_id: str,
    request: Request,
    service: ConversationService = Depends(get_service),
) -> dict:
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("image")
        if upload is not None and not isinstance(upload, UploadFile):
            raise ValueError("image must be an upload file")
        if upload is None:
            return service.send_message(
                conversation_id,
                str(form["customer_id"]),
                str(form["content"]),
            )
        return service.send_message_with_image(
            conversation_id,
            str(form["customer_id"]),
            str(form["content"]),
            upload.filename or "image",
            upload.content_type or "application/octet-stream",
            await upload.read(),
        )

    body = await request.json()
    message = MessageRequest.model_validate(body)
    return service.send_message(conversation_id, message.customer_id, message.content)


@router.get("/{conversation_id}/messages", response_model=ConversationMessagesResponse)
def list_messages(
    conversation_id: str,
    customer_id: str = Query(min_length=1),
    service: ConversationService = Depends(get_service),
) -> dict:
    return {"messages": service.list_messages(conversation_id, customer_id)}


@router.get("/{conversation_id}/runs", response_model=AgentRunsResponse)
def list_agent_runs(
    conversation_id: str,
    customer_id: str = Query(min_length=1),
    service: ConversationService = Depends(get_service),
) -> dict:
    return service.list_agent_runs(conversation_id, customer_id)


@router.post("/{conversation_id}/actions/confirm", response_model=ConversationWorkflowResponse)
def confirm_action(
    conversation_id: str,
    request: ConfirmRequest,
    service: ConversationService = Depends(get_service),
) -> dict:
    return service.confirm(
        conversation_id,
        request.customer_id,
        request.action_id,
        approved=request.approved,
    )
```

Remove route functions for `/messages-with-image` and `/tool-calls`.

- [ ] **Step 5: Add message history service support**

Add to `python-impl/src/smart_cs/application/conversation_service.py`:

```python
    def list_messages(self, conversation_id: str, customer_id: str) -> list[dict]:
        self.executor.require_conversation_owner(conversation_id, customer_id)
        messages = self.repository.list_messages(conversation_id, customer_id)
        return [
            {
                "id": message.id,
                "conversation_id": message.conversation_id,
                "customer_id": message.customer_id,
                "role": message.role,
                "content": message.content,
                "content_type": message.content_type,
                "asset_key": message.asset_key,
                "visual_evidence": message.visual_evidence,
            }
            for message in messages
        ]
```

Add to the repository interface:

```python
    def list_messages(self, conversation_id: str, customer_id: str) -> list[Message]: ...
```

Add to the SQLAlchemy repository:

```python
    def list_messages(self, conversation_id: str, customer_id: str) -> list[Message]:
        self.require_conversation_owner(conversation_id, customer_id)
        with self.session_factory() as session:
            rows = session.scalars(
                select(MessageRow)
                .where(MessageRow.conversation_id == conversation_id)
                .order_by(MessageRow.created_at.asc())
            ).all()
            return [self._message_from_row(row) for row in rows]
```

- [ ] **Step 6: Verify API exactness and commit**

Run:

```bash
cd python-impl
pytest tests/api/test_spec_api_contract.py tests/api/test_conversations.py tests/api/test_image_message.py -q
git add src/smart_cs tests/api/test_spec_api_contract.py tests/api/test_conversations.py tests/api/test_image_message.py
git commit -m "feat: align conversation api exactly to spec"
```

Expected: PASS; OpenAPI contains the required routes and does not contain `/messages-with-image` or `/tool-calls`.

### Task 2: Match The Spec Tool Set Exactly

**Files:**
- Create: `python-impl/tests/unit/test_spec_tool_contract.py`
- Modify: `python-impl/src/smart_cs/tools/customer_tools.py`
- Modify: `python-impl/src/smart_cs/tools/executor.py`
- Modify: `python-impl/src/smart_cs/domain/models.py`
- Modify: `python-impl/src/smart_cs/domain/repositories.py`
- Modify: `python-impl/src/smart_cs/infrastructure/repositories.py`
- Modify: `python-impl/tests/unit/test_tools.py`

- [ ] **Step 1: Write failing exact tool contract tests**

```python
# python-impl/tests/unit/test_spec_tool_contract.py
import pytest

from smart_cs.domain.errors import ToolPermissionError
from smart_cs.tools.executor import AuthorizedToolExecutor, ToolInvocationContext


SPEC_TOOLS = {
    "search_products",
    "get_product",
    "get_order",
    "get_shipment",
    "retrieve_knowledge",
    "get_ticket",
    "draft_after_sales",
    "draft_handoff",
    "submit_after_sales",
    "confirm_handoff",
}


def test_declared_tools_match_spec_exactly(repo) -> None:
    tools = AuthorizedToolExecutor(repo)
    assert set(tools.declared_tools) == SPEC_TOOLS
    assert "lookup_order" not in tools.declared_tools


def test_order_and_shipment_tools_return_business_facts(repo) -> None:
    tools = AuthorizedToolExecutor(repo)
    context = ToolInvocationContext(
        calling_agent="OrderAgent",
        authorized_tools={"get_order", "get_shipment"},
    )

    order = tools.invoke("get_order", {"customer_id": "C001", "order_id": "O1001"}, context=context)
    shipment = tools.invoke("get_shipment", {"customer_id": "C001", "order_id": "O1001"}, context=context)

    assert order["order_id"] == "O1001"
    assert order["customer_id"] == "C001"
    assert shipment["order_id"] == "O1001"
    assert shipment["tracking_events"]


def test_product_and_ticket_read_tools_exist(repo) -> None:
    tools = AuthorizedToolExecutor(repo)
    product = tools.invoke(
        "get_product",
        {"product_id": "P1001"},
        context=ToolInvocationContext(calling_agent="ProductAgent", authorized_tools={"get_product"}),
    )
    ticket = tools.invoke(
        "get_ticket",
        {"customer_id": "C001", "ticket_id": "missing"},
        context=ToolInvocationContext(calling_agent="HandoffAgent", authorized_tools={"get_ticket"}),
    )

    assert product["product_id"] == "P1001"
    assert ticket["status"] == "not_found"


def test_tool_call_records_required_audit_fields(repo) -> None:
    tools = AuthorizedToolExecutor(repo)

    with pytest.raises(ToolPermissionError):
        tools.invoke(
            "get_order",
            {"customer_id": "C002", "order_id": "O1001"},
            context=ToolInvocationContext(calling_agent="OrderAgent", authorized_tools={"get_order"}),
        )

    call = repo.list_tool_calls("C002")[-1]
    assert call.tool_name == "get_order"
    assert call.calling_agent == "OrderAgent"
    assert call.input_summary == {"customer_id": "C002", "order_id": "O1001"}
    assert call.status == "rejected"
    assert call.duration_ms >= 0
    assert call.error_type == "ToolPermissionError"


def test_unplanned_tool_is_rejected(repo) -> None:
    tools = AuthorizedToolExecutor(repo)

    with pytest.raises(ToolPermissionError):
        tools.invoke(
            "draft_after_sales",
            {"customer_id": "C001", "order_id": "O1001", "reason": "鞋底开胶"},
            context=ToolInvocationContext(calling_agent="OrderAgent", authorized_tools={"get_order"}),
        )
```

- [ ] **Step 2: Run the exact tool tests to verify red**

Run:

```bash
cd python-impl
pytest tests/unit/test_spec_tool_contract.py -q
```

Expected: FAIL because the declared tool set and audit metadata do not yet match the spec exactly.

- [ ] **Step 3: Declare only the spec tools**

Update `python-impl/src/smart_cs/tools/customer_tools.py` so `CUSTOMER_TOOL_SCHEMAS` contains exactly:

```python
from langchain.tools import tool


@tool
def search_products(query: str) -> dict:
    """Search product facts by a customer-visible query."""
    raise RuntimeError("Executed only through AuthorizedToolExecutor")


@tool
def get_product(product_id: str) -> dict:
    """Read one product by id."""
    raise RuntimeError("Executed only through AuthorizedToolExecutor")


@tool
def get_order(customer_id: str, order_id: str) -> dict:
    """Read an order owned by the current customer."""
    raise RuntimeError("Executed only through AuthorizedToolExecutor")


@tool
def get_shipment(customer_id: str, order_id: str) -> dict:
    """Read shipment events for an order owned by the current customer."""
    raise RuntimeError("Executed only through AuthorizedToolExecutor")


@tool
def retrieve_knowledge(query: str, category: str | None = None) -> dict:
    """Retrieve cited policy, FAQ, shipping, or product-guide knowledge."""
    raise RuntimeError("Executed only through AuthorizedToolExecutor")


@tool
def get_ticket(customer_id: str, ticket_id: str) -> dict:
    """Read a ticket owned by the current customer."""
    raise RuntimeError("Executed only through AuthorizedToolExecutor")


@tool
def draft_after_sales(customer_id: str, order_id: str, reason: str) -> dict:
    """Prepare an after-sales request that still needs confirmation."""
    raise RuntimeError("Executed only through AuthorizedToolExecutor")


@tool
def draft_handoff(customer_id: str, reason: str) -> dict:
    """Prepare a handoff request that still needs confirmation."""
    raise RuntimeError("Executed only through AuthorizedToolExecutor")


@tool
def submit_after_sales(action_id: str, customer_id: str) -> dict:
    """Submit a confirmed after-sales pending action."""
    raise RuntimeError("Executed only through AuthorizedToolExecutor")


@tool
def confirm_handoff(action_id: str, customer_id: str) -> dict:
    """Submit a confirmed handoff pending action."""
    raise RuntimeError("Executed only through AuthorizedToolExecutor")


CUSTOMER_TOOL_SCHEMAS = [
    search_products,
    get_product,
    get_order,
    get_shipment,
    retrieve_knowledge,
    get_ticket,
    draft_after_sales,
    draft_handoff,
    submit_after_sales,
    confirm_handoff,
]
```

- [ ] **Step 4: Add invocation context and exact authorization**

Update `python-impl/src/smart_cs/tools/executor.py`:

```python
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ToolInvocationContext:
    calling_agent: str
    authorized_tools: set[str] = field(default_factory=set)
```

Change `AuthorizedToolExecutor.invoke`:

```python
    def invoke(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        context: ToolInvocationContext,
        turn_fence: TurnFence | None = None,
    ) -> dict[str, Any]:
        if tool_name not in self.declared_tools:
            raise ValueError(f"Unknown customer tool: {tool_name}")
        if tool_name not in context.authorized_tools:
            raise ToolPermissionError("Tool is not authorized for this supervisor plan")
        provided_arguments = dict(arguments)
```

Update all internal callers to pass a `ToolInvocationContext` built from the Supervisor decision.

- [ ] **Step 5: Implement exact read and write handlers**

Use these handler maps:

```python
        self._read_handlers = {
            "search_products": self._search_products,
            "get_product": self._get_product,
            "get_order": self._get_order,
            "get_shipment": self._get_shipment,
            "retrieve_knowledge": self._retrieve_knowledge,
            "get_ticket": self._get_ticket,
        }
        self._write_handlers = {
            "draft_after_sales": self._draft_after_sales,
            "draft_handoff": self._draft_handoff,
            "submit_after_sales": self._submit_after_sales,
            "confirm_handoff": self._confirm_handoff,
        }
```

Add handlers:

```python
    def _get_product(self, arguments: dict[str, Any]) -> dict[str, Any]:
        product = self.repository.get_product(str(arguments["product_id"]))
        if product is None:
            return {"status": "not_found", "product_id": str(arguments["product_id"])}
        return self._product_result(product)

    def _get_order(self, arguments: dict[str, Any]) -> dict[str, Any]:
        customer_id = str(arguments["customer_id"])
        order_id = str(arguments["order_id"])
        return self._order_result(self._owned_order(customer_id, order_id))

    def _get_shipment(self, arguments: dict[str, Any]) -> dict[str, Any]:
        customer_id = str(arguments["customer_id"])
        order_id = str(arguments["order_id"])
        self._owned_order(customer_id, order_id)
        shipment = self.repository.get_shipment_for_order(customer_id, order_id)
        if shipment is None:
            return {"status": "not_found", "order_id": order_id, "tracking_events": []}
        return {
            "order_id": shipment.order_id,
            "carrier": shipment.carrier,
            "tracking_number": shipment.tracking_number,
            "tracking_events": shipment.tracking_events,
        }

    def _get_ticket(self, arguments: dict[str, Any]) -> dict[str, Any]:
        ticket = self.repository.get_ticket(str(arguments["customer_id"]), str(arguments["ticket_id"]))
        if ticket is None:
            return {"status": "not_found", "ticket_id": str(arguments["ticket_id"])}
        return {"status": ticket.status, "ticket_id": ticket.id, "customer_id": ticket.customer_id}
```

`retrieve_knowledge` may delegate to the existing `KnowledgeAgent` or retrieval service. It must not read orders, shipments, tickets, image evidence, or live business state.

- [ ] **Step 6: Persist exact ToolCall audit fields**

Add to the `ToolCall` domain and database row:

```python
calling_agent: str
input_summary: dict
duration_ms: int
error_type: str | None
```

Record success and rejection with:

```python
self.repository.record_tool_call(
    tool_name=tool_name,
    calling_agent=context.calling_agent,
    input_summary={key: value for key, value in arguments.items() if key != "idempotency_key"},
    customer_id=str(customer_id) if customer_id is not None else None,
    status=ToolCallStatus.SUCCEEDED.value,
    result=result,
    duration_ms=self._duration_ms(started),
)
```

Use `ToolCallStatus.REJECTED.value` and `error_type=type(error).__name__` for rejected calls.

- [ ] **Step 7: Verify exact tool set and commit**

Run:

```bash
cd python-impl
pytest tests/unit/test_spec_tool_contract.py tests/unit/test_tools.py -q
git add src/smart_cs tests/unit/test_spec_tool_contract.py tests/unit/test_tools.py
git commit -m "feat: align customer tools exactly to spec"
```

Expected: PASS; the declared tool set equals the spec and no `lookup_order` tool remains.

### Task 3: Match RAG Metadata, RetrievalLog, And Failure Behavior

**Files:**
- Create: `python-impl/tests/unit/test_rag_spec_metadata.py`
- Create: `python-impl/tests/integration/test_retrieval_log.py`
- Modify: `python-impl/src/smart_cs/rag/indexing.py`
- Modify: `python-impl/src/smart_cs/rag/retrieval.py`
- Modify: `python-impl/src/smart_cs/agents/knowledge.py`
- Modify: `python-impl/src/smart_cs/domain/models.py`
- Modify: `python-impl/src/smart_cs/domain/repositories.py`
- Modify: `python-impl/src/smart_cs/infrastructure/repositories.py`
- Modify: `python-impl/tests/unit/test_markdown_windows.py`
- Modify: `python-impl/tests/unit/test_query_policy.py`

- [ ] **Step 1: Write failing RAG metadata and log tests**

```python
# python-impl/tests/unit/test_rag_spec_metadata.py
from smart_cs.agents.knowledge import KnowledgeAgent
from smart_cs.rag.indexing import markdown_sentence_documents
from smart_cs.rag.retrieval import KnowledgeUnavailable


def test_sentence_document_uses_section_path_metadata() -> None:
    markdown = "# 售后政策\n## 七天无理由\n签收后七天内可以申请退货。商品应保持完好。"
    documents = markdown_sentence_documents("after_sales_policy", "after_sales", markdown)

    metadata = documents[0].metadata
    assert metadata["document_id"] == "after_sales_policy"
    assert metadata["category"] == "after_sales"
    assert metadata["section_path"] == "售后政策 > 七天无理由"
    assert "header_path" not in metadata
    assert "window_text" in metadata


class UnavailableStore:
    def similarity_search(self, *args, **kwargs):
        raise KnowledgeUnavailable("Milvus is unavailable")


def test_knowledge_agent_returns_safe_unavailable_reply() -> None:
    answer = KnowledgeAgent(UnavailableStore()).answer("退货需要几天内申请？")

    assert answer.answer == "知识服务暂时不可用，请重试或转人工确认。"
    assert answer.citations == []
    assert answer.contexts == []
```

```python
# python-impl/tests/integration/test_retrieval_log.py
def test_retrieval_log_records_query_filter_contexts_and_status(runtime, repo) -> None:
    runtime.invoke("conv-rag-log", "C001", "七天无理由退货有哪些条件？")

    logs = repo.list_retrieval_logs("conv-rag-log")
    assert logs
    log = logs[-1]
    assert log.query
    assert log.rewritten_query
    assert log.metadata_filter == 'category == "after_sales"'
    assert log.context_ids
    assert log.status in {"succeeded", "insufficient_evidence", "unavailable"}
```

- [ ] **Step 2: Run RAG tests to verify red**

Run:

```bash
cd python-impl
pytest tests/unit/test_rag_spec_metadata.py tests/integration/test_retrieval_log.py -q
```

Expected: FAIL because the current metadata or retrieval logging does not match the spec exactly.

- [ ] **Step 3: Use `section_path` as the only chapter-path metadata**

Update `python-impl/src/smart_cs/rag/indexing.py`:

```python
        section_path = " > ".join(
            section.metadata[key] for key in ("h1", "h2", "h3") if key in section.metadata
        )
```

Use this metadata:

```python
                    metadata={
                        "document_id": document_id,
                        "category": category,
                        "section_path": section_path,
                        "window_text": "".join(sentences[start:end]),
                    },
```

Do not add `header_path`.

- [ ] **Step 4: Add dependency-unavailable behavior**

Update `python-impl/src/smart_cs/rag/retrieval.py`:

```python
class KnowledgeUnavailable(RuntimeError):
    """Raised when Milvus or embeddings are unavailable for knowledge answers."""
```

Wrap store calls:

```python
        try:
            documents = store.similarity_search(
                rewritten_query,
                k=4,
                expr=category_expression,
                ranker_type="rrf",
                ranker_params={"k": 60},
            )
        except Exception as error:
            raise KnowledgeUnavailable("Milvus is unavailable") from error
```

Update `KnowledgeAgent.answer` to return:

```python
"知识服务暂时不可用，请重试或转人工确认。"
```

when `KnowledgeUnavailable` is raised.

- [ ] **Step 5: Persist RetrievalLog records**

Add a domain model:

```python
@dataclass
class RetrievalLog:
    id: str
    conversation_id: str
    query: str
    rewritten_query: str
    metadata_filter: str
    context_ids: list[str]
    status: str
```

Add repository methods:

```python
    def record_retrieval_log(
        self,
        *,
        conversation_id: str,
        query: str,
        rewritten_query: str,
        metadata_filter: str,
        context_ids: list[str],
        status: str,
    ) -> None: ...

    def list_retrieval_logs(self, conversation_id: str) -> list[RetrievalLog]: ...
```

Record `succeeded`, `insufficient_evidence`, or `unavailable` for every knowledge query.

- [ ] **Step 6: Verify RAG exactness and commit**

Run:

```bash
cd python-impl
pytest tests/unit/test_rag_spec_metadata.py tests/integration/test_retrieval_log.py tests/unit/test_markdown_windows.py tests/unit/test_query_policy.py tests/api/test_knowledge_reply.py -q
git add src/smart_cs tests/unit/test_rag_spec_metadata.py tests/integration/test_retrieval_log.py tests/unit/test_markdown_windows.py tests/unit/test_query_policy.py
git commit -m "feat: align rag metadata and logging to spec"
```

Expected: PASS; RAG uses `section_path`, persists `RetrievalLog`, and returns a safe unavailable response when Milvus or embeddings are unavailable.

### Task 4: Match Image Evidence And Handoff Rules Exactly

**Files:**
- Create: `python-impl/tests/integration/test_image_clarification_flow.py`
- Modify: `python-impl/src/smart_cs/application/agent_runtime.py`
- Modify: `python-impl/src/smart_cs/application/conversation_service.py`
- Modify: `python-impl/src/smart_cs/agents/guardrails.py`
- Modify: `python-impl/src/smart_cs/agents/specialists.py`
- Modify: `python-impl/src/smart_cs/domain/enums.py`
- Modify: `python-impl/src/smart_cs/domain/models.py`
- Modify: `python-impl/src/smart_cs/domain/repositories.py`
- Modify: `python-impl/src/smart_cs/infrastructure/repositories.py`
- Modify: `python-impl/tests/api/test_image_message.py`
- Modify: `python-impl/tests/unit/test_vision_agent.py`

- [ ] **Step 1: Write failing image and handoff rule tests**

```python
# python-impl/tests/integration/test_image_clarification_flow.py
def test_first_low_confidence_image_keeps_conversation_active_without_handoff(runtime, repo) -> None:
    first = runtime.invoke_with_image("conv-low-1", "C001", "O1001 鞋底好像坏了", "asset/blur.jpg")

    assert first["status"] == "active"
    assert first["pending_confirmation"] is None
    assert "请补拍" in first["reply"] or "请确认" in first["reply"]
    assert repo.list_tickets("C001") == []


def test_user_confirming_uncertain_recognition_can_continue_without_handoff(runtime, repo) -> None:
    runtime.invoke_with_image("conv-low-2", "C001", "O1001 鞋底好像坏了", "asset/blur.jpg")
    second = runtime.invoke("conv-low-2", "C001", "我确认图片里是鞋底开胶，请继续判断")

    assert second["status"] in {"active", "pending_confirmation"}
    assert repo.list_tickets("C001") == []


def test_conflicting_supplemented_evidence_can_create_handoff_draft(runtime, repo) -> None:
    runtime.invoke_with_image("conv-conflict", "C001", "O1001 鞋底开胶，申请售后", "asset/damage.jpg")
    conflict = runtime.invoke("conv-conflict", "C001", "补充说明：其实不是鞋底，是物流丢件导致没收到")

    assert conflict["status"] == "pending_handoff"
    assert conflict["pending_confirmation"]["action_type"] == "handoff"
    assert repo.list_tickets("C001") == []


def test_explicit_human_request_creates_handoff_draft(runtime, repo) -> None:
    result = runtime.invoke_with_image("conv-human", "C001", "我要人工客服处理", "asset/blur.jpg")

    assert result["status"] == "pending_handoff"
    assert result["pending_confirmation"]["action_type"] == "handoff"
    assert repo.list_tickets("C001") == []
```

- [ ] **Step 2: Run image rule tests to verify red**

Run:

```bash
cd python-impl
pytest tests/integration/test_image_clarification_flow.py -q
```

Expected: FAIL because current low-confidence or handoff behavior does not exactly match the spec.

- [ ] **Step 3: Keep only spec conversation states**

Update `python-impl/src/smart_cs/domain/enums.py`:

```python
class ConversationStatus(str, Enum):
    ACTIVE = "active"
    PENDING_CONFIRMATION = "pending_confirmation"
    PENDING_HANDOFF = "pending_handoff"
    CLOSED = "closed"
```

Do not add `needs_clarification` as a stored conversation state.

- [ ] **Step 4: Implement exact evidence routing**

Add this deterministic decision helper in `python-impl/src/smart_cs/agents/specialists.py`:

```python
def decide_image_evidence_path(
    *,
    explicit_handoff: bool,
    policy_or_business_conflict_after_supplement: bool,
    dependency_failure: bool,
    evidence_usable_for_after_sales: bool,
) -> str:
    if explicit_handoff or policy_or_business_conflict_after_supplement or dependency_failure:
        return "draft_handoff"
    if evidence_usable_for_after_sales:
        return "draft_after_sales"
    return "ask_for_clarification"
```

When the path is `ask_for_clarification`, keep `Conversation.status` as `active`, create no pending action, and return:

```python
{
    "status": "active",
    "pending_confirmation": None,
    "reply": "图片证据暂不能确认问题，请补拍清晰的商品问题部位，或确认识别内容后继续。",
    "visual_evidence": evidence.model_dump(),
}
```

- [ ] **Step 5: Keep handoff only for spec scenarios**

Update `python-impl/src/smart_cs/agents/guardrails.py`:

```python
IMAGE_CLARIFICATION_REPLY = "图片证据暂不能确认问题，请补拍清晰的商品问题部位，或确认识别内容后继续。"
PENDING_HANDOFF_REPLY = "已为您生成转人工交接草稿，请确认。"
```

Rules:

```text
low-confidence first-pass image: active conversation, IMAGE_CLARIFICATION_REPLY, no pending action.
explicit human request: pending_handoff, draft_handoff, PENDING_HANDOFF_REPLY.
uncovered or sensitive dispute: pending_handoff, draft_handoff, PENDING_HANDOFF_REPLY.
supplemented evidence still conflicts: pending_handoff, draft_handoff, PENDING_HANDOFF_REPLY.
tool or model failure that prevents safe continuation: pending_handoff, draft_handoff, PENDING_HANDOFF_REPLY.
```

- [ ] **Step 6: Verify image exactness and commit**

Run:

```bash
cd python-impl
pytest tests/integration/test_image_clarification_flow.py tests/api/test_image_message.py tests/unit/test_vision_agent.py -q
git add src/smart_cs tests/integration/test_image_clarification_flow.py tests/api/test_image_message.py tests/unit/test_vision_agent.py
git commit -m "feat: align image evidence handoff rules to spec"
```

Expected: PASS; low-confidence image evidence asks for recapture or user confirmation without handoff, and handoff is created only for the spec scenarios.

### Task 5: Remove Non-Spec Public Claims And Run Full Regression

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Modify: `docs/code-walkthrough.md`
- Modify: `docs/project-plan.md`
- Modify: `docs/interview/agent-project-qa.md`

- [ ] **Step 1: Scan for non-spec public claims**

Run:

```bash
rg -n -g '!docs/superpowers/plans/**' "messages-with-image|lookup_order|header_path|tool-calls|needs_clarification|低置信度转人工|兼容别名|alias|Compatibility" README.md docs python-impl/src
```

Expected: no matches in public documentation or API/tool declarations. A match inside removed-code history or comments must be deleted unless it is in Git history outside the working tree.

- [ ] **Step 2: Update architecture documents to exact spec wording**

`docs/architecture.md` must include:

```markdown
## API Surface

- `POST /api/conversations`
- `POST /api/conversations/{id}/messages`
- `GET /api/conversations/{id}/messages`
- `GET /api/conversations/{id}/runs`
- `POST /api/conversations/{id}/actions/confirm`
- `GET /health`

## Tool Set

- Read tools: `search_products`, `get_product`, `get_order`, `get_shipment`, `retrieve_knowledge`, `get_ticket`
- Draft tools: `draft_after_sales`, `draft_handoff`
- Confirmed side-effect tools: `submit_after_sales`, `confirm_handoff`

## Evidence Boundary

Low-confidence or unclear images ask for recapture or user confirmation. Handoff is only for explicit human requests, uncovered or sensitive disputes, conflicts after evidence supplementation, or dependency failure that prevents safe continuation.
```

- [ ] **Step 3: Update walkthrough and interview Q&A**

`docs/code-walkthrough.md` must describe this order:

```markdown
1. `api/routers/conversations.py` receives text or optional image input through `/messages`.
2. `application/conversation_service.py` persists `Message` and optional `Asset`.
3. `agents/vision.py` extracts `conversation_evidence` only for the current message.
4. `agents/router.py` and `agents/supervisor.py` choose the required specialist sequence.
5. `tools/executor.py` enforces the exact spec tool set and records ToolCall audit fields.
6. `agents/knowledge.py` uses Milvus text retrieval, `section_path`, Answerability Gate, citations, and RetrievalLog.
7. `agents/guardrails.py` prevents unsupported claims and unconfirmed side effects.
```

Add to `docs/interview/agent-project-qa.md`:

```markdown
## 低置信图片为什么不立即转人工？

低置信图片只说明当前会话证据不足，不能单独支持售后决策。系统先要求补拍或确认识别内容；只有用户明确要求人工、规则未覆盖或敏感争议、补充后证据仍冲突，或工具/模型失败导致无法安全继续时，才生成需确认的转人工草稿。
```

- [ ] **Step 4: Run full regression**

Run:

```bash
docker compose up -d etcd minio standalone
cd python-impl
python scripts/index_knowledge.py
pytest -q
python scripts/evaluate_rag.py
```

Expected: all tests PASS, Milvus-backed knowledge indexing completes, and `data/evaluation/latest_results.md` is regenerated with only Faithfulness, Answer Relevancy, Context Recall, and Context Precision.

- [ ] **Step 5: Commit documentation and exact-alignment verification**

Run:

```bash
git add README.md docs python-impl/data/evaluation/latest_results.json python-impl/data/evaluation/latest_results.md
git commit -m "docs: document exact spec-aligned behavior"
git status --short
```

Expected: empty git status after the commit. If local caches or service files appear, add them to `.gitignore` only when they are local artifacts and not project deliverables.

## Acceptance Checklist

- [ ] Public API surface exactly matches the spec route list.
- [ ] Public tool set exactly matches the spec tool list.
- [ ] Conversation states exactly match `active | pending_confirmation | pending_handoff | closed`.
- [ ] Text messages and single optional after-sales image messages are both handled through `POST /api/conversations/{id}/messages`.
- [ ] `GET /api/conversations/{id}/messages` returns message history.
- [ ] `GET /api/conversations/{id}/runs` returns AgentRun records and related ToolCall audit data.
- [ ] ToolCall records include tool name, calling Agent, input summary, result status, duration, and error type.
- [ ] RAG metadata uses `section_path`, not `header_path`.
- [ ] RetrievalLog records are persisted for knowledge queries.
- [ ] Knowledge dependency failures produce a safe unavailable response without policy fabrication.
- [ ] Low-confidence image evidence asks for recapture or user confirmation; handoff is only for the spec scenarios.
- [ ] Documentation does not claim non-spec routes, aliases, states, RAG metadata, production capabilities, or unmeasured results.

## Self-Review

- Spec coverage: this supplement covers missing API, exact tool set, ToolCall audit, RAG `section_path`, RetrievalLog, dependency failure behavior, and image/handoff rules.
- Over-scope check: removed compatibility routes, tool aliases, `header_path`, `tool-calls`, and non-spec conversation states from the required plan.
- Placeholder scan: this document contains no placeholder markers, deferred implementation notes, or unspecified test instructions.
- Type consistency: public names are stable across tasks: `ToolInvocationContext`, `section_path`, `RetrievalLog`, `KnowledgeUnavailable`, `get_order`, and `get_shipment`.
