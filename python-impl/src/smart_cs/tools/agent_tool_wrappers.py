# 将底层业务能力包装成可执行 LangChain 工具。

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from time import perf_counter
from typing import Any

from langchain.tools import tool

from smart_cs.agents.knowledge import KnowledgeService
from smart_cs.application.memory_selector import MemoryContextSelector, MemorySelectionInput
from smart_cs.application.policy import PolicyEngine
from smart_cs.domain.enums import ToolCallStatus
from smart_cs.domain.errors import ToolPermissionError
from smart_cs.tools.executor import AuthorizedToolExecutor, TurnFence


@dataclass(frozen=True)
class RuntimeToolContext:
    conversation_id: str
    customer_id: str
    request_id: str
    turn_fence: TurnFence | None
    visual_evidence: dict[str, Any] | None = None
    asset_key: str | None = None
    runtime_context: dict[str, Any] | None = None
    memory_store: Any | None = None
    memory_selector: MemoryContextSelector | None = None

    def idempotency_key(self, action_type: str, *parts: object) -> str:
        payload = "|".join([self.conversation_id, self.customer_id, action_type, *map(str, parts)])
        digest = sha256(payload.encode("utf-8")).hexdigest()[:24]
        return f"{self.conversation_id}:{action_type}:{digest}"


ContextProvider = Callable[[], RuntimeToolContext]


def build_pre_sales_tools(
    executor: AuthorizedToolExecutor,
    knowledge_service: KnowledgeService | None,
    context_provider: ContextProvider,
) -> list[Any]:
    return [
        make_search_products_tool(executor, context_provider, caller_agent="PreSalesAgent"),
        make_knowledge_rag_tool(
            executor,
            knowledge_service,
            context_provider,
            caller_agent="PreSalesAgent",
        ),
        make_recall_memory_tool(executor, context_provider, caller_agent="PreSalesAgent"),
    ]


def build_post_sales_tools(
    executor: AuthorizedToolExecutor,
    knowledge_service: KnowledgeService | None,
    context_provider: ContextProvider,
    policy_engine: PolicyEngine,
) -> list[Any]:
    return [
        make_lookup_order_tool(executor, context_provider),
        make_knowledge_rag_tool(
            executor,
            knowledge_service,
            context_provider,
            caller_agent="PostSalesAgent",
        ),
        make_request_after_sales_tool(
            executor,
            knowledge_service,
            context_provider,
            policy_engine,
        ),
        make_request_handoff_tool(executor, context_provider),
        make_recall_memory_tool(executor, context_provider, caller_agent="PostSalesAgent"),
    ]


def make_search_products_tool(
    executor: AuthorizedToolExecutor,
    context_provider: ContextProvider,
    *,
    caller_agent: str,
):
    @tool
    def search_products(query: str) -> dict[str, Any]:
        """Search customer-visible product facts by query."""

        ctx = context_provider()
        return executor.invoke(
            "search_products",
            {
                "query": query,
                "conversation_id": ctx.conversation_id,
                "customer_id": ctx.customer_id,
            },
            caller_agent=caller_agent,
        )

    return search_products


def make_lookup_order_tool(executor: AuthorizedToolExecutor, context_provider: ContextProvider):
    @tool
    def lookup_order(order_id: str) -> dict[str, Any]:
        """Look up an order that belongs to the current customer."""

        ctx = context_provider()
        return executor.invoke(
            "lookup_order",
            {
                "order_id": order_id,
                "conversation_id": ctx.conversation_id,
                "customer_id": ctx.customer_id,
            },
            caller_agent="PostSalesAgent",
        )

    return lookup_order


def make_knowledge_rag_tool(
    executor: AuthorizedToolExecutor,
    knowledge_service: KnowledgeService | None,
    context_provider: ContextProvider,
    *,
    caller_agent: str,
):
    @tool
    def knowledge_rag(query: str) -> dict[str, Any]:
        """Retrieve policy or product-rule knowledge with citations."""

        ctx = context_provider()
        return run_knowledge_rag(
            executor,
            knowledge_service,
            ctx,
            query=query,
            caller_agent=caller_agent,
        )

    return knowledge_rag


def make_recall_memory_tool(
    executor: AuthorizedToolExecutor,
    context_provider: ContextProvider,
    *,
    caller_agent: str,
):
    @tool
    def recall_memory(query: str, scope: str = "all") -> dict[str, Any]:
        """Recall short-term context and selected long-term customer memories."""

        ctx = context_provider()
        return run_recall_memory(
            executor,
            ctx,
            query=query,
            scope=scope,
            caller_agent=caller_agent,
        )

    return recall_memory


def make_request_after_sales_tool(
    executor: AuthorizedToolExecutor,
    knowledge_service: KnowledgeService | None,
    context_provider: ContextProvider,
    policy_engine: PolicyEngine,
):
    @tool
    def request_after_sales(order_id: str, reason: str) -> dict[str, Any]:
        """Submit an approved after-sales request for the current customer."""

        ctx = context_provider()
        draft = draft_after_sales_action(
            executor,
            knowledge_service,
            policy_engine,
            ctx,
            order_id=order_id,
            reason=reason,
        )
        if draft.get("status") != "pending_confirmation":
            return draft
        return executor.submit_confirmed_action(
            draft["action_id"],
            ctx.customer_id,
            caller_agent="ConfirmActionNode",
            turn_fence=ctx.turn_fence,
        )

    return request_after_sales


def make_request_handoff_tool(
    executor: AuthorizedToolExecutor,
    context_provider: ContextProvider,
):
    @tool
    def request_handoff(reason: str) -> dict[str, Any]:
        """Submit an approved human-handoff request for the current customer."""

        ctx = context_provider()
        draft = draft_handoff_action(executor, ctx, reason=reason)
        if draft.get("status") != "pending_confirmation":
            return draft
        return executor.submit_confirmed_action(
            draft["action_id"],
            ctx.customer_id,
            caller_agent="ConfirmActionNode",
            turn_fence=ctx.turn_fence,
        )

    return request_handoff


def draft_after_sales_action(
    executor: AuthorizedToolExecutor,
    knowledge_service: KnowledgeService | None,
    policy_engine: PolicyEngine,
    ctx: RuntimeToolContext,
    *,
    order_id: str,
    reason: str,
) -> dict[str, Any]:
    _authorize_wrapper_tool(executor, "request_after_sales", "PostSalesAgent")
    order_result = executor.invoke(
        "lookup_order",
        {
            "order_id": order_id,
            "conversation_id": ctx.conversation_id,
            "customer_id": ctx.customer_id,
        },
        caller_agent="PostSalesAgent",
    )
    knowledge_result = run_knowledge_rag(
        executor,
        knowledge_service,
        ctx,
        query=f"售后政策 {reason}",
        caller_agent="PostSalesAgent",
    )
    decision = policy_engine.evaluate_after_sales(
        order_result=order_result,
        knowledge_result=knowledge_result,
        visual_evidence=ctx.visual_evidence,
    )
    if decision.next_action == "explain":
        return {
            "status": "policy_explained",
            "message": decision.explanation,
            "reason_code": decision.reason_code,
        }
    if decision.next_action == "handoff":
        return draft_handoff_action(
            executor,
            ctx,
            reason=f"{decision.explanation} 用户诉求：{reason}",
        )
    return executor.invoke(
        "draft_after_sales",
        {
            "customer_id": ctx.customer_id,
            "conversation_id": ctx.conversation_id,
            "order_id": order_id,
            "reason": reason,
            "idempotency_key": ctx.idempotency_key("after_sales", order_id, reason),
        },
        caller_agent="PostSalesAgent",
        turn_fence=ctx.turn_fence,
    )


def draft_handoff_action(
    executor: AuthorizedToolExecutor,
    ctx: RuntimeToolContext,
    *,
    reason: str,
) -> dict[str, Any]:
    _authorize_wrapper_tool(executor, "request_handoff", "PostSalesAgent")
    return executor.invoke(
        "draft_handoff",
        {
            "customer_id": ctx.customer_id,
            "conversation_id": ctx.conversation_id,
            "reason": reason,
            "idempotency_key": ctx.idempotency_key("handoff", reason),
        },
        caller_agent="PostSalesAgent",
        turn_fence=ctx.turn_fence,
    )


def run_knowledge_rag(
    executor: AuthorizedToolExecutor,
    knowledge_service: KnowledgeService | None,
    ctx: RuntimeToolContext,
    *,
    query: str,
    caller_agent: str,
) -> dict[str, Any]:
    _authorize_wrapper_tool(executor, "knowledge_rag", caller_agent)
    arguments = {
        "query": query,
        "conversation_id": ctx.conversation_id,
        "customer_id": ctx.customer_id,
    }
    started = perf_counter()
    try:
        if knowledge_service is None:
            result = {
                "status": "knowledge_unavailable",
                "answer": "知识库未启用，无法提供政策依据。",
                "contexts": [],
                "citations": [],
            }
        else:
            result = knowledge_service.answer(query).as_result()
    except Exception as error:
        executor.repository.record_tool_call(
            tool_name="knowledge_rag",
            arguments=arguments,
            customer_id=ctx.customer_id,
            status=ToolCallStatus.REJECTED.value,
            error_type=type(error).__name__,
            duration_ms=_duration_ms(started),
        )
        raise
    executor.repository.record_tool_call(
        tool_name="knowledge_rag",
        arguments=arguments,
        customer_id=ctx.customer_id,
        status=ToolCallStatus.SUCCEEDED.value,
        result=result,
        duration_ms=_duration_ms(started),
    )
    return result


def run_recall_memory(
    executor: AuthorizedToolExecutor,
    ctx: RuntimeToolContext,
    *,
    query: str,
    scope: str,
    caller_agent: str,
) -> dict[str, Any]:
    _authorize_wrapper_tool(executor, "recall_memory", caller_agent)
    normalized_scope = scope if scope in {"short_term", "long_term", "all"} else "all"
    arguments = {
        "query": query,
        "scope": normalized_scope,
        "conversation_id": ctx.conversation_id,
        "customer_id": ctx.customer_id,
    }
    started = perf_counter()
    try:
        result: dict[str, Any] = {}
        if normalized_scope in {"short_term", "all"}:
            result["short_term"] = _short_term_memory(ctx)
        if normalized_scope in {"long_term", "all"}:
            result["long_term"] = _long_term_memory(ctx, query)
    except Exception as error:
        executor.repository.record_tool_call(
            tool_name="recall_memory",
            arguments=arguments,
            customer_id=ctx.customer_id,
            status=ToolCallStatus.REJECTED.value,
            error_type=type(error).__name__,
            duration_ms=_duration_ms(started),
        )
        raise
    executor.repository.record_tool_call(
        tool_name="recall_memory",
        arguments=arguments,
        customer_id=ctx.customer_id,
        status=ToolCallStatus.SUCCEEDED.value,
        result=result,
        duration_ms=_duration_ms(started),
    )
    return result


def _short_term_memory(ctx: RuntimeToolContext) -> dict[str, Any]:
    runtime_context = ctx.runtime_context or {}
    return {
        "session_facts": runtime_context.get("session_facts") or {},
        "recent_messages": runtime_context.get("recent_messages") or [],
        "conversation_summary": runtime_context.get("conversation_summary"),
        "pending_action": runtime_context.get("pending_confirmation"),
        "visual_evidence": runtime_context.get("visual_evidence") or ctx.visual_evidence,
    }


def _long_term_memory(ctx: RuntimeToolContext, query: str) -> dict[str, Any]:
    if ctx.memory_store is None:
        selected = []
    else:
        namespace = ("customer", ctx.customer_id, "memories")
        records = ctx.memory_store.search(namespace, query=query, limit=20)
        memories = [_memory_record_to_dict(record) for record in records]
        selector = ctx.memory_selector or MemoryContextSelector()
        selected = selector.select(
            MemorySelectionInput(
                query=query,
                intent=(ctx.runtime_context or {}).get("session_facts", {}).get("current_intent"),
                memories=memories,
                limit=5,
                max_chars=1200,
            )
        ).memories
    semantic = [memory.model_dump() for memory in selected if memory.memory_kind == "semantic"]
    episodic = [memory.model_dump() for memory in selected if memory.memory_kind == "episodic"]
    return {"semantic_memories": semantic, "episodic_memories": episodic}


def _memory_record_to_dict(record: Any) -> dict[str, Any]:
    if isinstance(record, dict):
        return dict(record)
    value = getattr(record, "value", None)
    if isinstance(value, dict):
        memory = dict(value)
    else:
        value_json = getattr(record, "value_json", None)
        memory = dict(value_json) if isinstance(value_json, dict) else {}
    memory.setdefault("memory_id", getattr(record, "key", None) or getattr(record, "id", None))
    memory.setdefault("key", getattr(record, "key", None))
    memory.setdefault("title", getattr(record, "title", None))
    memory.setdefault("description", getattr(record, "description", None))
    memory.setdefault("confidence", getattr(record, "confidence", None))
    memory.setdefault("risk_level", getattr(record, "risk_level", None))
    memory.setdefault("review_status", getattr(record, "review_status", None))
    expires_at = getattr(record, "expires_at", None)
    if expires_at is not None:
        memory.setdefault("expires_at", expires_at.isoformat())
    return {key: value for key, value in memory.items() if value is not None}


def _authorize_wrapper_tool(
    executor: AuthorizedToolExecutor,
    tool_name: str,
    caller_agent: str,
) -> None:
    policy = executor.tool_registry.get(tool_name)
    if caller_agent not in policy.allowed_agents:
        raise ToolPermissionError(f"Tool {tool_name} is not allowed for {caller_agent}")


def _duration_ms(started: float) -> int:
    return max(0, round((perf_counter() - started) * 1000))
