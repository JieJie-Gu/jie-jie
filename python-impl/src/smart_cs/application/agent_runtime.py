from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import logging
import sqlite3
from pathlib import Path
from threading import Condition, Event, Lock, Thread, local
from typing import Any
from uuid import uuid4

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from smart_cs.agents.knowledge import KnowledgeAgent
from smart_cs.agents.state import RuntimeState
from smart_cs.agents.subagents import create_post_sales_agent, create_pre_sales_agent
from smart_cs.application.memory import MemoryWriteback, SqlMemoryStoreAdapter
from smart_cs.application.policy import PolicyEngine
from smart_cs.domain.enums import ActionStatus
from smart_cs.domain.errors import ConversationLeaseLostError
from smart_cs.infrastructure.prompts import CUSTOMER_SERVICE_SUPERVISOR_PROMPT
from smart_cs.tools.agent_tool_wrappers import (
    RuntimeToolContext,
    build_post_sales_tools,
    build_pre_sales_tools,
    draft_after_sales_action,
    draft_handoff_action,
)
from smart_cs.tools.executor import AuthorizedToolExecutor, TurnFence
from smart_cs.tools.subagent_tools import make_post_sales_tool, make_pre_sales_tool


LOGGER = logging.getLogger(__name__)


class _TurnLeaseHeartbeat:
    def __init__(
        self,
        *,
        executor: AuthorizedToolExecutor,
        conversation_id: str,
        customer_id: str,
        token: str,
        ttl_seconds: float,
        renew_interval_seconds: float,
    ) -> None:
        self._executor = executor
        self._conversation_id = conversation_id
        self._customer_id = customer_id
        self._token = token
        self._ttl_seconds = ttl_seconds
        self._renew_interval_seconds = renew_interval_seconds
        self._stopped = Event()
        self._failure_lock = Lock()
        self._failure: Exception | None = None
        self._thread = Thread(
            target=self._run,
            name=f"smart-cs-turn-lease-heartbeat-{token}",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def turn_fence(self) -> TurnFence:
        return TurnFence(conversation_id=self._conversation_id, lease_token=self._token)

    def stop(self) -> None:
        self._stopped.set()
        self._thread.join()

    def renew_and_check(self) -> None:
        self.check()
        try:
            self._executor.renew_turn_lease(
                self._conversation_id,
                self._customer_id,
                self._token,
                ttl_seconds=self._ttl_seconds,
            )
        except Exception as error:
            self._record_failure(error)
            self.check()

    def check(self) -> None:
        with self._failure_lock:
            failure = self._failure
        if failure is None:
            return
        if isinstance(failure, ConversationLeaseLostError):
            raise failure
        raise ConversationLeaseLostError("Conversation turn lease heartbeat failed") from failure

    def _run(self) -> None:
        while not self._stopped.wait(self._renew_interval_seconds):
            try:
                self._executor.renew_turn_lease(
                    self._conversation_id,
                    self._customer_id,
                    self._token,
                    ttl_seconds=self._ttl_seconds,
                )
            except Exception as error:
                self._record_failure(error)
                return

    def _record_failure(self, error: Exception) -> None:
        with self._failure_lock:
            if self._failure is None:
                self._failure = error


class AgentRuntime:
    """Run the LangChain supervisor with sub-agents exposed as tools."""

    TURN_LEASE_TTL_SECONDS = 300.0

    def __init__(
        self,
        *,
        executor: AuthorizedToolExecutor,
        chat_model: Any,
        checkpoint_path: str | Path,
        knowledge_agent: KnowledgeAgent | None = None,
        policy_engine: PolicyEngine | None = None,
        memory_writeback: MemoryWriteback | None = None,
        graph_store: Any | None = None,
        memory_store: Any | None = None,
        turn_lease_ttl_seconds: float = TURN_LEASE_TTL_SECONDS,
        turn_lease_renew_interval_seconds: float | None = None,
    ) -> None:
        renew_interval_seconds = (
            turn_lease_renew_interval_seconds
            if turn_lease_renew_interval_seconds is not None
            else turn_lease_ttl_seconds / 3
        )
        if turn_lease_ttl_seconds <= 0:
            raise ValueError("Turn lease TTL must be positive")
        if renew_interval_seconds <= 0 or renew_interval_seconds >= turn_lease_ttl_seconds:
            raise ValueError("Turn lease renew interval must be positive and shorter than TTL")

        self.executor = executor
        self.chat_model = chat_model
        self.knowledge_agent = knowledge_agent
        self.policy_engine = policy_engine or PolicyEngine()
        self.memory_writeback = memory_writeback
        self.store = memory_store or SqlMemoryStoreAdapter(executor.repository)
        self._graph_store = graph_store or InMemoryStore()
        self._turn_lease_ttl_seconds = turn_lease_ttl_seconds
        self._turn_lease_renew_interval_seconds = renew_interval_seconds
        self._active_turn = local()
        self._lifecycle = Condition()
        self._active_turn_count = 0
        self._closing = False
        self._closed = False

        checkpoint_file = Path(checkpoint_path)
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        self._checkpoint_connection = sqlite3.connect(str(checkpoint_file), check_same_thread=False)
        self._checkpointer = SqliteSaver(self._checkpoint_connection)
        self.graph = self._build_supervisor()

    def invoke(
        self,
        conversation_id: str,
        customer_id: str,
        message: str,
        *,
        visual_evidence: dict[str, Any] | None = None,
        asset_key: str | None = None,
    ) -> dict[str, Any]:
        self.executor.claim_conversation(conversation_id, customer_id)
        with self._turn_lease(conversation_id, customer_id):
            pending_action = self.executor.pending_action_for_conversation(
                conversation_id, customer_id
            )
            if pending_action is not None:
                return self._pending_result(pending_action)

            previous_action = self.executor.latest_action_for_conversation(
                conversation_id, customer_id
            )
            previous_action_id = previous_action.get("action_id") if previous_action else None
            request_id = f"{conversation_id}:{uuid4()}"
            human_message = HumanMessage(id=f"{request_id}:human", content=message)
            ctx = RuntimeToolContext(
                conversation_id=conversation_id,
                customer_id=customer_id,
                request_id=request_id,
                turn_fence=self._current_turn_fence(),
                visual_evidence=visual_evidence,
                asset_key=asset_key,
            )
            with self._tool_context(ctx):
                graph_result = self.graph.invoke(
                    {
                        "messages": [human_message],
                        "conversation_id": conversation_id,
                        "customer_id": customer_id,
                        "request_id": request_id,
                        "message": message,
                        "has_image": visual_evidence is not None,
                        "visual_evidence": visual_evidence,
                        "asset_key": asset_key,
                    },
                    config=self._config(conversation_id),
                )

            interrupt_result = self._result_from_interrupt(
                graph_result,
                ctx,
                message=message,
                fallback_messages=[human_message],
            )
            if interrupt_result is not None:
                return interrupt_result

            result = self._completed_result_from_graph(
                graph_result,
                conversation_id=conversation_id,
                customer_id=customer_id,
                request_id=request_id,
                message=message,
                fallback_messages=[human_message],
                previous_action_id=previous_action_id,
            )
            self._write_memory(
                conversation_id=conversation_id,
                customer_id=customer_id,
                request_id=request_id,
                message=message,
                messages=graph_result.get("messages") or [human_message],
                business_result=result.get("result"),
            )
            return result

    def confirm(
        self,
        conversation_id: str,
        customer_id: str,
        action_id: str,
        *,
        approved: bool | None = None,
    ) -> dict[str, Any]:
        if type(approved) is not bool:
            raise ValueError("Confirmation requires boolean approval")
        self.executor.require_conversation_owner(conversation_id, customer_id)

        with self._turn_lease(conversation_id, customer_id):
            action = self.executor.action_for_conversation(conversation_id, customer_id, action_id)
            if action["status"] != ActionStatus.PENDING_CONFIRMATION.value:
                return self._completed_result(action, [])

            request_id = f"{conversation_id}:confirm:{uuid4()}"
            ctx = RuntimeToolContext(
                conversation_id=conversation_id,
                customer_id=customer_id,
                request_id=request_id,
                turn_fence=self._current_turn_fence(),
            )
            graph_result: dict[str, Any] | None = None
            if approved:
                graph_result = self._resume_with_decision(
                    conversation_id,
                    ctx,
                    {"type": "approve"},
                )
                result = self.executor.latest_action_for_conversation(conversation_id, customer_id)
                if result is None or result.get("action_id") != action_id:
                    result = self.executor.action_for_conversation(
                        conversation_id, customer_id, action_id
                    )
                if result["status"] == ActionStatus.PENDING_CONFIRMATION.value:
                    result = self.executor.submit_confirmed_action(
                        action_id,
                        customer_id,
                        caller_agent="ConfirmActionNode",
                        turn_fence=self._current_turn_fence(),
                    )
            else:
                try:
                    graph_result = self._resume_with_decision(
                        conversation_id,
                        ctx,
                        {"type": "reject", "message": "用户取消本次申请。"},
                    )
                finally:
                    result = self.executor.cancel_pending_action(
                        action_id,
                        customer_id,
                        caller_agent="ConfirmActionNode",
                        turn_fence=self._current_turn_fence(),
                    )

            messages = (graph_result or {}).get("messages") or []
            public = self._completed_result(result, messages)
            self._write_memory(
                conversation_id=conversation_id,
                customer_id=customer_id,
                request_id=request_id,
                message="",
                messages=messages,
                business_result=result,
            )
            return public

    def close(self) -> None:
        with self._lifecycle:
            self._closing = True
            while self._active_turn_count:
                self._lifecycle.wait()
            if self._closed:
                return
            self._checkpoint_connection.close()
            self._closed = True

    def _build_supervisor(self):
        pre_sales_agent = create_pre_sales_agent(
            self.chat_model,
            build_pre_sales_tools(
                self.executor,
                self.knowledge_agent,
                self._current_tool_context,
            ),
        )
        post_sales_agent = create_post_sales_agent(
            self.chat_model,
            build_post_sales_tools(
                self.executor,
                self.knowledge_agent,
                self._current_tool_context,
                self.policy_engine,
            ),
        )
        return create_agent(
            self.chat_model,
            tools=[
                make_pre_sales_tool(pre_sales_agent),
                make_post_sales_tool(post_sales_agent),
            ],
            system_prompt=CUSTOMER_SERVICE_SUPERVISOR_PROMPT,
            state_schema=RuntimeState,
            checkpointer=self._checkpointer,
            store=self._graph_store,
            name="customer_service_supervisor",
        )

    def _resume_with_decision(
        self,
        conversation_id: str,
        ctx: RuntimeToolContext,
        decision: dict[str, Any],
    ) -> dict[str, Any]:
        with self._tool_context(ctx):
            return self.graph.invoke(
                Command(resume={"decisions": [decision]}),
                config=self._config(conversation_id),
            )

    def _result_from_interrupt(
        self,
        graph_result: dict[str, Any],
        ctx: RuntimeToolContext,
        *,
        message: str,
        fallback_messages: list[BaseMessage],
    ) -> dict[str, Any] | None:
        interrupts = graph_result.get("__interrupt__")
        if not interrupts:
            return None
        payload = getattr(interrupts[0], "value", interrupts[0])
        action_request = self._first_action_request(payload)
        if action_request is None:
            return {
                "status": ActionStatus.PENDING_CONFIRMATION.value,
                "pending_confirmation": {},
                "reply": "该操作需要确认。",
                "agents_invoked": self._agents_from_messages(graph_result.get("messages") or []),
            }

        action = self._draft_action_for_interrupt(action_request, ctx)
        if action.get("status") != ActionStatus.PENDING_CONFIRMATION.value:
            return self._completed_result(action, graph_result.get("messages") or fallback_messages)

        self._write_memory(
            conversation_id=ctx.conversation_id,
            customer_id=ctx.customer_id,
            request_id=ctx.request_id,
            message=message,
            messages=graph_result.get("messages") or fallback_messages,
            business_result=action,
        )
        return self._pending_result(action, graph_result.get("messages") or fallback_messages)

    def _draft_action_for_interrupt(
        self,
        action_request: dict[str, Any],
        ctx: RuntimeToolContext,
    ) -> dict[str, Any]:
        existing = self.executor.pending_action_for_conversation(
            ctx.conversation_id, ctx.customer_id
        )
        if existing is not None:
            return existing

        name = action_request.get("name")
        args = dict(action_request.get("args") or {})
        if name == "request_after_sales":
            return draft_after_sales_action(
                self.executor,
                self.knowledge_agent,
                self.policy_engine,
                ctx,
                order_id=str(args.get("order_id", "")),
                reason=str(args.get("reason", "")),
            )
        if name == "request_handoff":
            return draft_handoff_action(
                self.executor,
                ctx,
                reason=str(args.get("reason", "用户请求转人工")),
            )
        return {
            "status": "policy_explained",
            "message": f"Unsupported confirmation tool: {name}",
        }

    @staticmethod
    def _first_action_request(payload: Any) -> dict[str, Any] | None:
        if isinstance(payload, dict):
            requests = payload.get("action_requests") or []
        else:
            requests = getattr(payload, "action_requests", [])
        if not requests:
            return None
        first = requests[0]
        if isinstance(first, dict):
            return first
        return {
            "name": getattr(first, "name", None),
            "args": getattr(first, "args", {}),
            "description": getattr(first, "description", None),
        }

    def _completed_result_from_graph(
        self,
        graph_result: dict[str, Any],
        *,
        conversation_id: str,
        customer_id: str,
        request_id: str,
        message: str,
        fallback_messages: list[BaseMessage],
        previous_action_id: str | None,
    ) -> dict[str, Any]:
        messages = graph_result.get("messages") or fallback_messages
        result = self.executor.latest_action_for_conversation(conversation_id, customer_id)
        if result is not None and result.get("action_id") == previous_action_id:
            result = None
        reply = self._last_reply(messages)
        if not reply and result is not None:
            reply = self._reply_for_action(result)
        if not reply:
            reply = "已处理。"
        return {
            "status": "completed",
            "reply": reply,
            "result": result,
            "agents_invoked": self._agents_from_messages(messages),
            "tools_invoked": self._tool_names_for_conversation(conversation_id, customer_id),
            "request_id": request_id,
            "message": message,
        }

    def _pending_result(
        self,
        action: dict[str, Any],
        messages: list[BaseMessage] | None = None,
    ) -> dict[str, Any]:
        return {
            "status": ActionStatus.PENDING_CONFIRMATION.value,
            "pending_confirmation": action,
            "reply": self._reply_for_action(action),
            "agents_invoked": self._agents_from_messages(messages or []),
        }

    def _completed_result(
        self,
        result: dict[str, Any],
        messages: list[BaseMessage],
    ) -> dict[str, Any]:
        reply = self._last_reply(messages) or self._reply_for_action(result)
        return {
            "status": "completed",
            "reply": reply,
            "result": result,
            "agents_invoked": self._agents_from_messages(messages),
        }

    def _write_memory(
        self,
        *,
        conversation_id: str,
        customer_id: str,
        request_id: str,
        message: str,
        messages: list[BaseMessage],
        business_result: dict[str, Any] | None,
    ) -> None:
        if self.memory_writeback is None:
            return
        self.memory_writeback.update(
            {
                "conversation_id": conversation_id,
                "customer_id": customer_id,
                "request_id": request_id,
                "message": message,
                "messages": messages,
                "business_result": business_result,
            },
            store=self.store,
        )

    @staticmethod
    def _config(conversation_id: str) -> dict[str, dict[str, str]]:
        return {"configurable": {"thread_id": conversation_id}}

    @staticmethod
    def _reply_for_action(action: dict[str, Any]) -> str:
        status = action.get("status")
        action_type = action.get("action_type")
        if status == ActionStatus.PENDING_CONFIRMATION.value:
            if action_type == "handoff":
                return "已生成转人工申请草稿，请确认是否提交。"
            return "已生成售后申请草稿，请确认是否提交。"
        if status == ActionStatus.SUBMITTED.value:
            ticket_id = action.get("ticket_id")
            if ticket_id:
                return f"已提交申请，工单号 {ticket_id}。"
            return "已提交申请。"
        if status == ActionStatus.CANCELLED.value:
            return "已取消本次申请。"
        return str(action.get("message") or "已处理。")

    @staticmethod
    def _last_reply(messages: list[BaseMessage]) -> str:
        for message in reversed(messages):
            if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
                text = AgentRuntime._message_text(message)
                if text:
                    return text
        for message in reversed(messages):
            text = AgentRuntime._message_text(message)
            if text:
                return text
        return ""

    @staticmethod
    def _message_text(message: BaseMessage | Any) -> str:
        content = getattr(message, "content", message)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if text:
                        parts.append(str(text))
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        return str(content) if content is not None else ""

    @staticmethod
    def _agents_from_messages(messages: list[BaseMessage]) -> list[str]:
        agents: list[str] = ["customer_service_supervisor"]
        for message in messages:
            for call in getattr(message, "tool_calls", []) or []:
                name = call.get("name")
                if name == "use_pre_sales_agent" and "pre_sales_agent" not in agents:
                    agents.append("pre_sales_agent")
                if name == "use_post_sales_agent" and "post_sales_agent" not in agents:
                    agents.append("post_sales_agent")
        return agents

    def _tool_names_for_conversation(self, conversation_id: str, customer_id: str) -> list[str]:
        names: list[str] = []
        for call in self.executor.repository.list_tool_calls(customer_id):
            if call.arguments.get("conversation_id") != conversation_id:
                continue
            if call.tool_name not in names:
                names.append(call.tool_name)
        return names

    @contextmanager
    def _tool_context(self, ctx: RuntimeToolContext) -> Iterator[None]:
        previous = getattr(self._active_turn, "tool_context", None)
        self._active_turn.tool_context = ctx
        try:
            yield
        finally:
            if previous is None:
                del self._active_turn.tool_context
            else:
                self._active_turn.tool_context = previous

    def _current_tool_context(self) -> RuntimeToolContext:
        ctx = getattr(self._active_turn, "tool_context", None)
        if ctx is None:
            raise RuntimeError("No active tool context")
        return ctx

    @contextmanager
    def _turn_lease(self, conversation_id: str, customer_id: str) -> Iterator[None]:
        self._begin_turn()
        token = str(uuid4())
        try:
            self.executor.acquire_turn_lease(
                conversation_id,
                customer_id,
                token,
                ttl_seconds=self._turn_lease_ttl_seconds,
            )
            heartbeat = _TurnLeaseHeartbeat(
                executor=self.executor,
                conversation_id=conversation_id,
                customer_id=customer_id,
                token=token,
                ttl_seconds=self._turn_lease_ttl_seconds,
                renew_interval_seconds=self._turn_lease_renew_interval_seconds,
            )
            previous_heartbeat = getattr(self._active_turn, "heartbeat", None)
            self._active_turn.heartbeat = heartbeat
            started = False
            try:
                heartbeat.start()
                started = True
                yield
            finally:
                if started:
                    heartbeat.stop()
                try:
                    if started:
                        heartbeat.renew_and_check()
                finally:
                    try:
                        self.executor.release_turn_lease(conversation_id, customer_id, token)
                    finally:
                        if previous_heartbeat is None:
                            del self._active_turn.heartbeat
                        else:
                            self._active_turn.heartbeat = previous_heartbeat
        finally:
            self._end_turn()

    def _begin_turn(self) -> None:
        with self._lifecycle:
            if self._closing or self._closed:
                raise RuntimeError("Agent runtime is closed")
            self._active_turn_count += 1

    def _end_turn(self) -> None:
        with self._lifecycle:
            self._active_turn_count -= 1
            if self._active_turn_count == 0:
                self._lifecycle.notify_all()

    def _current_turn_fence(self) -> TurnFence | None:
        heartbeat = getattr(self._active_turn, "heartbeat", None)
        if heartbeat is None:
            return None
        return heartbeat.turn_fence()
