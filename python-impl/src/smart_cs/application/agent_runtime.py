from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import logging
import sqlite3
from pathlib import Path
from threading import Condition, Event, Lock, Thread, local
from typing import Any, Protocol
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from smart_cs.agents.guardrails import ResponseGuard
from smart_cs.agents.knowledge import KnowledgeAgent
from smart_cs.agents.router import RouterAgent, RoutingDecisionModel
from smart_cs.agents.specialists import SpecialistDispatcher
from smart_cs.agents.state import RouteAnalysis, RuntimeState, SupervisorDecision
from smart_cs.agents.supervisor import PlanningDecisionModel, SupervisorAgent
from smart_cs.domain.evidence import VisualEvidence
from smart_cs.domain.enums import ActionStatus
from smart_cs.domain.errors import ConversationLeaseLostError
from smart_cs.infrastructure.model_factory import RulesDecisionModel
from smart_cs.tools.executor import AuthorizedToolExecutor, TurnFence


LOGGER = logging.getLogger(__name__)


class DecisionModel(RoutingDecisionModel, PlanningDecisionModel, Protocol):
    """同时满足路由分析和主管规划接口的决策模型。"""

    pass


class _TurnLeaseHeartbeat:
    """后台续租线程，保证同一会话同一时间只有一个有效处理轮次。"""

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
        """启动后台心跳线程，按固定间隔刷新 turn lease。"""

        self._thread.start()

    def turn_fence(self) -> TurnFence:
        """把当前租约包装成写工具可校验的 fence。"""

        return TurnFence(conversation_id=self._conversation_id, lease_token=self._token)

    def stop(self) -> None:
        """停止心跳线程，并等待线程退出。"""

        self._stopped.set()
        self._thread.join()

    def renew_and_check(self) -> None:
        """主动续租一次；如果后台心跳曾失败，这里会抛出租约丢失错误。"""

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
        """检查心跳线程是否记录过失败，失败时统一转换成会话租约错误。"""

        with self._failure_lock:
            failure = self._failure
        if failure is None:
            return
        if isinstance(failure, ConversationLeaseLostError):
            raise failure
        raise ConversationLeaseLostError("Conversation turn lease heartbeat failed") from failure

    def _run(self) -> None:
        """后台循环续租；一旦续租失败就记录错误并退出。"""

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
    """运行客服多 Agent 编排，并支持持久化确认暂停与恢复。"""

    # 每个对话轮次默认持有 5 分钟租约，避免并发请求同时修改同一会话。
    TURN_LEASE_TTL_SECONDS = 300.0

    def __init__(
        self,
        *,
        executor: AuthorizedToolExecutor,
        decision_model: DecisionModel,
        checkpoint_path: str | Path,
        knowledge_agent: KnowledgeAgent | None = None,
        turn_lease_ttl_seconds: float = TURN_LEASE_TTL_SECONDS,
        turn_lease_renew_interval_seconds: float | None = None,
    ) -> None:
        # 续租间隔默认是 TTL 的三分之一，确保长流程执行时租约不会过期。
        renew_interval_seconds = (
            turn_lease_renew_interval_seconds
            if turn_lease_renew_interval_seconds is not None
            else turn_lease_ttl_seconds / 3
        )
        if turn_lease_ttl_seconds <= 0:
            raise ValueError("Turn lease TTL must be positive")
        if renew_interval_seconds <= 0 or renew_interval_seconds >= turn_lease_ttl_seconds:
            raise ValueError("Turn lease renew interval must be positive and shorter than TTL")
        if isinstance(decision_model, RulesDecisionModel):
            LOGGER.warning(
                "RulesDecisionModel enabled: development non-evaluation mode; "
                "do not use this run for evaluation claims."
            )
        # executor 是唯一能触达业务工具和数据库写操作的入口。
        self.executor = executor
        # router 只分析意图、实体和风险；不负责工具授权。
        self.router = RouterAgent(decision_model)
        # supervisor 负责把路由结果转成 Agent 执行计划，并负责最终回复合成。
        self.supervisor = SupervisorAgent(decision_model)
        # specialists 根据 supervisor 的计划调用具体 Product/Order/Knowledge 等能力。
        self.specialists = SpecialistDispatcher(executor, knowledge_agent)
        # guard 负责把业务结果渲染成可对外回复的受控内容。
        self.guard = ResponseGuard()
        self._turn_lease_ttl_seconds = turn_lease_ttl_seconds
        self._turn_lease_renew_interval_seconds = renew_interval_seconds
        # thread-local 保存当前线程正在处理的 turn lease heartbeat。
        self._active_turn = local()
        # lifecycle 用于 close() 等待正在执行的轮次全部结束。
        self._lifecycle = Condition()
        self._active_turn_count = 0
        self._closing = False
        self._closed = False

        # LangGraph checkpoint 存到 SQLite，确认中断后可以按 conversation_id 恢复。
        checkpoint_file = Path(checkpoint_path)
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        self._checkpoint_connection = sqlite3.connect(str(checkpoint_file), check_same_thread=False)
        self._checkpointer = SqliteSaver(self._checkpoint_connection)
        self.graph = self._build_graph()

    def invoke(
        self,
        conversation_id: str,
        customer_id: str,
        message: str,
        *,
        visual_evidence: dict[str, Any] | None = None,
        asset_key: str | None = None,
    ) -> dict[str, Any]:
        """处理一轮用户消息，返回对外可见的执行结果。"""

        # 先绑定会话归属，后续查询/确认都要求 customer_id 匹配。
        self.executor.claim_conversation(conversation_id, customer_id)
        # 整个处理过程都放在 turn lease 内，防止并发轮次互相覆盖状态。
        with self._turn_lease(conversation_id, customer_id):
            # 如果会话已有待确认动作，本轮不重新规划，直接返回待确认状态。
            pending_action = self.executor.pending_action_for_conversation(
                conversation_id, customer_id
            )
            if pending_action is not None:
                # 从 checkpoint 取 LangGraph 状态，用来恢复 agents_invoked/reply 等上下文。
                state = self.graph.get_state(self._config(conversation_id)).values
                checkpoint_action = state.get("pending_confirmation") or {}
                if checkpoint_action.get("action_id") != pending_action["action_id"]:
                    state = {}
                return self._pending_result(pending_action, state)

            # 没有待确认动作时，从初始 RuntimeState 开始执行整条 LangGraph 工作流。
            result = self.graph.invoke(
                {
                    "conversation_id": conversation_id,
                    "customer_id": customer_id,
                    "request_id": f"{conversation_id}:{uuid4()}",
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
                },
                config=self._config(conversation_id),
            )
            return self._public_result(result)

    def confirm(
        self,
        conversation_id: str,
        customer_id: str,
        action_id: str,
        *,
        approved: bool | None = None,
    ) -> dict[str, Any]:
        """处理用户对 pending action 的确认或拒绝。"""

        config = self._config(conversation_id)
        self.executor.require_conversation_owner(conversation_id, customer_id)
        if type(approved) is not bool:
            raise ValueError("Confirmation requires boolean approval")

        with self._turn_lease(conversation_id, customer_id):
            # 从数据库读取真实 action 状态，数据库状态优先于 checkpoint。
            action = self.executor.action_for_conversation(conversation_id, customer_id, action_id)
            state = self._state_for_action(action_id, self.graph.get_state(config).values)
            if action["status"] != ActionStatus.PENDING_CONFIRMATION.value:
                return self._completed_result(action, state)

            # 如果 checkpoint 中断点还在同一个 action 上，就通过 Command(resume=...) 恢复图。
            interrupted_action = state.get("pending_confirmation")
            if interrupted_action is None or interrupted_action.get("action_id") != action["action_id"]:
                # checkpoint 不匹配时，直接根据数据库里的 pending action 做状态转换。
                result = self._transition_action(action["action_id"], customer_id, approved)
                return self._completed_result(result, state)

            result = self.graph.invoke(Command(resume={"approved": approved}), config=config)
            return self._public_result(result)

    def close(self) -> None:
        """关闭 runtime；先等待正在执行的轮次结束，再关闭 checkpoint 连接。"""

        with self._lifecycle:
            self._closing = True
            while self._active_turn_count:
                self._lifecycle.wait()
            if self._closed:
                return
            self._checkpoint_connection.close()
            self._closed = True

    @contextmanager
    def _turn_lease(self, conversation_id: str, customer_id: str) -> Iterator[None]:
        """为一次 invoke/confirm 获取租约，并在结束时释放。"""

        self._begin_turn()
        token = str(uuid4())
        try:
            # 数据库层记录租约 token，写工具会用 token 校验当前轮次仍然有效。
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
                # 长流程执行期间由后台线程自动续租。
                heartbeat.start()
                started = True
                yield
            finally:
                if started:
                    heartbeat.stop()
                try:
                    if started:
                        # 结束前主动续租并检查心跳错误，避免静默丢失租约。
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
        """登记一个活跃轮次；runtime 关闭后不再允许新轮次进入。"""

        with self._lifecycle:
            if self._closing or self._closed:
                raise RuntimeError("Agent runtime is closed")
            self._active_turn_count += 1

    def _end_turn(self) -> None:
        """结束一个活跃轮次，并唤醒等待 close() 的线程。"""

        with self._lifecycle:
            self._active_turn_count -= 1
            if self._active_turn_count == 0:
                self._lifecycle.notify_all()

    def _assert_turn_lease(self) -> None:
        """在关键节点前后检查租约仍然有效。"""

        heartbeat = getattr(self._active_turn, "heartbeat", None)
        if heartbeat is not None:
            heartbeat.renew_and_check()

    def _current_turn_fence(self) -> TurnFence | None:
        """返回当前轮次的写操作 fence；没有活跃轮次时返回 None。"""

        heartbeat = getattr(self._active_turn, "heartbeat", None)
        if heartbeat is None:
            return None
        return heartbeat.turn_fence()

    def _build_graph(self):
        """构建 LangGraph 编排图。"""

        workflow = StateGraph(RuntimeState)
        # 每个 node 都是一个可 checkpoint 的步骤，返回值会合并进 RuntimeState。
        workflow.add_node("router", self._router_node)
        workflow.add_node("supervisor", self._supervisor_node)
        workflow.add_node("specialists", self._specialists_node)
        workflow.add_node("validate_evidence", self._validate_evidence_node)
        workflow.add_node("confirm_action", self._confirm_action_node)
        workflow.add_node("guard", self._guard_node)
        workflow.add_node("synthesize", self._synthesize_node)
        # 主路径：路由 -> 规划 -> specialist 执行 -> 证据校验 -> guard -> 合成回复。
        workflow.add_edge(START, "router")
        workflow.add_edge("router", "supervisor")
        workflow.add_edge("supervisor", "specialists")
        workflow.add_edge("specialists", "validate_evidence")
        workflow.add_edge("validate_evidence", "guard")
        workflow.add_edge("confirm_action", "guard")
        workflow.add_edge("guard", "synthesize")
        # 如果合成后仍有 pending_confirmation，就进入 confirm_action 中断节点。
        workflow.add_conditional_edges(
            "synthesize",
            self._next_after_synthesis,
            {"confirm_action": "confirm_action", "end": END},
        )
        return workflow.compile(checkpointer=self._checkpointer)

    def _router_node(self, state: RuntimeState) -> dict[str, Any]:
        """Router 节点：分析意图、实体和风险。"""

        self._assert_turn_lease()
        route = self.router.analyze(state["message"])
        self._assert_turn_lease()
        return {"route": route.model_dump()}

    def _supervisor_node(self, state: RuntimeState) -> dict[str, Any]:
        """Supervisor 节点：根据路由结果规划实际要调用的 Agent 和动作。"""

        self._assert_turn_lease()
        route = RouteAnalysis.model_validate(state["route"])
        decision = self.supervisor.plan(
            state["message"], route, has_image=bool(state.get("has_image"))
        )
        self._assert_turn_lease()
        return {"decision": decision.model_dump()}

    def _specialists_node(self, state: RuntimeState) -> dict[str, Any]:
        """Specialists 节点：按规划执行具体 specialist，并产出业务结果。"""

        self._assert_turn_lease()
        execution = self.specialists.execute(
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
        self._assert_turn_lease()
        return {
            "agents_invoked": execution.agents_invoked,
            "specialist_results": execution.results,
            "business_result": execution.result,
            "pending_confirmation": execution.pending_confirmation,
        }

    def _validate_evidence_node(self, state: RuntimeState) -> dict[str, Any]:
        """售后图片证据校验节点：图片不可用时把售后草稿转成人工接入草稿。"""

        self._assert_turn_lease()
        result = state.get("business_result") or {}
        # 只有“带图售后草稿待确认”需要额外校验证据可用性。
        if (
            not state.get("has_image")
            or result.get("status") != ActionStatus.PENDING_CONFIRMATION.value
            or result.get("action_type") != "after_sales"
        ):
            return {}

        evidence_payload = state.get("visual_evidence") or {}
        evidence = VisualEvidence.model_validate(evidence_payload)
        if evidence.usable_for_draft:
            return {}

        # 图片证据不可靠时，取消原售后 pending action，再创建 handoff pending action。
        self._assert_turn_lease()
        with self.executor.repository.transaction() as session:
            self.executor.repository.require_active_turn_lease(
                state["conversation_id"],
                state["customer_id"],
                self._current_turn_fence().lease_token,
                session=session,
            )
            self.executor.repository.cancel_pending_action(
                result["action_id"],
                state["customer_id"],
                session=session,
            )
            action = self.executor.repository.create_pending_action(
                customer_id=state["customer_id"],
                conversation_id=state["conversation_id"],
                idempotency_key=f"{state['request_id']}:handoff",
                action_type="handoff",
                reason=f"图片证据暂不能确认问题。用户描述：{state['message']}",
                session=session,
            )
            handoff = self.executor._action_result(action)
            self.executor.repository.record_tool_call(
                tool_name="convert_after_sales_to_handoff",
                arguments={
                    "conversation_id": state["conversation_id"],
                    "customer_id": state["customer_id"],
                    "action_id": result["action_id"],
                    "reason": handoff["reason"],
                },
                customer_id=state["customer_id"],
                status="succeeded",
                result=handoff,
                session=session,
            )
        handoff["evidence_status"] = "uncertain"
        # 用 handoff 结果替换最后一个 specialist 结果，让最终回复基于新的 pending action。
        results = list(state.get("specialist_results", []))
        if results:
            results[-1] = handoff
        else:
            results = [handoff]
        agents_invoked = list(state.get("agents_invoked", []))
        if "HandoffAgent" not in agents_invoked:
            agents_invoked.append("HandoffAgent")
        self._assert_turn_lease()
        return {
            "agents_invoked": agents_invoked,
            "specialist_results": results,
            "business_result": handoff,
            "pending_confirmation": handoff,
        }

    @staticmethod
    def _next_after_synthesis(state: RuntimeState) -> str:
        """决定合成回复后是结束，还是进入确认中断节点。"""

        result = state.get("business_result") or {}
        if (
            state.get("pending_confirmation") is not None
            and result.get("status") == ActionStatus.PENDING_CONFIRMATION.value
        ):
            return "confirm_action"
        return "end"

    def _confirm_action_node(self, state: RuntimeState) -> dict[str, Any]:
        """确认节点：通过 LangGraph interrupt 暂停，等待用户批准或拒绝。"""

        self._assert_turn_lease()
        action = state["pending_confirmation"]
        if action is None:
            raise ValueError("Missing pending action for confirmation")
        # interrupt 会把当前图状态写入 checkpoint，并把 pending_confirmation 返回给调用方。
        approval = interrupt(
            {
                "status": "pending_confirmation",
                "pending_confirmation": action,
                "reply": state.get("reply") or self.guard.render(action),
                "agents_invoked": list(state.get("agents_invoked", [])),
            }
        )
        self._assert_turn_lease()
        if not isinstance(approval, dict) or type(approval.get("approved")) is not bool:
            raise ValueError("Confirmation requires boolean approval")
        if approval["approved"]:
            # 用户批准后，才真正提交之前创建的 pending action。
            result = self.executor.submit_confirmed_action(
                action["action_id"],
                state["customer_id"],
                turn_fence=self._current_turn_fence(),
            )
            self._assert_turn_lease()
            return {"business_result": result}
        # 用户拒绝时取消 pending action。
        result = self.executor.cancel_pending_action(
            action["action_id"],
            state["customer_id"],
            turn_fence=self._current_turn_fence(),
        )
        self._assert_turn_lease()
        return {"business_result": result}

    def _guard_node(self, state: RuntimeState) -> dict[str, Any]:
        """Guard 节点：把业务结果渲染成受控回复片段。"""

        self._assert_turn_lease()
        results = self._response_results(state)
        guarded_contents = self.guard.render_results(results)
        self._assert_turn_lease()
        return {"guarded_contents": guarded_contents}

    def _synthesize_node(self, state: RuntimeState) -> dict[str, Any]:
        """回复合成节点：由 Supervisor 汇总 specialist 结果和 guard 内容。"""

        self._assert_turn_lease()
        results = self._response_results(state)
        reply = self.supervisor.synthesize(results, state["guarded_contents"])
        self._assert_turn_lease()
        if self._next_after_synthesis(state) == "confirm_action":
            return {"status": ActionStatus.PENDING_CONFIRMATION.value, "reply": reply}
        return {"status": "completed", "reply": reply, "pending_confirmation": None}

    @staticmethod
    def _config(conversation_id: str) -> dict[str, dict[str, str]]:
        """LangGraph 使用 thread_id 区分不同会话的 checkpoint。"""

        return {"configurable": {"thread_id": conversation_id}}

    def _transition_action(
        self, action_id: str, customer_id: str, approved: bool
    ) -> dict[str, Any]:
        """不通过图恢复时，直接根据批准结果转换 pending action 状态。"""

        if approved:
            return self.executor.submit_confirmed_action(
                action_id, customer_id, turn_fence=self._current_turn_fence()
            )
        return self.executor.cancel_pending_action(
            action_id, customer_id, turn_fence=self._current_turn_fence()
        )

    def _pending_result(
        self, action: dict[str, Any], state: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """把已有待确认动作包装成公开响应。"""

        reply = self._synthesize_reply(action, state or {})
        return {
            "status": ActionStatus.PENDING_CONFIRMATION.value,
            "pending_confirmation": action,
            "reply": reply,
            "agents_invoked": list((state or {}).get("agents_invoked", [])),
        }

    def _completed_result(
        self, result: dict[str, Any], state: dict[str, Any]
    ) -> dict[str, Any]:
        """把已完成动作包装成公开响应。"""

        return {
            "status": "completed",
            "reply": self._synthesize_reply(result, state),
            "result": result,
            "agents_invoked": state.get("agents_invoked", []),
        }

    def _synthesize_reply(self, result: dict[str, Any], state: dict[str, Any]) -> str:
        """在图外路径中复用 guard + supervisor 合成最终回复。"""

        response_results = self._response_results(state, result)
        guarded_contents = self.guard.render_results(response_results)
        return self.supervisor.synthesize(response_results, guarded_contents)

    @staticmethod
    def _state_for_action(action_id: str, state: dict[str, Any]) -> dict[str, Any]:
        """只有 checkpoint 中的 action_id 匹配时，才复用这份状态。"""

        state_action = state.get("pending_confirmation") or state.get("business_result") or {}
        if state_action.get("action_id") == action_id:
            return state
        return {}

    @staticmethod
    def _response_results(
        state: dict[str, Any], result: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """取得用于 guard/render 的结果列表，并用终态结果替换最后一个 specialist 结果。"""

        terminal_result = result if result is not None else state.get("business_result")
        if terminal_result is None:
            raise ValueError("Missing business result for response rendering")
        results = list(state.get("specialist_results", []))
        if results:
            results[-1] = terminal_result
            return results
        return [terminal_result]

    @staticmethod
    def _public_result(state: dict[str, Any]) -> dict[str, Any]:
        """把 LangGraph 内部状态或 interrupt 转成 API 可返回的公开结果。"""

        interrupts = state.get("__interrupt__")
        if interrupts:
            return dict(interrupts[0].value)
        return {
            "status": state["status"],
            "reply": state["reply"],
            "result": state.get("business_result"),
            "agents_invoked": state.get("agents_invoked", []),
        }
