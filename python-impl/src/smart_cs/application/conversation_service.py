from __future__ import annotations

import base64
from typing import Any
from uuid import uuid4

from smart_cs.agents.vision import VisionAgent
from smart_cs.application.agent_runtime import AgentRuntime
from smart_cs.domain.errors import ToolPermissionError
from smart_cs.domain.models import AgentRun, ToolCall
from smart_cs.infrastructure.assets import LocalAssetStorage
from smart_cs.infrastructure.repositories import SqlRepository


class ConversationService:
    """Application boundary for HTTP conversation operations."""

    def __init__(
        self,
        *,
        repository: SqlRepository,
        runtime: AgentRuntime,
        vision_agent: VisionAgent | None = None,
        asset_storage: LocalAssetStorage | None = None,
    ) -> None:
        self.repository = repository
        self.runtime = runtime
        self.vision_agent = vision_agent
        self.asset_storage = asset_storage

    def create_conversation(self, customer_id: str) -> dict[str, str]:
        self._require_demo_customer(customer_id)
        conversation_id = str(uuid4())
        conversation = self.repository.claim_conversation(conversation_id, customer_id)
        return {"id": conversation.id, "customer_id": conversation.customer_id}

    def send_message(
        self, conversation_id: str, customer_id: str, message: str
    ) -> dict[str, Any]:
        self.repository.require_conversation_owner(conversation_id, customer_id)
        self.repository.record_message(conversation_id, customer_id, "user", message)
        result = self.runtime.invoke(conversation_id, customer_id, message)
        self._record_agent_run(conversation_id, customer_id, result)
        return self._http_result(result)

    def confirm(
        self,
        conversation_id: str,
        customer_id: str,
        action_id: str,
        *,
        approved: bool,
    ) -> dict[str, Any]:
        self.repository.require_conversation_owner(conversation_id, customer_id)
        result = self.runtime.confirm(
            conversation_id, customer_id, action_id, approved=approved
        )
        self._record_agent_run(conversation_id, customer_id, result, action_id=action_id)
        return self._http_result(result)

    def send_message_with_image(
        self,
        conversation_id: str,
        customer_id: str,
        message: str,
        filename: str,
        content_type: str,
        content: bytes,
    ) -> dict[str, Any]:
        self.repository.require_conversation_owner(conversation_id, customer_id)
        if self.vision_agent is None or self.asset_storage is None:
            raise ValueError("Image evidence processing is not configured")
        asset_key = self.asset_storage.save(conversation_id, filename, content_type, content)
        encoded_image = base64.b64encode(content).decode("ascii")
        evidence = self.vision_agent.inspect(
            f"data:{content_type};base64,{encoded_image}", message
        )
        self.repository.record_message(
            conversation_id,
            customer_id,
            "user",
            message,
            content_type="image",
            asset_key=asset_key,
            visual_evidence=evidence.model_dump(),
        )
        runtime_result = self.runtime.invoke(
            conversation_id,
            customer_id,
            message,
            visual_evidence=evidence.model_dump(),
            asset_key=asset_key,
        )
        self._record_agent_run(conversation_id, customer_id, runtime_result)
        result = self._http_result(runtime_result)
        result["visual_evidence"] = evidence.model_dump()
        result["asset_key"] = asset_key
        return result

    def list_tool_calls(self, conversation_id: str, customer_id: str) -> list[dict[str, Any]]:
        self.repository.require_conversation_owner(conversation_id, customer_id)
        return [
            self._tool_call_result(call)
            for call in self._conversation_tool_calls(conversation_id, customer_id)
        ]

    def list_agent_runs(self, conversation_id: str, customer_id: str) -> dict[str, Any]:
        self.repository.require_conversation_owner(conversation_id, customer_id)
        tool_calls = self._conversation_tool_calls(conversation_id, customer_id)
        return {
            "runs": [
                self._agent_run_result(run)
                for run in self.repository.list_agent_runs(conversation_id, customer_id)
            ],
            "tool_calls": [self._tool_call_result(call) for call in tool_calls],
        }

    def _conversation_tool_calls(
        self, conversation_id: str, customer_id: str
    ) -> list[ToolCall]:
        runs = self.repository.list_agent_runs(conversation_id, customer_id)
        action_ids = {
            run.pending_action_id for run in runs if run.pending_action_id is not None
        }
        return [
            call
            for call in self.repository.list_tool_calls(customer_id)
            if call.arguments.get("conversation_id") == conversation_id
            or call.arguments.get("action_id") in action_ids
            or (call.result or {}).get("action_id") in action_ids
        ]

    def _require_demo_customer(self, customer_id: str) -> None:
        if not self.repository.customer_exists(customer_id):
            raise ToolPermissionError("Customer is not available")

    def _record_agent_run(
        self,
        conversation_id: str,
        customer_id: str,
        result: dict[str, Any],
        *,
        agents: list[str] | None = None,
        action_id: str | None = None,
    ) -> None:
        pending_action = result.get("pending_confirmation")
        result_payload = result.get("result")
        pending_action_id = action_id
        if pending_action_id is None and isinstance(pending_action, dict):
            pending_action_id = pending_action.get("action_id")
        if pending_action_id is None and isinstance(result_payload, dict):
            pending_action_id = result_payload.get("action_id")

        run_agents = agents if agents is not None else list(result.get("agents_invoked", []))
        status = str(result.get("status", ""))
        if action_id is not None:
            updated = self.repository.update_agent_run_for_action(
                conversation_id=conversation_id,
                customer_id=customer_id,
                pending_action_id=action_id,
                status=status,
                reply=result.get("reply"),
                agents=run_agents or None,
            )
            if updated is not None:
                return

        self.repository.record_agent_run(
            conversation_id=conversation_id,
            customer_id=customer_id,
            agents=run_agents,
            status=status,
            pending_action_id=pending_action_id,
            reply=result.get("reply"),
        )

    @staticmethod
    def _http_result(result: dict[str, Any]) -> dict[str, Any]:
        if result.get("status") == "pending_confirmation":
            pending_action = result.get("pending_confirmation")
            if not isinstance(pending_action, dict):
                raise ValueError("Runtime did not return a pending action")
            return {
                "status": "pending_confirmation",
                "reply": str(result.get("reply", "")),
                "pending_action": pending_action,
                "agents_invoked": list(result.get("agents_invoked", [])),
            }
        return {
            "status": str(result["status"]),
            "reply": str(result["reply"]),
            "result": result.get("result"),
            "agents_invoked": list(result.get("agents_invoked", [])),
        }

    @staticmethod
    def _tool_call_result(call: ToolCall) -> dict[str, Any]:
        return {
            "id": call.id,
            "tool_name": call.tool_name,
            "customer_id": call.customer_id,
            "arguments": call.arguments,
            "result": call.result,
            "status": call.status,
            "error_type": call.error_type,
            "duration_ms": call.duration_ms,
            "created_at": call.created_at,
        }

    @staticmethod
    def _agent_run_result(run: AgentRun) -> dict[str, Any]:
        return {
            "id": run.id,
            "conversation_id": run.conversation_id,
            "agents": run.agents,
            "status": run.status,
            "pending_action_id": run.pending_action_id,
            "reply": run.reply,
            "created_at": run.created_at,
        }
