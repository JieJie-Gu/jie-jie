from __future__ import annotations

from typing import Any
from uuid import uuid4

from smart_cs.application.agent_runtime import AgentRuntime
from smart_cs.domain.errors import ToolPermissionError
from smart_cs.domain.models import ToolCall
from smart_cs.infrastructure.repositories import SqlRepository


class ConversationService:
    """Application boundary for HTTP conversation operations."""

    def __init__(self, *, repository: SqlRepository, runtime: AgentRuntime) -> None:
        self.repository = repository
        self.runtime = runtime

    def create_conversation(self, customer_id: str) -> dict[str, str]:
        self._require_demo_customer(customer_id)
        conversation_id = str(uuid4())
        conversation = self.repository.claim_conversation(conversation_id, customer_id)
        return {"id": conversation.id, "customer_id": conversation.customer_id}

    def send_message(
        self, conversation_id: str, customer_id: str, message: str
    ) -> dict[str, Any]:
        self.repository.require_conversation_owner(conversation_id, customer_id)
        result = self.runtime.invoke(conversation_id, customer_id, message)
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
        return self._http_result(result)

    def list_tool_calls(self, conversation_id: str, customer_id: str) -> list[dict[str, Any]]:
        self.repository.require_conversation_owner(conversation_id, customer_id)
        return [self._tool_call_result(call) for call in self.repository.list_tool_calls(customer_id)]

    def _require_demo_customer(self, customer_id: str) -> None:
        if not self.repository.customer_exists(customer_id):
            raise ToolPermissionError("Customer is not available")

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
