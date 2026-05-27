from __future__ import annotations

import base64
from typing import Any
from uuid import uuid4

from smart_cs.application.agent_runtime import AgentRuntime
from smart_cs.agents.vision import VisionAgent
from smart_cs.infrastructure.assets import LocalAssetStorage
from smart_cs.domain.errors import ToolPermissionError
from smart_cs.domain.models import ToolCall
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
        workflow_message = message
        if not evidence.usable_for_draft:
            workflow_message = f"转人工：图片证据暂不能确认问题。用户描述：{message}"
        result = self._http_result(
            self.runtime.invoke(conversation_id, customer_id, workflow_message)
        )
        if not evidence.usable_for_draft and result["status"] == "pending_confirmation":
            result["reply"] = "图片证据暂不能确认问题，已为您生成转人工申请草稿，请确认。"
        result["visual_evidence"] = evidence.model_dump()
        result["asset_key"] = asset_key
        return result

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
