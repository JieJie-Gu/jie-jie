from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests


DEFAULT_BACKEND_URL = "http://localhost:8000"
DEFAULT_CUSTOMER_ID = "C001"


JsonDict = dict[str, Any]
ChatHistory = list[tuple[str, str]]


class DemoApiError(RuntimeError):
    def __init__(self, status_code: int, payload: Any) -> None:
        self.status_code = status_code
        self.payload = payload
        super().__init__(format_error_message(status_code, payload))


class SmartCsApiClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def health(self) -> JsonDict:
        return self._request("GET", "/health")

    def create_conversation(self, customer_id: str) -> JsonDict:
        return self._request(
            "POST",
            "/api/conversations",
            json_payload={"customer_id": customer_id},
        )

    def send_message(self, conversation_id: str, customer_id: str, content: str) -> JsonDict:
        return self._request(
            "POST",
            f"/api/conversations/{conversation_id}/messages",
            json_payload={"customer_id": customer_id, "content": content},
        )

    def send_message_with_image(
        self,
        conversation_id: str,
        customer_id: str,
        content: str,
        image_path: str,
    ) -> JsonDict:
        with Path(image_path).open("rb") as image_file:
            return self._request(
                "POST",
                f"/api/conversations/{conversation_id}/messages-with-image",
                data={"customer_id": customer_id, "content": content},
                files={"image": (Path(image_path).name, image_file, _content_type(image_path))},
            )

    def confirm_action(
        self,
        conversation_id: str,
        customer_id: str,
        action_id: str,
        *,
        approved: bool,
    ) -> JsonDict:
        return self._request(
            "POST",
            f"/api/conversations/{conversation_id}/actions/confirm",
            json_payload={
                "customer_id": customer_id,
                "action_id": action_id,
                "approved": approved,
            },
        )

    def list_runs(self, conversation_id: str, customer_id: str) -> JsonDict:
        return self._request(
            "GET",
            f"/api/conversations/{conversation_id}/runs",
            params={"customer_id": customer_id},
        )

    def list_tool_calls(self, conversation_id: str, customer_id: str) -> JsonDict:
        return self._request(
            "GET",
            f"/api/conversations/{conversation_id}/tool-calls",
            params={"customer_id": customer_id},
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_payload: JsonDict | None = None,
        params: JsonDict | None = None,
        data: JsonDict | None = None,
        files: JsonDict | None = None,
    ) -> JsonDict:
        try:
            response = requests.request(
                method,
                f"{self.base_url}{path}",
                json=json_payload,
                params=params,
                data=data,
                files=files,
                timeout=30,
            )
        except requests.RequestException as error:
            raise DemoApiError(0, {"detail": f"无法连接后端：{error}"}) from error

        try:
            payload: Any = response.json()
        except ValueError:
            payload = {"detail": response.text}

        if response.status_code >= 400:
            raise DemoApiError(response.status_code, payload)
        if not isinstance(payload, dict):
            return {"value": payload}
        return payload


def extract_pending_action(response: JsonDict | None) -> JsonDict | None:
    if not response:
        return None
    action = response.get("pending_action")
    return action if isinstance(action, dict) else None


def append_chat_entry(
    history: ChatHistory,
    *,
    user_text: str,
    response: JsonDict,
    image_path: str | None = None,
) -> ChatHistory:
    display_text = user_text
    if image_path:
        display_text = f"{display_text}\n\n[已上传图片：{Path(image_path).name}]"
    reply = str(response.get("reply") or response.get("detail") or "后端没有返回 reply。")
    return [*history, (display_text, reply)]


def format_pending_action(action: JsonDict | None) -> str:
    if not action:
        return "当前没有待确认动作。"
    fields = ("action_type", "action_id", "order_id", "reason", "status")
    lines = ["### Pending Action"]
    for field in fields:
        if field in action and action[field] is not None:
            lines.append(f"- **{field}:** {action[field]}")
    return "\n".join(lines)


def format_error_message(status_code: int, payload: Any) -> str:
    if isinstance(payload, dict) and "detail" in payload:
        detail = payload["detail"]
    else:
        detail = payload
    if status_code == 0:
        return str(detail)
    return f"HTTP {status_code}: {detail}"


def to_pretty_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _content_type(image_path: str) -> str:
    suffix = Path(image_path).suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    return "application/octet-stream"
