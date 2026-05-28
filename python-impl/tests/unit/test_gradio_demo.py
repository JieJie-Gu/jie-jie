from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def load_demo_module():
    script_path = Path(__file__).parents[2] / "scripts" / "gradio_demo.py"
    spec = importlib.util.spec_from_file_location("gradio_demo", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_extract_pending_action_returns_action_or_none() -> None:
    demo = load_demo_module()

    action = {"action_id": "A1", "action_type": "after_sales"}

    assert demo.extract_pending_action({"pending_action": action}) == action
    assert demo.extract_pending_action({"pending_action": None}) is None
    assert demo.extract_pending_action({"reply": "ok"}) is None


def test_append_chat_entry_marks_image_messages() -> None:
    demo = load_demo_module()

    history = demo.append_chat_entry(
        [],
        user_text="O1001 鞋底开胶",
        response={"reply": "已为您生成售后申请草稿，请确认后提交。"},
        image_path="damage.jpg",
    )

    assert history == [
        (
            "O1001 鞋底开胶\n\n[已上传图片：damage.jpg]",
            "已为您生成售后申请草稿，请确认后提交。",
        )
    ]


def test_format_pending_action_renders_compact_markdown() -> None:
    demo = load_demo_module()

    markdown = demo.format_pending_action(
        {
            "action_type": "after_sales",
            "action_id": "A1",
            "order_id": "O1001",
            "reason": "鞋底开胶",
            "status": "pending_confirmation",
        }
    )

    assert "**action_type:** after_sales" in markdown
    assert "**action_id:** A1" in markdown
    assert "**order_id:** O1001" in markdown
    assert "**status:** pending_confirmation" in markdown


def test_format_pending_action_handles_empty_action() -> None:
    demo = load_demo_module()

    assert demo.format_pending_action(None) == "当前没有待确认动作。"


def test_format_error_message_prefers_backend_detail() -> None:
    demo = load_demo_module()

    assert demo.format_error_message(403, {"detail": "Conversation is not available"}) == (
        "HTTP 403: Conversation is not available"
    )
    assert demo.format_error_message(500, {"error": "boom"}) == "HTTP 500: {'error': 'boom'}"


class FakeResponse:
    def __init__(self, status_code: int, payload, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def test_client_maps_request_exception_to_demo_api_error(monkeypatch) -> None:
    demo = load_demo_module()

    def fake_request(*_args, **_kwargs):
        raise demo.requests.RequestException("connection refused")

    monkeypatch.setattr(demo.requests, "request", fake_request)

    with pytest.raises(demo.DemoApiError) as error_info:
        demo.SmartCsApiClient("http://backend").health()

    assert error_info.value.status_code == 0
    assert error_info.value.payload == {"detail": "无法连接后端：connection refused"}
    assert str(error_info.value) == "无法连接后端：connection refused"


def test_client_raises_demo_api_error_for_json_error_payload(monkeypatch) -> None:
    demo = load_demo_module()

    def fake_request(*_args, **_kwargs):
        return FakeResponse(403, {"detail": "Conversation is not available"})

    monkeypatch.setattr(demo.requests, "request", fake_request)

    with pytest.raises(demo.DemoApiError) as error_info:
        demo.SmartCsApiClient("http://backend").health()

    assert error_info.value.status_code == 403
    assert error_info.value.payload == {"detail": "Conversation is not available"}
    assert str(error_info.value) == "HTTP 403: Conversation is not available"


def test_client_uses_text_detail_for_non_json_error_body(monkeypatch) -> None:
    demo = load_demo_module()

    def fake_request(*_args, **_kwargs):
        return FakeResponse(503, ValueError("not json"), text="backend unavailable")

    monkeypatch.setattr(demo.requests, "request", fake_request)

    with pytest.raises(demo.DemoApiError) as error_info:
        demo.SmartCsApiClient("http://backend").health()

    assert error_info.value.status_code == 503
    assert error_info.value.payload == {"detail": "backend unavailable"}
    assert str(error_info.value) == "HTTP 503: backend unavailable"


def test_client_wraps_non_dict_success_payload(monkeypatch) -> None:
    demo = load_demo_module()

    def fake_request(*_args, **_kwargs):
        return FakeResponse(200, ["ok"])

    monkeypatch.setattr(demo.requests, "request", fake_request)

    assert demo.SmartCsApiClient("http://backend").health() == {"value": ["ok"]}


def test_send_message_with_image_passes_file_tuple_and_closes_file(
    monkeypatch,
    tmp_path: Path,
) -> None:
    demo = load_demo_module()
    image_path = tmp_path / "damage.jpg"
    image_path.write_bytes(b"image-bytes")
    captured = {}

    def fake_request(method, url, **kwargs):
        filename, image_file, content_type = kwargs["files"]["image"]
        captured["method"] = method
        captured["url"] = url
        captured["data"] = kwargs["data"]
        captured["timeout"] = kwargs["timeout"]
        captured["filename"] = filename
        captured["content_type"] = content_type
        captured["file"] = image_file
        captured["file_closed_during_request"] = image_file.closed
        captured["file_bytes"] = image_file.read()
        return FakeResponse(200, {"reply": "ok"})

    monkeypatch.setattr(demo.requests, "request", fake_request)

    response = demo.SmartCsApiClient("http://backend").send_message_with_image(
        "conv-1",
        "C001",
        "O1001 鞋底开胶",
        str(image_path),
    )

    assert response == {"reply": "ok"}
    assert captured["method"] == "POST"
    assert captured["url"] == "http://backend/api/conversations/conv-1/messages-with-image"
    assert captured["data"] == {"customer_id": "C001", "content": "O1001 鞋底开胶"}
    assert captured["timeout"] == 30
    assert captured["filename"] == "damage.jpg"
    assert captured["content_type"] == "image/jpeg"
    assert captured["file_closed_during_request"] is False
    assert captured["file_bytes"] == b"image-bytes"
    assert captured["file"].closed is True


class FakeClient:
    def __init__(self) -> None:
        self.confirmed: list[tuple[str, str, str, bool]] = []

    def create_conversation(self, customer_id: str):
        return {"id": "conv-1", "customer_id": customer_id}

    def send_message(self, conversation_id: str, customer_id: str, content: str):
        return {
            "status": "pending_confirmation",
            "reply": "宸蹭负鎮ㄧ敓鎴愬敭鍚庣敵璇疯崏绋匡紝璇风‘璁ゅ悗鎻愪氦銆?",
            "pending_action": {
                "action_type": "after_sales",
                "action_id": "A1",
                "order_id": "O1001",
                "reason": content,
                "status": "pending_confirmation",
            },
        }

    def send_message_with_image(
        self,
        conversation_id: str,
        customer_id: str,
        content: str,
        image_path: str,
    ):
        return self.send_message(conversation_id, customer_id, content)

    def confirm_action(
        self,
        conversation_id: str,
        customer_id: str,
        action_id: str,
        *,
        approved: bool,
    ):
        self.confirmed.append((conversation_id, customer_id, action_id, approved))
        return {
            "status": "completed",
            "reply": "鍞悗鐢宠宸插彈鐞嗭紝宸ュ崟缂栧彿涓?T1銆?",
            "result": {"status": "submitted", "ticket_id": "T1"},
            "pending_action": None,
        }

    def list_runs(self, conversation_id: str, customer_id: str):
        return {"runs": [{"conversation_id": conversation_id, "agents": ["OrderAgent"]}], "tool_calls": []}

    def list_tool_calls(self, conversation_id: str, customer_id: str):
        return {"tool_calls": [{"tool_name": "draft_after_sales", "customer_id": customer_id}]}


def test_create_conversation_callback_returns_initial_state() -> None:
    demo = load_demo_module()

    result = demo.create_conversation_callback(
        "http://backend",
        "C001",
        client_factory=lambda _base_url: FakeClient(),
    )

    assert result.conversation_id == "conv-1"
    assert result.pending_action is None
    assert result.chat_history[-1][1] == "宸插垱寤轰細璇?conv-1銆?"


def test_send_callback_creates_conversation_when_missing_and_refreshes_audit() -> None:
    demo = load_demo_module()

    result = demo.send_message_callback(
        backend_url="http://backend",
        customer_id="C001",
        conversation_id="",
        message="O1001 闉嬪簳寮€鑳?",
        image_path=None,
        chat_history=[],
        client_factory=lambda _base_url: FakeClient(),
    )

    assert result.conversation_id == "conv-1"
    assert result.pending_action["action_id"] == "A1"
    assert result.chat_history[-1][1] == FakeClient().send_message("conv-1", "C001", "")["reply"]
    assert "OrderAgent" in result.runs_json
    assert "draft_after_sales" in result.tool_calls_json


def test_confirm_callback_requires_pending_action() -> None:
    demo = load_demo_module()

    result = demo.confirm_action_callback(
        backend_url="http://backend",
        customer_id="C001",
        conversation_id="conv-1",
        pending_action=None,
        approved=True,
        chat_history=[],
        client_factory=lambda _base_url: FakeClient(),
    )

    assert result.raw_json == "娌℃湁寰呯‘璁ゅ姩浣溿€?"
    assert result.chat_history[-1][1] == "娌℃湁寰呯‘璁ゅ姩浣溿€?"


def test_confirm_callback_submits_pending_action_and_clears_panel() -> None:
    demo = load_demo_module()

    result = demo.confirm_action_callback(
        backend_url="http://backend",
        customer_id="C001",
        conversation_id="conv-1",
        pending_action={"action_id": "A1"},
        approved=True,
        chat_history=[],
        client_factory=lambda _base_url: FakeClient(),
    )

    assert result.pending_action is None
    assert "鍞悗鐢宠宸插彈鐞" in result.chat_history[-1][1]
    assert "ticket_id" in result.raw_json
