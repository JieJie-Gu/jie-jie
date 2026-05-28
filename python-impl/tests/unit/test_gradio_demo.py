from __future__ import annotations

import importlib.util
from pathlib import Path


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
