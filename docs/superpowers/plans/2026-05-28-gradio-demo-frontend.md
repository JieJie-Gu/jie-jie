# Gradio Demo Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a small Gradio demo frontend that exercises the existing FastAPI customer-service workflow over HTTP.

**Architecture:** Add a standalone `python-impl/scripts/gradio_demo.py` script. It runs Gradio Blocks, calls the FastAPI backend with synchronous HTTP requests, stores `conversation_id` and `pending_action` in Gradio state, and refreshes AgentRun / ToolCall panels after every workflow operation.

**Tech Stack:** Python 3.11, Gradio, requests, FastAPI HTTP API, pytest

---

## Scope

This plan implements the approved core closed-loop demo:

- Create conversation for customer `C001`.
- Send text messages through `POST /api/conversations/{id}/messages`.
- Send text plus one image through `POST /api/conversations/{id}/messages-with-image`.
- Display assistant replies in a chat transcript.
- Show latest pending action.
- Confirm or reject the pending action through `POST /api/conversations/{id}/actions/confirm`.
- Show current `AgentRun`, `ToolCall`, and raw response JSON.

This plan does not modify backend API behavior.

## File Map

Create:

```text
python-impl/scripts/gradio_demo.py
python-impl/tests/unit/test_gradio_demo.py
python-impl/tests/unit/test_demo_dependencies.py
```

Modify:

```text
python-impl/pyproject.toml
README.md
```

Do not modify:

```text
python-impl/src/smart_cs/config.py
python-impl/src/smart_cs/main.py
python-impl/src/smart_cs/api/routers/conversations.py
```

## Task 1: Add Demo Dependencies

**Files:**
- Create: `python-impl/tests/unit/test_demo_dependencies.py`
- Modify: `python-impl/pyproject.toml`

- [ ] **Step 1: Write the failing dependency contract test**

Create `python-impl/tests/unit/test_demo_dependencies.py`:

```python
from __future__ import annotations

import tomllib
from pathlib import Path


def test_demo_optional_dependencies_include_gradio_and_requests() -> None:
    pyproject = Path(__file__).parents[2] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    optional = data["project"]["optional-dependencies"]
    demo_dependencies = optional["demo"]

    assert "gradio>=4,<6" in demo_dependencies
    assert "requests>=2.32,<3" in demo_dependencies
```

- [ ] **Step 2: Run the dependency test to verify red**

Run:

```powershell
cd python-impl
pytest tests/unit/test_demo_dependencies.py -q
```

Expected: FAIL with `KeyError: 'demo'`.

- [ ] **Step 3: Add the optional dependency group**

Modify `python-impl/pyproject.toml` so `[project.optional-dependencies]` contains both `test` and `demo`:

```toml
[project.optional-dependencies]
test = [
    "httpx>=0.28,<1",
    "pytest>=8.3,<9",
]
demo = [
    "gradio>=4,<6",
    "requests>=2.32,<3",
]
```

- [ ] **Step 4: Run the dependency test to verify green**

Run:

```powershell
cd python-impl
pytest tests/unit/test_demo_dependencies.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

Run:

```powershell
git add python-impl/pyproject.toml python-impl/tests/unit/test_demo_dependencies.py
git commit -m "feat: add gradio demo dependencies"
```

## Task 2: Build HTTP Client And Formatting Helpers

**Files:**
- Create: `python-impl/tests/unit/test_gradio_demo.py`
- Create: `python-impl/scripts/gradio_demo.py`

- [ ] **Step 1: Write failing helper tests**

Create `python-impl/tests/unit/test_gradio_demo.py`:

```python
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
```

- [ ] **Step 2: Run the helper tests to verify red**

Run:

```powershell
cd python-impl
pytest tests/unit/test_gradio_demo.py -q
```

Expected: FAIL with `FileNotFoundError` for `scripts/gradio_demo.py`.

- [ ] **Step 3: Create `gradio_demo.py` with helper logic and HTTP client**

Create `python-impl/scripts/gradio_demo.py`:

```python
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
```

- [ ] **Step 4: Run helper tests to verify green**

Run:

```powershell
cd python-impl
pytest tests/unit/test_gradio_demo.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

Run:

```powershell
git add python-impl/scripts/gradio_demo.py python-impl/tests/unit/test_gradio_demo.py
git commit -m "feat: add gradio demo client helpers"
```

## Task 3: Add Gradio Blocks UI

**Files:**
- Modify: `python-impl/scripts/gradio_demo.py`
- Modify: `python-impl/tests/unit/test_gradio_demo.py`

- [ ] **Step 1: Add failing tests for UI callback behavior**

Append to `python-impl/tests/unit/test_gradio_demo.py`:

```python
class FakeClient:
    def __init__(self) -> None:
        self.confirmed: list[tuple[str, str, str, bool]] = []

    def create_conversation(self, customer_id: str):
        return {"id": "conv-1", "customer_id": customer_id}

    def send_message(self, conversation_id: str, customer_id: str, content: str):
        return {
            "status": "pending_confirmation",
            "reply": "已为您生成售后申请草稿，请确认后提交。",
            "pending_action": {
                "action_type": "after_sales",
                "action_id": "A1",
                "order_id": "O1001",
                "reason": content,
                "status": "pending_confirmation",
            },
        }

    def send_message_with_image(self, conversation_id: str, customer_id: str, content: str, image_path: str):
        return self.send_message(conversation_id, customer_id, content)

    def confirm_action(self, conversation_id: str, customer_id: str, action_id: str, *, approved: bool):
        self.confirmed.append((conversation_id, customer_id, action_id, approved))
        return {
            "status": "completed",
            "reply": "售后申请已受理，工单编号为 T1。",
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
    assert result.chat_history[-1][1] == "已创建会话 conv-1。"


def test_send_callback_creates_conversation_when_missing_and_refreshes_audit() -> None:
    demo = load_demo_module()

    result = demo.send_message_callback(
        backend_url="http://backend",
        customer_id="C001",
        conversation_id="",
        message="O1001 鞋底开胶",
        image_path=None,
        chat_history=[],
        client_factory=lambda _base_url: FakeClient(),
    )

    assert result.conversation_id == "conv-1"
    assert result.pending_action["action_id"] == "A1"
    assert "售后申请草稿" in result.chat_history[-1][1]
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

    assert result.raw_json == "没有待确认动作。"
    assert result.chat_history[-1][1] == "没有待确认动作。"


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
    assert "售后申请已受理" in result.chat_history[-1][1]
    assert "ticket_id" in result.raw_json
```

- [ ] **Step 2: Run callback tests to verify red**

Run:

```powershell
cd python-impl
pytest tests/unit/test_gradio_demo.py -q
```

Expected: FAIL with `AttributeError` for missing callback functions.

- [ ] **Step 3: Add callback result type and callbacks**

Append this code to `python-impl/scripts/gradio_demo.py` after `_content_type`:

```python
from dataclasses import dataclass


ClientFactory = Any


@dataclass(frozen=True)
class UiUpdate:
    conversation_id: str
    chat_history: ChatHistory
    pending_action: JsonDict | None
    pending_markdown: str
    runs_json: str
    tool_calls_json: str
    raw_json: str


def create_conversation_callback(
    backend_url: str,
    customer_id: str,
    *,
    client_factory: ClientFactory = SmartCsApiClient,
) -> UiUpdate:
    client = client_factory(backend_url)
    response = client.create_conversation(customer_id)
    conversation_id = str(response["id"])
    history = [("创建会话", f"已创建会话 {conversation_id}。")]
    runs, tool_calls = refresh_audit(client, conversation_id, customer_id)
    return UiUpdate(
        conversation_id=conversation_id,
        chat_history=history,
        pending_action=None,
        pending_markdown=format_pending_action(None),
        runs_json=to_pretty_json(runs),
        tool_calls_json=to_pretty_json(tool_calls),
        raw_json=to_pretty_json(response),
    )


def send_message_callback(
    *,
    backend_url: str,
    customer_id: str,
    conversation_id: str,
    message: str,
    image_path: str | None,
    chat_history: ChatHistory,
    client_factory: ClientFactory = SmartCsApiClient,
) -> UiUpdate:
    client = client_factory(backend_url)
    active_conversation_id = conversation_id.strip()
    active_history = list(chat_history)
    if not active_conversation_id:
        created = client.create_conversation(customer_id)
        active_conversation_id = str(created["id"])
        active_history.append(("创建会话", f"已创建会话 {active_conversation_id}。"))

    if image_path:
        response = client.send_message_with_image(
            active_conversation_id,
            customer_id,
            message,
            image_path,
        )
    else:
        response = client.send_message(active_conversation_id, customer_id, message)

    pending_action = extract_pending_action(response)
    runs, tool_calls = refresh_audit(client, active_conversation_id, customer_id)
    return UiUpdate(
        conversation_id=active_conversation_id,
        chat_history=append_chat_entry(
            active_history,
            user_text=message,
            response=response,
            image_path=image_path,
        ),
        pending_action=pending_action,
        pending_markdown=format_pending_action(pending_action),
        runs_json=to_pretty_json(runs),
        tool_calls_json=to_pretty_json(tool_calls),
        raw_json=to_pretty_json(response),
    )


def confirm_action_callback(
    *,
    backend_url: str,
    customer_id: str,
    conversation_id: str,
    pending_action: JsonDict | None,
    approved: bool,
    chat_history: ChatHistory,
    client_factory: ClientFactory = SmartCsApiClient,
) -> UiUpdate:
    if not pending_action or not pending_action.get("action_id"):
        message = "没有待确认动作。"
        return UiUpdate(
            conversation_id=conversation_id,
            chat_history=[*chat_history, ("确认动作", message)],
            pending_action=None,
            pending_markdown=format_pending_action(None),
            runs_json="{}",
            tool_calls_json="{}",
            raw_json=message,
        )

    client = client_factory(backend_url)
    response = client.confirm_action(
        conversation_id,
        customer_id,
        str(pending_action["action_id"]),
        approved=approved,
    )
    next_pending = extract_pending_action(response)
    runs, tool_calls = refresh_audit(client, conversation_id, customer_id)
    return UiUpdate(
        conversation_id=conversation_id,
        chat_history=[*chat_history, ("确认动作", str(response.get("reply", "已完成确认。")))],
        pending_action=next_pending,
        pending_markdown=format_pending_action(next_pending),
        runs_json=to_pretty_json(runs),
        tool_calls_json=to_pretty_json(tool_calls),
        raw_json=to_pretty_json(response),
    )


def refresh_audit(client: SmartCsApiClient, conversation_id: str, customer_id: str) -> tuple[JsonDict, JsonDict]:
    if not conversation_id:
        return {}, {}
    runs = client.list_runs(conversation_id, customer_id)
    tool_calls = client.list_tool_calls(conversation_id, customer_id)
    return runs, tool_calls
```

- [ ] **Step 4: Add Gradio UI factory and launcher**

Append this code below the callbacks in `python-impl/scripts/gradio_demo.py`:

```python
def build_app():
    import gradio as gr

    with gr.Blocks(title="Smart CS Multi-Agent Demo") as app:
        gr.Markdown(
            "# Smart CS Multi-Agent Demo\n"
            "左侧体验客服对话，右侧查看 pending action、AgentRun、ToolCall 和原始 JSON。"
        )
        conversation_state = gr.State("")
        pending_action_state = gr.State(None)

        with gr.Row():
            with gr.Column(scale=5):
                backend_url = gr.Textbox(
                    label="FastAPI Backend URL",
                    value=DEFAULT_BACKEND_URL,
                )
                customer_id = gr.Textbox(label="Customer ID", value=DEFAULT_CUSTOMER_ID)
                create_button = gr.Button("创建会话", variant="primary")
                conversation_id = gr.Textbox(label="Conversation ID", interactive=False)
                chatbot = gr.Chatbot(label="客服对话", height=420)
                message = gr.Textbox(
                    label="客户消息",
                    placeholder="例如：推荐一双跑鞋 / 查询订单 O1001 / O1001 鞋底开胶了，申请售后",
                    lines=3,
                )
                image = gr.Image(label="可选图片", type="filepath")
                send_button = gr.Button("发送", variant="primary")

            with gr.Column(scale=4):
                pending_markdown = gr.Markdown("当前没有待确认动作。")
                with gr.Row():
                    approve_button = gr.Button("确认提交", variant="primary")
                    reject_button = gr.Button("拒绝")
                with gr.Tabs():
                    with gr.Tab("AgentRun"):
                        runs_json = gr.Code(label="AgentRun JSON", language="json", lines=16)
                    with gr.Tab("ToolCall"):
                        tool_calls_json = gr.Code(label="ToolCall JSON", language="json", lines=16)
                    with gr.Tab("Raw JSON"):
                        raw_json = gr.Code(label="Latest Raw Response", language="json", lines=16)

        def create_ui(backend_url_value: str, customer_id_value: str):
            try:
                result = create_conversation_callback(backend_url_value, customer_id_value)
            except DemoApiError as error:
                raw = to_pretty_json(error.payload)
                return "", "", [("创建会话", str(error))], None, format_pending_action(None), "{}", "{}", raw
            return (
                result.conversation_id,
                result.conversation_id,
                result.chat_history,
                result.pending_action,
                result.pending_markdown,
                result.runs_json,
                result.tool_calls_json,
                result.raw_json,
            )

        def send_ui(
            backend_url_value: str,
            customer_id_value: str,
            conversation_id_value: str,
            message_value: str,
            image_path_value: str | None,
            chat_history_value: ChatHistory | None,
        ):
            history = chat_history_value or []
            try:
                result = send_message_callback(
                    backend_url=backend_url_value,
                    customer_id=customer_id_value,
                    conversation_id=conversation_id_value,
                    message=message_value,
                    image_path=image_path_value,
                    chat_history=history,
                )
            except DemoApiError as error:
                error_response = {"reply": str(error), "detail": error.payload}
                return (
                    conversation_id_value,
                    conversation_id_value,
                    append_chat_entry(history, user_text=message_value, response=error_response, image_path=image_path_value),
                    None,
                    format_pending_action(None),
                    "{}",
                    "{}",
                    to_pretty_json(error.payload),
                    "",
                    None,
                )
            return (
                result.conversation_id,
                result.conversation_id,
                result.chat_history,
                result.pending_action,
                result.pending_markdown,
                result.runs_json,
                result.tool_calls_json,
                result.raw_json,
                "",
                None,
            )

        def confirm_ui(
            backend_url_value: str,
            customer_id_value: str,
            conversation_id_value: str,
            pending_action_value: JsonDict | None,
            chat_history_value: ChatHistory | None,
            approved: bool,
        ):
            history = chat_history_value or []
            try:
                result = confirm_action_callback(
                    backend_url=backend_url_value,
                    customer_id=customer_id_value,
                    conversation_id=conversation_id_value,
                    pending_action=pending_action_value,
                    approved=approved,
                    chat_history=history,
                )
            except DemoApiError as error:
                return (
                    conversation_id_value,
                    history + [("确认动作", str(error))],
                    pending_action_value,
                    format_pending_action(pending_action_value),
                    "{}",
                    "{}",
                    to_pretty_json(error.payload),
                )
            return (
                result.conversation_id,
                result.chat_history,
                result.pending_action,
                result.pending_markdown,
                result.runs_json,
                result.tool_calls_json,
                result.raw_json,
            )

        create_button.click(
            create_ui,
            inputs=[backend_url, customer_id],
            outputs=[
                conversation_id,
                conversation_state,
                chatbot,
                pending_action_state,
                pending_markdown,
                runs_json,
                tool_calls_json,
                raw_json,
            ],
        )
        send_button.click(
            send_ui,
            inputs=[backend_url, customer_id, conversation_state, message, image, chatbot],
            outputs=[
                conversation_id,
                conversation_state,
                chatbot,
                pending_action_state,
                pending_markdown,
                runs_json,
                tool_calls_json,
                raw_json,
                message,
                image,
            ],
        )
        approve_button.click(
            lambda backend, customer, conv, action, history: confirm_ui(
                backend, customer, conv, action, history, True
            ),
            inputs=[backend_url, customer_id, conversation_state, pending_action_state, chatbot],
            outputs=[
                conversation_state,
                chatbot,
                pending_action_state,
                pending_markdown,
                runs_json,
                tool_calls_json,
                raw_json,
            ],
        )
        reject_button.click(
            lambda backend, customer, conv, action, history: confirm_ui(
                backend, customer, conv, action, history, False
            ),
            inputs=[backend_url, customer_id, conversation_state, pending_action_state, chatbot],
            outputs=[
                conversation_state,
                chatbot,
                pending_action_state,
                pending_markdown,
                runs_json,
                tool_calls_json,
                raw_json,
            ],
        )

    return app


def main() -> None:
    build_app().launch()


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run Gradio unit tests**

Run:

```powershell
cd python-impl
pytest tests/unit/test_gradio_demo.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 3**

Run:

```powershell
git add python-impl/scripts/gradio_demo.py python-impl/tests/unit/test_gradio_demo.py
git commit -m "feat: add gradio demo ui"
```

## Task 4: Document The Demo Frontend

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README with Gradio instructions**

Add this section after the current “快速启动” section in `README.md`:

````markdown
## Gradio 演示前端

Gradio 前端是一个面试演示层，独立调用 FastAPI HTTP 接口。先启动后端：

```powershell
cd d:\LLM\smart-cs-multi-agent\python-impl
conda activate customer_service
pip install -e ".[demo,test]"
python scripts/seed_demo_data.py
uvicorn smart_cs.main:app --app-dir src --host 0.0.0.0 --port 8000
```

再打开第二个 PowerShell 窗口启动 Gradio：

```powershell
cd d:\LLM\smart-cs-multi-agent\python-impl
conda activate customer_service
python scripts/gradio_demo.py
```

Gradio 默认会输出本地访问地址，通常是：

```text
http://127.0.0.1:7860
```

推荐演示顺序：

1. 点击“创建会话”，默认客户为 `C001`。
2. 发送“推荐一双跑鞋”，展示商品查询。
3. 发送“查询订单 O1001”，展示订单工具查询。
4. 发送“O1001 鞋底开胶了，申请售后”，展示 pending action。
5. 点击“确认提交”，展示确认后才创建工单。
6. 查看右侧 AgentRun、ToolCall、Raw JSON，讲清楚 Agent 编排和工具审计。
7. 上传一张图片并发送售后消息，展示图片证据链路。
````

- [ ] **Step 2: Check README diff**

Run:

```powershell
git diff -- README.md
```

Expected: README contains a Gradio frontend section and does not remove existing FastAPI instructions.

- [ ] **Step 3: Commit Task 4**

Run:

```powershell
git add README.md
git commit -m "docs: add gradio demo instructions"
```

## Task 5: Verify End To End

**Files:**
- No code changes expected.

- [ ] **Step 1: Run focused tests**

Run:

```powershell
cd python-impl
pytest tests/unit/test_demo_dependencies.py tests/unit/test_gradio_demo.py -q
```

Expected: PASS.

- [ ] **Step 2: Run backend API smoke tests**

Run:

```powershell
cd python-impl
pytest tests/api/test_health.py tests/api/test_conversations.py tests/api/test_image_message.py -q
```

Expected: PASS.

- [ ] **Step 3: Manually launch backend**

Run in PowerShell window 1:

```powershell
cd d:\LLM\smart-cs-multi-agent\python-impl
conda activate customer_service
python scripts/seed_demo_data.py
uvicorn smart_cs.main:app --app-dir src --host 0.0.0.0 --port 8000
```

Expected: Uvicorn serves the backend on `http://localhost:8000`.

- [ ] **Step 4: Manually launch Gradio**

Run in PowerShell window 2:

```powershell
cd d:\LLM\smart-cs-multi-agent\python-impl
conda activate customer_service
python scripts/gradio_demo.py
```

Expected: Gradio prints a local URL such as `http://127.0.0.1:7860`.

- [ ] **Step 5: Manual browser smoke test**

In the Gradio UI:

1. Click `创建会话`.
2. Send `推荐一双跑鞋`.
3. Send `查询订单 O1001`.
4. Send `O1001 鞋底开胶了，申请售后`.
5. Confirm the pending action.
6. Confirm the right-side `AgentRun`, `ToolCall`, and `Raw JSON` panels update.
7. Upload a JPG or PNG and send `O1001 鞋底开胶了，上传图片申请售后`.

Expected: The chat transcript updates, pending actions can be confirmed, and audit tabs show JSON from the backend.

- [ ] **Step 6: Final status**

Run:

```powershell
git status --short
```

Expected: no uncommitted changes from this plan. If user-owned changes remain, report them separately and do not modify them.

## Self-Review

- Spec coverage: Tasks cover dependencies, HTTP client, text/image sends, pending action display, confirm/reject, AgentRun/ToolCall JSON, raw JSON, README, and verification.
- Scope check: The plan remains a single demo frontend and does not add login, backend API changes, or production UI complexity.
- Type consistency: Callback result fields are consistently named `conversation_id`, `chat_history`, `pending_action`, `pending_markdown`, `runs_json`, `tool_calls_json`, and `raw_json`.
- Placeholder scan: This plan contains no deferred implementation markers or unspecified test instructions.
