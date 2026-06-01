# 启动面向演示和调试的 Gradio 前端，通过 FastAPI 后端体验客服 Agent。

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable

import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BACKEND_URL = "http://localhost:8000"
DEFAULT_CUSTOMER_ID = "C001"


JsonDict = dict[str, Any]
ChatHistory = list[tuple[str, str]]
ClientFactory = Callable[[str], Any]


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
            timeout=120,
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
                timeout=180,
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
            timeout=120,
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

    def current_context(
        self,
        conversation_id: str,
        customer_id: str,
        *,
        query: str | None = None,
    ) -> JsonDict:
        params: JsonDict = {"customer_id": customer_id}
        if query:
            params["query"] = query
        return self._request(
            "GET",
            f"/api/conversations/{conversation_id}/context",
            params=params,
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
        timeout: int = 60,
    ) -> JsonDict:
        try:
            response = requests.request(
                method,
                f"{self.base_url}{path}",
                json=json_payload,
                params=params,
                data=data,
                files=files,
                timeout=timeout,
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
    fields = ("action_type", "action_id", "order_id", "reason", "status", "ticket_id")
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


@dataclass(frozen=True)
class UiUpdate:
    conversation_id: str
    chat_history: ChatHistory
    pending_action: JsonDict | None
    pending_markdown: str
    runs_json: str
    tool_calls_json: str
    context_json: str
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
    runs, tool_calls, context = refresh_observability(client, conversation_id, customer_id)
    return UiUpdate(
        conversation_id=conversation_id,
        chat_history=history,
        pending_action=None,
        pending_markdown=format_pending_action(None),
        runs_json=to_pretty_json(runs),
        tool_calls_json=to_pretty_json(tool_calls),
        context_json=to_pretty_json(context),
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
    runs, tool_calls, context, warning = refresh_observability_safely(
        client,
        active_conversation_id,
        customer_id,
    )
    raw_payload = response if warning is None else {"response": response, "warning": warning}
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
        context_json=to_pretty_json(context),
        raw_json=to_pretty_json(raw_payload),
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
            context_json="{}",
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
    runs, tool_calls, context = refresh_observability(client, conversation_id, customer_id)
    return UiUpdate(
        conversation_id=conversation_id,
        chat_history=[
            *chat_history,
            ("确认动作", str(response.get("reply", "已完成确认。"))),
        ],
        pending_action=next_pending,
        pending_markdown=format_pending_action(next_pending),
        runs_json=to_pretty_json(runs),
        tool_calls_json=to_pretty_json(tool_calls),
        context_json=to_pretty_json(context),
        raw_json=to_pretty_json(response),
    )


def refresh_observability(
    client: Any,
    conversation_id: str,
    customer_id: str,
) -> tuple[JsonDict, JsonDict, JsonDict]:
    if not conversation_id:
        return {}, {}, {}
    runs = client.list_runs(conversation_id, customer_id)
    tool_calls = client.list_tool_calls(conversation_id, customer_id)
    context = client.current_context(conversation_id, customer_id)
    return runs, tool_calls, context


def refresh_observability_safely(
    client: Any,
    conversation_id: str,
    customer_id: str,
) -> tuple[JsonDict, JsonDict, JsonDict, str | None]:
    try:
        runs, tool_calls, context = refresh_observability(client, conversation_id, customer_id)
    except DemoApiError as error:
        warning = f"Observability refresh failed: {error}"
        payload = {"error": warning, "detail": error.payload}
        return payload, payload, payload, warning
    return runs, tool_calls, context, None


def refresh_observability_callback(
    backend_url: str,
    conversation_id: str,
    customer_id: str,
    *,
    client_factory: ClientFactory = SmartCsApiClient,
) -> tuple[str, str, str, str]:
    if not conversation_id:
        return "{}", "{}", "{}", "请先创建会话。"
    try:
        client = client_factory(backend_url)
        runs, tool_calls, context = refresh_observability(client, conversation_id, customer_id)
    except DemoApiError as error:
        error_json = to_pretty_json(error.payload)
        return error_json, error_json, error_json, str(error)
    return (
        to_pretty_json(runs),
        to_pretty_json(tool_calls),
        to_pretty_json(context),
        "已刷新工程观测面板。",
    )


def health_check_callback(
    backend_url: str,
    *,
    client_factory: ClientFactory = SmartCsApiClient,
) -> str:
    try:
        return to_pretty_json(client_factory(backend_url).health())
    except DemoApiError as error:
        return to_pretty_json({"error": str(error), "detail": error.payload})


def run_rag_index_callback() -> str:
    return run_project_script("index_knowledge.py")


def run_rag_evaluation_callback(offline: bool) -> str:
    args = ["evaluate_rag.py"]
    if offline:
        args.append("--offline")
    return run_project_script(*args)


def run_project_script(*script_args: str, timeout_seconds: int = 900) -> str:
    command = [sys.executable, str(ROOT / "scripts" / script_args[0]), *script_args[1:]]
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except Exception as error:
        return f"运行失败：{error}"

    output = []
    output.append("$ " + " ".join(command))
    if completed.stdout:
        output.append("\n[stdout]\n" + completed.stdout.strip())
    if completed.stderr:
        output.append("\n[stderr]\n" + completed.stderr.strip())
    output.append(f"\nexit_code={completed.returncode}")
    return "\n".join(output)


def build_app():
    import gradio as gr

    with gr.Blocks(title="Smart CS Agent Demo") as app:
        gr.Markdown(
            "# Smart CS Agent Demo\n"
            "左侧是客服对话操作区，右侧是工程观测面板。"
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
                with gr.Row():
                    create_button = gr.Button("创建会话", variant="primary")
                    health_button = gr.Button("健康检查")
                conversation_id = gr.Textbox(label="Conversation ID", interactive=False)
                chatbot = gr.Chatbot(label="客服对话", height=460, type="tuples")
                message = gr.Textbox(
                    label="客户消息",
                    placeholder=(
                        "例如：推荐一双通勤跑鞋 / 查询订单 O1001 / "
                        "O1001 鞋底开胶了，我想申请售后"
                    ),
                    lines=3,
                )
                image = gr.Image(label="可选图片证据", type="filepath")
                send_button = gr.Button("发送", variant="primary")

            with gr.Column(scale=6):
                pending_markdown = gr.Markdown(format_pending_action(None))
                with gr.Row():
                    approve_button = gr.Button("确认提交", variant="primary")
                    reject_button = gr.Button("拒绝")
                    refresh_button = gr.Button("刷新观测")

                ops_output = gr.Textbox(label="操作输出", lines=4, interactive=False)

                with gr.Tabs():
                    with gr.Tab("当前上下文"):
                        context_json = gr.Code(
                            label="Runtime Context",
                            language="json",
                            lines=22,
                        )
                    with gr.Tab("ToolCall"):
                        tool_calls_json = gr.Code(
                            label="ToolCall JSON",
                            language="json",
                            lines=22,
                        )
                    with gr.Tab("AgentRun"):
                        runs_json = gr.Code(
                            label="AgentRun JSON",
                            language="json",
                            lines=22,
                        )
                    with gr.Tab("Raw JSON"):
                        raw_json = gr.Code(
                            label="Latest Raw Response",
                            language="json",
                            lines=22,
                        )
                    with gr.Tab("RAG 运维"):
                        gr.Markdown(
                            "重建索引会读取 `data/knowledge/*.md` 并写入 Milvus；"
                            "RAG 评估会生成 `data/evaluation/latest_results.*`。"
                        )
                        offline_eval = gr.Checkbox(
                            label="离线评估，仅检查 Markdown/评估脚本，不作为 Milvus 全链路验收",
                            value=False,
                        )
                        with gr.Row():
                            rebuild_rag_button = gr.Button("重建 RAG 索引")
                            evaluate_rag_button = gr.Button("运行 RAG 评估")
                        rag_output = gr.Textbox(
                            label="RAG 操作输出",
                            lines=18,
                            interactive=False,
                        )

        def create_ui(backend_url_value: str, customer_id_value: str):
            try:
                result = create_conversation_callback(backend_url_value, customer_id_value)
            except DemoApiError as error:
                raw = to_pretty_json(error.payload)
                return "", "", [("创建会话", str(error))], None, format_pending_action(None), "{}", "{}", "{}", raw
            return (
                result.conversation_id,
                result.conversation_id,
                result.chat_history,
                result.pending_action,
                result.pending_markdown,
                result.runs_json,
                result.tool_calls_json,
                result.context_json,
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
                    append_chat_entry(
                        history,
                        user_text=message_value,
                        response=error_response,
                        image_path=image_path_value,
                    ),
                    None,
                    format_pending_action(None),
                    "{}",
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
                result.context_json,
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
                result.context_json,
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
                context_json,
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
                context_json,
                raw_json,
                message,
                image,
            ],
        )
        approve_button.click(
            lambda backend, customer, conv, action, history: confirm_ui(
                backend,
                customer,
                conv,
                action,
                history,
                True,
            ),
            inputs=[backend_url, customer_id, conversation_state, pending_action_state, chatbot],
            outputs=[
                conversation_state,
                chatbot,
                pending_action_state,
                pending_markdown,
                runs_json,
                tool_calls_json,
                context_json,
                raw_json,
            ],
        )
        reject_button.click(
            lambda backend, customer, conv, action, history: confirm_ui(
                backend,
                customer,
                conv,
                action,
                history,
                False,
            ),
            inputs=[backend_url, customer_id, conversation_state, pending_action_state, chatbot],
            outputs=[
                conversation_state,
                chatbot,
                pending_action_state,
                pending_markdown,
                runs_json,
                tool_calls_json,
                context_json,
                raw_json,
            ],
        )
        refresh_button.click(
            refresh_observability_callback,
            inputs=[backend_url, conversation_state, customer_id],
            outputs=[runs_json, tool_calls_json, context_json, ops_output],
        )
        health_button.click(
            health_check_callback,
            inputs=[backend_url],
            outputs=[ops_output],
        )
        rebuild_rag_button.click(
            run_rag_index_callback,
            outputs=[rag_output],
        )
        evaluate_rag_button.click(
            run_rag_evaluation_callback,
            inputs=[offline_eval],
            outputs=[rag_output],
        )

    return app


def main() -> None:
    build_app().launch()


if __name__ == "__main__":
    main()
