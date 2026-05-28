# Gradio Demo Frontend Design

## Goal

Build a small Gradio frontend for interview and local demonstration of the Smart CS Multi-Agent backend.

The frontend is not a production UI. It is a demo console that lets a reviewer experience the complete customer-service loop while also seeing the engineering evidence behind the response: pending actions, AgentRun records, ToolCall audit records, and raw API responses.

## Scope

The first version is the core closed-loop demo:

- Create a conversation for a customer, defaulting to `C001`.
- Send text messages to the existing FastAPI backend.
- Upload one image with a message through the existing image endpoint.
- Display the assistant reply as a chat transcript.
- Display the latest pending action, if any.
- Confirm or reject the pending action.
- Refresh and display AgentRun and ToolCall data for the current conversation.
- Show the latest raw JSON response for debugging and interview explanation.

Out of scope for the first version:

- Authentication or user management.
- A persistent multi-conversation inbox.
- Production styling or responsive mobile polish.
- Starting or managing the FastAPI backend process automatically.
- Directly importing backend application services.
- Changing backend API contracts.

## Architecture

Create a single script:

```text
python-impl/scripts/gradio_demo.py
```

The script runs a Gradio app and calls the FastAPI service over HTTP. The default backend URL is:

```text
http://localhost:8000
```

The user starts the backend separately with:

```powershell
uvicorn smart_cs.main:app --app-dir src --host 0.0.0.0 --port 8000
```

Then starts the frontend with:

```powershell
python scripts/gradio_demo.py
```

This preserves a clean boundary:

- FastAPI remains the real application service.
- Gradio is only an interview/demo client.
- The frontend exercises the same HTTP API that Swagger and external clients use.

## UI Layout

Use a two-column Gradio Blocks layout.

Left column: customer experience

- Backend URL textbox.
- Customer ID textbox, default `C001`.
- Create conversation button.
- Conversation ID display.
- Chatbot transcript.
- Message textbox.
- Optional image upload.
- Send button.

Right column: interview and debug evidence

- Pending action panel.
- Confirm button.
- Reject button.
- AgentRun tab.
- ToolCall tab.
- Raw JSON tab.

The UI should be utilitarian and dense enough for repeated demo use. It should avoid a marketing-style landing page. The first screen is the usable console.

## Data Flow

### Create Conversation

When the user clicks Create Conversation:

```text
POST /api/conversations
```

Request:

```json
{"customer_id": "C001"}
```

The returned conversation ID is stored in Gradio state and displayed.

### Send Text

If no image is attached, the Send button calls:

```text
POST /api/conversations/{id}/messages
```

Request:

```json
{"customer_id": "C001", "content": "..."}
```

The frontend appends the user message and backend reply to the chat transcript.

### Send Image

If an image is attached, the Send button calls:

```text
POST /api/conversations/{id}/messages-with-image
```

Form fields:

- `customer_id`
- `content`
- `image`

The frontend appends the user message and backend reply to the chat transcript and displays visual evidence in the raw JSON tab.

### Confirm Or Reject

If the latest response contains `pending_action`, the pending action panel shows:

- `action_type`
- `action_id`
- `order_id`
- `reason`
- `status`

Confirm calls:

```text
POST /api/conversations/{id}/actions/confirm
```

with `approved: true`.

Reject calls the same endpoint with `approved: false`.

After either operation, the frontend updates the chat transcript, pending action panel, AgentRun tab, ToolCall tab, and raw JSON tab.

### Refresh Audit Data

After create, send, confirm, or reject, the frontend calls:

```text
GET /api/conversations/{id}/runs?customer_id=C001
GET /api/conversations/{id}/tool-calls?customer_id=C001
```

The response is rendered as JSON in the right-side tabs.

## Error Handling

The frontend should handle common demo errors without crashing:

- Backend unavailable: show a clear message asking the user to start `uvicorn`.
- Missing conversation ID: create a conversation automatically or ask the user to click Create Conversation.
- HTTP 4xx / 5xx: show the backend error detail in the chat and raw JSON tab.
- Missing pending action on confirm/reject: show a short message explaining there is no pending action.
- Image upload with unsupported file type: pass the backend error through visibly.

The frontend should not hide raw errors during a demo; the point is to make backend behavior inspectable.

## Dependencies

Add Gradio as a demo dependency. Prefer a dedicated optional dependency group:

```toml
[project.optional-dependencies]
demo = [
    "gradio>=4,<6",
    "requests>=2.32,<3",
]
```

The project already uses `httpx` in test dependencies, but `requests` is acceptable for a small synchronous demo script. If the repo maintainers prefer fewer dependencies, the implementation may use `urllib.request` from the standard library instead, but Gradio itself is still required.

## Testing

Add focused tests for helper logic if the script exposes small pure functions:

- Extracting a pending action from an API response.
- Formatting chat transcript entries.
- Formatting HTTP errors.

Do not overbuild browser automation for the first version. Manual verification is enough for Gradio rendering:

1. Start FastAPI.
2. Start Gradio.
3. Create conversation.
4. Send product message.
5. Send order message.
6. Send after-sales message.
7. Confirm pending action.
8. Verify AgentRun and ToolCall tabs update.
9. Send a message with an image.

## README Update

Update README with a short “Gradio demo frontend” section:

```powershell
cd d:\LLM\smart-cs-multi-agent\python-impl
conda activate customer_service
pip install -e ".[demo,test]"
uvicorn smart_cs.main:app --app-dir src --host 0.0.0.0 --port 8000
python scripts/gradio_demo.py
```

Mention that the Gradio app defaults to `http://localhost:8000` and opens on Gradio's default local URL.

## Acceptance Criteria

- The Gradio frontend starts locally.
- It can create a conversation for `C001`.
- It can send text messages through FastAPI.
- It can send one image with a message through FastAPI.
- It displays backend replies in a chat transcript.
- It displays pending actions and supports confirm/reject.
- It displays AgentRun and ToolCall JSON for the current conversation.
- It displays raw JSON for the latest backend response.
- README explains how to run the FastAPI backend and Gradio frontend.
- Existing backend API tests still pass.
