# Vision After-Sales And Interview Delivery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 增加会话级图片售后证据理解，完成安全售后闭环和可解释运行轨迹，并将项目文档重写为一周内能掌握、面试可据实讲述的材料。

**Architecture:** 图片只存储在会话资产目录，由 `VisionAgent` 通过 LangChain 标准多模态 `HumanMessage` 和结构化输出提取证据，不进入 Milvus。Supervisor 将图片证据、订单工具结果和知识引用组合成售后草稿；第一阶段已有的 LangGraph 中断节点仍然是唯一提交入口。系统记录调用过的 Agent 与工具，文档只引用实际测试和自动生成的 RAG 报告。

**Tech Stack:** Python 3.11, FastAPI multipart upload, LangChain multimodal messages, LangGraph, Pydantic v2, SQLAlchemy, pytest, Markdown

---

## Prerequisites And Scope

依次完成：

1. [2026-05-25-agent-foundation-orchestration.md](./2026-05-25-agent-foundation-orchestration.md)
2. [2026-05-25-text-rag-milvus-evaluation.md](./2026-05-25-text-rag-milvus-evaluation.md)

本计划只交付：

- 一张用户上传图片的会话级保存和结构化解析。
- 售后草稿必须同时具备订单事实、政策引用和可用图片证据。
- 低置信度、证据冲突、系统失败或用户明确要求时转人工草稿。
- `AgentRun` 和 `ToolCall` 可检查执行轨迹。
- 中文 README、架构说明、七日学习记录和面试问答。

不交付图片知识库、图片 embedding、浏览器页面、生产对象存储、真实退款系统或虚构效果数字。

## Official Component Decisions

| Need | Adopted official component | Application code retained |
| --- | --- | --- |
| 图片输入消息 | `langchain.messages.HumanMessage` standard content blocks | 本地资产路径和 MIME 校验 |
| 结构化视觉证据 | chat model `.with_structured_output(VisualEvidence)` | 置信度门禁 |
| 动作确认 | 第一计划已有 `interrupt` + `Command` | 售后证据校验 |
| 编排状态 | 第一计划已有 `StateGraph` | `AgentRun` 审计记录 |

官方参考：

- <https://docs.langchain.com/oss/python/langchain/messages>
- <https://docs.langchain.com/oss/python/integrations/chat/openai>
- <https://docs.langchain.com/oss/python/langgraph/interrupts>

## File Map

Create:

```text
python-impl/src/smart_cs/domain/evidence.py
python-impl/src/smart_cs/infrastructure/assets.py
python-impl/src/smart_cs/agents/vision.py
python-impl/tests/unit/test_vision_agent.py
python-impl/tests/integration/test_image_after_sales.py
python-impl/tests/api/test_image_message.py
docs/interview/learning-log.md
docs/interview/agent-project-resume.md
docs/interview/agent-project-qa.md
```

Modify:

```text
python-impl/src/smart_cs/config.py
python-impl/src/smart_cs/domain/models.py
python-impl/src/smart_cs/domain/repositories.py
python-impl/src/smart_cs/infrastructure/repositories.py
python-impl/src/smart_cs/infrastructure/model_factory.py
python-impl/src/smart_cs/agents/state.py
python-impl/src/smart_cs/agents/supervisor.py
python-impl/src/smart_cs/agents/specialists.py
python-impl/src/smart_cs/agents/guardrails.py
python-impl/src/smart_cs/application/agent_runtime.py
python-impl/src/smart_cs/application/conversation_service.py
python-impl/src/smart_cs/api/schemas.py
python-impl/src/smart_cs/api/routers/conversations.py
README.md
docs/architecture.md
docs/code-walkthrough.md
docs/project-plan.md
docs/interview/resume-template.md
docs/interview/star-method.md
docs/interview/project-qa.md
docs/interview/baguwen.md
docker-compose.yml
```

Delete after final regression passes:

```text
python-impl/agents/
python-impl/api/
python-impl/memory/
python-impl/mcp/
python-impl/tracing/
python-impl/requirements.txt
```

### Task 1: Store Scoped Images And Extract Structured Visual Evidence

**Files:**
- Create: `python-impl/src/smart_cs/domain/evidence.py`
- Create: `python-impl/src/smart_cs/infrastructure/assets.py`
- Create: `python-impl/src/smart_cs/agents/vision.py`
- Modify: `python-impl/src/smart_cs/config.py`
- Modify: `python-impl/src/smart_cs/infrastructure/model_factory.py`
- Modify: `python-impl/src/smart_cs/domain/models.py`
- Modify: `python-impl/src/smart_cs/infrastructure/repositories.py`
- Create: `python-impl/tests/unit/test_vision_agent.py`
- Create: `python-impl/tests/integration/test_image_after_sales.py`

- [ ] **Step 1: Write red tests for storage isolation and low-confidence evidence**

```python
# python-impl/tests/integration/test_image_after_sales.py
from smart_cs.infrastructure.assets import LocalAssetStorage


def test_image_is_stored_in_conversation_directory(tmp_path) -> None:
    storage = LocalAssetStorage(tmp_path / "assets")
    key = storage.save("conv-1", "damage.jpg", "image/jpeg", b"jpeg-data")
    assert key == "conv-1/damage.jpg"
    assert (tmp_path / "assets" / key).read_bytes() == b"jpeg-data"
```

```python
# python-impl/tests/unit/test_vision_agent.py
from smart_cs.domain.evidence import VisualEvidence
from smart_cs.agents.vision import VisionAgent


class FakeVisionModel:
    def examine(self, image_data_url: str, user_message: str) -> VisualEvidence:
        return VisualEvidence(
            visible_issue="uncertain",
            affected_part="shoe",
            summary="图片不清晰，无法确认问题部位",
            confidence=0.42,
            needs_clarification=True,
        )


def test_low_confidence_image_cannot_support_after_sales_draft() -> None:
    evidence = VisionAgent(FakeVisionModel()).inspect("data:image/jpeg;base64,eA==", "鞋底坏了")
    assert evidence.usable_for_draft is False
```

- [ ] **Step 2: Implement evidence and storage boundary**

```python
# python-impl/src/smart_cs/domain/evidence.py
from pydantic import BaseModel, Field


class VisualEvidence(BaseModel):
    visible_issue: str = Field(description="Visible issue category or uncertain")
    affected_part: str = Field(description="Visible affected product part")
    summary: str = Field(description="Short factual description of visible evidence")
    confidence: float = Field(ge=0.0, le=1.0)
    needs_clarification: bool
    extracted_text: str | None = None

    @property
    def usable_for_draft(self) -> bool:
        return self.confidence >= 0.8 and not self.needs_clarification
```

```python
# python-impl/src/smart_cs/infrastructure/assets.py
from pathlib import Path


class LocalAssetStorage:
    ALLOWED_TYPES = {"image/jpeg": ".jpg", "image/png": ".png"}

    def __init__(self, root: Path) -> None:
        self.root = root

    def save(self, conversation_id: str, filename: str, content_type: str, content: bytes) -> str:
        if content_type not in self.ALLOWED_TYPES:
            raise ValueError("Only JPEG and PNG evidence images are accepted")
        suffix = self.ALLOWED_TYPES[content_type]
        safe_stem = Path(filename).stem.replace("/", "_")
        key = f"{conversation_id}/{safe_stem}{suffix}"
        target = self.root / key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return key
```

Persist `asset_key` and serialized `VisualEvidence` against the originating message id, never in knowledge tables.

- [ ] **Step 3: Use LangChain multimodal messages and structured output**

```python
# python-impl/src/smart_cs/agents/vision.py
from langchain.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

from smart_cs.domain.evidence import VisualEvidence


class LangChainVisionModel:
    def __init__(self, model: BaseChatModel) -> None:
        self.model = model.with_structured_output(VisualEvidence)

    def examine(self, image_data_url: str, user_message: str) -> VisualEvidence:
        return self.model.invoke([
            SystemMessage(content="Extract visible after-sales evidence only. Never approve a refund."),
            HumanMessage(content=[
                {"type": "text", "text": user_message},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ]),
        ])


class VisionAgent:
    def __init__(self, vision_model) -> None:
        self.vision_model = vision_model

    def inspect(self, image_data_url: str, user_message: str) -> VisualEvidence:
        return self.vision_model.examine(image_data_url, user_message)
```

In rule mode, images return `needs_clarification=True` with a visible `"规则模式不解析图片"` explanation. This prevents a no-model run from pretending to understand an image.

- [ ] **Step 4: Run tests and commit**

```bash
cd python-impl
pytest tests/unit/test_vision_agent.py tests/integration/test_image_after_sales.py -q
git add src/smart_cs tests
git commit -m "feat: extract scoped visual evidence through langchain messages"
```

Expected: PASS; image assets remain separate from RAG documents.

### Task 2: Coordinate Vision, Policy And Confirmation With Traceable Runs

**Files:**
- Modify: `python-impl/src/smart_cs/agents/state.py`
- Modify: `python-impl/src/smart_cs/agents/supervisor.py`
- Modify: `python-impl/src/smart_cs/agents/specialists.py`
- Modify: `python-impl/src/smart_cs/agents/guardrails.py`
- Modify: `python-impl/src/smart_cs/application/agent_runtime.py`
- Modify: `python-impl/src/smart_cs/application/conversation_service.py`
- Modify: `python-impl/src/smart_cs/domain/models.py`
- Modify: `python-impl/src/smart_cs/domain/repositories.py`
- Modify: `python-impl/src/smart_cs/infrastructure/repositories.py`
- Create: `python-impl/tests/integration/test_safe_image_workflow.py`

- [ ] **Step 1: Test safe composition and run tracing**

```python
def test_clear_image_still_requires_user_confirmation(runtime, repo) -> None:
    first = runtime.invoke_with_image("conv-2", "C001", "O1001 鞋底开胶，申请售后", "asset/damage.jpg")
    assert first["pending_confirmation"]["action_type"] == "after_sales"
    assert repo.list_tickets("C001") == []

    run = repo.latest_agent_run("conv-2")
    assert run.agents == ["RouterAgent", "SupervisorAgent", "VisionAgent", "OrderAgent", "KnowledgeAgent", "AfterSalesAgent"]


def test_uncertain_image_routes_to_handoff_draft_without_refund(runtime, repo) -> None:
    first = runtime.invoke_with_image("conv-3", "C001", "想退款", "asset/blur.jpg")
    assert first["pending_confirmation"]["action_type"] == "handoff"
    assert repo.list_tickets("C001") == []
```

- [ ] **Step 2: Extend structured plan and validate multimodal requirements**

Add `"VisionAgent"` to the allowed `SupervisorDecision.agents`. At planning time, change `validate_decision(decision, has_image)` with these deterministic rules:

```text
When has_image is true and action is draft_after_sales:
  require VisionAgent, OrderAgent, KnowledgeAgent and AfterSalesAgent.
Every draft_after_sales action requires confirmation regardless of confidence.
```

After `VisionAgent` executes, add a `validate_evidence` graph node before the existing confirmation node:

```text
When a draft_after_sales result has no evidence or evidence.usable_for_draft is false:
  replace the unsubmitted draft with draft_handoff and require confirmation.
When evidence is usable:
  preserve draft_after_sales and route it to confirmation.
```

The configured Supervisor still chooses order and relevant specialists; deterministic nodes add mandatory evidence steps and inspect the resulting evidence rather than silently trusting a model.

- [ ] **Step 3: Persist a real AgentRun record**

Add table and repository operations:

```python
class AgentRunRecord:
    id: str
    conversation_id: str
    agents: list[str]
    status: str
    pending_action_id: str | None
    reply: str | None
```

`AgentRuntime` accumulates executed node names in state and `ConversationService` stores one record after an interrupt is surfaced and updates it after resume. Add:

```text
GET /api/conversations/{conversation_id}/runs
```

The endpoint returns `AgentRunRecord` entries plus related `ToolCall` records; it does not expose checkpoint internals.

- [ ] **Step 4: Feed the existing confirmation node, not a new submit path**

Image-backed after-sales output must be a `draft_after_sales` result consumed by the existing LangGraph `confirm_action` node from plan 1. The only function allowed to create a ticket remains `AuthorizedToolExecutor.submit_confirmed_action`.

- [ ] **Step 5: Test and commit**

```bash
cd python-impl
pytest tests/integration/test_safe_image_workflow.py tests/integration/test_action_confirmation.py -q
git add src/smart_cs tests
git commit -m "feat: coordinate evidence and persist agent run traces"
```

Expected: PASS; no ticket exists before confirmation in text or image paths.

### Task 3: Publish Multipart API And Guardrail Behavior

**Files:**
- Modify: `python-impl/src/smart_cs/api/schemas.py`
- Modify: `python-impl/src/smart_cs/api/routers/conversations.py`
- Modify: `python-impl/src/smart_cs/agents/guardrails.py`
- Create: `python-impl/tests/api/test_image_message.py`

- [ ] **Step 1: Add API tests**

```python
def test_image_message_returns_pending_action_and_evidence_summary(client, clear_damage_jpeg) -> None:
    conversation = client.post("/api/conversations", json={"customer_id": "C001"}).json()
    response = client.post(
        f"/api/conversations/{conversation['id']}/messages-with-image",
        data={"content": "订单 O1001 鞋底开胶，申请售后"},
        files={"image": ("damage.jpg", clear_damage_jpeg, "image/jpeg")},
    )
    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "pending_confirmation"
    assert body["visual_evidence"]["summary"]


def test_guard_never_says_refund_completed_before_confirmation(client, clear_damage_jpeg) -> None:
    conversation = client.post("/api/conversations", json={"customer_id": "C001"}).json()
    body = client.post(
        f"/api/conversations/{conversation['id']}/messages-with-image",
        data={"content": "退钱"},
        files={"image": ("damage.jpg", clear_damage_jpeg, "image/jpeg")},
    ).json()
    assert "退款已完成" not in body["reply"]
```

- [ ] **Step 2: Add a multipart endpoint**

Provide:

```text
POST /api/conversations/{conversation_id}/messages-with-image
Content fields: content, image
Accepted MIME types: image/jpeg, image/png
```

The endpoint stores the original image, builds its data URL for `VisionAgent`, saves returned evidence, and invokes the same workflow used by text requests.

- [ ] **Step 3: Enforce response phrases**

`ResponseGuard` must render:

```text
pending after-sales: 已为您生成售后申请草稿，请确认后提交。
submitted after-sales: 售后申请已受理，工单编号为 {ticket_id}。
uncertain evidence: 图片证据暂不能确认问题，已为您生成转人工申请草稿，请确认。
```

It must reject model output containing `退款已完成` when the stored action status is not `submitted`.

- [ ] **Step 4: Verify API and commit**

```bash
cd python-impl
pytest tests/api/test_image_message.py tests/api/test_conversations.py -q
git add src/smart_cs tests
git commit -m "feat: expose guarded image evidence conversation api"
```

Expected: PASS.

### Task 4: Remove Stale Claims And Produce Interview Material

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Modify: `docs/code-walkthrough.md`
- Modify: `docs/project-plan.md`
- Modify: `docs/interview/resume-template.md`
- Modify: `docs/interview/star-method.md`
- Modify: `docs/interview/project-qa.md`
- Modify: `docs/interview/baguwen.md`
- Create: `docs/interview/learning-log.md`
- Create: `docs/interview/agent-project-resume.md`
- Create: `docs/interview/agent-project-qa.md`
- Modify: `docker-compose.yml`

- [ ] **Step 1: Detect claims that cannot remain**

Run:

```bash
rg -n -g '!docs/superpowers/plans/**' "金融|Java|Go|企业级|日均|准确率|FCR|CSAT|token.*40|FAISS|十万|92%|生产环境|事故" README.md docs python-impl
```

Expected: matches in legacy explanatory documents; each match must be removed or explicitly described as prior deleted demo code.

- [ ] **Step 2: Replace public project description with truthful content**

`README.md` must contain these sections and no performance claims:

```markdown
# Smart CS Multi-Agent

面向 AI 应用 / Agent 开发岗位学习与面试复盘的电商客服后端项目。项目实现
Supervisor 编排的多 Agent 工作流、文本知识 RAG、会话级售后图片证据理解，以及
需用户确认的售后提交流程。

## 已实现范围

- Python FastAPI 后端与 LangGraph 状态工作流。
- 独立 RouterAgent；SupervisorAgent 负责规划、协调与汇总。
- Markdown 知识库，Milvus dense + BM25 + RRF 检索。
- 图片仅作为当前会话售后证据，不进入知识库。
- SQLite 业务数据、动作确认、AgentRun 与 ToolCall 审计。

## 边界

这是学习型工程，不连接真实订单或退款系统，不包含生产认证与部署结论。
评估结果以 `python-impl/data/evaluation/latest_results.md` 的实际运行产物为准。
```

`docs/architecture.md` must document the actual graph:

```text
Input -> RouterAgent -> SupervisorAgent -> Specialist execution
Specialists: Product | Order | Knowledge | Vision | AfterSales | Handoff
Draft side effect -> interrupt confirmation -> authorised tool submission -> ResponseGuard
```

`docs/code-walkthrough.md` must link to `python-impl/src/smart_cs` modules in execution order. `docs/project-plan.md` must state the seven-day study route from Task 5.

- [ ] **Step 3: Replace old interview documents rather than retaining numbers**

Replace each legacy file `resume-template.md`, `star-method.md`, `project-qa.md`, and `baguwen.md` with this exact pointer:

```markdown
# 已归档

旧版内容描述了已移除的演示实现，不能作为本项目经历陈述。
请使用 [agent-project-resume.md](./agent-project-resume.md) 与
[agent-project-qa.md](./agent-project-qa.md)，并以代码测试和自动生成评估报告为准。
```

Create `docs/interview/agent-project-resume.md`:

```markdown
# 简历项目表述

## 电商客服多 Agent 后端

基于 Python、LangGraph 与 LangChain 构建电商客服学习项目：拆分独立 Router 与
Supervisor，通过受控工具完成订单查询和需确认的售后申请；以 Markdown、
Milvus dense + BM25 + RRF 实现有引用的政策检索；增加会话级图片证据解析和
低置信度转人工门禁。项目指标请引用
`python-impl/data/evaluation/latest_results.md` 中实际生成的四项 RAG 评估结果。
```

Create `docs/interview/agent-project-qa.md` with five questions and code-linked answers:

```markdown
# 面试问答

## Router 与 Supervisor 为什么分开？
Router 无副作用地识别意图、实体和风险；Supervisor 决定调用哪些子 Agent 及顺序。
写动作仍由确定性授权与确认节点拦截，避免模型直接提交售后。

## 为什么不用 RAG 查订单状态？
知识库适合政策和产品说明；订单状态是实时业务事实，通过鉴权后的工具读取。

## 为什么采用 RRF？
稠密检索覆盖语义表达，BM25 覆盖商品名和规则关键词；RRF 基于排名合并，
避免直接比较两类分数尺度。

## 图片如何控制风险？
图片是当前会话证据，不进入公共索引；视觉结构化结果低置信度时只生成转人工草稿。

## 如何评价 RAG？
执行 `python scripts/evaluate_rag.py` 生成报告，只报告忠实度、答案相关性、
上下文召回和上下文精确四项实际结果。
```

- [ ] **Step 4: Keep deployment description aligned with implemented services**

`docker-compose.yml` must contain only Milvus standalone dependency services and the Python API service used in plan 2. Remove Redis, Java, Go and Jaeger entries from the final study project description because the completed Python implementation does not depend on them.

- [ ] **Step 5: Scan and commit documentation cleanup**

```bash
rg -n -g '!docs/superpowers/plans/**' "金融|Java|Go|企业级|日均|准确率|FCR|CSAT|token.*40|FAISS|十万|92%|生产环境|事故" README.md docs python-impl
git add README.md docs docker-compose.yml
git commit -m "docs: publish truthful agent interview material"
```

Expected: search returns no unsupported project claim; a remaining match is permitted only when an explicit archive or scope boundary identifies removed legacy content without asserting results.

### Task 5: Complete Seven-Day Verification And Learning Log

**Files:**
- Create: `docs/interview/learning-log.md`
- Modify: `docs/project-plan.md`

- [ ] **Step 1: Run final technical acceptance**

```bash
docker compose up -d etcd minio standalone
cd python-impl
python scripts/index_knowledge.py
pytest -q
python scripts/evaluate_rag.py
```

Expected: all tests PASS and `data/evaluation/latest_results.md` is regenerated.

- [ ] **Step 2: Remove displaced demo packages and rerun tests**

Delete only the legacy paths listed in this plan, then run:

```bash
cd python-impl
pytest -q
```

Expected: PASS with imports exclusively from `src/smart_cs`.

- [ ] **Step 3: Write a seven-day mastery checklist**

Create `docs/interview/learning-log.md` containing:

```markdown
# 七日掌握记录

| 日程 | 能解释的主题 | 可执行验证 |
| --- | --- | --- |
| Day 1 | 项目边界、Router 与 Supervisor 职责 | `pytest tests/unit/test_router_supervisor.py -q` |
| Day 2 | 工具鉴权与 SQLite 业务状态 | `pytest tests/unit/test_tools.py -q` |
| Day 3 | LangGraph 中断确认和 API 流程 | `pytest tests/integration/test_action_confirmation.py -q` |
| Day 4 | Markdown 分块、窗口元数据与 Milvus 混合检索 | `python scripts/index_knowledge.py` |
| Day 5 | 图片证据、低置信度转人工 | `pytest tests/unit/test_vision_agent.py -q` |
| Day 6 | 四项 RAG 评估及结果边界 | `python scripts/evaluate_rag.py` |
| Day 7 | 用 STAR 与架构图完整复盘 | `pytest -q` |

## 面试陈述边界

只陈述代码已实现的功能和本机生成的评估报告；不陈述线上流量、业务提升、
生产部署或没有测量的准确率。
```

- [ ] **Step 4: Commit final deliverable**

```bash
git add python-impl docs README.md docker-compose.yml
git commit -m "feat: complete interview focused multimodal customer agent"
git status --short
```

Expected: empty git status and a committed generated evaluation report.

## Acceptance Checklist

- [ ] 图片保存在会话目录而非向量库；VisionAgent 使用 LangChain 多模态消息和结构化输出。
- [ ] 售后写操作在文本和图片路径中均通过同一个 LangGraph 确认节点。
- [ ] 低置信度证据不能触发售后提交，只能澄清或转人工草稿。
- [ ] `AgentRun` 与 `ToolCall` 展示可讲解的执行轨迹。
- [ ] 文档不声称未测量的效果、语言实现或生产能力。
- [ ] 学习记录把一星期掌握目标映射到可运行命令。
