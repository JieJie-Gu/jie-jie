# Smart CS Multi-Agent

面向 AI 应用 / Agent 开发岗位学习与面试复盘的电商客服后端项目。项目实现了
Supervisor 编排的多 Agent 工作流、文本知识 RAG、会话级售后图片证据处理、动作确认、
SQLite 状态持久化和 ToolCall / AgentRun 审计。

这份 README 按“先跑起来，再完整体验，再用于面试讲解”的顺序组织。

## 一句话介绍

这是一个电商客服 Agent 后端：用户发起订单、商品、政策、售后或转人工问题后，
`RouterAgent` 识别意图，`SupervisorAgent` 规划 specialist 执行顺序，
各 specialist 只能调用被授权的工具；售后提交和转人工不会自动生效，必须先生成草稿，
再由用户确认后才真正创建工单。

## 已实现功能

- FastAPI 后端与 Swagger 调试入口。
- LangGraph 工作流、checkpoint 和并发 turn lease。
- `RouterAgent`、`SupervisorAgent`、`ProductAgent`、`OrderAgent`、`KnowledgeAgent`、`VisionAgent`、`AfterSalesAgent`、`HandoffAgent`。
- SQLite 演示业务数据：客户 `C001`、商品 `P1001`、已签收订单 `O1001`。
- Markdown 知识库、Sentence Window Metadata、Milvus dense + BM25 混合检索、RRF 排序和引用返回。
- 图片售后证据上传：本地规则模式不假装识别图片；真实模型模式可用多模态模型抽取结构化证据。
- 售后 / 转人工动作确认：先 draft，确认后 submit，拒绝则 cancel。
- `ToolCall` 和 `AgentRun` 审计：可复盘每轮调用了哪些 Agent、工具入参、结果和状态。

## 快速启动

以下命令默认使用你本地的 conda 环境 `customer_service`。

```powershell
cd d:\LLM\smart-cs-multi-agent\python-impl
conda activate customer_service
pip install -e ".[test]"
python scripts/seed_demo_data.py
uvicorn smart_cs.main:app --app-dir src --host 0.0.0.0 --port 8000
```

启动后打开：

- API 文档：<http://localhost:8000/docs>
- 健康检查：<http://localhost:8000/health>

默认配置是 `SMART_CS_MODEL_MODE=rules`、`SMART_CS_RAG_ENABLED=false`，不需要外部模型或
Milvus 就能体验商品、订单、售后草稿、确认、图片上传和审计链路。

## 5 分钟体验路线

在另一个 PowerShell 窗口执行下面的命令。

### 1. 创建会话

```powershell
$base = "http://localhost:8000"
$conv = Invoke-RestMethod "$base/api/conversations" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"customer_id":"C001"}'
$conv
```

记住返回的 `$conv.id`。后面的命令会直接复用它。

### 2. 商品推荐

```powershell
Invoke-RestMethod "$base/api/conversations/$($conv.id)/messages" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"customer_id":"C001","content":"推荐一双跑鞋"}'
```

看点：`ProductAgent` 调用 `search_products`，返回演示商品 `轻量跑鞋`。

### 3. 订单查询

```powershell
Invoke-RestMethod "$base/api/conversations/$($conv.id)/messages" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"customer_id":"C001","content":"查询订单 O1001"}'
```

看点：订单状态来自授权业务工具，不从 RAG 文档编造实时状态。

### 4. 售后申请先生成草稿

```powershell
$pending = Invoke-RestMethod "$base/api/conversations/$($conv.id)/messages" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"customer_id":"C001","content":"O1001 鞋底开胶了，申请售后"}'
$pending
$actionId = $pending.pending_action.action_id
```

看点：返回 `pending_action`，但此时还没有真正创建售后工单。

### 5. 用户确认后才提交

```powershell
$body = @{
  customer_id = "C001"
  action_id = $actionId
  approved = $true
} | ConvertTo-Json

Invoke-RestMethod "$base/api/conversations/$($conv.id)/actions/confirm" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

看点：确认后才调用提交工具，返回 `ticket_id`。这是面试里最值得讲的安全边界。

### 6. 查看 AgentRun 和 ToolCall 审计

```powershell
Invoke-RestMethod "$base/api/conversations/$($conv.id)/runs?customer_id=C001"

Invoke-RestMethod "$base/api/conversations/$($conv.id)/tool-calls?customer_id=C001"
```

看点：可以复盘每轮使用了哪些 Agent、是否 pending、用了哪些工具、工具结果是什么。

### 7. 图片售后证据上传

准备一张本地图片，例如 `D:\tmp\damage.jpg`，再执行：

```powershell
curl.exe -X POST "$base/api/conversations/$($conv.id)/messages-with-image" `
  -F "customer_id=C001" `
  -F "content=O1001 鞋底开胶了，上传图片申请售后" `
  -F "image=@D:\tmp\damage.jpg;type=image/jpeg"
```

看点：

- 默认规则模式不会假装看懂图片，会返回需要核验的视觉证据。
- 设置真实多模态模型后，`VisionAgent` 会抽取结构化 `visual_evidence`，再进入订单、知识和售后流程。

## 体验 RAG / Milvus 混合检索

需要 Docker 已启动并能访问 Docker socket。

```powershell
cd d:\LLM\smart-cs-multi-agent
docker compose up -d etcd minio standalone

cd python-impl
conda activate customer_service
$env:SMART_CS_RAG_ENABLED = "true"
python scripts/index_knowledge.py
python scripts/evaluate_rag.py
uvicorn smart_cs.main:app --app-dir src --host 0.0.0.0 --port 8000
```

然后发送政策类问题：

```powershell
Invoke-RestMethod "$base/api/conversations/$($conv.id)/messages" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"customer_id":"C001","content":"签收后退货期限是几天？"}'
```

当前 RAG 评估报告在 [latest_results.md](./python-impl/data/evaluation/latest_results.md)，报告只展示四项指标：
Faithfulness、Answer Relevancy、Context Recall、Context Precision。

## 使用真实模型

本地规则模式适合面试演示工程链路。需要体验 LLM 路由和多模态图片证据时，设置：

```powershell
$env:SMART_CS_MODEL_MODE = "llm"
$env:SMART_CS_LLM_MODEL = "gpt-4o-mini"
$env:SMART_CS_LLM_API_KEY = "<your-api-key>"
# 如使用兼容 OpenAI 的网关，再设置：
# $env:SMART_CS_LLM_BASE_URL = "https://your-compatible-endpoint/v1"
```

随后重新启动 `uvicorn`。

## 常用验证命令

```powershell
cd d:\LLM\smart-cs-multi-agent\python-impl
conda activate customer_service
pytest -q
pytest tests/api/test_conversations.py tests/api/test_image_message.py -q
pytest tests/integration/test_milvus_hybrid.py -q
python scripts/evaluate_rag.py
```

没有 Milvus 时，`test_milvus_hybrid.py` 会跳过或提示启动服务；只想看离线报告可运行：

```powershell
python scripts/evaluate_rag.py --offline
```

## 主要接口

- `POST /api/conversations`：创建或声明会话归属。
- `POST /api/conversations/{id}/messages`：发送文本消息。
- `POST /api/conversations/{id}/messages-with-image`：发送文本加单张图片。
- `POST /api/conversations/{id}/actions/confirm`：确认或拒绝 pending action。
- `GET /api/conversations/{id}/runs?customer_id=C001`：查看 AgentRun 和关联工具调用。
- `GET /api/conversations/{id}/tool-calls?customer_id=C001`：查看会话相关工具审计。
- `GET /health`：健康检查。

## 面试讲解顺序

1. 先讲问题边界：电商客服里有商品、订单、政策、售后和转人工，多数读操作可自动完成，写操作必须确认。
2. 再讲架构：FastAPI 接入层，Application 编排会话，LangGraph 负责多 Agent 状态流，Domain 定义业务实体，Infrastructure 落 SQLite / Milvus。
3. 重点讲安全：Supervisor 只规划 Agent，不直接批准动作；工具有授权边界；售后和转人工先 draft，再 confirm。
4. 讲 RAG：政策问题走知识库，订单状态走业务工具，避免用静态文档回答实时事实。
5. 讲图片：图片只作为当前会话证据，规则模式保守，多模态模式只抽证据不做责任裁定。
6. 讲可观测性：`AgentRun` 看一轮用了哪些 Agent，`ToolCall` 看工具入参、结果、状态和错误。
7. 最后讲验证：单元、API、集成测试覆盖路由、确认闭环、并发租约、Milvus 混合检索和 RAG 指标。

## 安全边界

- 售后申请与转人工申请均先生成草稿，只有用户确认后才调用提交工具。
- 订单状态来自授权工具，不从 RAG 文档回答。
- 默认规则模式不会解析图片像素；它只返回需要人工核验的证据结果。
- 项目不连接真实订单、退款或客服系统，不声明线上生产效果。
- RAG 指标来自固定评测集，适合作为工程演示证据，不等价于线上业务指标。
- RAG 报告见 [latest_results.md](./python-impl/data/evaluation/latest_results.md)；当前提交中的报告已明确标为离线基线，不能作为 Milvus 检索成绩。

## 快速验证

以下命令默认从仓库根目录运行，并使用 Conda 环境 `customer_service`；如果本地环境名不同，请替换为自己的环境名。

```bash
cd python-impl
conda activate customer_service
pytest tests -q
python scripts/evaluate_rag.py --offline
```

需要验证 Milvus 混合检索时，先确保当前用户具有 Docker 权限：

```bash
docker compose up -d etcd minio standalone
cd python-impl
conda activate customer_service
python scripts/index_knowledge.py
pytest tests/integration/test_milvus_hybrid.py -q
python scripts/evaluate_rag.py
```

API 启动：

```bash
cd python-impl
conda activate customer_service
uvicorn smart_cs.main:app --app-dir src --host 0.0.0.0 --port 8000
```

主要接口：

- `POST /api/conversations`
- `POST /api/conversations/{id}/messages`
- `POST /api/conversations/{id}/messages-with-image`
- `POST /api/conversations/{id}/actions/confirm`
- `GET /api/conversations/{id}/runs`
- `GET /api/conversations/{id}/tool-calls`

## Gradio 演示前端

Gradio 是面向面试展示的演示层，通过 FastAPI HTTP APIs 调用后端能力。

以下命令同样从仓库根目录运行。

后端启动：

```powershell
cd python-impl
conda activate customer_service
pip install -e ".[demo,test]"
python scripts/seed_demo_data.py
uvicorn smart_cs.main:app --app-dir src --host 0.0.0.0 --port 8000
```

第二个 PowerShell 窗口启动 Gradio：

```powershell
cd python-impl
conda activate customer_service
python scripts/gradio_demo.py
```

Gradio 通常会打开在 <http://127.0.0.1:7860>。

推荐演示顺序：

1. 创建会话。
2. 商品查询。
3. 订单查询。
4. 售后待确认动作。
5. 确认提交。
6. 查看 AgentRun、ToolCall、Raw JSON。
7. 上传图片。

## 文档

- [架构说明](./docs/architecture.md)
- [代码走读](./docs/code-walkthrough.md)
- [七日学习计划](./docs/project-plan.md)
- [简历表述](./docs/interview/agent-project-resume.md)
- [面试问答](./docs/interview/agent-project-qa.md)
- [学习记录](./docs/interview/learning-log.md)
