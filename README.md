# Smart CS Multi-Agent 用户使用手册

这是一个电商客服多 Agent 后端示例项目，主体代码在 `python-impl/`。它演示了如何用 LangChain / LangGraph 构建客服对话系统：顶层 supervisor 负责判断该找哪个业务子 Agent，中层子 Agent 通过受控工具完成商品、订单、知识库、售后和记忆检索等任务，写操作必须先生成待确认动作，用户确认后才真正提交。

本文按“从零启动、准备数据、构建 RAG、体验功能、运行测试、根据结果优化”的顺序编写。所有运行和验收路径都使用 **LLM 模式**，不再提供 `rules` 模式路线。

## 1. 项目能力

当前主架构是官方 sub-agent-as-tool 模式：

```text
customer_service_supervisor
├── use_pre_sales_agent
│   └── pre_sales_agent
│       ├── search_products
│       ├── knowledge_rag
│       └── recall_memory
└── use_post_sales_agent
    └── post_sales_agent
        ├── lookup_order
        ├── knowledge_rag
        ├── recall_memory
        ├── request_after_sales
        └── request_handoff
```

主要能力：

- 商品咨询：根据演示商品库返回商品信息和推荐。
- 订单查询：通过授权业务工具查询订单，不从 RAG 文档编造实时状态。
- 知识库问答：Markdown 知识库入 Milvus，`knowledge_rag` 返回 `answer / contexts / citations`。
- 售后申请：先生成 `pending_action`，用户确认后才提交售后工单。
- 转人工申请：同样先 pending，再确认提交。
- 图片证据：`VisionAgent` 只做图片证据预处理，输出 `VisualEvidence`，再进入 PostSalesAgent 上下文和 PolicyEngine。
- 长期记忆：只注入 approved active memory；`memory_candidates` 不会进入 prompt。
- 审计：每轮可查看 `AgentRun` 和 `ToolCall`，复盘 Agent、工具、参数、结果和错误。

不会恢复的旧路径：

- `RouterAgent`
- `SpecialistDispatcher`
- `ResponseGuard`
- `RulesDecisionModel`
- supervisor 直接调度 `KnowledgeAgent` 或 `MemoryAgent`

## 2. 环境准备

以下命令默认在 Windows PowerShell 中执行。Linux / macOS 用户可以把 `$env:NAME="value"` 换成 `export NAME=value`。

进入项目：

```powershell
cd D:\LLM\smart-cs-multi-agent
```

创建并激活 Conda 环境：

```powershell
conda create -n customer_service python=3.11 -y
conda activate customer_service
```

安装项目和测试 / Demo 依赖：

```powershell
cd python-impl
python -m pip install -e ".[test,demo]"
```

## 3. LLM 配置

本项目现在按 LLM 模式运行。请在 `python-impl/.env` 中写入你的模型配置，或者在 PowerShell 中设置同名环境变量。

最小 `.env` 示例：

```env
SMART_CS_MODEL_MODE=llm
SMART_CS_LLM_MODEL=your-chat-model
SMART_CS_LLM_API_KEY=your-api-key
SMART_CS_LLM_BASE_URL=https://your-openai-compatible-endpoint/v1

SMART_CS_DATABASE_URL=sqlite:///./data/smart_cs.db
SMART_CS_CHECKPOINT_PATH=data/checkpoints.db

SMART_CS_RAG_ENABLED=true
SMART_CS_MILVUS_URI=http://localhost:19530
SMART_CS_MILVUS_COLLECTION=smart_cs_knowledge
SMART_CS_EMBEDDING_MODEL=BAAI/bge-m3

SMART_CS_MEMORY_VECTOR_ENABLED=false
SMART_CS_MEMORY_MILVUS_COLLECTION=smart_cs_memories
```

如果某个任务模型没有单独配置，会回退到 `SMART_CS_LLM_MODEL`：

```env
SMART_CS_AGENT_MODEL=
SMART_CS_EXTRACTION_MODEL=
SMART_CS_SUMMARY_MODEL=
SMART_CS_MEMORY_MODEL=
SMART_CS_RAG_MODEL=
SMART_CS_VISION_MODEL=
```

建议第一次跑通时先让这些任务共用一个模型。稳定后再按任务拆分模型。

## 4. 准备演示业务数据

演示数据会写入 SQLite，包含：

- 客户：`C001`
- 商品：演示鞋类商品
- 订单：`O1001`
- 售后 / 审计所需基础表结构

执行：

```powershell
cd D:\LLM\smart-cs-multi-agent\python-impl
conda activate customer_service
python scripts/seed_demo_data.py
```

如果你删除了 `data/smart_cs.db`，重新执行这个脚本即可恢复演示数据。

## 5. 准备 RAG 知识库

知识库 Markdown 文件位于：

```text
python-impl/data/knowledge/
├── after_sales_policy.md
├── faq.md
├── product_guide.md
└── shipping_policy.md
```

入库策略：

```text
Markdown 文件
→ MarkdownHeaderTextSplitter 按 # / ## / ### 切 section
→ 每个 section 生成一个 LangChain Document
→ 写入 knowledge Milvus collection
→ knowledge_rag 检索后返回 answer / contexts / citations
→ 最终回复由 LLM 子 Agent 结合工具结果生成
```

启动 Milvus：

```powershell
cd D:\LLM\smart-cs-multi-agent
docker compose up -d etcd minio standalone
```

检查 Milvus 容器：

```powershell
docker ps
```

建立知识库索引：

```powershell
cd D:\LLM\smart-cs-multi-agent\python-impl
conda activate customer_service
python scripts/index_knowledge.py
```

每次修改 `data/knowledge/*.md` 后，都需要重新执行 `scripts/index_knowledge.py`。

## 6. 启动后端服务

确保已经配置 LLM、准备业务数据，并且如果 `SMART_CS_RAG_ENABLED=true`，Milvus 已启动且知识库已入库。

启动 FastAPI：

```powershell
cd D:\LLM\smart-cs-multi-agent\python-impl
conda activate customer_service
uvicorn smart_cs.main:app --app-dir src --host 0.0.0.0 --port 8000
```

打开：

- 健康检查：<http://localhost:8000/health>
- Swagger：<http://localhost:8000/docs>

如果启动时报 `rules mode has been removed`，说明环境变量里仍有旧配置：

```powershell
$env:SMART_CS_MODEL_MODE = "llm"
```

## 7. HTTP API 体验全流程

下面所有命令在第二个 PowerShell 窗口执行。

设置基础地址：

```powershell
$base = "http://localhost:8000"
```

### 7.1 创建会话

```powershell
$conv = Invoke-RestMethod `
  -Uri "$base/api/conversations" `
  -Method Post `
  -ContentType "application/json" `
  -Body (@{ customer_id = "C001" } | ConvertTo-Json)

$conv
```

后续用 `$conv.id` 作为会话 ID。

### 7.2 商品咨询

```powershell
Invoke-RestMethod `
  -Uri "$base/api/conversations/$($conv.id)/messages" `
  -Method Post `
  -ContentType "application/json" `
  -Body (@{
    customer_id = "C001"
    content = "推荐一双适合通勤和轻量跑步的鞋"
  } | ConvertTo-Json)
```

预期观察点：

- supervisor 调用 `use_pre_sales_agent`
- pre-sales 子 Agent 可以调用 `search_products`
- 回复来自 LLM 对工具结果的整合

### 7.3 订单查询

```powershell
Invoke-RestMethod `
  -Uri "$base/api/conversations/$($conv.id)/messages" `
  -Method Post `
  -ContentType "application/json" `
  -Body (@{
    customer_id = "C001"
    content = "帮我查一下订单 O1001"
  } | ConvertTo-Json)
```

预期观察点：

- supervisor 调用 `use_post_sales_agent`
- post-sales 子 Agent 调用 `lookup_order`
- 订单状态来自 SQL 业务数据，不来自 RAG 文档

### 7.4 知识库问答

```powershell
Invoke-RestMethod `
  -Uri "$base/api/conversations/$($conv.id)/messages" `
  -Method Post `
  -ContentType "application/json" `
  -Body (@{
    customer_id = "C001"
    content = "签收后几天内可以申请退货？需要什么凭证？"
  } | ConvertTo-Json)
```

预期观察点：

- 子 Agent 调用 `knowledge_rag`
- ToolCall 中可以看到 `contexts` 和 `citations`
- 最终客户回复由 LLM 根据知识库证据生成

### 7.5 售后申请：先 pending

```powershell
$pending = Invoke-RestMethod `
  -Uri "$base/api/conversations/$($conv.id)/messages" `
  -Method Post `
  -ContentType "application/json" `
  -Body (@{
    customer_id = "C001"
    content = "O1001 鞋底开胶了，我想申请售后"
  } | ConvertTo-Json)

$pending
$actionId = $pending.pending_action.action_id
```

预期观察点：

- 返回 `pending_action`
- 此时不应直接创建最终售后工单
- 写操作需要 HITL 确认

### 7.6 确认提交

```powershell
Invoke-RestMethod `
  -Uri "$base/api/conversations/$($conv.id)/actions/confirm" `
  -Method Post `
  -ContentType "application/json" `
  -Body (@{
    customer_id = "C001"
    action_id = $actionId
    approved = $true
  } | ConvertTo-Json)
```

如果要拒绝：

```powershell
Invoke-RestMethod `
  -Uri "$base/api/conversations/$($conv.id)/actions/confirm" `
  -Method Post `
  -ContentType "application/json" `
  -Body (@{
    customer_id = "C001"
    action_id = $actionId
    approved = $false
  } | ConvertTo-Json)
```

### 7.7 查看审计

查看 AgentRun：

```powershell
Invoke-RestMethod `
  -Uri "$base/api/conversations/$($conv.id)/runs?customer_id=C001" `
  -Method Get
```

查看 ToolCall：

```powershell
Invoke-RestMethod `
  -Uri "$base/api/conversations/$($conv.id)/tool-calls?customer_id=C001" `
  -Method Get
```

审计重点：

- 调用了哪个高级 sub-agent tool
- 子 Agent 调用了哪些底层工具
- 工具入参、结果、状态、错误类型
- 售后是否先 pending，再 confirm
- `vision_evidence`、`knowledge_rag`、`recall_memory` 是否有记录

## 8. 图片售后体验

准备一张本地图片，例如：

```text
D:\tmp\damage.jpg
```

上传图片并发送消息：

```powershell
curl.exe -X POST "$base/api/conversations/$($conv.id)/messages-with-image" `
  -F "customer_id=C001" `
  -F "content=O1001 鞋底开胶了，我上传图片申请售后" `
  -F "image=@D:\tmp\damage.jpg;type=image/jpeg"
```

图片链路：

```text
ConversationService.send_message_with_image()
→ VisionAgent.inspect()
→ VisualEvidence
→ ToolCall: vision_evidence
→ PostSalesAgent prompt 注入 visual_evidence
→ request_after_sales 工具层再次走 PolicyEngine
```

规则：

- `VisionAgent` 是图片证据预处理 Agent，不是 supervisor handoff worker。
- 低置信度、模糊或需要澄清的图片不能被描述成“已确认质量问题”。
- `visual_evidence.usable_for_draft=false` 时，工具层不得创建 after-sales draft，应转人工或要求补充证据。

## 9. Gradio 可视化体验

Gradio 只是演示前端，所有能力仍通过 FastAPI API 调用后端。

先启动后端，再开第二个 PowerShell：

```powershell
cd D:\LLM\smart-cs-multi-agent\python-impl
conda activate customer_service
python scripts/gradio_demo.py
```

打开：

```text
http://127.0.0.1:7860
```

推荐演示顺序：

1. 创建会话。
2. 问商品推荐。
3. 查订单 `O1001`。
4. 问退货政策，观察 RAG。
5. 发起售后，观察 pending action。
6. 点击确认或拒绝。
7. 查看 AgentRun / ToolCall / Raw JSON。
8. 上传图片，观察 `visual_evidence`。

## 10. Memory 使用说明

Memory 有两条使用路径，但共享同一个检索服务：

```text
自动注入路径:
RuntimeContextBuilder
→ MemoryRetrievalService.search_active_memories()
→ compact customer_memories
→ supervisor / sub-agent prompt

主动工具路径:
recall_memory(scope="long_term")
→ MemoryRetrievalService.search_active_memories()
→ compact tool result
→ LLM 子 Agent 整合
```

Memory vector 开启后的检索流程：

```text
Milvus memory collection
→ 命中 memory_id candidates
→ SQL hydrate MemoryRecord
→ 校验 customer / namespace / review_status / expires_at / risk
→ MemoryContextSelector 重排和安全投影
→ 注入 prompt 或返回 recall_memory 工具结果
```

边界：

- 只注入 `customer/{customer_id}/memories` 下的 approved active memory。
- 不注入 `memory_candidates`。
- 不注入完整 `value / evidence / before_json / after_json / conflict payload`。
- `recall_memory` 只补充客户上下文，不替代 `lookup_order`、`knowledge_rag` 或 `PolicyEngine`。
- `SMART_CS_MEMORY_VECTOR_ENABLED=false` 时，自动 fallback 到 SQL 检索。

开启 memory vector：

```powershell
$env:SMART_CS_MEMORY_VECTOR_ENABLED = "true"
$env:SMART_CS_MEMORY_MILVUS_COLLECTION = "smart_cs_memories"
```

然后重启后端。Memory vector 默认关闭，避免第一次体验时把 Milvus memory 索引也作为必需步骤。

## 11. 测试

本文所有测试命令都显式使用 LLM 模式。单元测试中部分模型会用 fake / stub 保证测试稳定，但运行入口和配置仍然是 LLM 模式，不走 `rules`。

PowerShell 中先设置：

```powershell
cd D:\LLM\smart-cs-multi-agent\python-impl
conda activate customer_service

$env:SMART_CS_MODEL_MODE = "llm"
$env:SMART_CS_LLM_MODEL = "your-chat-model"
$env:SMART_CS_LLM_API_KEY = "your-api-key"
$env:SMART_CS_LLM_BASE_URL = "https://your-openai-compatible-endpoint/v1"
```

只跑单元测试：

```powershell
python -m pytest tests/unit/ -q --tb=short
```

跑 API 测试：

```powershell
python -m pytest tests/api/ -q --tb=short
```

跑 Milvus 集成测试前先确保 Milvus 已启动，并已执行 `scripts/index_knowledge.py`：

```powershell
python -m pytest tests/integration/test_milvus_hybrid.py -q --tb=short
```

跑全部测试：

```powershell
python -m pytest tests/ -q --tb=short
```

重点测试文件：

```text
tests/unit/test_agent_runtime.py
tests/unit/test_agent_tool_wrappers.py
tests/unit/test_context_builder.py
tests/unit/test_memory.py
tests/unit/test_memory_retrieval.py
tests/unit/test_recall_memory_tool.py
tests/unit/test_vision_agent.py
tests/integration/test_milvus_hybrid.py
```

## 12. RAG 评估和优化

评估用例在：

```text
python-impl/data/evaluation/rag_cases.json
```

Milvus 全链路评估：

```powershell
cd D:\LLM\smart-cs-multi-agent
docker compose up -d etcd minio standalone

cd python-impl
conda activate customer_service
$env:SMART_CS_MODEL_MODE = "llm"
python scripts/index_knowledge.py
python scripts/evaluate_rag.py
```

输出：

```text
python-impl/data/evaluation/latest_results.json
python-impl/data/evaluation/latest_results.md
```

指标含义：

- Faithfulness：回答是否忠于检索到的证据。
- Answer Relevancy：回答是否回应了问题。
- Context Recall：应该命中的证据是否被召回。
- Context Precision：召回内容是否足够精准。

根据结果优化：

- Context Recall 低：检查 Markdown 标题层级、section 是否过大或过小、query rewrite 类别是否正确。
- Context Precision 低：检查知识文档是否混杂多个主题，必要时拆分 section。
- Faithfulness 低：检查 prompt 是否要求引用证据，检查 `KnowledgeService` 是否返回了足够 contexts。
- Answer Relevancy 低：补充 FAQ、增加评估用例、观察 ToolCall 中的 query 和 citations。

可选诊断：

```powershell
python scripts/evaluate_rag.py --offline
```

`--offline` 只用于检查评估脚本和 Markdown 解析，不代表 Milvus / LLM 全链路验收结果。

## 13. 常见问题

### 13.1 后端启动时报 rules mode

说明环境中仍有旧变量：

```powershell
$env:SMART_CS_MODEL_MODE = "llm"
```

同时检查 `python-impl/.env` 中是否写了 `SMART_CS_MODEL_MODE=rules`。

### 13.2 LLM 请求失败

检查：

- `SMART_CS_LLM_MODEL`
- `SMART_CS_LLM_API_KEY`
- `SMART_CS_LLM_BASE_URL`
- 当前网络是否能访问模型服务
- 模型服务是否兼容 OpenAI Chat Completions 接口

### 13.3 RAG 查不到结果

检查：

```powershell
docker ps
python scripts/index_knowledge.py
python scripts/evaluate_rag.py
```

再查看 ToolCall 中 `knowledge_rag` 的 `contexts` 和 `citations`。

### 13.4 售后没有生成 pending_action

检查：

- 用户消息是否包含订单号，例如 `O1001`
- 订单是否属于当前客户 `C001`
- ToolCall 中是否调用了 `lookup_order`
- PolicyEngine 是否因为证据不足转人工
- 图片证据是否 `usable_for_draft=false`

### 13.5 图片上传失败

检查：

- 文件路径是否存在
- 图片格式是否为 jpg / jpeg / png
- multipart 字段是否为 `image`
- ToolCall 中是否有 `vision_evidence`

### 13.6 Memory 没有召回

检查：

- 是否是 approved active memory
- 是否属于当前 customer scope
- 是否过期或高风险
- `SMART_CS_MEMORY_VECTOR_ENABLED` 是否按预期开启
- ToolCall 中 `recall_memory` 是否返回了 compact memories

## 14. 目录结构

```text
python-impl/
├── src/smart_cs/api/              # FastAPI 路由和 schema
├── src/smart_cs/application/      # AgentRuntime、ConversationService、Memory、ContextBuilder
├── src/smart_cs/agents/           # sub-agents、VisionAgent、KnowledgeService、RuntimeState
├── src/smart_cs/tools/            # LangChain tools、AuthorizedToolExecutor、ToolPolicy
├── src/smart_cs/rag/              # Markdown indexing、Milvus hybrid store、retrieval、evaluation
├── src/smart_cs/infrastructure/   # 数据库、仓库、模型工厂、资源存储
├── data/knowledge/                # RAG Markdown 知识库
├── data/evaluation/               # RAG 评估集和结果
├── scripts/                       # 数据准备、知识库入库、RAG 评估、Gradio demo
└── tests/                         # unit / api / integration 测试
```

## 15. 推荐首次运行路线

第一次完整体验建议按这个顺序：

1. 配置 LLM `.env`。
2. 安装依赖。
3. 启动 Milvus。
4. 执行 `python scripts/seed_demo_data.py`。
5. 执行 `python scripts/index_knowledge.py`。
6. 启动 `uvicorn`。
7. 用 Swagger 或 PowerShell API 走商品、订单、RAG、售后 pending、confirm、审计。
8. 上传图片体验 VisionAgent。
9. 运行 `python -m pytest tests/unit/ -q --tb=short`。
10. 运行 `python -m pytest tests/ -q --tb=short`。
11. 运行 `python scripts/evaluate_rag.py` 查看 RAG 评估。
12. 根据 ToolCall、测试失败信息和 RAG 指标优化文档、prompt、工具或策略。

这条路线能覆盖：LLM agent 编排、业务数据、RAG 入库、Milvus 检索、HITL 确认、Vision 证据、Memory 上下文、审计和测试验收。
