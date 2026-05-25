# Python 电商客服多 Agent 核心重建设计

## 1. 背景与结论

项目目标是建设一个可持续二次开发的、多模态电商客服多 Agent 系统。
首期建立文本与图片客服闭环：用户能够以文字或图片咨询，运营人员能够以统一的 Markdown 知识模型导入文本文件与图片，系统通过可引用证据回答并尽量减少人工接管。

现有 `python-impl` 规模较小、工作树干净，适合作为 Python 实现落点；但当前代码属于演示骨架，不能直接作为长期内核：

- `Supervisor`、工具层、内存存储与 RAG 尚未形成一致的业务执行链路。
- 工单使用内存实现，缺少可替换 Repository 与可审计持久化。
- RAG 使用演示性哈希嵌入，示例数据偏金融而非电商。
- 没有测试体系、明确领域模型或可执行的多模态消息契约。

决定采用的方案是：**在 `smart-cs-multi-agent/python-impl` 内重建清晰的分层结构，保留 FastAPI、Supervisor 多 Agent 和 Tool/MCP 可扩展方向，不保持当前内部组织或 API 兼容性。**

## 2. 目标与首期范围

### 2.1 目标

- 构建职责清晰、可单元测试、可替换基础设施的 Python 后端骨架。
- 落实真实的 Supervisor 型多 Agent 执行链路，而非仅多个提示词文件。
- 提供电商客服关键业务闭环与副作用安全门禁。
- 提供 Canonical Markdown 与 Milvus 多模态检索链路，支撑可追溯知识回答。
- 保持未来语音与真实电商系统接入的扩展空间。

### 2.2 首期实现范围

- 文本与图片会话消息、原始资产保存和视觉解析。
- 商品咨询。
- 订单与物流查询。
- 售后/退款申请：用户确认后受理并创建工单。
- 转人工：用户确认后建立交接请求。
- 多模态 RAG：文档/图片导入、Canonical Markdown、分块、Milvus 混合检索、引用与 Answerability Gate。
- 轻量知识库管理页和最小聊天演示页。
- SQLite 持久化、Repository 接口、演示数据脚本。
- 以会话访问令牌隔离演示会话以及订单、物流、工单读取，防止已建立会话之间越界读取。
- Swagger、自动化测试、AgentRun 与 ToolCall 审计记录。

### 2.3 首期明确不做

- 语音转写和视频输入。
- 完整运营后台或人工客服工作台；首期页面仅服务知识管理与聊天演示。
- 完整账号体系、后台角色管理或生产级身份认证；首期只提供会话作用域访问令牌。
- 对接真实电商、物流或 CRM API。
- 文档版本发布审批、GraphRAG、Text-to-SQL 与高级 RAG 策略实验。
- 自主执行有副作用动作的开放式 Agent 循环。

## 3. 重建策略

采用原地重建而不是增量堆叠或新建平行实现：

| 路径 | 取舍 | 结论 |
|---|---|---|
| 在 `python-impl` 内建立新边界并替换演示实现 | 与目标落点一致，能形成长期结构；需要重写旧实现 | 采用 |
| 保持平铺目录后追加 Agent 与 SQLite | 短期出效果快，但会继续积累耦合 | 不采用 |
| 新建 `python-v2` 并保留旧实现 | 最干净，但产生两套 Python 维护对象 | 不采用 |

原有代码可作为业务场景和 Supervisor 思路参考；内存工单、伪 RAG 和金融示例不纳入首期新内核。

## 4. 工程结构与依赖规则

目标目录结构：

```text
python-impl/
  pyproject.toml
  src/smart_cs/
    main.py
    config.py

    api/
      routers/
        conversations.py
        catalogs.py
        knowledge.py
        assets.py
        operations.py
      schemas/
      dependencies.py

    domain/
      models.py
      enums.py
      repositories.py
      errors.py

    application/
      conversation_service.py
      knowledge_service.py
      asset_service.py
      agent_runtime.py
      dto.py

    agents/
      state.py
      supervisor.py
      router.py
      specialists/
        product.py
        order.py
        logistics.py
        after_sales.py
        handoff.py
        knowledge.py
        vision.py
      guardrails.py

    tools/
      registry.py
      specifications.py
      handlers.py

    rag/
      loaders.py
      markdown.py
      sentence_window.py
      retrieval.py
      answerability.py
      types.py

    infrastructure/
      database.py
      repositories/
      llm.py
      assets.py
      vision.py
      embeddings.py
      milvus.py
      observability.py

  web/                       # 薄 UI：知识管理页与聊天演示页
  tests/
    unit/
    integration/
    api/
  scripts/
    seed_demo_data.py
    evaluate_rag.py
```

依赖方向必须保持单向：

```text
api -> application -> domain
agents -> domain + tools interfaces
tools handlers -> domain repository interfaces
infrastructure -> domain interfaces
main/config -> application + infrastructure 装配
```

约束如下：

- `domain` 不依赖 FastAPI、LangGraph、SQLAlchemy、SQLite 或具体 LLM SDK。
- `api` 只承担请求校验、依赖获取、调用应用用例和响应序列化。
- `application` 负责一次会话回合的事务、Agent 编排调用、状态变更与日志保存。
- `agents` 输出结构化判断、计划、工具请求和候选答复，不直接操作数据库。
- `rag` 负责知识转换后的索引、检索、引用和证据充分性判断，不查询实时订单事实。
- `tools` 是 Agent 使用业务能力的唯一入口，并对风险动作实施权限与确认控制。
- `infrastructure` 提供 SQLite Repository、LocalAssetStorage、Milvus 与 Provider 适配器，可被替换。
- `web` 仅调用 API 并渲染状态，不保存业务规则或 Agent 逻辑。

## 5. 多 Agent 模型与职责

### 5.1 架构类型

系统采用 **Supervisor 型多 Agent 架构**。`SupervisorAgent` 是主编排 Agent，拥有理解、规划、调度、协调和结果汇总职责；`RouterAgent` 保持为独立分析型子 Agent，提供结构化的路由建议但不取得最终调度权。

```text
User Message
  -> SupervisorAgent
       -> RouterAgent
       -> SupervisorAgent.plan
            -> ProductAgent
            -> OrderAgent
            -> LogisticsAgent
            -> AfterSalesAgent
            -> HandoffAgent
            -> KnowledgeAgent
            -> VisionAgent
       -> ResponseGuardAgent
       -> SupervisorAgent.synthesize
  -> Assistant Message
```

### 5.2 Agent 职责

| Agent | 职责 | 允许的业务能力 |
|---|---|---|
| `SupervisorAgent` | 理解整体请求；调用 Router；制定一个或多个专业 Agent 的执行顺序；限制工具范围；汇总回复 | 调度与汇总，不直接写业务数据 |
| `RouterAgent` | 识别单意图/多意图、实体、置信度、风险信号，返回建议 Agent | 无业务副作用工具 |
| `ProductAgent` | 商品信息、规格、库存咨询 | `search_products`, `get_product` |
| `OrderAgent` | 用户自身订单状态查询 | `get_order` |
| `LogisticsAgent` | 配送轨迹、预计送达和物流异常解释 | `get_order`, `get_shipment` |
| `AfterSalesAgent` | 退货、退款和售后诉求处理；结合 `KnowledgeAgent` 返回的政策证据生成动作 | `get_order`, `draft_after_sales`, `submit_after_sales` |
| `HandoffAgent` | 转人工申请与交接摘要生成 | `draft_handoff`, `confirm_handoff` |
| `KnowledgeAgent` | 检索政策、FAQ、商品说明和图片知识，基于证据形成答复片段 | `retrieve_knowledge` |
| `VisionAgent` | 理解知识库图片或用户会话图片，输出 OCR、Markdown 描述、结构化属性与置信度 | 读取授权图片资产，无业务副作用 |
| `ResponseGuardAgent` | 检查隐私泄露、无依据事实、越权承诺和未确认执行 | 审查，不执行业务动作 |

### 5.3 多 Agent 协作示例

请求：

```text
我买的鞋还没收到，明天就要出差了，没到的话帮我退掉。
```

执行链路：

```text
SupervisorAgent
  -> RouterAgent:
       intents = [logistics_query, conditional_refund]
       risk_flags = [refund_requires_confirmation]
  -> SupervisorAgent.plan:
       step 1: LogisticsAgent 查询物流事实
       step 2: AfterSalesAgent 说明退款条件并生成待确认动作（如用户当前已明确申请）
  -> ResponseGuardAgent
  -> SupervisorAgent 汇总物流状态和下一步确认方式
```

这一设计允许多个专业 Agent 按序协作，但不允许子 Agent 自主绕过 Supervisor 或确认门禁执行退款、建单或转人工。

图片售后请求可以增加视觉和知识证据节点：

```text
用户上传开胶照片并询问退款
  -> SupervisorAgent 调用 VisionAgent 提取本次会话图片证据
  -> RouterAgent 识别 after_sales 意图
  -> KnowledgeAgent 检索质量问题退货政策并返回引用
  -> OrderAgent 校验订单事实
  -> AfterSalesAgent 生成符合规则的售后申请摘要
  -> 用户确认后自动受理，不默认转人工
```

## 6. 领域模型与状态

首期核心实体：

```text
Customer
Conversation
Message
AttachmentRef
Asset
Product
Order
Shipment
AfterSalesRequest
HandoffRequest
Ticket
KnowledgeBase
KnowledgeDocument
KnowledgeAsset
KnowledgeChunk
RetrievalLog
AgentRun
ToolCall
```

关键状态：

```text
Conversation:
  active | pending_confirmation | pending_handoff | human_active | closed

AfterSalesRequest:
  draft | pending_confirmation | submitted | rejected | completed

HandoffRequest:
  draft | confirmed | assigned | cancelled

Ticket:
  open | processing | resolved | closed

Message.content_type:
  text | image | audio | mixed

KnowledgeDocument.conversion_status:
  pending | processing | indexed | failed

KnowledgeAsset.processing_status:
  stored | parsing | indexed | failed

EvidenceLevel:
  authoritative_text | vision_generated | conversation_evidence
```

`Message` 设计中包含 `attachments: list[AttachmentRef]`。首期支持文本和图片附件；语音类型仅保留契约，不开放处理链路。

知识文档采用统一 Markdown 数据契约：

- 人工提交的 Markdown 以及由 PDF、TXT、DOCX 转换得到的正文标记为 `authoritative_text`，可以独立支撑政策和说明答复。
- 知识库图片单独保存原图，`VisionAgent` 生成 OCR、Markdown 描述、结构化属性与置信度，标记为 `vision_generated`；低置信解析结果不能单独支撑最终结论。
- 用户会话图片标记为 `conversation_evidence`，只用于当前咨询或售后事实，不自动写回公共知识库。

创建 `Conversation` 时生成不可预测的会话访问令牌，仅向调用方返回一次并在存储中保留哈希。除公开商品列表和健康检查外，客户侧会话及关联资源接口都要求该令牌。该机制只证明请求方持有当前演示会话，不验证其现实身份；首期服务不得作为公开生产客服入口部署，后续必须由正式登录与授权体系替代或补强。

## 7. 工具层与动作安全

### 7.1 工具分类

只读工具可以在计划允许的 Agent 中直接执行：

```text
search_products
get_product
get_order
get_shipment
retrieve_knowledge
get_ticket
```

风险动作分为生成草稿和确认后执行：

```text
draft_after_sales
draft_handoff

submit_after_sales
confirm_handoff
```

### 7.2 工具执行规则

`ToolRegistry` 负责：

- 注册工具描述、输入 schema、调用权限、风险等级和确认要求。
- 校验当前 Specialist 是否被本轮 Supervisor 计划授权调用对应工具。
- 校验参数格式及资源所属关系，例如订单是否属于当前客户。
- 在执行前校验是否存在匹配且已确认的 pending action。
- 保存每次工具调用的输入摘要、结果、耗时、成功状态和失败原因。
- 将基础设施异常转化为稳定的业务错误，不向模型暴露内部异常详情。

知识检索与业务事实严格分流：

- `retrieve_knowledge` 仅检索政策、FAQ、商品说明及知识库图片证据。
- 订单状态、库存、物流轨迹和售后进度必须通过业务工具读取，不能由 RAG 推断。
- `AfterSalesAgent` 可以同时消费订单工具结果和 `KnowledgeAgent` 的政策引用，以自动受理标准售后申请。

### 7.3 副作用闭环

退款/退货请求：

```text
用户提出售后请求
  -> AfterSalesAgent 调用 draft_after_sales
  -> 保存草稿与待确认动作
  -> 回复申请摘要和确认提示
  -> Conversation = pending_confirmation

用户明确确认
  -> Supervisor 验证 pending action
  -> AfterSalesAgent 调用 submit_after_sales
  -> 创建/更新 AfterSalesRequest 与 Ticket
  -> 返回受理编号
```

`Ticket` 表示业务追踪记录，不等于人工接管。标准退货/退款在规则满足且用户确认后由系统自动受理；仅在规则不覆盖、证据冲突、系统故障、敏感争议或用户明确要求时，才进入 `HandoffRequest`。任何模型输出都不能替代确认状态校验。

## 8. 请求数据流

核心对话端点每轮执行：

```text
POST /api/conversations/{id}/messages
  -> 校验会话可接收消息
  -> 持久化用户 Message 与可选图片 Asset
  -> application 加载会话、最近消息、客户可见业务上下文
  -> SupervisorAgent 启动 AgentRun
       -> 如存在图片，VisionAgent 返回当前会话视觉证据
       -> RouterAgent 返回意图、实体、建议 Agent 与风险信号
       -> SupervisorAgent 生成有序执行计划和授权工具集
       -> SpecialistAgent(s) 生成工具请求并消费工具结果
       -> ResponseGuardAgent 审查候选回复与证据
       -> SupervisorAgent 汇总最终回复
  -> application 原子保存助手 Message、状态变更、AgentRun、ToolCall
  -> 返回本轮结果
```

回复包含可调试但不暴露内部推理的执行摘要：

```json
{
  "reply": "您的订单当前运输中，预计送达时间以最新物流轨迹为准。",
  "conversation_status": "active",
  "run_id": "run_xxx",
  "agents_invoked": [
    "SupervisorAgent",
    "RouterAgent",
    "LogisticsAgent",
    "ResponseGuardAgent"
  ],
  "citations": [],
  "pending_action": null
}
```

## 9. 多模态 RAG 主链路

### 9.1 使用边界

RAG 负责可沉淀、可引用的知识：

```text
售后政策、物流规则、FAQ、商品说明、尺码表、知识库图片解析内容
```

RAG 不负责实时业务事实：

```text
库存数量、用户订单状态、当前物流轨迹、售后处理进度
```

实时事实只能通过受控业务工具查询。复合问题由 `SupervisorAgent` 编排，例如图片售后请求同时调用 `VisionAgent`、`KnowledgeAgent`、`OrderAgent` 与 `AfterSalesAgent`。

### 9.2 数据加载与统一 Markdown

首期知识入库路径：

```text
知识库管理页上传
  -> LocalAssetStorage 保存原始文件或原始图片
  -> Loader / Converter / VisionProvider
       Markdown          -> 规范化为 Canonical Markdown
       PDF/TXT/DOCX      -> 提取文本与结构并转换为 Canonical Markdown
       Image             -> OCR + Markdown 描述 + 结构化元数据 + confidence
  -> SQL 保存 KnowledgeDocument / KnowledgeAsset / 元数据
  -> 分块、Embedding、Milvus 索引
```

统一索引输入是 Markdown 文本和元数据，而不是原始二进制文件。图片原件保存在 `LocalAssetStorage`，其视觉解析 Markdown 用于文本检索，原图向量用于跨模态检索。每条文档或资产元数据至少包含：

```text
knowledge_base_id
document_id / asset_id
source_type
storage_key
section_path
evidence_level
confidence
document_status
checksum
citation_metadata
```

### 9.3 分块与上下文窗口

首期只使用以下两项，不加入递归分块或语义分块：

```text
MarkdownHeaderTextSplitter
  -> 依据 Markdown 标题切出带 section_path 的章节文档

Sentence Window Metadata
  -> 在章节内按句子生成基础检索节点
  -> 给每个句子节点记录前后句窗口
```

Milvus 与 BM25 检索索引的是定位精确的句子节点；向 LLM 提供上下文时，将命中句子替换为其窗口文本，并保留文档标题、章节路径和资产引用。这样在检索阶段减少主题稀释，在回答阶段避免断章取义。

### 9.4 Embedding 与 Milvus 索引

代码依赖 Provider 接口，运行时可通过配置接入本地或云端 HTTP 服务，自动化测试使用 fake provider：

```text
VisionProvider
  -> 图片 OCR、Markdown 描述、属性和置信度

TextEmbeddingProvider
  -> Markdown 句子节点与文本 query 向量

MultimodalEmbeddingProvider
  -> 原始图片、文本节点与文本/图片 query 的共享空间向量

VectorStore
  -> Milvus 实现
```

Milvus 是首期 RAG 的必需依赖，collection 至少承载：

```text
text_dense_vector       # 文本语义召回
multimodal_dense_vector # 图文共享空间召回
text_sparse_vector      # BM25 关键词召回
raw_text
metadata fields         # knowledge_base、document_status、evidence_level 等过滤字段
```

Milvus 的 BM25 sparse 检索和 multi-vector hybrid search 能在同一检索层组合 sparse、文本 dense 与图片 dense 结果，并使用 RRF 合并排名。首期不保留“Milvus 不可用时仅靠 SQLite 检索并继续回答”的降级路径；Embedding 或 Milvus 不可用时，知识回答报告依赖故障并进入安全兜底。

文本提问生成 `text_dense_vector` 查询向量、共享空间文本查询向量以及 BM25 文本查询；带图片的提问还会生成共享空间图片查询向量，并将视觉解析出的 Markdown 作为查询辅助信息。因此图片不仅通过 OCR 文本被召回，也可以通过原图与文本/图片之间的跨模态相似度被召回。

### 9.5 查询增强与检索

查询增强只实现两项：

```text
Query Rewrite
  -> 将口语化提问转换为适合知识检索的短查询

Metadata Filter
  -> 由允许字段生成过滤条件，例如 knowledge_base、category、
     document_status、evidence_level、product_type
```

元数据过滤必须经过白名单与值校验，不能将模型生成的任意表达式直接交给 Milvus。

`KnowledgeAgent` 的检索流程：

```text
用户问题 / 图片解析证据
  -> Query Rewrite
  -> Metadata Filter
  -> 并行召回
       Milvus text dense search
       Milvus multimodal dense search
       Milvus BM25 sparse search
  -> RRF 融合排序
  -> Sentence Window 上下文替换与去重
  -> Answerability Gate
  -> 基于引用生成答复片段
```

首期不实现多查询分解、HyDE、Step-back、Cross-Encoder、RankLLM、ColBERT、C-RAG Web 搜索、Text-to-SQL 或 GraphRAG。

### 9.6 RRF 与 Answerability Gate

重排策略只采用 `RRF`，不加入权重学习或模型二次排序：

```text
final_score(result) = sum(1 / (k + rank_in_each_retrieval_list))
```

证据可靠性不修改 RRF 分数，而由 `Answerability Gate` 在生成前判定：

- `authoritative_text` 可以独立支撑知识答复。
- 高置信 `vision_generated` 可以参与回答和引用。
- 低置信 `vision_generated` 不能单独支撑退款条件、赔付或商品事实结论，应要求补充信息或图片。
- `conversation_evidence` 只支撑当前会话中的图片事实，必须结合政策知识或业务工具后才能产生售后动作。
- 检索结果不足、冲突或依赖服务故障时不得生成无依据答复。

为了降低人工客服接管率，证据不足但可以补全的场景优先澄清，例如请求订单号、补拍照片或确认图片识别内容；只有补全仍不足、发生争议或用户明确要求时才转人工。

### 9.7 RAG 评估

首期仅以四项 RAG 质量指标作为检索与生成评估目标：

| 指标 | 含义 |
|---|---|
| `Faithfulness` 忠实度 | 回答是否基于检索上下文，不添加无证据结论 |
| `Answer Relevancy` 答案相关性 | 回答是否解决用户提出的问题 |
| `Context Recall` 上下文召回 | 应被检索到的相关证据是否被召回 |
| `Context Precision` 上下文精确 | 被召回的上下文中相关证据所占比例 |

人工客服接管率与标准售后自动受理率属于整体业务指标，不替代上述 RAG 指标；错误自动执行率和无依据回答必须作为安全约束持续观察。

## 10. API 与页面设计

首期接口：

```text
POST /api/conversations
GET  /api/conversations/{id}
POST /api/conversations/{id}/messages
GET  /api/conversations/{id}/messages
GET  /api/conversations/{id}/runs

GET  /api/products
GET  /api/conversations/{id}/orders/{order_no}
GET  /api/conversations/{id}/shipments/{order_no}
GET  /api/conversations/{id}/tickets/{ticket_no}

POST /api/knowledge-bases
POST /api/knowledge-bases/{id}/documents
GET  /api/knowledge-bases/{id}/documents
GET  /api/knowledge-bases/{id}/chunks
POST /api/knowledge-bases/{id}/retrieval-tests
GET  /api/retrieval-logs

POST /api/demo/seed
GET  /health
```

约束：

- 本期不承诺兼容原演示接口。
- `POST /api/conversations` 返回一次性明文 `access_token`；服务端只保存哈希。
- 客户聊天页发送消息时支持文本与图片附件。
- 除 `GET /api/products`、知识管理演示接口、`POST /api/demo/seed` 与 `GET /health` 外，首期客户侧接口要求 `X-Conversation-Token`，并校验令牌属于路径中的会话。
- 查询订单、物流和工单时还必须校验资源属于会话绑定的客户。
- `POST /api/demo/seed` 仅在 demo 配置下启用，并返回用于演示会话的测试客户引用；首期会话令牌不构成生产身份认证。
- Swagger 展示结构化请求、响应和确认动作示例。
- 知识管理接口首期属于演示管理入口；部署到非本地环境前必须补充正式管理员认证。

首期提供两个薄页面，页面不包含业务决策：

```text
知识库管理页
  -> 上传 Markdown / PDF / TXT / DOCX / 图片
  -> 查看 Canonical Markdown、图片解析结果、置信度、chunk 与索引状态
  -> 输入问题测试召回、引用和 Answerability

聊天演示页
  -> 创建或恢复演示会话
  -> 发送文本与图片
  -> 展示 Agent 回复、引用、图片解析摘要和待确认动作
  -> 确认或取消标准售后申请
  -> 显示是否进入人工处理
```

## 11. 持久化与基础设施

首期使用 SQLite + SQLAlchemy，并通过领域定义的接口访问数据：

```text
ConversationRepository
MessageRepository
AssetRepository
CatalogRepository
OrderRepository
ShipmentRepository
AfterSalesRepository
HandoffRepository
TicketRepository
KnowledgeRepository
RetrievalLogRepository
AgentRunRepository
ToolCallRepository
```

SQLite 表：

```text
customers
conversations
messages
assets
products
orders
shipments
knowledge_bases
knowledge_documents
knowledge_assets
knowledge_chunks
retrieval_logs
after_sales_requests
handoff_requests
tickets
agent_runs
tool_calls
```

设计边界：

- `infrastructure.repositories` 提供 SQLite 实现，业务层只面向 Repository 接口。
- `conversations` 保存会话访问令牌哈希而非明文；验证由 API dependency 统一处理。
- `LocalAssetStorage` 实现 `AssetStorage` 接口，首期把原始文档和图片保存到 `data/assets/knowledge/` 与 `data/assets/conversations/`；领域层只保存相对存储键和元数据，不直接操作文件路径。
- `knowledge_documents` 保存原始来源、Canonical Markdown 和转换状态；`knowledge_assets` 保存图片解析信息、证据等级、置信度和原始资产引用；`knowledge_chunks` 保存句子节点、窗口文本与 Milvus 标识。
- Milvus 承载文本 dense、多模态 dense 和 BM25 sparse 索引，并按元数据过滤后执行 RRF hybrid search。
- 演示数据由 `scripts/seed_demo_data.py` 幂等导入。
- `LLMProvider`、`VisionProvider`、`TextEmbeddingProvider` 与 `MultimodalEmbeddingProvider` 通过配置接入 HTTP 模型服务，测试使用 fake provider。
- 模型与 Milvus 的具体部署可在环境配置中替换；核心应用不绑定云厂商或本地模型进程。

## 12. 错误处理与降级

| 情况 | 系统行为 |
|---|---|
| 意图置信度不足 | Supervisor 回复澄清问题或提供人工入口 |
| 会话令牌缺失或不匹配 | 返回未授权，不执行 Agent 或业务查询 |
| 订单不存在或不属于会话客户 | 返回统一的不可查询提示，不泄露资源是否属于其他客户 |
| 工具参数无效 | 拒绝调用，写入失败 ToolCall，返回可纠正提示 |
| 文档转换或图片解析失败 | 保留原始资产及失败状态，管理页可查看失败原因，不建立可查询索引 |
| 图片解析置信度不足 | 在会话中请求补图或确认解析内容；不能据此直接承诺售后结论 |
| Embedding 或 Milvus 不可用 | 知识链路返回不可用状态，不以不完整关键词结果继续生成知识答复 |
| Repository/数据库失败 | 回滚本轮副作用，返回暂时无法处理或转人工提示 |
| LLM 不可用/输出不可解析 | 不伪造业务结果；只读查询可使用确定性模板回复，否则降级 |
| Guard 检测越权承诺或隐私风险 | 阻止候选回复，生成安全答复并记录 AgentRun 风险状态 |
| 用户未确认副作用操作 | 保留 pending action，仅返回确认提示 |

对外响应不暴露堆栈、SQL、模型提示内容或内部调用凭据。

## 13. 测试与验收

### 13.1 自动化测试

```text
tests/unit/
  RouterAgent 输出解析、多意图与风险信号
  SupervisorAgent 计划和 Specialist 顺序控制
  Specialist 工具请求及允许工具约束
  ToolRegistry 参数校验、授权与确认门禁
  ResponseGuardAgent 越权承诺与隐私拦截
  MarkdownHeaderTextSplitter 与 Sentence Window Metadata
  Query Rewrite 与 metadata filter 白名单校验
  RRF 融合和 Answerability Gate 证据规则

tests/integration/
  SQLite Repository 的读写与资源可见性
  会话访问令牌哈希保存与校验
  LocalAssetStorage 文档/图片存取
  文本文件 -> Canonical Markdown -> chunk -> Milvus 索引
  图片 -> VisionProvider -> 文本/多模态索引
  Milvus dense/BM25/multimodal hybrid retrieval + RRF
  Message -> AgentRun -> ToolCall 持久化
  物流查询并追加售后草稿的组合链路
  确认后提交售后并生成 Ticket

tests/api/
  知识库上传、转换状态、检索测试和引用结果
  创建会话和发送商品咨询
  上传商品图片或破损图片并获得证据化答复
  订单与物流查询
  无效会话令牌及跨客户资源查询拒绝
  退款请求 -> 待确认 -> 受理编号
  转人工请求 -> 待确认 -> 交接状态
  降级和未授权资源查询
```

所有 Agent、视觉与 Embedding 单元测试使用固定响应的 fake provider，不依赖公网服务或实际模型密钥。Milvus 集成测试使用隔离测试 collection；没有运行 Milvus 时可被测试环境显式跳过，但完整验收必须执行该组测试。

### 13.2 首期验收场景

| 场景 | 预期结果 |
|---|---|
| “这双跑鞋有没有 42 码？” | `ProductAgent` 查询演示商品/库存并基于数据回复 |
| “订单 O1001 到哪了？” | `OrderAgent`/`LogisticsAgent` 返回该客户可见物流事实 |
| 管理页上传售后政策 Markdown/PDF | 转换、分块、Milvus 索引成功，检索测试返回引用 |
| 管理页上传商品尺码图片 | 原图保留，视觉摘要可见，可由文本问题跨模态检索 |
| 用户上传开胶照片并申请退货 | 图片作为本次会话证据，政策由 `KnowledgeAgent` 引用，生成待确认售后动作 |
| “订单 O1001 一直没到，我要退货” | 多 Agent 顺序处理并创建待确认售后动作 |
| 用户确认上述申请 | `submit_after_sales` 执行，产生售后记录和工单号 |
| “转人工”并确认 | 建立人工交接请求，不由模型虚构已接入客服 |
| 模型尝试承诺立即退款 | `ResponseGuardAgent` 拦截或改写 |

### 13.3 质量指标

RAG 指标：

```text
Faithfulness
Answer Relevancy
Context Recall
Context Precision
```

整体业务指标：

```text
人工客服接管率
标准售后自动受理率
错误自动执行率
无依据回答率
```

人工接管率应尽可能低，但不能通过降低 `Answerability Gate` 或副作用确认门禁来获得。

## 14. 后续演进

首期完成后，按独立增量扩展：

1. 将 `LocalAssetStorage` 替换为 S3 兼容对象存储实现，并补充正式管理鉴权。
2. 增加真实电商/物流/工单 API 的 Repository 或 Tool handler 实现。
3. 在评测证明有效后，再引入高级检索策略或 GraphRAG，不把实验策略提前放入主链路。
4. 扩展语音输入与客服工作台。
5. 在只读查询场景内评估有限 ReAct 工具循环；退款、建单、转人工继续保持确认后执行。

## 15. 设计决策摘要

- 开发落点：`smart-cs-multi-agent/python-impl`。
- 工程方式：在现有目录内重建分层结构，不维护原演示接口兼容。
- 交付定位：可持续二次开发的后端骨架。
- 输入范围：首期支持文本和图片，语音仅预留内容类型。
- 业务范围：商品、订单、物流、售后/退款、转人工与工单。
- 执行架构：Supervisor 型多 Agent；`RouterAgent` 保持独立子 Agent，Supervisor 持有最终规划和调度权。
- RAG 主链路：Canonical Markdown + `MarkdownHeaderTextSplitter` + Sentence Window Metadata + Query Rewrite + Metadata Filter + Milvus 文本/多模态/BM25 召回 + RRF + Answerability Gate。
- 存储与工具：Repository 接口 + SQLite + `LocalAssetStorage` + Milvus；Agent 仅通过受控工具访问业务能力。
- 模型边界：配置化 HTTP Provider 接口，支持 LLM、视觉、文本 Embedding 与多模态 Embedding；测试使用 fake provider。
- 风险治理：副作用两阶段确认；最终回复统一经 `ResponseGuardAgent`。
- 人工参与策略：符合规则且经用户确认的标准售后自动受理，只有异常、争议、失败或明确要求才接入人工。
- 交付界面：REST API、Swagger、知识库管理页、聊天演示页、演示数据脚本与自动化测试。

## 16. 技术参考

- Milvus BM25 Function: <https://milvus.io/docs/bm25-function.md>
- Milvus Multi-Vector Hybrid Search: <https://milvus.io/docs/multi-vector-search.md>
