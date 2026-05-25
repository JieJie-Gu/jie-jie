# Python 电商客服多 Agent 面试型工程设计

## 1. 项目定位

本项目定位为：**面向电商客服场景的多模态多 Agent 决策系统**，用于展示 AI 应用 / Agent 开发工程师岗位相关的核心能力：

- Supervisor 多 Agent 编排与职责划分。
- 受控工具调用与有副作用动作的确认门禁。
- 基于 Milvus 的文本 RAG 检索、引用生成与质量评估。
- 售后场景中的用户图片证据理解与多 Agent 协作。
- 可追踪的执行日志、测试和设计取舍。

项目不以面试现场演示或建设完整客服产品为目标。首期实现必须满足以下约束：

- 一周内能够理解代码、运行主要场景、解释架构决策和复述关键实现。
- 简历陈述必须与实际代码、测试和评估结果一致，不使用未测量的准确率或性能数字。
- 功能数量服从可解释性，保留面试可讨论的主链路，移除不必要的页面和产品化范围。

现有 `smart-cs-multi-agent/python-impl` 是实现落点。当前演示骨架中的伪 RAG、内存业务状态和未连接的 Router 不作为新内核保留，但 FastAPI、Supervisor 图编排与工具扩展方向可以延续。

## 2. 方案选择

考虑过三种首期范围：

| 方案 | 范围 | 取舍 |
|---|---|---|
| 文本最小版 | 多 Agent + 文本 RAG + 工具调用 | 最容易掌握，但不能体现原定的多模态特色 |
| 平衡版 | 多 Agent + 文本 RAG + 用户售后图片证据理解 + 评估脚本 | 保留差异化场景，仍可在一周内掌握 |
| 完整产品版 | 图片知识库、跨模态向量、文件上传转换、前端页面与产品化设施 | 范围过宽，面试中难以深入解释 |

采用 **平衡版**。

首期的“多模态”明确指：用户在售后对话中上传商品问题图片，系统提取结构化证据并辅助售后决策。首期不把图片存入知识库，不实现图文跨模态向量检索。

## 3. 首期范围

### 3.1 必须完成

- 文本消息和单张售后图片消息处理。
- 商品咨询、订单/物流查询、售后退换申请、异常或明确请求下的人工交接。
- Supervisor 型多 Agent 调度，`RouterAgent` 作为独立分析 Agent。
- 标准售后动作的草稿、用户确认、自动提交与业务追踪单创建。
- Markdown 知识库的文本 RAG：分块、Milvus 混合检索、RRF、引用和可回答性判断。
- `VisionAgent` 对用户会话图片的结构化证据提取。
- SQLite 业务持久化、LocalAssetStorage 图片保存、AgentRun/ToolCall 审计记录。
- 演示数据脚本、RAG 评估脚本、关键自动化测试。
- 面向面试复习的架构说明、设计决策和实测结果记录。

### 3.2 明确不做

- 知识库管理网页、聊天演示网页或人工客服工作台。
- PDF、DOCX 等文件上传转换流程；首期知识材料直接维护为 Markdown。
- 知识库图片、图片向量索引、图文共享向量或图片检索。
- 语音、视频和 OCR 文档导入流水线。
- 真实电商、物流、支付或 CRM API 接入。
- 账号体系、生产鉴权、部署扩缩容和运营后台。
- GraphRAG、Text-to-SQL、HyDE、Cross-Encoder、ColBERT、Web 搜索校正等扩展检索策略。
- 自主循环执行退款、赔付、建单或转人工等副作用。

这些内容可以作为后续扩展方向讨论，但不得写成首期已经实现的能力。

## 4. 可交付成果

首期交付不是页面，而是可验证、可讲述的工程资产：

```text
1. 可运行的 FastAPI 后端接口与 Swagger 调试入口
2. 多 Agent 核心执行链路及工具安全门禁
3. Markdown 测试知识库和电商演示业务数据
4. Milvus 文本混合检索及引用返回
5. 用户图片售后证据处理链路
6. tests/ 自动化测试
7. scripts/evaluate_rag.py 评估输出
8. 架构说明、简历表述草稿与面试问题清单
```

接口和 Swagger 用于本地验证实现，不作为面试现场演示承诺。

## 5. 架构总览

系统采用 **Supervisor 型多 Agent 架构**。`SupervisorAgent` 是最终编排者，`RouterAgent` 是独立的意图与风险分析子 Agent；两者不得合并。

```text
User Message (+ optional after-sales image)
  -> ConversationService
  -> SupervisorAgent
       -> VisionAgent            # 仅当本轮包含图片
       -> RouterAgent            # 分析意图、实体、风险、建议 Agent
       -> SupervisorAgent.plan   # 决定实际执行步骤和工具权限
            -> ProductAgent
            -> OrderAgent        # 同时处理物流查询
            -> KnowledgeAgent
            -> AfterSalesAgent
            -> HandoffAgent      # 仅在满足接入条件时
       -> ResponseGuardAgent
       -> SupervisorAgent.synthesize
  -> Assistant Reply + citations + pending action + audit summary
```

核心边界：

- 知识回答依赖 RAG 证据。
- 订单、物流、售后进度等实时业务事实依赖工具，不由 RAG 推断。
- 用户图片只作为当前会话证据，不能作为公共知识事实。
- 副作用动作必须在用户确认后由受控工具执行。
- 人工接入应作为异常路径，而不是标准售后默认路径。

## 6. Agent 职责

| Agent | 核心职责 | 可用能力 |
|---|---|---|
| `SupervisorAgent` | 理解用户请求；调用 Router；制定多步计划；授权工具；汇总最终答案 | 调度与汇总，不直接写业务数据 |
| `RouterAgent` | 输出意图、实体、置信度、风险信号和建议 Agent | 无副作用工具，无最终派发权 |
| `ProductAgent` | 回答商品基本信息、规格和演示库存问题 | `search_products`, `get_product` |
| `OrderAgent` | 查询订单状态和物流轨迹 | `get_order`, `get_shipment` |
| `KnowledgeAgent` | 检索售后政策、FAQ、商品说明，生成带引用的证据片段 | `retrieve_knowledge` |
| `VisionAgent` | 对用户上传的售后图片提取现象、部位、可见问题和置信度 | 读取当前消息图片，无业务副作用 |
| `AfterSalesAgent` | 综合订单事实、政策引用和图片证据，生成或提交售后申请 | `draft_after_sales`, `submit_after_sales` |
| `HandoffAgent` | 在争议、失败、证据不可判定或用户明确要求时生成交接记录 | `draft_handoff`, `confirm_handoff` |
| `ResponseGuardAgent` | 校验引用、承诺、隐私和未确认副作用风险 | 审查，不执行业务动作 |

`OrderAgent` 首期同时承担物流查询，避免为了概念完整而增加一个仅转发工具的 `LogisticsAgent`。后续若物流异常处理形成独立复杂流程，再拆分为专业 Agent。

## 7. 关键业务流程

### 7.1 政策知识问答

```text
用户：退货需要在几天内申请？
  -> SupervisorAgent
  -> RouterAgent: knowledge_query
  -> KnowledgeAgent: 查询重写、元数据过滤、Milvus 检索、RRF
  -> ResponseGuardAgent: 校验回答被引用支持
  -> 返回政策结论和引用章节
```

### 7.2 订单及物流查询

```text
用户：我的订单 O1001 到哪了？
  -> RouterAgent: order_query + logistics_query
  -> OrderAgent: get_order + get_shipment
  -> ResponseGuardAgent
  -> 返回工具获得的实时事实
```

该流程不得使用知识库文档猜测具体订单状态。

### 7.3 图片售后与自动受理

```text
用户上传开胶照片，并请求退货
  -> VisionAgent: evidence = {issue: "sole_detachment", confidence: 0.91}
  -> RouterAgent: after_sales_request
  -> OrderAgent: 校验订单及可售后状态
  -> KnowledgeAgent: 检索质量问题退货政策
  -> AfterSalesAgent: draft_after_sales
  -> ResponseGuardAgent: 检查证据、政策和待确认状态
  -> 回复售后摘要并要求用户确认

用户确认
  -> SupervisorAgent 校验 pending action
  -> AfterSalesAgent: submit_after_sales
  -> 创建 AfterSalesRequest 与 Ticket
  -> 返回受理编号
```

图片识别置信度低、订单不匹配或证据与用户描述冲突时，系统优先要求补充信息或补拍图片，而不是立即转人工。

### 7.4 人工交接

仅以下场景进入 `HandoffAgent`：

- 用户明确要求人工客服。
- 规则未覆盖或产生敏感争议。
- 图片/业务/政策证据补充后仍冲突。
- 工具或模型服务失败导致无法安全继续。

标准售后符合规则且用户确认后自动受理，以降低不必要的人工参与率。

## 8. 领域状态与数据模型

首期核心实体：

```text
Customer
Conversation
Message
Asset
Product
Order
Shipment
AfterSalesRequest
HandoffRequest
Ticket
KnowledgeDocument
KnowledgeChunk
RetrievalLog
AgentRun
ToolCall
```

关键状态：

```text
Conversation:
  active | pending_confirmation | pending_handoff | closed

AfterSalesRequest:
  draft | pending_confirmation | submitted | rejected | completed

HandoffRequest:
  draft | confirmed | closed

Ticket:
  open | processing | resolved | closed

Message.content_type:
  text | image | mixed

EvidenceLevel:
  authoritative_text | conversation_evidence
```

边界说明：

- `authoritative_text` 是维护在 Markdown 知识库中的政策、FAQ 或商品说明，可支撑知识结论。
- `conversation_evidence` 是用户本轮上传图片经 `VisionAgent` 得到的结构化证据，只服务当前会话和售后申请。
- 图片原文件保存到本地资产目录，数据库只记录存储键、类型、校验值和与消息的关系。

## 9. 工具调用与安全门禁

### 9.1 工具集合

只读工具：

```text
search_products
get_product
get_order
get_shipment
retrieve_knowledge
get_ticket
```

创建待确认动作的工具：

```text
draft_after_sales
draft_handoff
```

仅在用户显式确认后执行的副作用工具：

```text
submit_after_sales
confirm_handoff
```

### 9.2 执行规则

`ToolRegistry` 必须执行以下约束：

- 工具调用只能来自本轮 Supervisor 计划授权的 Agent。
- 订单和物流工具必须校验资源属于当前演示客户上下文。
- `submit_after_sales` 与 `confirm_handoff` 只能消费与当前会话匹配且用户已确认的 pending action。
- 每次工具调用记录工具名、调用 Agent、输入摘要、结果状态、耗时和错误类型。
- 模型输出不能绕过持久化状态检查直接宣称退款成功、已建单或已接入人工。

首期是本地学习项目，不实现生产用户身份认证。API 通过演示会话绑定的 `customer_id` 执行业务资源隔离，并在文档中明确该机制不能作为生产认证方案。

## 10. 文本 RAG 设计

### 10.1 使用边界

RAG 只处理稳定、可维护、可引用的文本知识：

```text
售后政策
配送规则
FAQ
商品规格说明
```

RAG 不处理订单状态、物流轨迹、库存变化、售后处理进度或用户上传图片。图片证据由 `VisionAgent` 直接输出给当前工作流消费。

### 10.2 数据准备

首期知识材料直接以 Markdown 文件维护：

```text
data/knowledge/
  after_sales_policy.md
  shipping_policy.md
  product_guide.md
  faq.md
```

每个文档保留以下可过滤元数据：

```text
document_id
title
category
product_type
status
source_path
section_path
```

首期不建设上传和格式转换流水线。未来接入 PDF 或其他文本文件时，仍应先转换为相同的 Canonical Markdown 契约，再进入以下检索链路。

### 10.3 分块策略

分块仅采用已确认的两步策略：

```text
MarkdownHeaderTextSplitter
  -> 按 Markdown 标题保留章节结构和 section_path

Sentence Window Metadata
  -> 在章节内建立句子级检索节点
  -> 为每个节点保存前后句窗口上下文
```

检索匹配句子节点，生成回答前用窗口上下文替换命中句，以降低主题稀释和断章取义。

### 10.4 Milvus 混合检索

首期只需要文本 Embedding Provider，不引入多模态 Embedding Provider：

```text
LLMProvider
  -> Router、Supervisor、查询重写、回答生成及 Guard

VisionProvider
  -> 用户售后图片证据提取

TextEmbeddingProvider
  -> Markdown 节点和文本 query 的 dense vector

VectorStore
  -> Milvus 实现
```

Milvus collection 支持：

```text
text_dense_vector   # 文本语义召回
text_sparse_vector  # BM25 关键词召回
raw_text
window_text
metadata fields     # category、product_type、status、section_path 等
```

Milvus 与 Embedding 是知识问答链路的必需依赖。依赖不可用时，应返回知识服务暂时不可用，不使用不完整的临时检索结果生成政策结论。

### 10.5 查询增强与排序

查询增强仅实现：

```text
Query Rewrite
  -> 把口语化问题改写成简洁检索 query

Metadata Filter
  -> 基于允许字段生成过滤条件
```

元数据过滤字段和值必须经过白名单校验，模型不得直接生成任意数据库表达式。

检索流程：

```text
用户问题
  -> Query Rewrite
  -> Metadata Filter
  -> Milvus dense search + BM25 sparse search
  -> RRF 融合
  -> Sentence Window 上下文替换
  -> Answerability Gate
  -> 基于引用生成回答
```

重排只使用 RRF：

```text
final_score(result) = sum(1 / (k + rank_in_each_result_list))
```

### 10.6 Answerability Gate

回答生成前进行证据判断：

- 检索到可支撑结论的 `authoritative_text` 时，才生成政策或说明答案并附引用。
- 证据不足时，回复缺少的信息或建议进一步确认，不虚构政策。
- 售后决策必须同时具有业务工具事实和政策依据；涉及图片时，还需要可接受置信度的会话证据。
- 检索结果冲突、依赖故障或风险动作未确认时，不输出已经执行的结论。

### 10.7 评估

`scripts/evaluate_rag.py` 使用固定的电商问答评测集，记录每次实际运行结果。首期只报告四项指标：

| 指标 | 含义 |
|---|---|
| `Faithfulness` 忠实度 | 回答是否得到检索上下文支持 |
| `Answer Relevancy` 答案相关性 | 回答是否解决输入问题 |
| `Context Recall` 上下文召回 | 标注相关证据是否被检索到 |
| `Context Precision` 上下文精确 | 返回上下文中相关证据的占比 |

简历只能填写实际评估得到并可复现的结果。若尚未执行评估，应表述为“构建评估链路”，不能声明已达到某个数字。

## 11. 图片证据处理

图片链路刻意保持简单，服务售后 Agent 协作而不是展示图像检索能力：

```text
POST conversation message + image
  -> LocalAssetStorage 保存图片
  -> VisionAgent / VisionProvider
       visible_issue
       affected_part
       extracted_text (optional)
       evidence_summary
       confidence
       needs_clarification
  -> 保存为当前 Message 的 conversation_evidence
  -> AfterSalesAgent 消费该证据
```

示例结构化输出：

```json
{
  "visible_issue": "sole_detachment",
  "affected_part": "right_shoe_forefoot",
  "evidence_summary": "鞋底前掌边缘可见分离",
  "confidence": 0.91,
  "needs_clarification": false
}
```

安全约束：

- 图片识别内容不能独立决定退款、赔付或商品真伪。
- 低置信或不清晰图片先要求补拍或用户确认识别内容。
- 图片不会自动进入公共知识库或向量索引。
- 测试使用固定结果的 fake vision provider，真实模型接入由配置替换。

## 12. 工程结构

目标目录：

```text
python-impl/
  pyproject.toml
  data/
    knowledge/
    assets/conversations/
    evaluation/
  src/smart_cs/
    main.py
    config.py
    api/
      routers/
        conversations.py
        operations.py
      schemas.py
      dependencies.py
    domain/
      models.py
      enums.py
      repositories.py
      errors.py
    application/
      conversation_service.py
      knowledge_service.py
      agent_runtime.py
      dto.py
    agents/
      state.py
      supervisor.py
      router.py
      product.py
      order.py
      knowledge.py
      vision.py
      after_sales.py
      handoff.py
      guardrails.py
    tools/
      registry.py
      specifications.py
      handlers.py
    rag/
      indexing.py
      sentence_window.py
      retrieval.py
      answerability.py
      types.py
    infrastructure/
      database.py
      repositories/
      assets.py
      providers.py
      milvus.py
      observability.py
  tests/
    unit/
    integration/
    api/
  scripts/
    seed_demo_data.py
    index_knowledge.py
    evaluate_rag.py
```

依赖方向：

```text
api -> application -> domain
agents -> domain + tools interfaces + rag interfaces
tools handlers -> domain repository interfaces
rag -> provider/vector store interfaces
infrastructure -> domain/rag interfaces
main/config -> application + infrastructure 装配
```

约束：

- `domain` 不依赖 FastAPI、LangGraph、SQLAlchemy 或具体模型 SDK。
- `application` 管理一次消息处理的事务、Agent 调用与日志保存。
- `agents` 不直接访问数据库或 Milvus，只调用受控接口。
- `rag` 不读取实时订单、物流或售后状态。
- `infrastructure` 承载 SQLite、Milvus、本地资产与可替换 Provider 实现。

## 13. API 与基础设施

首期提供最小 API：

```text
POST /api/conversations
POST /api/conversations/{id}/messages       # 支持可选图片
GET  /api/conversations/{id}/messages
GET  /api/conversations/{id}/runs
POST /api/conversations/{id}/actions/confirm
GET  /health
```

知识数据准备通过脚本完成，而不是管理页面或管理 API：

```text
python scripts/seed_demo_data.py
python scripts/index_knowledge.py
python scripts/evaluate_rag.py
```

基础设施：

| 能力 | 首期实现 |
|---|---|
| 业务持久化 | SQLite + Repository 接口 |
| 图片保存 | `LocalAssetStorage`，仅保存会话图片 |
| 知识向量库 | Milvus，文本 dense + BM25 sparse |
| 模型调用 | 配置化 `LLMProvider`、`VisionProvider`、`TextEmbeddingProvider` |
| 测试替身 | fake providers 和隔离测试数据 |
| 可观测性 | `AgentRun`、`ToolCall`、`RetrievalLog` |

首期本地运行，不承诺公网部署安全。未来若提供公开服务，必须增加正式身份认证、管理员权限和资产访问保护。

## 14. 测试与验收

### 14.1 自动化测试

```text
tests/unit/
  RouterAgent 意图、实体与风险结构化输出
  SupervisorAgent 多步计划与授权工具集合
  ToolRegistry 的参数校验与确认门禁
  ResponseGuardAgent 对无引用结论和越权承诺的拦截
  MarkdownHeader + Sentence Window 处理
  Metadata Filter 白名单与 RRF 融合
  Answerability Gate 判定

tests/integration/
  SQLite Repository 与会话业务数据隔离
  Markdown -> chunk -> Milvus dense/BM25 索引和检索
  Conversation image -> VisionProvider -> conversation evidence
  售后草稿 -> 用户确认 -> 售后单与 Ticket
  AgentRun、ToolCall 和 RetrievalLog 持久化

tests/api/
  商品咨询
  订单/物流查询
  政策问题带引用回答
  上传破损图片形成待确认售后申请
  确认后自动受理
  请求转人工
  知识或模型依赖不可用时的安全回复
```

### 14.2 验收场景

| 场景 | 应证明的能力 |
|---|---|
| “七天无理由退货有哪些条件？” | 文本 RAG、RRF、引用和忠实回答 |
| “订单 O1001 到哪里了？” | 实时事实通过工具而非 RAG 查询 |
| “订单没到，我想退款” | 多 Agent 顺序协调与副作用确认 |
| 上传鞋底开胶图片并申请退货 | 图片证据、政策证据和售后工具协作 |
| 用户确认售后申请 | 自动受理并产生 Ticket，不默认转人工 |
| 用户明确要求人工 | 有控制的 Handoff 流程 |
| 模型输出无依据退款承诺 | Guard 拦截 |

验收完成后记录实际测试和 RAG 评估结果，为简历和面试陈述提供可核查依据。

## 15. 一周掌握路径

实施和学习以能够解释为终点，而不是仅完成代码：

| 天数 | 实现重点 | 必须能说明的问题 |
|---|---|---|
| 第 1 天 | 领域模型、目录结构、演示数据和 SQLite Repository | 为什么领域、应用、基础设施分层 |
| 第 2 天 | Supervisor、Router、状态与简单 Specialist | Router 为什么独立但不拥有调度权 |
| 第 3 天 | ToolRegistry、订单/物流查询、副作用确认 | Agent 如何安全调用真实业务动作 |
| 第 4 天 | Markdown 分块、Milvus dense/BM25、RRF 与引用 | RAG 与实时工具数据如何分流 |
| 第 5 天 | VisionAgent 与图片售后流程 | 多模态如何服务决策而不扩大幻觉风险 |
| 第 6 天 | Guard、测试、日志和 RAG 评估 | 如何验证答案可信与动作安全 |
| 第 7 天 | 重跑场景、整理实测结果、准备简历和面试问答 | 架构取舍、缺陷和后续演进 |

一周结束时，应能从入口追踪一次请求的 Agent、工具、检索、状态变化和安全判断，而不是只会运行接口。

## 16. 简历与面试陈述边界

### 16.1 可陈述能力

完成首期并取得实测结果后，可以围绕以下内容形成简历描述：

```text
设计并实现电商客服 Supervisor 多 Agent 系统，将意图路由、知识检索、
订单工具调用、图片售后证据分析与安全审查拆分为可追踪执行链路；
使用 Milvus dense + BM25 混合检索与 RRF 融合构建带引用的 Markdown
知识问答，并通过 Faithfulness、Answer Relevancy、Context Recall、
Context Precision 评估结果验证效果；对退换货等副作用采用用户确认
门禁与审计记录。
```

其中任何“提升百分比”“准确率”或“降低人工率”的数字，只有在脚本实际产出、评测集定义清楚并可复跑时才能补入。

### 16.2 应准备的面试问题

- 单 Agent + tools 与 Supervisor 多 Agent 的取舍是什么？
- `RouterAgent` 与 `SupervisorAgent` 为什么分离？
- RAG 为什么只用于政策知识，而不用于订单和物流状态？
- 为什么选 MarkdownHeader + Sentence Window，而没有采用更多分块算法？
- 为什么采用 dense + BM25 + RRF，而不是更复杂重排？
- 图片证据为什么不直接触发退款？
- 如何降低人工参与率，同时防止错误自动执行？
- 如何评估 RAG，如何避免在简历中夸大效果？

## 17. 后续演进

首期掌握后再按实际需要增加：

1. 文本文件到 Canonical Markdown 的导入转换。
2. 真实电商与物流 API 工具适配。
3. 正式认证与可公开部署的管理能力。
4. 图片知识库和跨模态检索，仅在业务需要与评估证明收益后引入。
5. 更复杂检索或语音输入，不提前放入首期主链路。

## 18. 决策摘要

- 开发落点：`smart-cs-multi-agent/python-impl`。
- 岗位定位：AI 应用 / Agent 开发工程师面试型项目。
- 时间边界：首期一周内可以理解、运行、测试并完成面试复习。
- 架构：Supervisor 型多 Agent；`RouterAgent` 独立提供分析建议，Supervisor 最终调度。
- 多模态范围：只处理用户售后图片证据，不做图片知识库或跨模态检索。
- RAG：Markdown + `MarkdownHeaderTextSplitter` + Sentence Window Metadata + Query Rewrite + Metadata Filter + Milvus dense/BM25 + RRF + Answerability Gate。
- 工具安全：副作用草稿与用户确认后执行；标准售后自动受理，异常场景才转人工。
- 交付：后端 API、Swagger、脚本、测试、评估结果与面试材料，不提供前端页面。
- 陈述原则：只写已实现、已测试和已测量的能力与指标。

## 19. 技术参考

- Milvus BM25 Function: <https://milvus.io/docs/bm25-function.md>
- Milvus Multi-Vector Hybrid Search: <https://milvus.io/docs/multi-vector-search.md>
