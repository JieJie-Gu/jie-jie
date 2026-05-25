# Python 电商客服多 Agent 核心重建设计

## 1. 背景与结论

项目目标是建设一个可持续二次开发的、多模态可扩展的电商客服多 Agent 系统。
首期不实现图片或语音能力，而是在文本客服闭环中建立稳定的工程边界，并为后续附件输入保留数据契约。

现有 `python-impl` 规模较小、工作树干净，适合作为 Python 实现落点；但当前代码属于演示骨架，不能直接作为长期内核：

- `Supervisor`、工具层、内存存储与 RAG 尚未形成一致的业务执行链路。
- 工单使用内存实现，缺少可替换 Repository 与可审计持久化。
- RAG 使用演示性哈希嵌入，示例数据偏金融而非电商。
- 没有测试体系、明确领域模型或未来多模态消息契约。

决定采用的方案是：**在 `smart-cs-multi-agent/python-impl` 内重建清晰的分层结构，保留 FastAPI、Supervisor 多 Agent 和 Tool/MCP 可扩展方向，不保持当前内部组织或 API 兼容性。**

## 2. 目标与首期范围

### 2.1 目标

- 构建职责清晰、可单元测试、可替换基础设施的 Python 后端骨架。
- 落实真实的 Supervisor 型多 Agent 执行链路，而非仅多个提示词文件。
- 提供电商客服关键业务闭环与副作用安全门禁。
- 保持未来图片、语音与真实电商系统接入的扩展空间。

### 2.2 首期实现范围

- 文本会话与消息 API。
- 商品咨询。
- 订单与物流查询。
- 售后/退款申请：用户确认后受理并创建工单。
- 转人工：用户确认后建立交接请求。
- SQLite 持久化、Repository 接口、演示数据脚本。
- 以会话访问令牌隔离演示会话以及订单、物流、工单读取，防止已建立会话之间越界读取。
- Swagger、自动化测试、AgentRun 与 ToolCall 审计记录。

### 2.3 首期明确不做

- 图片上传、视觉理解、语音转写和视频输入。
- 用户端聊天页面或人工客服工作台。
- 完整账号体系、后台角色管理或生产级身份认证；首期只提供会话作用域访问令牌。
- 对接真实电商、物流或 CRM API。
- 向量检索主链路、复杂知识库管理后台。
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
      guardrails.py

    tools/
      registry.py
      specifications.py
      handlers.py

    infrastructure/
      database.py
      repositories/
      llm.py
      knowledge.py
      observability.py

  tests/
    unit/
    integration/
    api/
  scripts/
    seed_demo_data.py
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
- `tools` 是 Agent 使用业务能力的唯一入口，并对风险动作实施权限与确认控制。
- `infrastructure` 提供 SQLite Repository、模型适配器、知识查询和日志实现，可被替换。

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
| `AfterSalesAgent` | 退货、退款和售后诉求处理 | `get_after_sales_policy`, `draft_after_sales`, `submit_after_sales` |
| `HandoffAgent` | 转人工申请与交接摘要生成 | `draft_handoff`, `confirm_handoff` |
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

## 6. 领域模型与状态

首期核心实体：

```text
Customer
Conversation
Message
AttachmentRef
Product
Order
Shipment
AfterSalesRequest
HandoffRequest
Ticket
KnowledgeArticle
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
```

`Message` 设计中包含 `attachments: list[AttachmentRef]`。首期所有消息均为文本且附件集合为空；后续添加 Vision Agent 或 ASR 时不需要重写会话契约和存储边界。

创建 `Conversation` 时生成不可预测的会话访问令牌，仅向调用方返回一次并在存储中保留哈希。除公开商品列表和健康检查外，客户侧会话及关联资源接口都要求该令牌。该机制只证明请求方持有当前演示会话，不验证其现实身份；首期服务不得作为公开生产客服入口部署，后续必须由正式登录与授权体系替代或补强。

## 7. 工具层与动作安全

### 7.1 工具分类

只读工具可以在计划允许的 Agent 中直接执行：

```text
search_products
get_product
get_order
get_shipment
get_after_sales_policy
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

转人工请求使用相同的确认原则。任何模型输出都不能替代确认状态校验。

## 8. 请求数据流

核心对话端点每轮执行：

```text
POST /api/conversations/{id}/messages
  -> 校验会话可接收消息
  -> 持久化用户 Message
  -> application 加载会话、最近消息、客户可见业务上下文
  -> SupervisorAgent 启动 AgentRun
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
  "pending_action": null
}
```

## 9. API 设计

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

POST /api/demo/seed
GET  /health
```

约束：

- 本期不承诺兼容原演示接口。
- `POST /api/conversations` 返回一次性明文 `access_token`；服务端只保存哈希。
- 除 `GET /api/products`、`POST /api/demo/seed` 与 `GET /health` 外，首期客户侧接口要求 `X-Conversation-Token`，并校验令牌属于路径中的会话。
- 查询订单、物流和工单时还必须校验资源属于会话绑定的客户。
- `POST /api/demo/seed` 仅在 demo 配置下启用，并返回用于演示会话的测试客户引用；首期会话令牌不构成生产身份认证。
- Swagger 展示结构化请求、响应和确认动作示例。
- 后续多模态通过消息请求扩展 `attachments` 字段，而不是引入独立对话执行路径。

## 10. 持久化与基础设施

首期使用 SQLite + SQLAlchemy，并通过领域定义的接口访问数据：

```text
ConversationRepository
MessageRepository
CatalogRepository
OrderRepository
ShipmentRepository
AfterSalesRepository
HandoffRepository
TicketRepository
AgentRunRepository
ToolCallRepository
```

SQLite 表：

```text
customers
conversations
messages
products
orders
shipments
knowledge_articles
after_sales_requests
handoff_requests
tickets
agent_runs
tool_calls
```

设计边界：

- `infrastructure.repositories` 提供 SQLite 实现，业务层只面向 Repository 接口。
- `conversations` 保存会话访问令牌哈希而非明文；验证由 API dependency 统一处理。
- 演示数据由 `scripts/seed_demo_data.py` 幂等导入。
- LLM 通过 provider 接口注入，测试使用 fake provider。
- 首期知识查询使用可测试的文章/结构化查找接口；向量检索可后续作为 `KnowledgeRepository` 的替代实现加入。

## 11. 错误处理与降级

| 情况 | 系统行为 |
|---|---|
| 意图置信度不足 | Supervisor 回复澄清问题或提供人工入口 |
| 会话令牌缺失或不匹配 | 返回未授权，不执行 Agent 或业务查询 |
| 订单不存在或不属于会话客户 | 返回统一的不可查询提示，不泄露资源是否属于其他客户 |
| 工具参数无效 | 拒绝调用，写入失败 ToolCall，返回可纠正提示 |
| Repository/数据库失败 | 回滚本轮副作用，返回暂时无法处理或转人工提示 |
| LLM 不可用/输出不可解析 | 不伪造业务结果；只读查询可使用确定性模板回复，否则降级 |
| Guard 检测越权承诺或隐私风险 | 阻止候选回复，生成安全答复并记录 AgentRun 风险状态 |
| 用户未确认副作用操作 | 保留 pending action，仅返回确认提示 |

对外响应不暴露堆栈、SQL、模型提示内容或内部调用凭据。

## 12. 测试与验收

### 12.1 自动化测试

```text
tests/unit/
  RouterAgent 输出解析、多意图与风险信号
  SupervisorAgent 计划和 Specialist 顺序控制
  Specialist 工具请求及允许工具约束
  ToolRegistry 参数校验、授权与确认门禁
  ResponseGuardAgent 越权承诺与隐私拦截

tests/integration/
  SQLite Repository 的读写与资源可见性
  会话访问令牌哈希保存与校验
  Message -> AgentRun -> ToolCall 持久化
  物流查询并追加售后草稿的组合链路
  确认后提交售后并生成 Ticket

tests/api/
  创建会话和发送商品咨询
  订单与物流查询
  无效会话令牌及跨客户资源查询拒绝
  退款请求 -> 待确认 -> 受理编号
  转人工请求 -> 待确认 -> 交接状态
  降级和未授权资源查询
```

所有 Agent 测试使用固定响应的 fake LLM provider，不依赖公网服务或实际模型密钥。

### 12.2 首期验收场景

| 场景 | 预期结果 |
|---|---|
| “这双跑鞋有没有 42 码？” | `ProductAgent` 查询演示商品/库存并基于数据回复 |
| “订单 O1001 到哪了？” | `OrderAgent`/`LogisticsAgent` 返回该客户可见物流事实 |
| “订单 O1001 一直没到，我要退货” | 多 Agent 顺序处理并创建待确认售后动作 |
| 用户确认上述申请 | `submit_after_sales` 执行，产生售后记录和工单号 |
| “转人工”并确认 | 建立人工交接请求，不由模型虚构已接入客服 |
| 模型尝试承诺立即退款 | `ResponseGuardAgent` 拦截或改写 |

## 13. 后续演进

首期完成后，按独立增量扩展：

1. 图片消息与对象存储，添加 `VisionAgent` 解析商品图、物流截图和破损图。
2. 将视觉摘要作为结构化观察交给现有 `SupervisorAgent` 与 Specialist，不改变业务副作用门禁。
3. 增加真实电商/物流/工单 API 的 Repository 或 Tool handler 实现。
4. 增加知识库向量检索与基于证据的回答引用。
5. 在只读查询场景内评估有限 ReAct 工具循环；退款、建单、转人工继续保持确认后执行。

## 14. 设计决策摘要

- 开发落点：`smart-cs-multi-agent/python-impl`。
- 工程方式：在现有目录内重建分层结构，不维护原演示接口兼容。
- 交付定位：可持续二次开发的后端骨架。
- 输入范围：首期文本，数据模型预留附件与内容类型。
- 业务范围：商品、订单、物流、售后/退款、转人工与工单。
- 执行架构：Supervisor 型多 Agent；`RouterAgent` 保持独立子 Agent，Supervisor 持有最终规划和调度权。
- 存储与工具：Repository 接口 + SQLite 演示实现；Agent 仅通过受控工具访问业务能力。
- 风险治理：副作用两阶段确认；最终回复统一经 `ResponseGuardAgent`。
- 交付界面：REST API、Swagger、演示数据脚本与自动化测试，无前端。
