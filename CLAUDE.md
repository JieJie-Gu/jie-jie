# CLAUDE.md

本文件为 claude code 在本仓库中工作时提供指引。

## 项目概览

电商客服多 Agent 后端。本项目采用python实现。主体实现在 `python-impl/`；`go-impl/` 和 `java-impl/` 可以忽略不用看。`docs/` 目录包含架构文档、面试准备材料，以及 `superpowers/` 下的近期工作 spec/plan。

## 开发准则

采用python语言，利用langchain，langgraph等python包，能直接调包部分直接调包，确保代码简洁可读性高，不要重复造轮子。

## 常用命令

除非特别说明，所有命令在 `python-impl/` 目录下执行。Windows 下使用 conda 环境 `customer_service`（Python 3.11+）。

```bash
cd python-impl

# 安装（可编辑模式 + 测试依赖）
pip install -e ".[test]"

# 灌入演示数据（商品、客户、订单、知识库文档）
python scripts/seed_demo_data.py

# 启动服务
uvicorn smart_cs.main:app --app-dir src --host 0.0.0.0 --port 8000

# 运行全部测试
pytest -q

# 运行单个测试文件
pytest tests/unit/test_router_supervisor.py -q

# 运行单个测试用例
pytest tests/unit/test_router_supervisor.py::test_router_classifies_intent -q

# 只跑 unit + API 测试（跳过需要 Milvus 的 integration 测试）
pytest tests/unit tests/api -q

# RAG 评估（需 Milvus 已启动；加 --offline 做离线评估）
python scripts/evaluate_rag.py
python scripts/evaluate_rag.py --offline
```

Docker（从仓库根目录执行）：
```bash
docker compose up -d           # 启动 Milvus 全家桶（etcd + minio + milvus）
docker compose up python-agent  # 构建并运行 Agent 容器
```

## 架构

### Agent Supervisor 模式

核心工作流定义在 `application/agent_runtime.py`，是一个 **LangGraph StateGraph**：

```
输入 → RouterAgent → SupervisorAgent → Specialist(s) → ResponseGuard → 合成回复
```

`agents/` 下的关键 Agent：
- **RouterAgent** — 意图分类（`product|order|knowledge|after_sales|handoff`）、实体提取、风险评估。
- **SupervisorAgent** — 规划需要调用哪些 Specialist，从 Specialist 结果合成最终回复。
- **SpecialistDispatcher**（`specialists.py`）— 执行 Supervisor 的计划，分发到具体 Agent（ProductAgent、OrderAgent、KnowledgeAgent、AfterSalesAgent、HandoffAgent、VisionAgent）。
- **ResponseGuard**（`guardrails.py`）— 将业务结果渲染为受控回复内容；绝不编造信息。
- **KnowledgeAgent** — 基于 Milvus 混合检索（dense + BM25）的 RAG。仅在 `SMART_CS_RAG_ENABLED=true` 时激活。

### 两种模型模式

通过 `SMART_CS_MODEL_MODE` 配置：
- **`rules`** — 确定性规则决策，无需 LLM。适合演示和测试。
- **`llm`** — 使用 LangChain ChatModel（通过 `configured_chat_model()`）。需要设置 `SMART_CS_LLM_*` 环境变量。

决策模型同时实现 `RoutingDecisionModel` 和 `PlanningDecisionModel` 两个 Protocol。

### 安全模式：先草稿、后确认

写操作（售后、转人工）先创建 `PendingAction` 记录。LangGraph `interrupt()` 暂停执行；用户必须通过 `/confirm` API 显式批准，由 `Command(resume=...)` 恢复执行。用户也可以拒绝取消。

### 分层结构（`python-impl/src/smart_cs/`）

| 层 | 包 | 职责 |
|---|---|---|
| API | `api/` | FastAPI 路由、请求/响应 schema |
| Application | `application/` | `AgentRuntime`（图构建、轮次租约、checkpoint）、`ConversationService` |
| Domain | `domain/` | SQLAlchemy ORM 模型、枚举、异常、仓库协议 |
| Infrastructure | `infrastructure/` | 数据库、模型工厂、资源存储、仓库实现 |
| Agents | `agents/` | Router、Supervisor、Specialist、Guardrails、状态 schema |
| Tools | `tools/` | LangChain `@tool` 装饰器、`AuthorizedToolExecutor` |
| RAG | `rag/` | 向量嵌入、Milvus 向量库、索引、检索、评估 |

### 关键设计决策

- **AuthorizedToolExecutor** 是调用业务工具的唯一入口。所有工具调用都有审计记录（ToolCall 表）。写操作工具需要 `TurnFence`（5 分钟租约 + 心跳）。
- **RuntimeState**（`agents/state.py`）是 LangGraph 的 `TypedDict` — 所有图节点读写此共享状态。
- **Pydantic schema**（`RouteAnalysis`、`SupervisorDecision`）集中在 `agents/state.py`。
- 所有配置通过 pydantic-settings 使用 `SMART_CS_` 前缀（`config.py`）。

### 测试

`tests/` 下三层测试：
- `unit/` — 路由、主管、工具、视觉、RAG（无外部依赖）
- `api/` — 使用 httpx `TestClient` 的 HTTP 级别测试
- `integration/` — 需要后端服务（Milvus 测试在服务不可用时自动跳过）

标记：需要 Milvus 的测试使用 `@pytest.mark.integration`。

### 进行中的计划（`docs/superpowers/`）

`plans/` 和 `specs/` 目录包含近期 P0 工作的实现计划。处理 Agent 工程相关任务时请查阅 — 它们定义了 schema 扩展、prompt 分层、消息裁剪、工具策略、状态管理的目标架构。
