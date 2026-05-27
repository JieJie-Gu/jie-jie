# 代码走读

## 启动入口

- `python-impl/src/smart_cs/main.py`：装配数据库、runtime、可选 RAG 和视觉能力，暴露 FastAPI。
- `python-impl/src/smart_cs/api/routers/conversations.py`：文本消息、图片消息、确认与工具审计接口。
- `python-impl/src/smart_cs/application/conversation_service.py`：HTTP 边界、图片资产保存与响应整形。

## 编排主线

1. `agents/router.py` 调用决策模型生成 `RouteAnalysis`，只识别意图和实体。
2. `agents/supervisor.py` 校验计划，强制写动作需要确认。
3. `agents/specialists.py` 调用授权工具或 `KnowledgeAgent`。
4. `agents/guardrails.py` 仅渲染工具结果和有证据的检索结果。
5. `application/agent_runtime.py` 构建 LangGraph，使用 `interrupt` 暂停待确认动作。
6. `tools/executor.py` 与 `infrastructure/repositories.py` 执行鉴权、租约和审计写入。

## 知识问答

- `rag/indexing.py`：将四个 Markdown 文件切为带邻句窗口的文档节点。
- `rag/vector_store.py`：以官方 Milvus integration 建立 dense + BM25 存储和连接。
- `rag/retrieval.py`：查询重写和类别过滤 allow-list。
- `agents/knowledge.py`：调用 RRF 检索并输出上下文和引用。
- `rag/evaluation.py` 与 `scripts/evaluate_rag.py`：计算四项评估指标并生成报告。

## 图片证据

- `domain/evidence.py`：视觉证据结构和可用性阈值。
- `infrastructure/assets.py`：限制 MIME、大小与会话目录的本地存储。
- `agents/vision.py`：LangChain 多模态消息适配；规则模式保守降级。

## 建议走读顺序

先运行 `tests/api/test_conversations.py` 理解确认闭环，再运行
`tests/api/test_knowledge_reply.py` 与 `tests/api/test_image_message.py`，最后对照
`python-impl/data/evaluation/latest_results.md` 解释评估边界。
