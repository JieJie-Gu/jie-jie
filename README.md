# Smart CS Multi-Agent

面向 AI 应用 / Agent 开发岗位学习与面试复盘的电商客服后端项目。项目实现
Supervisor 编排的多 Agent 工作流、文本知识 RAG、会话级售后图片证据处理，以及
需用户确认的售后提交流程。

## 已实现范围

- Python FastAPI API 与 LangGraph 状态工作流。
- 独立 `RouterAgent` 识别意图；`SupervisorAgent` 规划 specialist 顺序并汇总结果。
- `ProductAgent`、`OrderAgent`、`KnowledgeAgent`、`AfterSalesAgent` 与 `HandoffAgent`。
- Markdown 知识源，`MarkdownHeaderTextSplitter` 与 Sentence Window Metadata。
- Milvus dense + BM25 混合检索、RRF 排序、allow-list 元数据过滤与证据引用。
- 图片仅保存为当前会话售后资产；低置信度视觉证据进入需确认的转人工流程。
- SQLite 业务状态、LangGraph checkpoint、动作确认和 `ToolCall` 审计。

## 安全边界

- 售后申请与转人工申请均先生成草稿，只有用户确认后才调用提交工具。
- 订单状态来自授权工具，不从 RAG 文档回答。
- 规则模式不会解析图片内容；它只返回需要人工核验的证据结果。
- 这是学习型工程，不连接真实订单或退款系统，也不声明线上效果。
- RAG 报告见 [latest_results.md](./python-impl/data/evaluation/latest_results.md)；当前提交中的报告已明确标为离线基线，不能作为 Milvus 检索成绩。

## 快速验证

```bash
cd python-impl
conda run -n agent pytest tests -q
conda run -n agent python scripts/evaluate_rag.py --offline
```

需要验证 Milvus 混合检索时，先确保当前用户具有 Docker 权限：

```bash
docker compose up -d etcd minio standalone
cd python-impl
conda run -n agent python scripts/index_knowledge.py
conda run -n agent pytest tests/integration/test_milvus_hybrid.py -q
conda run -n agent python scripts/evaluate_rag.py
```

API 启动：

```bash
cd python-impl
conda run -n agent uvicorn smart_cs.main:app --app-dir src --host 0.0.0.0 --port 8000
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

后端启动：

```powershell
cd d:\LLM\smart-cs-multi-agent\python-impl
conda activate customer_service
pip install -e ".[demo,test]"
python scripts/seed_demo_data.py
uvicorn smart_cs.main:app --app-dir src --host 0.0.0.0 --port 8000
```

第二个 PowerShell 窗口启动 Gradio：

```powershell
cd d:\LLM\smart-cs-multi-agent\python-impl
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
