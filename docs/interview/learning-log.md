# 七日掌握记录

| 日程 | 能解释的主题 | 可执行验证 |
| --- | --- | --- |
| Day 1 | 项目边界、Router 与 Supervisor 职责 | `pytest tests/unit/test_router_supervisor.py -q` |
| Day 2 | 工具鉴权与 SQLite 业务状态 | `pytest tests/unit/test_tools.py -q` |
| Day 3 | LangGraph 中断确认和 API 流程 | `pytest tests/integration/test_action_confirmation.py -q` |
| Day 4 | Markdown 分块、窗口元数据与 Milvus 混合检索 | `python scripts/index_knowledge.py` |
| Day 5 | 图片证据、低置信度转人工 | `pytest tests/unit/test_vision_agent.py -q` |
| Day 6 | 四项 RAG 评估及结果边界 | `python scripts/evaluate_rag.py --offline` |
| Day 7 | 用架构图和问答完整复盘 | `pytest tests -q` |

## 面试陈述边界

只陈述代码已实现的功能和本机生成的评估报告；不陈述线上流量、业务提升、真实部署
或没有测量的效果。
