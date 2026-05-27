# 七日掌握计划

项目目标是在一周内能够据实讲解一个多 Agent 电商客服后端，而不是背诵没有实现的
系统能力。每天都对应现有代码和可执行验证。

| 日程 | 学习重点 | 验证命令 |
| --- | --- | --- |
| Day 1 | 项目边界；Router 与 Supervisor 职责拆分 | `pytest tests/unit/test_router_supervisor.py -q` |
| Day 2 | 工具鉴权、SQLite 状态与审计 | `pytest tests/unit/test_tools.py -q` |
| Day 3 | LangGraph 中断、确认恢复、并发租约 | `pytest tests/integration/test_action_confirmation.py -q` |
| Day 4 | Markdown 标题切分和邻句窗口 | `pytest tests/unit/test_markdown_windows.py -q` |
| Day 5 | Milvus dense + BM25 + RRF 与查询过滤 | `pytest tests/integration/test_milvus_hybrid.py -q` |
| Day 6 | 图片证据和低置信度转人工 | `pytest tests/api/test_image_message.py -q` |
| Day 7 | 四项评估、架构复盘和简历表达 | `pytest tests -q` |

## 讲解主线

1. 为什么 Router 与 Supervisor 分开，以及写动作如何被确定性代码约束。
2. 为什么订单状态走工具、政策说明走 RAG。
3. 为什么选择 Markdown 窗口、混合检索和 RRF。
4. 为什么图片单独存储、低置信度只能转人工。
5. 报告只引用实际生成的四项评估结果，并说明运行模式。

## 环境备注

Milvus 混合检索需要 Standalone 服务。若当前用户没有 Docker socket 权限，可以运行
离线评估验证报告流程，但面试表述中必须说明它不是 Milvus 检索成绩。
