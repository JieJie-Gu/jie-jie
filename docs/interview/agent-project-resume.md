# 简历项目表述

## 电商客服多 Agent 后端

基于 Python、LangGraph 与 LangChain 构建电商客服学习项目：拆分独立 Router 与
Supervisor，通过鉴权工具完成订单查询和需确认的售后申请；以 Markdown、
Milvus dense + BM25 + RRF 实现有引用的政策检索；增加会话级图片证据处理和
低置信度转人工门禁。

## 可展开的技术点

- 使用 LangGraph `interrupt` 与 SQLite checkpoint 承载确认暂停和恢复。
- 为业务写操作增加客户归属校验、动作幂等约束与会话租约。
- 只允许应用生成元数据过滤表达式，知识答复返回 Markdown 章节引用。
- 将图片保存在会话资产目录；规则模式不作虚假视觉判断。

项目指标仅引用 [RAG 实际生成报告](../../python-impl/data/evaluation/latest_results.md)
中的四项结果，并同时说明报告采用的检索模式。
