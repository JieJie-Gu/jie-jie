# 面试问答

## Router 与 Supervisor 为什么分开？

`RouterAgent` 无副作用地识别意图、实体和风险；`SupervisorAgent` 决定 specialist
调用顺序并汇总 guard 后的结果。模型不能直接提交写动作，确认节点和授权工具才有
提交权限。对应代码为 `agents/router.py`、`agents/supervisor.py` 和
`application/agent_runtime.py`。

## 为什么不用 RAG 查订单状态？

知识库用于政策和产品说明，订单状态属于受客户权限约束的业务事实。本项目通过
`tools/executor.py` 查询订单，而 `agents/knowledge.py` 只返回 Markdown 引用。

## 为什么采用 RRF？

稠密检索覆盖语义表达，BM25 覆盖商品名与政策关键词。RRF 基于排名融合，不要求把
两类分数校准到同一尺度。实现位于 `rag/vector_store.py` 与 `agents/knowledge.py`。

## 图片如何控制风险？

图片只保存在当前会话的资产目录，不进入知识索引。`VisionAgent` 输出结构化证据；
低置信度结果只能生成待确认的转人工草稿。规则模式明确不解析图片。

## 如何评价 RAG？

`scripts/evaluate_rag.py` 生成忠实度、答案相关性、上下文召回和上下文精确四项
结果。当前提交中的报告是离线基线；启动 Milvus 并不带 `--offline` 运行脚本后，
才可讨论混合检索表现。
