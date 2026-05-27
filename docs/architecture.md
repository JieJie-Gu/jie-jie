# 架构说明

## 目标与边界

该实现是可复盘的电商客服 Agent 后端：查询事实、检索政策、生成需确认的业务草稿，
并处理会话级图片证据。订单与售后工单使用 SQLite 演示数据；图片不进入知识索引。

## Agent 分工

| 组件 | 职责 | 是否产生写动作 |
| --- | --- | --- |
| `RouterAgent` | 无副作用识别意图、实体与风险 | 否 |
| `SupervisorAgent` | 决定 specialist 执行顺序并汇总 guard 后内容 | 否 |
| `ProductAgent` | 查询演示商品 | 否 |
| `OrderAgent` | 按客户权限读取订单 | 否 |
| `KnowledgeAgent` | 检索 Markdown 知识并返回引用 | 否 |
| `AfterSalesAgent` | 创建待确认售后草稿 | 草稿 |
| `HandoffAgent` | 创建待确认转人工草稿 | 草稿 |
| `VisionAgent` | 提取当前图片的结构化可见证据 | 否 |

## 文本工作流

```text
Input -> RouterAgent -> SupervisorAgent -> Specialist execution
      -> ResponseGuard -> Supervisor synthesis
      -> [draft action?] interrupt confirmation -> authorized submission -> ResponseGuard
```

`AgentRuntime` 使用 LangGraph checkpoint 保存中断状态。`AuthorizedToolExecutor` 与
SQLite 仓储约束客户归属、动作状态和活动会话租约，避免确认重放或并发重复提交。

## RAG 链路

```text
Markdown files
  -> MarkdownHeaderTextSplitter
  -> sentence nodes + neighbor window metadata
  -> local dense embedding + Milvus built-in BM25
  -> metadata allow-list filter + RRF
  -> KnowledgeAgent answer + context citations
```

查询增强仅包含确定性查询重写和固定类别过滤表达式。用户输入不会直接成为 Milvus
表达式。四项验收指标由 `scripts/evaluate_rag.py` 生成：忠实度、答案相关性、
上下文召回、上下文精确。

## 图片路径

```text
multipart image -> LocalAssetStorage(conversation scope)
                -> VisionAgent structured evidence
                -> usable evidence: existing after-sales flow
                -> uncertain evidence: existing handoff draft flow
                -> interrupt confirmation
```

规则模式始终把图片判为需要核验；配置支持多模态 chat model 后才能提取可用证据。
图片资产保存在 `data/assets/{conversation_id}/`，不会写入 Milvus。

## 可验证边界

`ToolCall` 审计记录已实现。当前版本没有持久化独立的 Agent 执行轨迹表，也没有真实
外部订单系统、真实退款动作或网页界面。
