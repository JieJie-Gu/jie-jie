# 测试 RAG 服务命名和知识问答服务兼容行为。

from __future__ import annotations

from smart_cs.agents.knowledge import KnowledgeAgent, KnowledgeService


def test_knowledge_agent_alias_points_to_knowledge_service() -> None:
    assert KnowledgeAgent is KnowledgeService
