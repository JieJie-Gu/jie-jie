from __future__ import annotations

from fastapi.testclient import TestClient
from langchain_core.documents import Document

from smart_cs.agents.knowledge import KnowledgeAgent
from smart_cs.config import Settings
from smart_cs.main import create_app
from smart_cs.rag.retrieval import RuleBasedQueryRewriter


class FakeKnowledgeStore:
    def similarity_search(self, _query: str, **_kwargs):
        return [
            Document(
                page_content="签收后七天内可以申请退货。",
                metadata={
                    "document_id": "after_sales_policy",
                    "context_id": "after_sales_policy:售后政策 > 七天无理由:0",
                    "category": "after_sales",
                    "header_path": "售后政策 > 七天无理由",
                    "window_text": "签收后七天内可以申请退货。商品应保持完好。",
                },
            )
        ]


def test_knowledge_reply_contains_cited_window_without_creating_action(tmp_path) -> None:
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'api.db'}",
        checkpoint_path=tmp_path / "checkpoints.db",
        model_mode="rules",
        rag_enabled=False,
    )
    knowledge_agent = KnowledgeAgent(FakeKnowledgeStore(), RuleBasedQueryRewriter())

    with TestClient(create_app(settings, knowledge_agent=knowledge_agent)) as client:
        conversation_id = client.post(
            "/api/conversations", json={"customer_id": "C001"}
        ).json()["id"]
        response = client.post(
            f"/api/conversations/{conversation_id}/messages",
            json={"customer_id": "C001", "content": "签收后退货期限是什么？"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["agents_invoked"] == ["KnowledgeAgent"]
    assert body["status"] == "completed"
    assert "签收后七天" in body["reply"]
    assert body["result"]["citations"][0]["document_id"] == "after_sales_policy"
    assert body["pending_action"] is None
