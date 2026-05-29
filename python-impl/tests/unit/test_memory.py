from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from smart_cs.application.memory import (
    ConversationSummarizer,
    MemoryExtractor,
    MemoryPolicy,
    MemoryWriteback,
)


class DummyRepository:
    def __init__(self) -> None:
        self.summaries = []

    def upsert_conversation_summary(self, *args, **kwargs) -> None:
        self.summaries.append((args, kwargs))


class RecordingStore:
    def __init__(self) -> None:
        self.writes = []

    def put(self, namespace, key, value) -> None:
        self.writes.append((namespace, key, value))

    def search(self, namespace, query: str, limit: int):
        return []


def test_memory_extractor_creates_preference_candidate_from_user_message() -> None:
    candidates = MemoryExtractor().extract(
        {
            "conversation_id": "conv-1",
            "customer_id": "C001",
            "message": "我一般穿42码",
            "business_result": {},
        }
    )

    preference = next(candidate for candidate in candidates if candidate["memory_type"] == "preference")
    assert preference["key"] == "shoe_size"
    assert preference["value"] == {"shoe_size": "42"}
    assert preference["evidence"][0]["text"] == "我一般穿42码"


def test_memory_writeback_keeps_candidates_out_of_active_memories() -> None:
    repository = DummyRepository()
    store = RecordingStore()

    MemoryWriteback(repository=repository).update(
        {
            "conversation_id": "conv-1",
            "customer_id": "C001",
            "message": "我喜欢黑色",
            "messages": [],
            "business_result": {},
            "route": {"intent": "product", "entities": {}, "risk": "low"},
        },
        store=store,
    )

    assert store.writes
    namespace, _key, value = store.writes[0]
    assert namespace == ("customer", "C001", "memory_candidates")
    assert value["memory_decision"]["action"] == "candidate"


def test_service_event_writes_conversation_events_namespace() -> None:
    store = RecordingStore()

    MemoryWriteback(repository=DummyRepository()).update(
        {
            "conversation_id": "conv-1",
            "customer_id": "C001",
            "message": "确认提交",
            "messages": [],
            "business_result": {
                "action_id": "A1",
                "action_type": "after_sales",
                "status": "submitted",
                "ticket_id": "T1",
                "order_id": "O1001",
            },
            "route": {"intent": "after_sales", "entities": {"order_id": "O1001"}, "risk": "medium"},
        },
        store=store,
    )

    namespace, key, value = store.writes[0]
    assert namespace == ("conversation", "conv-1", "events")
    assert key == "after_sales:A1:submitted"
    assert value["memory_decision"]["action"] == "write"


def test_memory_policy_routes_sensitive_and_badcase_to_human_review() -> None:
    policy = MemoryPolicy()

    assert policy.decide({"memory_type": "sensitive_label", "risk_level": "high"}).action == "human_review"
    assert policy.decide({"memory_type": "badcase_candidate", "risk_level": "medium"}).action == "human_review"


def test_summarizer_removes_only_human_and_ai_messages() -> None:
    messages = [
        HumanMessage(id="h1", content="查询订单 O1001"),
        ToolMessage(id="t1", content="tool output", tool_call_id="call-1"),
        AIMessage(id="a1", content="订单 O1001 当前状态为 delivered。"),
        HumanMessage(id="h2", content="继续"),
        AIMessage(id="a2", content="好的"),
    ]

    removable = ConversationSummarizer(summary_keep_last=2).removable_messages(messages)

    assert [message.id for message in removable] == ["h1", "a1"]


def test_memory_writeback_without_real_summarizer_does_not_emit_remove_messages() -> None:
    update = MemoryWriteback(
        repository=DummyRepository(),
        summarizer=ConversationSummarizer(summary_keep_last=1),
    ).update(
        {
            "conversation_id": "conv-1",
            "customer_id": "C001",
            "message": "continue",
            "messages": [
                HumanMessage(id="h1", content="first"),
                AIMessage(id="a1", content="second"),
                HumanMessage(id="h2", content="third"),
            ],
            "business_result": {},
        },
        store=RecordingStore(),
    )

    assert update["conversation_summary"]
    assert update["messages"] == []


class FakeSummarizer:
    def invoke(self, _payload):
        return AIMessage(content="real summary")


def test_memory_writeback_with_real_summarizer_can_emit_remove_messages() -> None:
    update = MemoryWriteback(
        repository=DummyRepository(),
        summarizer=ConversationSummarizer(
            summary_keep_last=1,
            summarizer=FakeSummarizer(),
        ),
    ).update(
        {
            "conversation_id": "conv-1",
            "customer_id": "C001",
            "message": "continue",
            "messages": [
                HumanMessage(id="h1", content="first"),
                ToolMessage(id="t1", content="tool", tool_call_id="call-1"),
                AIMessage(id="a1", content="second"),
                HumanMessage(id="h2", content="third"),
            ],
            "business_result": {},
        },
        store=RecordingStore(),
    )

    assert [message.id for message in update["messages"]] == ["h1", "a1"]
