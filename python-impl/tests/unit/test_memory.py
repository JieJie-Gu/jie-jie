# 测试记忆提取、策略分层和摘要写回行为。

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from smart_cs.application.memory import (
    ConversationSummarizer,
    MemoryCandidate,
    MemoryDecision,
    MemoryExtractor,
    MemoryPolicy,
    MemoryWriter,
    MemoryWriteback,
)


class DummyRepository:
    def __init__(self) -> None:
        self.summaries = []
        self.tool_calls = []

    def upsert_conversation_summary(self, *args, **kwargs) -> None:
        self.summaries.append((args, kwargs))

    def record_tool_call(self, **kwargs) -> None:
        self.tool_calls.append(kwargs)


class RecordingStore:
    def __init__(self) -> None:
        self.writes = []
        self.records = {}

    def put(self, namespace, key, value) -> None:
        self.writes.append((namespace, key, value))
        self.records[(namespace, key)] = value

    def get(self, namespace, key):
        return self.records.get((namespace, key))

    def search(self, namespace, query: str, limit: int):
        return []


class SequenceExtractor:
    def __init__(self, candidates):
        self.candidates = list(candidates)

    def extract(self, _state):
        return [self.candidates.pop(0)]


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
    assert preference["memory_kind"] == "semantic"
    assert preference["key"] == "preference:shoe_size"
    assert preference["value"] == {"shoe_size": "42"}
    assert preference["evidence"][0]["text"] == "我一般穿42码"


def test_memory_writeback_writes_high_confidence_preference_to_active_memories() -> None:
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

    namespace, _key, value = store.writes[0]
    assert namespace == ("customer", "C001", "memories")
    assert value["memory_kind"] == "semantic"
    assert value["memory_decision"]["action"] == "write"


def test_service_event_writes_customer_memories_namespace_for_recall() -> None:
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
    assert namespace == ("customer", "C001", "memories")
    assert key == "episode:after_sales_event:A1:submitted"
    assert value["memory_kind"] == "episodic"
    assert value["value"]["conversation_id"] == "conv-1"
    assert value["memory_decision"]["action"] == "write"


def test_memory_policy_routes_sensitive_and_badcase_to_human_review() -> None:
    policy = MemoryPolicy()

    assert policy.decide({"memory_type": "sensitive_label", "risk_level": "high"}).action == "human_review"
    assert policy.decide({"memory_type": "badcase_candidate", "risk_level": "medium"}).action == "human_review"


def test_memory_writer_semantic_same_key_merges_evidence_and_sets_ttl() -> None:
    store = RecordingStore()
    writer = MemoryWriter()
    decision = MemoryDecision(action="write", reason="approved_semantic_memory")
    first = MemoryCandidate(
        scope="customer",
        owner_id="C001",
        memory_kind="semantic",
        memory_type="preference",
        key="preference:shoe_size",
        title="Shoe size",
        description="User wears size 42.",
        value={"shoe_size": "42"},
        evidence=[{"text": "我一般穿42码"}],
        source="llm_extraction",
        confidence="high",
        risk_level="low",
        review_status="approved",
    )
    second = first.model_copy(update={"evidence": [{"text": "通常42码"}]})

    writer.write(first, decision, store)
    writer.write(second, decision, store)

    namespace, key, value = store.writes[-1]
    assert namespace == ("customer", "C001", "memories")
    assert key == "preference:shoe_size"
    assert value["review_status"] == "approved"
    assert value["expires_at"] is not None
    assert len(value["evidence"]) == 2


def test_memory_writer_semantic_conflict_preserves_active_and_writes_candidate() -> None:
    store = RecordingStore()
    writer = MemoryWriter()
    decision = MemoryDecision(action="write", reason="approved_semantic_memory")
    first = MemoryCandidate(
        scope="customer",
        owner_id="C001",
        memory_kind="semantic",
        memory_type="preference",
        key="preference:shoe_size",
        title="Shoe size",
        description="User wears size 42.",
        value={"shoe_size": "42"},
        evidence=[{"text": "我一般穿42码"}],
        source="llm_extraction",
        confidence="high",
        risk_level="low",
        review_status="approved",
    )
    second = first.model_copy(update={"value": {"shoe_size": "43"}})

    writer.write(first, decision, store)
    writer.write(second, decision, store)

    active = store.records[(("customer", "C001", "memories"), "preference:shoe_size")]
    namespace, _key, value = store.writes[-1]
    assert namespace == ("customer", "C001", "memory_candidates")
    assert active["value"] == {"shoe_size": "42"}
    assert active["review_status"] == "approved"
    assert value["conflict"] is True
    assert value["review_status"] == "pending"
    assert value["confidence"] == "medium"
    assert value["value"]["previous_value"] == {"shoe_size": "42"}
    assert value["value"]["proposed_value"] == {"shoe_size": "43"}
    assert value["value"]["conflict_with"] == "preference:shoe_size"


def test_memory_writer_episodic_same_key_is_idempotent_append() -> None:
    store = RecordingStore()
    writer = MemoryWriter()
    decision = MemoryDecision(action="write", reason="approved_episodic_memory")
    event = MemoryCandidate(
        scope="customer",
        owner_id="C001",
        memory_kind="episodic",
        memory_type="after_sales_event",
        key="episode:after_sales_event:A1:submitted",
        title="After sales submitted",
        description="Ticket submitted.",
        value={"action_id": "A1", "status": "submitted"},
        evidence=[{"action_id": "A1"}],
        source="tool_result",
        confidence="high",
        risk_level="low",
        review_status="approved",
    )

    writer.write(event, decision, store)
    writer.write(event, decision, store)

    namespace, key, value = store.writes[-1]
    assert namespace == ("customer", "C001", "memories")
    assert key == "episode:after_sales_event:A1:submitted"
    assert value["memory_kind"] == "episodic"
    assert value["expires_at"] is not None
    assert len(value["evidence"]) == 1


def test_memory_writeback_audits_before_after_for_upsert_and_conflict() -> None:
    repository = DummyRepository()
    store = RecordingStore()
    first = MemoryCandidate(
        scope="customer",
        owner_id="C001",
        memory_kind="semantic",
        memory_type="preference",
        key="preference:shoe_size",
        title="Shoe size",
        description="User wears size 42.",
        value={"shoe_size": "42"},
        evidence=[{"text": "first"}],
        source="llm_extraction",
        confidence="high",
        risk_level="low",
        review_status="approved",
    ).model_dump()
    second = {**first, "value": {"shoe_size": "43"}, "evidence": [{"text": "second"}]}
    extractor = SequenceExtractor([first, second])
    writeback = MemoryWriteback(repository=repository, extractor=extractor)
    writeback.update(
        {
            "conversation_id": "conv-1",
            "customer_id": "C001",
            "message": "first",
            "messages": [],
            "business_result": {},
        },
        store=store,
    )
    writeback.update(
        {
            "conversation_id": "conv-1",
            "customer_id": "C001",
            "message": "second",
            "messages": [],
            "business_result": {},
        },
        store=store,
    )

    write_calls = [call for call in repository.tool_calls if call["tool_name"] == "memory_write"]
    conflict_calls = [call for call in repository.tool_calls if call["tool_name"] == "memory_conflict"]
    assert write_calls[0]["result"]["operation"] == "semantic_upsert"
    assert write_calls[0]["result"]["before_json"] is None
    assert write_calls[0]["result"]["after_json"]["value"] == {"shoe_size": "42"}
    assert conflict_calls[0]["result"]["operation"] == "semantic_conflict"
    assert conflict_calls[0]["result"]["before_json"]["value"] == {"shoe_size": "42"}
    assert conflict_calls[0]["result"]["after_json"]["value"]["proposed_value"] == {"shoe_size": "43"}


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
            "conversation_summary": "old summary",
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

    assert update["conversation_summary"] == "old summary"
    assert update["messages"] == []


class FakeSummarizer:
    def __init__(self) -> None:
        self.payload = None

    def invoke(self, _payload):
        self.payload = _payload
        return AIMessage(content="real summary")


def test_memory_writeback_with_real_summarizer_can_emit_remove_messages() -> None:
    summarizer_model = FakeSummarizer()
    update = MemoryWriteback(
        repository=DummyRepository(),
        summarizer=ConversationSummarizer(
            summary_keep_last=1,
            summarizer=summarizer_model,
        ),
    ).update(
        {
            "conversation_id": "conv-1",
            "customer_id": "C001",
            "conversation_summary": "old summary",
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
    assert isinstance(summarizer_model.payload[0], SystemMessage)
    assert isinstance(summarizer_model.payload[1], HumanMessage)
    payload = json.loads(summarizer_model.payload[1].content)
    assert payload["existing_summary"] == "old summary"
    assert "first" in payload["new_messages"]
    assert update["conversation_summary"] == "real summary"


class FailingSummarizer:
    def invoke(self, _payload):
        raise RuntimeError("summary model unavailable")


def test_failing_real_summarizer_keeps_old_summary_and_does_not_remove_messages() -> None:
    update = MemoryWriteback(
        repository=DummyRepository(),
        summarizer=ConversationSummarizer(
            summary_keep_last=1,
            summarizer=FailingSummarizer(),
        ),
    ).update(
        {
            "conversation_id": "conv-1",
            "customer_id": "C001",
            "conversation_summary": "old summary",
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

    assert update["conversation_summary"] == "old summary"
    assert update["messages"] == []
