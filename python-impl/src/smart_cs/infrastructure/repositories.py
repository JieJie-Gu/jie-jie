# 实现基于 SQLAlchemy 的客服事实仓库。

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from sqlalchemy import or_, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from smart_cs.domain.enums import ActionStatus, OrderStatus, TicketStatus, ToolCallStatus
from smart_cs.domain.errors import (
    ConversationBusyError,
    ConversationLeaseLostError,
    InvalidActionState,
    ToolPermissionError,
)
from smart_cs.domain.models import (
    AgentRun,
    Base,
    Conversation,
    ConversationSummary,
    Customer,
    Message,
    MemoryRecord,
    Order,
    PendingAction,
    Product,
    Ticket,
    ToolCall,
    utc_now,
)
from smart_cs.infrastructure.database import Database


class SqlRepository:
    """SQLite-backed persistence with operations matching business state changes."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def create_schema(self) -> None:
        Base.metadata.create_all(self.database.engine)

    @contextmanager
    def transaction(self) -> Iterator[Session]:
        with self.database.session() as session:
            yield session

    def seed_demo_data(self) -> None:
        with self.database.session() as session:
            if session.get(Customer, "C001") is None:
                session.add(Customer(id="C001", name="演示客户"))
            if session.get(Customer, "C002") is None:
                session.add(Customer(id="C002", name="其他客户"))
            if session.get(Product, "P1001") is None:
                session.add(
                    Product(
                        id="P1001",
                        name="轻量跑鞋",
                        description="适合日常训练的轻量缓震跑鞋",
                        price_cents=39900,
                    )
                )
            if session.get(Order, "O1001") is None:
                session.add(
                    Order(
                        id="O1001",
                        customer_id="C001",
                        product_id="P1001",
                        status=OrderStatus.DELIVERED.value,
                        quantity=1,
                        total_cents=39900,
                    )
                )

    def claim_conversation(
        self, conversation_id: str, customer_id: str, *, session: Session | None = None
    ) -> Conversation:
        if session is not None:
            return self._claim_conversation(session, conversation_id, customer_id)
        with self.transaction() as managed_session:
            return self._claim_conversation(managed_session, conversation_id, customer_id)

    def require_conversation_owner(
        self, conversation_id: str, customer_id: str, *, session: Session | None = None
    ) -> Conversation:
        if session is not None:
            return self._require_conversation_owner(session, conversation_id, customer_id)
        with self.transaction() as managed_session:
            return self._require_conversation_owner(managed_session, conversation_id, customer_id)

    def acquire_turn_lease(
        self,
        conversation_id: str,
        customer_id: str,
        token: str,
        *,
        ttl_seconds: float,
        session: Session | None = None,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("Turn lease TTL must be positive")
        if session is not None:
            self._acquire_turn_lease(
                session, conversation_id, customer_id, token, ttl_seconds=ttl_seconds
            )
            return
        with self.transaction() as managed_session:
            self._acquire_turn_lease(
                managed_session, conversation_id, customer_id, token, ttl_seconds=ttl_seconds
            )

    def renew_turn_lease(
        self,
        conversation_id: str,
        customer_id: str,
        token: str,
        *,
        ttl_seconds: float,
        session: Session | None = None,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("Turn lease TTL must be positive")
        if session is not None:
            self._renew_turn_lease(
                session, conversation_id, customer_id, token, ttl_seconds=ttl_seconds
            )
            return
        with self.transaction() as managed_session:
            self._renew_turn_lease(
                managed_session, conversation_id, customer_id, token, ttl_seconds=ttl_seconds
            )

    def require_active_turn_lease(
        self,
        conversation_id: str,
        customer_id: str,
        token: str,
        *,
        session: Session | None = None,
    ) -> None:
        if session is not None:
            self._require_active_turn_lease(session, conversation_id, customer_id, token)
            return
        with self.transaction() as managed_session:
            self._require_active_turn_lease(managed_session, conversation_id, customer_id, token)

    def release_turn_lease(
        self,
        conversation_id: str,
        customer_id: str,
        token: str,
        *,
        session: Session | None = None,
    ) -> None:
        if session is not None:
            self._release_turn_lease(session, conversation_id, customer_id, token)
            return
        with self.transaction() as managed_session:
            self._release_turn_lease(managed_session, conversation_id, customer_id, token)

    def customer_exists(self, customer_id: str, *, session: Session | None = None) -> bool:
        if session is not None:
            return session.get(Customer, customer_id) is not None
        with self.transaction() as managed_session:
            return managed_session.get(Customer, customer_id) is not None

    def record_message(
        self,
        conversation_id: str,
        customer_id: str,
        role: str,
        content: str,
        content_type: str = "text",
        asset_key: str | None = None,
        visual_evidence: dict[str, Any] | None = None,
        session: Session | None = None,
    ) -> Message:
        if session is not None:
            return self._record_message(
                session,
                conversation_id=conversation_id,
                customer_id=customer_id,
                role=role,
                content=content,
                content_type=content_type,
                asset_key=asset_key,
                visual_evidence=visual_evidence,
            )
        with self.transaction() as managed_session:
            return self._record_message(
                managed_session,
                conversation_id=conversation_id,
                customer_id=customer_id,
                role=role,
                content=content,
                content_type=content_type,
                asset_key=asset_key,
                visual_evidence=visual_evidence,
            )

    def latest_message(self, conversation_id: str) -> Message | None:
        with self.database.session() as session:
            return session.scalar(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at.desc(), Message.id.desc())
            )

    def list_recent_messages(
        self,
        conversation_id: str,
        customer_id: str,
        *,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        with self.database.session() as session:
            self._require_conversation_owner(session, conversation_id, customer_id)
            statement = (
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at.desc(), Message.id.desc())
                .limit(limit)
            )
            rows = list(session.scalars(statement))
        return [
            {
                "role": row.role,
                "content": row.content,
                "content_type": row.content_type,
                "asset_key": row.asset_key,
                "visual_evidence": row.visual_evidence,
                "created_at": row.created_at.isoformat(),
            }
            for row in reversed(rows)
        ]

    def search_products(self, query: str) -> list[Product]:
        raw_text = query.strip()
        text = raw_text
        terms = [raw_text] if raw_text else []
        for intent_word in ("推荐", "商品", "产品", "价格", "请", "帮我", "一下"):
            text = text.replace(intent_word, " ")
        terms.extend(term for term in text.split() if term and term not in terms)

        with self.database.session() as session:
            statement = select(Product).where(Product.active.is_(True))
            if terms:
                statement = statement.where(
                    or_(
                        *(
                            condition
                            for term in terms
                            for condition in (
                                Product.name.contains(term),
                                Product.description.contains(term),
                            )
                        )
                    )
                )
            return list(session.scalars(statement.order_by(Product.id)))

    def get_owned_order(
        self, customer_id: str, order_id: str, *, session: Session | None = None
    ) -> Order | None:
        if session is not None:
            return session.scalar(
                select(Order).where(Order.id == order_id, Order.customer_id == customer_id)
            )
        with self.transaction() as managed_session:
            return managed_session.scalar(
                select(Order).where(Order.id == order_id, Order.customer_id == customer_id)
            )

    def create_pending_action(
        self,
        customer_id: str,
        action_type: str,
        reason: str,
        order_id: str | None = None,
        conversation_id: str | None = None,
        idempotency_key: str | None = None,
        *,
        session: Session | None = None,
    ) -> PendingAction:
        if session is not None:
            return self._create_pending_action(
                session,
                customer_id=customer_id,
                action_type=action_type,
                reason=reason,
                order_id=order_id,
                conversation_id=conversation_id,
                idempotency_key=idempotency_key,
            )
        with self.transaction() as managed_session:
            return self._create_pending_action(
                managed_session,
                customer_id=customer_id,
                action_type=action_type,
                reason=reason,
                order_id=order_id,
                conversation_id=conversation_id,
                idempotency_key=idempotency_key,
            )

    def get_pending_action(
        self, conversation_id: str, customer_id: str, *, session: Session | None = None
    ) -> PendingAction | None:
        if session is not None:
            return self._get_pending_action(session, conversation_id, customer_id)
        with self.transaction() as managed_session:
            return self._get_pending_action(managed_session, conversation_id, customer_id)

    def get_latest_action(
        self, conversation_id: str, customer_id: str, *, session: Session | None = None
    ) -> PendingAction | None:
        if session is not None:
            return self._get_latest_action(session, conversation_id, customer_id)
        with self.transaction() as managed_session:
            return self._get_latest_action(managed_session, conversation_id, customer_id)

    def get_action(
        self,
        conversation_id: str,
        customer_id: str,
        action_id: str,
        *,
        session: Session | None = None,
    ) -> PendingAction:
        if session is not None:
            return self._get_action(session, conversation_id, customer_id, action_id)
        with self.transaction() as managed_session:
            return self._get_action(managed_session, conversation_id, customer_id, action_id)

    def get_ticket_for_action(self, action_id: str, *, session: Session | None = None) -> Ticket | None:
        if session is not None:
            return session.scalar(select(Ticket).where(Ticket.action_id == action_id))
        with self.transaction() as managed_session:
            return managed_session.scalar(select(Ticket).where(Ticket.action_id == action_id))

    def submit_pending_action(
        self, action_id: str, customer_id: str, *, session: Session | None = None
    ) -> tuple[PendingAction, Ticket]:
        if session is not None:
            return self._submit_pending_action(session, action_id, customer_id)
        with self.transaction() as managed_session:
            return self._submit_pending_action(managed_session, action_id, customer_id)

    def cancel_pending_action(
        self, action_id: str, customer_id: str, *, session: Session | None = None
    ) -> PendingAction:
        if session is not None:
            return self._cancel_pending_action(session, action_id, customer_id)
        with self.transaction() as managed_session:
            return self._cancel_pending_action(managed_session, action_id, customer_id)

    def list_tickets(self, customer_id: str) -> list[Ticket]:
        with self.database.session() as session:
            return list(
                session.scalars(
                    select(Ticket).where(Ticket.customer_id == customer_id).order_by(Ticket.created_at)
                )
            )

    def list_tool_calls(self, customer_id: str | None = None) -> list[ToolCall]:
        with self.database.session() as session:
            statement = select(ToolCall)
            if customer_id is not None:
                statement = statement.where(ToolCall.customer_id == customer_id)
            return list(session.scalars(statement.order_by(ToolCall.id)))

    def record_agent_run(
        self,
        conversation_id: str,
        customer_id: str,
        agents: list[str],
        status: str,
        pending_action_id: str | None = None,
        reply: str | None = None,
        session: Session | None = None,
    ) -> AgentRun:
        run = AgentRun(
            id=str(uuid4()),
            conversation_id=conversation_id,
            agents=agents,
            status=status,
            pending_action_id=pending_action_id,
            reply=reply,
        )
        if session is not None:
            self._require_conversation_owner(session, conversation_id, customer_id)
            session.add(run)
            session.flush()
            return run
        with self.transaction() as managed_session:
            self._require_conversation_owner(managed_session, conversation_id, customer_id)
            managed_session.add(run)
            managed_session.flush()
        return run

    def list_agent_runs(self, conversation_id: str, customer_id: str) -> list[AgentRun]:
        with self.transaction() as session:
            self._require_conversation_owner(session, conversation_id, customer_id)
            return list(
                session.scalars(
                    select(AgentRun)
                    .where(AgentRun.conversation_id == conversation_id)
                    .order_by(AgentRun.created_at.desc(), AgentRun.id.desc())
                )
            )

    def update_agent_run_for_action(
        self,
        conversation_id: str,
        customer_id: str,
        pending_action_id: str,
        status: str,
        reply: str | None = None,
        agents: list[str] | None = None,
        session: Session | None = None,
    ) -> AgentRun | None:
        if session is not None:
            return self._update_agent_run_for_action(
                session,
                conversation_id=conversation_id,
                customer_id=customer_id,
                pending_action_id=pending_action_id,
                status=status,
                reply=reply,
                agents=agents,
            )
        with self.transaction() as managed_session:
            return self._update_agent_run_for_action(
                managed_session,
                conversation_id=conversation_id,
                customer_id=customer_id,
                pending_action_id=pending_action_id,
                status=status,
                reply=reply,
                agents=agents,
            )

    @classmethod
    def _update_agent_run_for_action(
        cls,
        session: Session,
        *,
        conversation_id: str,
        customer_id: str,
        pending_action_id: str,
        status: str,
        reply: str | None,
        agents: list[str] | None,
    ) -> AgentRun | None:
        cls._require_conversation_owner(session, conversation_id, customer_id)
        run = session.scalar(
            select(AgentRun)
            .where(
                AgentRun.conversation_id == conversation_id,
                AgentRun.pending_action_id == pending_action_id,
            )
            .order_by(AgentRun.created_at.desc(), AgentRun.id.desc())
        )
        if run is None:
            return None
        run.status = status
        run.reply = reply
        if agents is not None:
            run.agents = agents
        session.flush()
        return run

    def record_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        status: str,
        customer_id: str | None = None,
        result: dict[str, Any] | None = None,
        error_type: str | None = None,
        duration_ms: int = 0,
        session: Session | None = None,
    ) -> ToolCall:
        call = ToolCall(
            tool_name=tool_name,
            customer_id=customer_id,
            arguments=arguments,
            result=result,
            status=status,
            error_type=error_type,
            duration_ms=duration_ms,
        )
        if session is not None:
            session.add(call)
            session.flush()
            return call
        with self.transaction() as managed_session:
            managed_session.add(call)
            managed_session.flush()
        return call

    def upsert_conversation_summary(
        self,
        conversation_id: str,
        customer_id: str,
        summary: str,
        *,
        open_items: dict[str, Any] | None = None,
        last_intent: str | None = None,
        last_entities: dict[str, str] | None = None,
    ) -> ConversationSummary:
        with self.transaction() as session:
            self._require_conversation_owner(session, conversation_id, customer_id)
            row = session.get(ConversationSummary, conversation_id)
            if row is None:
                row = ConversationSummary(
                    conversation_id=conversation_id,
                    customer_id=customer_id,
                    summary=summary,
                    open_items=open_items or {},
                    last_intent=last_intent,
                    last_entities=last_entities or {},
                )
                session.add(row)
            else:
                row.summary = summary
                row.open_items = open_items or {}
                row.last_intent = last_intent
                row.last_entities = last_entities or {}
            session.flush()
            return row

    def get_conversation_summary(
        self, conversation_id: str, customer_id: str
    ) -> ConversationSummary | None:
        with self.transaction() as session:
            self._require_conversation_owner(session, conversation_id, customer_id)
            return session.get(ConversationSummary, conversation_id)

    def put_memory(
        self,
        namespace: tuple[str, str, str],
        key: str,
        value: dict[str, Any],
        *,
        scope: str,
        owner_id: str,
        memory_type: str,
        source: str,
        confidence: str,
        risk_level: str,
        created_by: str,
    ) -> MemoryRecord:
        namespace_text = "/".join(namespace)
        memory_id = f"{namespace_text}:{key}"
        expires_at = self._parse_datetime(value.get("expires_at"))
        with self.transaction() as session:
            row = session.get(MemoryRecord, memory_id)
            if row is None:
                row = MemoryRecord(
                    id=memory_id,
                    namespace=namespace_text,
                    scope=scope,
                    owner_id=owner_id,
                    memory_type=memory_type,
                    key=key,
                    title=str(value.get("title", key)),
                    description=str(value.get("description", "")),
                    value_json=value,
                    evidence_json=list(value.get("evidence", [])),
                    source=source,
                    confidence=confidence,
                    risk_level=risk_level,
                    review_status=str(value.get("review_status", "pending")),
                    created_by=created_by,
                    expires_at=expires_at,
                )
                session.add(row)
            else:
                row.title = str(value.get("title", key))
                row.description = str(value.get("description", ""))
                row.value_json = value
                row.evidence_json = list(value.get("evidence", []))
                row.confidence = confidence
                row.risk_level = risk_level
                row.source = source
                row.review_status = str(value.get("review_status", "pending"))
                row.expires_at = expires_at
            session.flush()
            return row

    def get_memory(self, namespace: tuple[str, str, str], key: str) -> MemoryRecord | None:
        namespace_text = "/".join(namespace)
        memory_id = f"{namespace_text}:{key}"
        with self.transaction() as session:
            return session.get(MemoryRecord, memory_id)

    def search_memories(
        self, namespace: tuple[str, str, str], query: str, limit: int
    ) -> list[MemoryRecord]:
        namespace_text = "/".join(namespace)
        with self.transaction() as session:
            statement = (
                select(MemoryRecord)
                .where(MemoryRecord.namespace == namespace_text)
                .order_by(MemoryRecord.updated_at.desc(), MemoryRecord.id.desc())
                .limit(limit)
            )
            return list(session.scalars(statement))

    def list_memory_candidates(
        self,
        *,
        customer_id: str | None = None,
        status: str = "pending",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        with self.transaction() as session:
            statement = select(MemoryRecord).where(MemoryRecord.review_status == status)
            if customer_id is not None:
                statement = statement.where(
                    MemoryRecord.namespace == f"customer/{customer_id}/memory_candidates"
                )
            else:
                statement = statement.where(MemoryRecord.namespace.like("%/memory_candidates"))
            statement = statement.order_by(MemoryRecord.updated_at.desc(), MemoryRecord.id.desc()).limit(limit)
            return [self._memory_record_result(record) for record in session.scalars(statement)]

    def approve_memory_candidate(
        self,
        *,
        candidate_key: str,
        customer_id: str,
        reviewer_id: str,
        edited_value: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        candidate_namespace = ("customer", customer_id, "memory_candidates")
        active_namespace = ("customer", customer_id, "memories")
        with self.transaction() as session:
            candidate = session.get(
                MemoryRecord,
                f"{'/'.join(candidate_namespace)}:{candidate_key}",
            )
            if candidate is None:
                raise ToolPermissionError("Memory candidate is not available")
            before = self._memory_record_result(candidate)
            value = dict(candidate.value_json)
            if edited_value is not None:
                if any(key in edited_value for key in {"memory_type", "memory_kind", "value"}):
                    value.update(edited_value)
                else:
                    value["value"] = edited_value
            value["review_status"] = "approved"
            value["source"] = "human_review"
            active_id = f"{'/'.join(active_namespace)}:{candidate_key}"
            active = session.get(MemoryRecord, active_id)
            if active is None:
                active = MemoryRecord(
                    id=active_id,
                    namespace="/".join(active_namespace),
                    scope="customer",
                    owner_id=customer_id,
                    memory_type=str(value.get("memory_type", candidate.memory_type)),
                    key=candidate_key,
                    title=str(value.get("title", candidate.title)),
                    description=str(value.get("description", candidate.description)),
                    value_json=value,
                    evidence_json=list(value.get("evidence", candidate.evidence_json)),
                    source="human_review",
                    confidence=str(value.get("confidence", candidate.confidence)),
                    risk_level=str(value.get("risk_level", candidate.risk_level)),
                    review_status="approved",
                    created_by=candidate.created_by,
                    approved_by=reviewer_id,
                    expires_at=self._parse_datetime(value.get("expires_at")),
                )
                session.add(active)
            else:
                active.title = str(value.get("title", candidate.title))
                active.description = str(value.get("description", candidate.description))
                active.value_json = value
                active.evidence_json = list(value.get("evidence", candidate.evidence_json))
                active.source = "human_review"
                active.confidence = str(value.get("confidence", candidate.confidence))
                active.risk_level = str(value.get("risk_level", candidate.risk_level))
                active.review_status = "approved"
                active.approved_by = reviewer_id
                active.expires_at = self._parse_datetime(value.get("expires_at"))
            candidate.review_status = "approved"
            candidate.value_json = value
            candidate.approved_by = reviewer_id
            session.flush()
            after = self._memory_record_result(active)
            self.record_tool_call(
                tool_name="memory_review",
                arguments={
                    "candidate_key": candidate_key,
                    "customer_id": customer_id,
                    "reviewer_id": reviewer_id,
                    "decision": "approve",
                },
                customer_id=customer_id,
                status=ToolCallStatus.SUCCEEDED.value,
                result={"before": before, "after": after},
                session=session,
            )
            return after

    def reject_memory_candidate(
        self,
        *,
        candidate_key: str,
        customer_id: str,
        reviewer_id: str,
        reason: str,
    ) -> dict[str, Any]:
        namespace = ("customer", customer_id, "memory_candidates")
        with self.transaction() as session:
            candidate = session.get(MemoryRecord, f"{'/'.join(namespace)}:{candidate_key}")
            if candidate is None:
                raise ToolPermissionError("Memory candidate is not available")
            before = self._memory_record_result(candidate)
            value = dict(candidate.value_json)
            value["review_status"] = "rejected"
            value["review_reason"] = reason
            candidate.review_status = "rejected"
            candidate.value_json = value
            candidate.approved_by = reviewer_id
            session.flush()
            after = self._memory_record_result(candidate)
            self.record_tool_call(
                tool_name="memory_review",
                arguments={
                    "candidate_key": candidate_key,
                    "customer_id": customer_id,
                    "reviewer_id": reviewer_id,
                    "decision": "reject",
                    "reason": reason,
                },
                customer_id=customer_id,
                status=ToolCallStatus.SUCCEEDED.value,
                result={"before": before, "after": after},
                session=session,
            )
            return after

    @staticmethod
    def _memory_record_result(record: MemoryRecord) -> dict[str, Any]:
        return {
            "id": record.id,
            "namespace": record.namespace,
            "scope": record.scope,
            "owner_id": record.owner_id,
            "memory_type": record.memory_type,
            "key": record.key,
            "title": record.title,
            "description": record.description,
            "value": record.value_json,
            "evidence": record.evidence_json,
            "source": record.source,
            "confidence": record.confidence,
            "risk_level": record.risk_level,
            "review_status": record.review_status,
            "expires_at": record.expires_at.isoformat() if record.expires_at else None,
            "updated_at": record.updated_at.isoformat() if record.updated_at else None,
        }

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        if value in (None, ""):
            return None
        if isinstance(value, datetime):
            return value
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None

    @staticmethod
    def _claim_conversation(session: Session, conversation_id: str, customer_id: str) -> Conversation:
        session.execute(
            sqlite_insert(Conversation)
            .values(id=conversation_id, customer_id=customer_id, created_at=utc_now())
            .on_conflict_do_nothing(index_elements=[Conversation.id])
        )
        return SqlRepository._require_conversation_owner(session, conversation_id, customer_id)

    @classmethod
    def _record_message(
        cls,
        session: Session,
        *,
        conversation_id: str,
        customer_id: str,
        role: str,
        content: str,
        content_type: str,
        asset_key: str | None,
        visual_evidence: dict[str, Any] | None,
    ) -> Message:
        cls._require_conversation_owner(session, conversation_id, customer_id)
        message = Message(
            conversation_id=conversation_id,
            customer_id=customer_id,
            role=role,
            content=content,
            content_type=content_type,
            asset_key=asset_key,
            visual_evidence=visual_evidence,
        )
        session.add(message)
        session.flush()
        return message

    @staticmethod
    def _require_conversation_owner(
        session: Session, conversation_id: str, customer_id: str
    ) -> Conversation:
        conversation = session.get(Conversation, conversation_id)
        if conversation is None or conversation.customer_id != customer_id:
            raise ToolPermissionError("Conversation is not available to this customer")
        return conversation

    @classmethod
    def _acquire_turn_lease(
        cls,
        session: Session,
        conversation_id: str,
        customer_id: str,
        token: str,
        *,
        ttl_seconds: float,
    ) -> None:
        cls._require_conversation_owner(session, conversation_id, customer_id)
        now = utc_now()
        acquired = session.execute(
            update(Conversation)
            .where(
                Conversation.id == conversation_id,
                Conversation.customer_id == customer_id,
                or_(
                    Conversation.turn_lease_token.is_(None),
                    Conversation.turn_lease_expires_at.is_(None),
                    Conversation.turn_lease_expires_at <= now,
                ),
            )
            .values(
                turn_lease_token=token,
                turn_lease_expires_at=now + timedelta(seconds=ttl_seconds),
            )
            .execution_options(synchronize_session=False)
        )
        if acquired.rowcount != 1:
            raise ConversationBusyError("Conversation is busy with another active turn")

    @classmethod
    def _renew_turn_lease(
        cls,
        session: Session,
        conversation_id: str,
        customer_id: str,
        token: str,
        *,
        ttl_seconds: float,
    ) -> None:
        cls._require_conversation_owner(session, conversation_id, customer_id)
        now = utc_now()
        renewed = session.execute(
            update(Conversation)
            .where(
                Conversation.id == conversation_id,
                Conversation.customer_id == customer_id,
                Conversation.turn_lease_token == token,
                Conversation.turn_lease_expires_at.is_not(None),
                Conversation.turn_lease_expires_at > now,
            )
            .values(turn_lease_expires_at=now + timedelta(seconds=ttl_seconds))
            .execution_options(synchronize_session=False)
        )
        if renewed.rowcount != 1:
            raise ConversationLeaseLostError("Conversation turn lease was lost")

    @classmethod
    def _require_active_turn_lease(
        cls, session: Session, conversation_id: str, customer_id: str, token: str
    ) -> None:
        cls._require_conversation_owner(session, conversation_id, customer_id)
        fenced = session.execute(
            update(Conversation)
            .where(
                Conversation.id == conversation_id,
                Conversation.customer_id == customer_id,
                Conversation.turn_lease_token == token,
                Conversation.turn_lease_expires_at.is_not(None),
                Conversation.turn_lease_expires_at > utc_now(),
            )
            .values(turn_lease_token=token)
            .execution_options(synchronize_session=False)
        )
        if fenced.rowcount != 1:
            raise ConversationLeaseLostError("Conversation turn lease was lost")

    @classmethod
    def _release_turn_lease(
        cls, session: Session, conversation_id: str, customer_id: str, token: str
    ) -> None:
        cls._require_conversation_owner(session, conversation_id, customer_id)
        session.execute(
            update(Conversation)
            .where(
                Conversation.id == conversation_id,
                Conversation.customer_id == customer_id,
                Conversation.turn_lease_token == token,
            )
            .values(turn_lease_token=None, turn_lease_expires_at=None)
            .execution_options(synchronize_session=False)
        )

    @staticmethod
    def _create_pending_action(
        session: Session,
        *,
        customer_id: str,
        action_type: str,
        reason: str,
        order_id: str | None,
        conversation_id: str | None,
        idempotency_key: str | None,
    ) -> PendingAction:
        if conversation_id is None and idempotency_key is None:
            action = PendingAction(
                id=str(uuid4()),
                customer_id=customer_id,
                action_type=action_type,
                order_id=order_id,
                reason=reason,
                status=ActionStatus.PENDING_CONFIRMATION.value,
            )
            session.add(action)
            session.flush()
            return action

        now = utc_now()
        session.execute(
            sqlite_insert(PendingAction)
            .values(
                id=str(uuid4()),
                customer_id=customer_id,
                conversation_id=conversation_id,
                idempotency_key=idempotency_key,
                action_type=action_type,
                order_id=order_id,
                reason=reason,
                status=ActionStatus.PENDING_CONFIRMATION.value,
                created_at=now,
                updated_at=now,
            )
            .on_conflict_do_nothing()
        )
        if idempotency_key is not None:
            action = session.scalar(
                select(PendingAction).where(PendingAction.idempotency_key == idempotency_key)
            )
            if action is not None:
                if (
                    action.customer_id != customer_id
                    or action.conversation_id != conversation_id
                ):
                    raise ToolPermissionError("Pending action is not available to this customer")
                if (
                    action.action_type != action_type
                    or action.order_id != order_id
                    or action.reason != reason
                ):
                    raise InvalidActionState(
                        "Idempotency key conflicts with another action payload"
                    )
                return action
        if conversation_id is not None:
            action = session.scalar(
                select(PendingAction).where(
                    PendingAction.conversation_id == conversation_id,
                    PendingAction.customer_id == customer_id,
                    PendingAction.status == ActionStatus.PENDING_CONFIRMATION.value,
                )
            )
            if action is not None:
                return action
        raise InvalidActionState("Unable to create or recover pending action")

    @classmethod
    def _get_pending_action(
        cls, session: Session, conversation_id: str, customer_id: str
    ) -> PendingAction | None:
        cls._require_conversation_owner(session, conversation_id, customer_id)
        return session.scalar(
            select(PendingAction).where(
                PendingAction.conversation_id == conversation_id,
                PendingAction.customer_id == customer_id,
                PendingAction.status == ActionStatus.PENDING_CONFIRMATION.value,
            )
        )

    @classmethod
    def _get_latest_action(
        cls, session: Session, conversation_id: str, customer_id: str
    ) -> PendingAction | None:
        cls._require_conversation_owner(session, conversation_id, customer_id)
        return session.scalar(
            select(PendingAction)
            .where(
                PendingAction.conversation_id == conversation_id,
                PendingAction.customer_id == customer_id,
            )
            .order_by(PendingAction.created_at.desc(), PendingAction.id.desc())
        )

    @classmethod
    def _get_action(
        cls, session: Session, conversation_id: str, customer_id: str, action_id: str
    ) -> PendingAction:
        cls._require_conversation_owner(session, conversation_id, customer_id)
        action = session.scalar(
            select(PendingAction).where(
                PendingAction.id == action_id,
                PendingAction.conversation_id == conversation_id,
                PendingAction.customer_id == customer_id,
            )
        )
        if action is None:
            raise ToolPermissionError("Pending action is not available to this conversation")
        return action

    def _submit_pending_action(
        self, session: Session, action_id: str, customer_id: str
    ) -> tuple[PendingAction, Ticket]:
        transitioned = session.execute(
            update(PendingAction)
            .where(
                PendingAction.id == action_id,
                PendingAction.customer_id == customer_id,
                PendingAction.status == ActionStatus.PENDING_CONFIRMATION.value,
            )
            .values(status=ActionStatus.SUBMITTED.value)
            .execution_options(synchronize_session=False)
        )
        if transitioned.rowcount == 1:
            action = self._owned_action(session, action_id, customer_id)
            ticket = Ticket(
                id=f"T{uuid4().hex[:12].upper()}",
                customer_id=customer_id,
                action_id=action.id,
                ticket_type=action.action_type,
                status=TicketStatus.OPEN.value,
                summary=action.reason,
            )
            session.add(ticket)
            session.flush()
            return action, ticket

        action = self._owned_action(session, action_id, customer_id)
        if action.status == ActionStatus.SUBMITTED.value:
            ticket = session.scalar(select(Ticket).where(Ticket.action_id == action.id))
            if ticket is not None:
                return action, ticket
        raise InvalidActionState(f"Action {action_id} cannot be submitted from {action.status}")

    def _cancel_pending_action(self, session: Session, action_id: str, customer_id: str) -> PendingAction:
        transitioned = session.execute(
            update(PendingAction)
            .where(
                PendingAction.id == action_id,
                PendingAction.customer_id == customer_id,
                PendingAction.status == ActionStatus.PENDING_CONFIRMATION.value,
            )
            .values(status=ActionStatus.CANCELLED.value)
            .execution_options(synchronize_session=False)
        )
        if transitioned.rowcount == 1:
            return self._owned_action(session, action_id, customer_id)

        action = self._owned_action(session, action_id, customer_id)
        if action.status == ActionStatus.CANCELLED.value:
            return action
        raise InvalidActionState(f"Action {action_id} cannot be cancelled from {action.status}")

    @staticmethod
    def _owned_action(session, action_id: str, customer_id: str) -> PendingAction:
        action = session.get(PendingAction, action_id)
        if action is None or action.customer_id != customer_id:
            raise ToolPermissionError("Pending action is not available to this customer")
        return action
