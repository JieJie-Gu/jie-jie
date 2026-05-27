from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timedelta
from typing import Any
from uuid import uuid4

from sqlalchemy import or_, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from smart_cs.domain.enums import ActionStatus, OrderStatus, TicketStatus
from smart_cs.domain.errors import (
    ConversationBusyError,
    ConversationLeaseLostError,
    InvalidActionState,
    ToolPermissionError,
)
from smart_cs.domain.models import (
    Base,
    Conversation,
    Customer,
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

    @staticmethod
    def _claim_conversation(session: Session, conversation_id: str, customer_id: str) -> Conversation:
        session.execute(
            sqlite_insert(Conversation)
            .values(id=conversation_id, customer_id=customer_id, created_at=utc_now())
            .on_conflict_do_nothing(index_elements=[Conversation.id])
        )
        return SqlRepository._require_conversation_owner(session, conversation_id, customer_id)

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
