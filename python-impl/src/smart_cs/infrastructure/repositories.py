from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from uuid import uuid4

from sqlalchemy import or_, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from smart_cs.domain.enums import ActionStatus, OrderStatus, TicketStatus
from smart_cs.domain.errors import InvalidActionState, ToolPermissionError
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

    def customer_exists(self, customer_id: str, *, session: Session | None = None) -> bool:
        if session is not None:
            return session.get(Customer, customer_id) is not None
        with self.transaction() as managed_session:
            return managed_session.get(Customer, customer_id) is not None

    def search_products(self, query: str) -> list[Product]:
        text = query.strip()
        with self.database.session() as session:
            statement = select(Product).where(Product.active.is_(True))
            if text:
                statement = statement.where(
                    or_(Product.name.contains(text), Product.description.contains(text))
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

    def list_tool_calls(self) -> list[ToolCall]:
        with self.database.session() as session:
            return list(session.scalars(select(ToolCall).order_by(ToolCall.id)))

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
