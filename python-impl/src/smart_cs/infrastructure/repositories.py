from __future__ import annotations

from typing import Any
from uuid import uuid4

from sqlalchemy import or_, select
from sqlalchemy.exc import IntegrityError

from smart_cs.domain.enums import ActionStatus, OrderStatus, TicketStatus
from smart_cs.domain.errors import InvalidActionState, ToolPermissionError
from smart_cs.domain.models import Base, Customer, Order, PendingAction, Product, Ticket, ToolCall
from smart_cs.infrastructure.database import Database


class SqlRepository:
    """SQLite-backed persistence with operations matching business state changes."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def create_schema(self) -> None:
        Base.metadata.create_all(self.database.engine)

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

    def customer_exists(self, customer_id: str) -> bool:
        with self.database.session() as session:
            return session.get(Customer, customer_id) is not None

    def search_products(self, query: str) -> list[Product]:
        text = query.strip()
        with self.database.session() as session:
            statement = select(Product).where(Product.active.is_(True))
            if text:
                statement = statement.where(
                    or_(Product.name.contains(text), Product.description.contains(text))
                )
            return list(session.scalars(statement.order_by(Product.id)))

    def get_owned_order(self, customer_id: str, order_id: str) -> Order | None:
        with self.database.session() as session:
            return session.scalar(
                select(Order).where(Order.id == order_id, Order.customer_id == customer_id)
            )

    def create_pending_action(
        self, customer_id: str, action_type: str, reason: str, order_id: str | None = None
    ) -> PendingAction:
        action = PendingAction(
            id=str(uuid4()),
            customer_id=customer_id,
            action_type=action_type,
            order_id=order_id,
            reason=reason,
            status=ActionStatus.PENDING_CONFIRMATION.value,
        )
        with self.database.session() as session:
            session.add(action)
        return action

    def submit_pending_action(self, action_id: str, customer_id: str) -> tuple[PendingAction, Ticket]:
        try:
            with self.database.session() as session:
                action = self._owned_action(session, action_id, customer_id)
                existing = session.scalar(select(Ticket).where(Ticket.action_id == action.id))
                if action.status == ActionStatus.SUBMITTED.value and existing is not None:
                    return action, existing
                if action.status != ActionStatus.PENDING_CONFIRMATION.value:
                    raise InvalidActionState(f"Action {action_id} cannot be submitted from {action.status}")

                ticket = existing or Ticket(
                    id=f"T{uuid4().hex[:12].upper()}",
                    customer_id=customer_id,
                    action_id=action.id,
                    ticket_type=action.action_type,
                    status=TicketStatus.OPEN.value,
                    summary=action.reason,
                )
                if existing is None:
                    session.add(ticket)
                action.status = ActionStatus.SUBMITTED.value
                session.flush()
                return action, ticket
        except IntegrityError:
            # The unique action/ticket constraint resolves simultaneous confirmation attempts.
            with self.database.session() as session:
                action = self._owned_action(session, action_id, customer_id)
                ticket = session.scalar(select(Ticket).where(Ticket.action_id == action.id))
                if action.status == ActionStatus.SUBMITTED.value and ticket is not None:
                    return action, ticket
            raise

    def cancel_pending_action(self, action_id: str, customer_id: str) -> PendingAction:
        with self.database.session() as session:
            action = self._owned_action(session, action_id, customer_id)
            if action.status == ActionStatus.CANCELLED.value:
                return action
            if action.status != ActionStatus.PENDING_CONFIRMATION.value:
                raise InvalidActionState(f"Action {action_id} cannot be cancelled from {action.status}")
            action.status = ActionStatus.CANCELLED.value
            session.flush()
            return action

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
        with self.database.session() as session:
            session.add(call)
        return call

    @staticmethod
    def _owned_action(session, action_id: str, customer_id: str) -> PendingAction:
        action = session.get(PendingAction, action_id)
        if action is None or action.customer_id != customer_id:
            raise ToolPermissionError("Pending action is not available to this customer")
        return action
