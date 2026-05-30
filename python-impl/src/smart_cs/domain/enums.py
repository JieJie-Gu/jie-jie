# 定义订单、动作和工具调用等领域枚举。

from enum import Enum


class OrderStatus(str, Enum):
    DELIVERED = "delivered"


class ActionType(str, Enum):
    AFTER_SALES = "after_sales"
    HANDOFF = "handoff"


class ActionStatus(str, Enum):
    PENDING_CONFIRMATION = "pending_confirmation"
    SUBMITTED = "submitted"
    CANCELLED = "cancelled"


class TicketStatus(str, Enum):
    OPEN = "open"


class ToolCallStatus(str, Enum):
    SUCCEEDED = "succeeded"
    REJECTED = "rejected"
