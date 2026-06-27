# 定义客服系统领域异常类型。

class ToolPermissionError(PermissionError):
    """Raised when a tool request cannot access the customer's resources."""


class BusinessToolRejection(RuntimeError):
    """Represent a safe, auditable business rejection returned to an agent."""

    def __init__(self, result: dict[str, object]) -> None:
        self.result = result
        super().__init__(str(result.get("message") or result.get("reason_code") or "rejected"))


class InvalidActionState(RuntimeError):
    """Raised when a pending action cannot make the requested transition."""


class ConversationBusyError(RuntimeError):
    """Raised when another turn currently owns a conversation graph thread."""


class ConversationLeaseLostError(RuntimeError):
    """Raised when an active turn no longer holds its conversation lease."""
