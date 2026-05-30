# 定义客服系统领域异常类型。

class ToolPermissionError(PermissionError):
    """Raised when a tool request cannot access the customer's resources."""


class InvalidActionState(RuntimeError):
    """Raised when a pending action cannot make the requested transition."""


class ConversationBusyError(RuntimeError):
    """Raised when another turn currently owns a conversation graph thread."""


class ConversationLeaseLostError(RuntimeError):
    """Raised when an active turn no longer holds its conversation lease."""
