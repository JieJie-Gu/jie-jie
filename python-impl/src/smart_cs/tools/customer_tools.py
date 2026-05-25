from langchain.tools import tool


@tool
def search_products(query: str) -> dict:
    """Search product facts by a customer-visible query."""
    raise RuntimeError("Executed only through AuthorizedToolExecutor")


@tool
def lookup_order(customer_id: str, order_id: str) -> dict:
    """Read an order owned by the current customer."""
    raise RuntimeError("Executed only through AuthorizedToolExecutor")


@tool
def draft_after_sales(customer_id: str, order_id: str, reason: str) -> dict:
    """Prepare a return or refund request that still needs confirmation."""
    raise RuntimeError("Executed only through AuthorizedToolExecutor")


@tool
def draft_handoff(customer_id: str, reason: str) -> dict:
    """Prepare an escalation request that still needs confirmation."""
    raise RuntimeError("Executed only through AuthorizedToolExecutor")


CUSTOMER_TOOL_SCHEMAS = (search_products, lookup_order, draft_after_sales, draft_handoff)
