# 实现 RAG 查询策略、检索、重排和上下文组装。

from __future__ import annotations

from typing import Protocol


class QueryRewriter(Protocol):
    def rewrite_query(self, query: str) -> str: ...


class RuleBasedQueryRewriter:
    """Normalize common customer phrasing without requiring an external model."""

    def rewrite_query(self, query: str) -> str:
        rewritten = " ".join(query.strip().split())
        if "退货" in rewritten and any(term in rewritten for term in ("几天", "多久", "截止", "期限")):
            return f"{rewritten} 退货期限"
        if "物流" in rewritten and "更新" in rewritten:
            return f"{rewritten} 配送状态说明"
        return rewritten


class QueryPolicy:
    """Translate rewritten queries only to allow-listed document categories."""

    CATEGORY_TERMS = {
        "after_sales": ("退货", "退款", "售后", "换货"),
        "shipping": ("配送", "物流", "运费", "发货"),
        "product": ("尺码", "材质", "保养", "产品", "商品"),
    }

    def __init__(self, rewrite_model: QueryRewriter | None = None) -> None:
        self.rewrite_model = rewrite_model

    def prepare(self, query: str) -> tuple[str, str]:
        rewritten = (
            self.rewrite_model.rewrite_query(query)
            if self.rewrite_model is not None
            else query.strip()
        )
        category = next(
            (
                name
                for name, terms in self.CATEGORY_TERMS.items()
                if any(term in rewritten for term in terms)
            ),
            "faq",
        )
        return rewritten, f'category == "{category}"'
