# 实现 RAG 查询改写和安全的分类过滤表达式生成。
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
        if "物流" in rewritten and any(term in rewritten for term in ("更新", "不更新", "没动")):
            return f"{rewritten} 配送状态说明"
        if any(term in rewritten for term in ("清洁", "清洗", "沾污")):
            return f"{rewritten} 鞋类清洁保养"
        return rewritten


class QueryPolicy:
    """Translate rewritten queries only to allow-listed document categories."""

    CATEGORY_TERMS = {
        "after_sales": (
            "退货",
            "退款",
            "售后",
            "换货",
            "开胶",
            "破损",
            "脱线",
            "损坏",
            "质量问题",
            "凭证",
        ),
        "shipping": (
            "配送",
            "物流",
            "运费",
            "发货",
            "派送",
            "签收",
            "地址",
            "包裹",
        ),
        "product": (
            "尺码",
            "材质",
            "保养",
            "清洁",
            "清洗",
            "鞋面",
            "鞋类",
            "跑鞋",
            "通勤",
            "防滑",
            "宽脚",
            "商品",
            "产品",
        ),
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
