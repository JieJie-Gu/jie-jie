# 测试 Markdown 知识库按标题 section 生成 RAG 检索块。
from pathlib import Path
from collections import Counter

from smart_cs.rag.indexing import load_knowledge_documents, markdown_section_documents


def test_markdown_headers_create_section_documents() -> None:
    markdown = (
        "# 售后政策\n"
        "## 七天无理由\n"
        "签收后七天内可以申请退货。商品应保持完好。运费按规则承担。\n"
    )

    documents = markdown_section_documents("after_sales_policy", "after_sales", markdown)

    assert len(documents) == 1
    assert "签收后七天内可以申请退货" in documents[0].page_content
    assert "商品应保持完好" in documents[0].page_content
    assert documents[0].metadata["document_id"] == "after_sales_policy"
    assert documents[0].metadata["context_id"] == "after_sales_policy:售后政策 > 七天无理由:0"
    assert documents[0].metadata["category"] == "after_sales"
    assert documents[0].metadata["header_path"] == "售后政策 > 七天无理由"
    assert "window_text" not in documents[0].metadata


def test_knowledge_documents_have_enough_section_chunks() -> None:
    root = Path(__file__).parents[2] / "data" / "knowledge"
    documents = load_knowledge_documents(root)
    counts = Counter(document.metadata["document_id"] for document in documents)

    assert counts == {
        "after_sales_policy": 10,
        "shipping_policy": 10,
        "product_guide": 10,
        "faq": 10,
    }
    for document in documents:
        assert document.page_content.strip()
        assert document.metadata["document_id"]
        assert document.metadata["category"] in {"after_sales", "shipping", "product", "faq"}
        assert document.metadata["header_path"]
        assert document.metadata["context_id"].startswith(f"{document.metadata['document_id']}:")
        assert "window_text" not in document.metadata
