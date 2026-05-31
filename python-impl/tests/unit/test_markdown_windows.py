# 测试 Markdown 知识文档按标题 section 生成检索块。
from smart_cs.rag.indexing import markdown_section_documents


def test_markdown_headers_create_section_documents() -> None:
    markdown = (
        "# 售后政策\n"
        "## 七天无理由\n"
        "签收后七天内可以申请退货。商品应保持完好。运费按规则承担。"
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
