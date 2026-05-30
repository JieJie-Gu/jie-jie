# 测试 Windows 环境下 Markdown 相关行为。

from smart_cs.rag.indexing import markdown_sentence_documents


def test_headers_and_neighbor_sentence_window_are_metadata() -> None:
    markdown = "# 售后政策\n## 七天无理由\n签收后七天内可以申请退货。商品应保持完好。运费按规则承担。"

    documents = markdown_sentence_documents("after_sales_policy", "after_sales", markdown)

    assert documents[0].page_content == "签收后七天内可以申请退货。"
    assert documents[0].metadata["document_id"] == "after_sales_policy"
    assert documents[0].metadata["context_id"] == "after_sales_policy:售后政策 > 七天无理由:0"
    assert documents[0].metadata["category"] == "after_sales"
    assert documents[0].metadata["header_path"] == "售后政策 > 七天无理由"
    assert "商品应保持完好。" in documents[0].metadata["window_text"]
