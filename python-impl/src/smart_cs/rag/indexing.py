# 解析 Markdown 知识文档，并按标题 section 生成检索索引记录。
from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter


HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]
KNOWLEDGE_CATEGORIES = {
    "after_sales_policy": "after_sales",
    "shipping_policy": "shipping",
    "product_guide": "product",
    "faq": "faq",
}


def markdown_section_documents(
    document_id: str, category: str, markdown: str
) -> list[Document]:
    sections = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS, strip_headers=True
    ).split_text(markdown)
    documents: list[Document] = []
    for index, section in enumerate(sections):
        content = section.page_content.strip()
        if not content:
            continue
        header_path = " > ".join(
            section.metadata[key] for key in ("h1", "h2", "h3") if key in section.metadata
        )
        context_section = header_path or "root"
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "document_id": document_id,
                    "context_id": f"{document_id}:{context_section}:{index}",
                    "category": category,
                    "header_path": header_path,
                },
            )
        )
    return documents

def load_knowledge_documents(knowledge_directory: Path) -> list[Document]:
    documents: list[Document] = []
    for document_id, category in KNOWLEDGE_CATEGORIES.items():
        markdown = (knowledge_directory / f"{document_id}.md").read_text(encoding="utf-8")
        documents.extend(markdown_section_documents(document_id, category, markdown))
    return documents
