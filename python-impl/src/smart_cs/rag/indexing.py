# 解析、切分并准备知识文档索引记录。

from __future__ import annotations

import re
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


def _sentences(text: str) -> list[str]:
    return [
        part.strip()
        for part in re.split(r"(?<=[。！？!?])\s*", text.strip())
        if part.strip()
    ]


def markdown_sentence_documents(
    document_id: str, category: str, markdown: str
) -> list[Document]:
    sections = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS, strip_headers=True
    ).split_text(markdown)
    documents: list[Document] = []
    for section in sections:
        sentences = _sentences(section.page_content)
        header_path = " > ".join(
            section.metadata[key] for key in ("h1", "h2", "h3") if key in section.metadata
        )
        for index, sentence in enumerate(sentences):
            start = max(0, index - 1)
            end = min(len(sentences), index + 2)
            documents.append(
                Document(
                    page_content=sentence,
                    metadata={
                        "document_id": document_id,
                        "context_id": f"{document_id}:{header_path}:{index}",
                        "category": category,
                        "header_path": header_path,
                        "window_text": "".join(sentences[start:end]),
                    },
                )
            )
    return documents


def load_knowledge_documents(knowledge_directory: Path) -> list[Document]:
    documents: list[Document] = []
    for document_id, category in KNOWLEDGE_CATEGORIES.items():
        markdown = (knowledge_directory / f"{document_id}.md").read_text(encoding="utf-8")
        documents.extend(markdown_sentence_documents(document_id, category, markdown))
    return documents
