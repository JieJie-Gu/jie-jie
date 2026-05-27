# Text RAG With Milvus And Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为知识问答增加 Markdown 检索链路：标题分块、Sentence Window 上下文、查询重写、元数据过滤、Milvus dense + BM25 混合检索、RRF 排序和四项可复现评估。

**Architecture:** 知识源只维护 Markdown。索引流程使用 LangChain `MarkdownHeaderTextSplitter` 产生章节文档，再由轻量应用函数把每个句子扩展成可回填的窗口元数据。`langchain_milvus.Milvus` 与 `BM25BuiltInFunction` 管理稠密和稀疏向量；生产检索发起一次带 allow-list metadata expression 的混合搜索，并使用 RRF。`KnowledgeAgent` 返回引用证据；订单事实仍只读取工具。

**Tech Stack:** Python 3.11, LangChain Documents and Text Splitters, `langchain-milvus`, Milvus Standalone 2.5+, Sentence Transformers, LangGraph, pytest

---

## Prerequisite And Scope

先完成 [2026-05-25-agent-foundation-orchestration.md](./2026-05-25-agent-foundation-orchestration.md)。

本计划只实现已确认的 RAG 子集：

```text
MarkdownHeaderTextSplitter + Sentence Window Metadata
Query Rewrite + Metadata Filter
Milvus dense search + BM25 sparse search + RRF
Faithfulness + Answer Relevancy + Context Recall + Context Precision
```

不加入 PDF/DOCX 转换、图片索引、HyDE、语义分块、Cross-encoder、GraphRAG、Text-to-SQL 或管理页面。

## Official Component Decisions

| Need | Adopted official component | Application code retained |
| --- | --- | --- |
| Markdown 章节切分 | `langchain_text_splitters.MarkdownHeaderTextSplitter` | 句子窗口元数据 |
| 标准文档载体 | `langchain_core.documents.Document` | 电商 metadata allow-list |
| Milvus 混合存储 | `langchain_milvus.Milvus.from_documents` + `BM25BuiltInFunction` | collection 配置 |
| 检索过滤 | Milvus `expr` passed through LangChain integration | query-to-category mapping |
| 融合排序 | Milvus/LangChain hybrid search with `ranker_type="rrf"` | 评估用结果序列 |

官方参考：

- <https://docs.langchain.com/oss/python/integrations/vectorstores/milvus>
- <https://reference.langchain.com/python/langchain-milvus/function/BM25BuiltInFunction>
- <https://reference.langchain.com/python/langchain-text-splitters/>
- <https://milvus.io/docs/bm25-function.md>
- <https://milvus.io/docs/multi-vector-search.md>
- <https://milvus.io/docs/reranking.md>

`BM25BuiltInFunction` 需要 Milvus Standalone 或 Distributed；本项目不把 Milvus Lite 宣称为该功能的运行环境。

## File Map

Create:

```text
python-impl/data/knowledge/after_sales_policy.md
python-impl/data/knowledge/shipping_policy.md
python-impl/data/knowledge/product_guide.md
python-impl/data/knowledge/faq.md
python-impl/data/evaluation/rag_cases.json
python-impl/src/smart_cs/rag/indexing.py
python-impl/src/smart_cs/rag/embeddings.py
python-impl/src/smart_cs/rag/vector_store.py
python-impl/src/smart_cs/rag/retrieval.py
python-impl/src/smart_cs/rag/evaluation.py
python-impl/src/smart_cs/agents/knowledge.py
python-impl/scripts/index_knowledge.py
python-impl/scripts/evaluate_rag.py
python-impl/tests/unit/test_markdown_windows.py
python-impl/tests/unit/test_query_policy.py
python-impl/tests/integration/test_milvus_hybrid.py
python-impl/tests/api/test_knowledge_reply.py
```

Modify:

```text
python-impl/pyproject.toml
python-impl/.env.example
python-impl/src/smart_cs/config.py
python-impl/src/smart_cs/agents/state.py
python-impl/src/smart_cs/application/agent_runtime.py
python-impl/src/smart_cs/api/dependencies.py
docker-compose.yml
```

### Task 1: Index Markdown Sections Into Sentence Windows

**Files:**
- Create: `python-impl/data/knowledge/after_sales_policy.md`
- Create: `python-impl/data/knowledge/shipping_policy.md`
- Create: `python-impl/data/knowledge/product_guide.md`
- Create: `python-impl/data/knowledge/faq.md`
- Create: `python-impl/src/smart_cs/rag/indexing.py`
- Create: `python-impl/tests/unit/test_markdown_windows.py`
- Modify: `python-impl/pyproject.toml`

- [x] **Step 1: Write the failing indexing test**

```python
# python-impl/tests/unit/test_markdown_windows.py
from smart_cs.rag.indexing import markdown_sentence_documents


def test_headers_and_neighbor_sentence_window_are_metadata() -> None:
    markdown = "# 售后政策\n## 七天无理由\n签收后七天内可以申请退货。商品应保持完好。运费按规则承担。"
    documents = markdown_sentence_documents("after_sales_policy", "after_sales", markdown)

    assert documents[0].page_content == "签收后七天内可以申请退货。"
    assert documents[0].metadata["category"] == "after_sales"
    assert documents[0].metadata["header_path"] == "售后政策 > 七天无理由"
    assert "商品应保持完好。" in documents[0].metadata["window_text"]
```

- [x] **Step 2: Add dependencies and run red test**

Add:

```toml
  "langchain-text-splitters>=1.0,<2",
```

Run:

```bash
cd python-impl
pytest tests/unit/test_markdown_windows.py -q
```

Expected: FAIL importing `smart_cs.rag.indexing`.

- [x] **Step 3: Implement header splitting with window enrichment**

```python
# python-impl/src/smart_cs/rag/indexing.py
import re
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[。！？!?])\s*", text.strip()) if part.strip()]


def markdown_sentence_documents(document_id: str, category: str, markdown: str) -> list[Document]:
    sections = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS, strip_headers=True).split_text(markdown)
    documents: list[Document] = []
    for section in sections:
        sentences = _sentences(section.page_content)
        header_path = " > ".join(section.metadata[key] for key in ("h1", "h2", "h3") if key in section.metadata)
        for index, sentence in enumerate(sentences):
            start, end = max(0, index - 1), min(len(sentences), index + 2)
            documents.append(Document(
                page_content=sentence,
                metadata={
                    "document_id": document_id,
                    "category": category,
                    "header_path": header_path,
                    "window_text": "".join(sentences[start:end]),
                },
            ))
    return documents
```

- [x] **Step 4: Write four small curated Markdown files and verify**

Documents must use `#` and `##` headings and contain only demonstrable study data: return period, evidence requirement, shipping status explanation, and sample product maintenance facts. Do not insert real-company claims.

```bash
cd python-impl
pytest tests/unit/test_markdown_windows.py -q
git add pyproject.toml data/knowledge src/smart_cs/rag/indexing.py tests/unit/test_markdown_windows.py
git commit -m "feat: split markdown knowledge into sentence windows"
```

Expected: PASS.

### Task 2: Build LangChain Milvus Dense And BM25 Hybrid Store

**Files:**
- Create: `python-impl/src/smart_cs/rag/embeddings.py`
- Create: `python-impl/src/smart_cs/rag/vector_store.py`
- Create: `python-impl/scripts/index_knowledge.py`
- Create: `python-impl/tests/integration/test_milvus_hybrid.py`
- Modify: `python-impl/pyproject.toml`
- Modify: `python-impl/src/smart_cs/config.py`
- Modify: `python-impl/.env.example`
- Modify: `docker-compose.yml`

- [x] **Step 1: Add a Milvus integration test**

```python
# python-impl/tests/integration/test_milvus_hybrid.py
import pytest

from smart_cs.rag.vector_store import build_hybrid_store


@pytest.mark.integration
def test_hybrid_search_returns_filtered_after_sales_sentence(settings, embeddings, after_sales_documents) -> None:
    store = build_hybrid_store(settings, embeddings, after_sales_documents, drop_old=True)
    results = store.similarity_search(
        "鞋子退货期限",
        k=2,
        expr='category == "after_sales"',
        ranker_type="rrf",
        ranker_params={"k": 60},
    )
    assert results
    assert all(item.metadata["category"] == "after_sales" for item in results)
```

- [x] **Step 2: Configure a real local embedding and Milvus Standalone**

Add:

```toml
  "langchain-milvus>=0.3.3,<1",
  "sentence-transformers>=3.4,<4",
```

Add settings:

```python
    milvus_uri: str = "http://localhost:19530"
    milvus_collection: str = "smart_cs_knowledge"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
```

Add `.env.example` values:

```dotenv
SMART_CS_MILVUS_URI=http://localhost:19530
SMART_CS_MILVUS_COLLECTION=smart_cs_knowledge
SMART_CS_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
```

Replace the repository root `docker-compose.yml` with Milvus official standalone dependencies (`etcd`, `minio`, `standalone`) and the Python API service; pin the same Milvus version used in the test environment and expose port `19530`.

- [x] **Step 3: Wrap local embeddings in the LangChain interface**

```python
# python-impl/src/smart_cs/rag/embeddings.py
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


class LocalSentenceEmbeddings(Embeddings):
    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
```

- [x] **Step 4: Use the official Milvus integration rather than constructing sparse vectors**

```python
# python-impl/src/smart_cs/rag/vector_store.py
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_milvus import BM25BuiltInFunction, Milvus


def build_hybrid_store(settings, embeddings: Embeddings, documents: list[Document], drop_old: bool) -> Milvus:
    return Milvus.from_documents(
        documents=documents,
        embedding=embeddings,
        builtin_function=BM25BuiltInFunction(analyzer_params={"type": "chinese"}),
        vector_field=["dense", "sparse"],
        connection_args={"uri": settings.milvus_uri},
        collection_name=settings.milvus_collection,
        consistency_level="Strong",
        drop_old=drop_old,
    )
```

`scripts/index_knowledge.py` reads the four Markdown files, calls `markdown_sentence_documents`, builds `LocalSentenceEmbeddings`, and calls `build_hybrid_store(..., drop_old=True)`. No application module creates BM25 vocabulary, sparse vectors, or its own reciprocal-rank algorithm.

- [x] **Step 5: Start Milvus, index, test, and commit**

```bash
docker compose up -d etcd minio standalone
cd python-impl
python scripts/index_knowledge.py
pytest tests/integration/test_milvus_hybrid.py -q
git add pyproject.toml .env.example src/smart_cs/rag scripts/index_knowledge.py tests/integration/test_milvus_hybrid.py ../docker-compose.yml
git commit -m "feat: index knowledge with langchain milvus hybrid search"
```

Expected: the integration test PASSes with a result in the `after_sales` category using `ranker_type="rrf"` from the pinned official integration. Do not replace it with a handwritten hybrid-search adapter.

### Task 3: Retrieve With Query Rewrite, Metadata Filter And Citations

**Files:**
- Create: `python-impl/src/smart_cs/rag/retrieval.py`
- Create: `python-impl/src/smart_cs/agents/knowledge.py`
- Create: `python-impl/tests/unit/test_query_policy.py`
- Modify: `python-impl/src/smart_cs/agents/state.py`
- Modify: `python-impl/src/smart_cs/application/agent_runtime.py`
- Modify: `python-impl/src/smart_cs/api/dependencies.py`
- Create: `python-impl/tests/api/test_knowledge_reply.py`

- [x] **Step 1: Test allow-listed filtering and citation response**

```python
def test_query_category_filter_is_not_user_supplied_expression() -> None:
    policy = QueryPolicy()
    rewritten, expression = policy.prepare("退货什么时候截止")
    assert "退货" in rewritten
    assert expression == 'category == "after_sales"'


def test_knowledge_answer_exposes_window_citation(fake_store) -> None:
    answer = KnowledgeAgent(fake_store, RuleBasedRewriter()).answer("退货期限")
    assert answer.citations[0].header_path == "售后政策 > 七天无理由"
    assert "签收后七天" in answer.contexts[0]
```

- [x] **Step 2: Implement only approved query enhancement**

```python
# python-impl/src/smart_cs/rag/retrieval.py
class QueryPolicy:
    CATEGORY_TERMS = {
        "after_sales": ("退货", "退款", "售后", "换货"),
        "shipping": ("配送", "物流", "运费", "发货"),
        "product": ("尺码", "材质", "保养", "产品"),
    }

    def __init__(self, rewrite_model=None) -> None:
        self.rewrite_model = rewrite_model

    def prepare(self, query: str) -> tuple[str, str]:
        rewritten = self.rewrite_model.rewrite_query(query) if self.rewrite_model else query.strip()
        category = next(
            (name for name, terms in self.CATEGORY_TERMS.items() if any(term in rewritten for term in terms)),
            "faq",
        )
        return rewritten, f'category == "{category}"'
```

`KnowledgeAgent.answer` calls:

```python
documents = store.similarity_search(
    rewritten_query,
    k=4,
    expr=category_expression,
    ranker_type="rrf",
    ranker_params={"k": 60},
)
contexts = [document.metadata["window_text"] for document in documents]
```

It answers only when retrieved evidence contains relevant policy text, returns `document_id` and `header_path` citations, and returns a clarification/handoff message when evidence is insufficient. It never returns order status from these documents.

- [x] **Step 3: Connect the existing Supervisor decision to `KnowledgeAgent`**

Add `KnowledgeAgent` to the specialist registry. When the Supervisor plan includes it, invoke the knowledge agent and send citations to `ResponseGuard`. Do not introduce another router or free-form retrieval tool.

- [x] **Step 4: Verify API and commit**

```bash
cd python-impl
pytest tests/unit/test_query_policy.py tests/api/test_knowledge_reply.py -q
git add src/smart_cs tests
git commit -m "feat: answer knowledge questions with cited hybrid retrieval"
```

Expected: PASS; response citations identify the Markdown section and window evidence.

### Task 4: Generate Four RAG Acceptance Metrics

**Files:**
- Create: `python-impl/data/evaluation/rag_cases.json`
- Create: `python-impl/src/smart_cs/rag/evaluation.py`
- Create: `python-impl/scripts/evaluate_rag.py`
- Create: `python-impl/tests/unit/test_rag_evaluation.py`

- [x] **Step 1: Add deterministic scoring tests**

```python
def test_context_precision_and_recall_on_labelled_case() -> None:
    case = EvaluationCase(expected_context_ids=["policy-returns-1", "policy-condition-1"])
    result = score_contexts(case, retrieved_ids=["policy-returns-1", "shipping-2"])
    assert result.context_precision == 0.5
    assert result.context_recall == 0.5
```

- [x] **Step 2: Add a small labelled evaluation set**

Create eight Chinese questions distributed over `after_sales`, `shipping`, `product`, and `faq`. Each record contains:

```json
{
  "question": "签收后几天可以申请退货？",
  "expected_answer_points": ["签收后七天内"],
  "expected_context_ids": ["after_sales_policy:七天无理由:0"],
  "category": "after_sales"
}
```

- [x] **Step 3: Implement and report exactly four metrics**

`evaluate_rag.py` indexes the curated knowledge, queries `KnowledgeAgent`, computes:

- `context_precision`: relevant retrieved context identifiers divided by retrieved identifiers.
- `context_recall`: relevant retrieved context identifiers divided by expected identifiers.
- `faithfulness`: fraction of generated answer points that can be matched in retrieved context text.
- `answer_relevancy`: fraction of labelled expected answer points matched in the answer.

It writes both:

```text
python-impl/data/evaluation/latest_results.json
python-impl/data/evaluation/latest_results.md
```

The Markdown file is generated from the numeric JSON in the same run and includes case count, model mode, embedding model and timestamp. Documentation in plan 3 links to this generated artifact; it does not hard-code invented scores.

- [x] **Step 4: Execute and commit measured output**

```bash
cd python-impl
pytest tests/unit/test_rag_evaluation.py -q
python scripts/evaluate_rag.py
git add data/evaluation src/smart_cs/rag/evaluation.py scripts/evaluate_rag.py tests/unit/test_rag_evaluation.py
git commit -m "test: add measured rag acceptance report"
```

Expected: both result files exist and contain only the four approved metric names.

## Acceptance Checklist

- [x] Raw text knowledge is Markdown; image evidence is not written to Milvus.
- [x] Indexing visibly uses `MarkdownHeaderTextSplitter` and sentence windows.
- [x] Retrieval visibly uses `BM25BuiltInFunction`, dense embeddings and RRF.
- [x] User input cannot provide arbitrary Milvus expressions; filtering is allow-listed.
- [x] Knowledge answers contain citations and do not fabricate real-time order status.
- [x] Evaluation output reports only measured Faithfulness, Answer Relevancy, Context Recall and Context Precision.
