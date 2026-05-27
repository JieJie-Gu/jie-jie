from __future__ import annotations

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


class LocalSentenceEmbeddings(Embeddings):
    """Expose a local Sentence Transformer through the LangChain contract."""

    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
