"""
Hybrid search module for harry_rag.
Combines BM25 lexical search with dense vector retrieval and fuses scores
using Reciprocal Rank Fusion (RRF).
Optionally reranks results with BAAI/bge-reranker-large.
"""

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from ingestion.embed_store import EmbedStore
from retrieval.retriever import RAGRetriever

logger = logging.getLogger(__name__)

_RERANKER_MODEL = "BAAI/bge-reranker-large"
_RRF_K = 60  # RRF constant


class HybridSearcher:
    """
    Executes hybrid search over a corpus of Documents using BM25 + vector
    retrieval fused via Reciprocal Rank Fusion (RRF).

    Optionally applies a cross-encoder reranker as a final step.

    Args:
        documents:       Full list of Documents (used to build the BM25 index).
        embed_store:     Initialised EmbedStore for dense retrieval.
        k:               Number of documents to return after fusion.
        use_reranker:    Whether to apply the BGE reranker.
        reranker_model:  HuggingFace model path / name for the reranker.
    """

    def __init__(
        self,
        documents: list[Document],
        embed_store: EmbedStore,
        k: int = 5,
        use_reranker: bool = False,
        reranker_model: str = _RERANKER_MODEL,
    ) -> None:
        self.k = k
        self.use_reranker = use_reranker

        logger.info("Building BM25 index over %d documents …", len(documents))
        self.bm25_retriever = BM25Retriever.from_documents(documents, k=k * 2)

        self.dense_retriever = RAGRetriever(embed_store=embed_store, k=k * 2)

        self._reranker = None
        if use_reranker:
            self._reranker = self._load_reranker(reranker_model)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        book_filter: str | None = None,
    ) -> list[Document]:
        """
        Run hybrid search and return up to *k* reranked documents.

        Args:
            query:       User question.
            book_filter: Optional book-level metadata filter for dense leg.

        Returns:
            Fused (and optionally reranked) list of Documents.
        """
        logger.info("Hybrid search | query=%r book_filter=%r", query, book_filter)

        bm25_docs = self._bm25_retrieve(query)
        dense_docs = self.dense_retriever.similarity_search(query, book_filter=book_filter)

        fused = self._reciprocal_rank_fusion([bm25_docs, dense_docs])
        top_k = fused[: self.k]

        if self._reranker is not None:
            top_k = self._rerank(query, top_k)

        logger.info("Hybrid search returning %d documents.", len(top_k))
        return top_k

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _bm25_retrieve(self, query: str) -> list[Document]:
        try:
            return self.bm25_retriever.get_relevant_documents(query)
        except Exception as exc:
            logger.warning("BM25 retrieval failed: %s — returning empty list.", exc)
            return []

    @staticmethod
    def _reciprocal_rank_fusion(
        ranked_lists: list[list[Document]],
        k: int = _RRF_K,
    ) -> list[Document]:
        """
        Combine multiple ranked lists of Documents into one using RRF.
        Documents are identified by their page_content (hash).
        """
        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for ranked in ranked_lists:
            for rank, doc in enumerate(ranked, start=1):
                doc_id = str(hash(doc.page_content))
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
                doc_map[doc_id] = doc

        sorted_ids = sorted(scores, key=lambda d: scores[d], reverse=True)
        return [doc_map[doc_id] for doc_id in sorted_ids]

    def _rerank(self, query: str, documents: list[Document]) -> list[Document]:
        """Apply cross-encoder reranking to *documents*."""
        if not documents:
            return documents

        pairs = [[query, doc.page_content] for doc in documents]
        try:
            scores: list[float] = self._reranker.predict(pairs).tolist()
            ranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
            reranked = [doc for _, doc in ranked]
            logger.debug("Reranker scores: %s", [round(s, 4) for s, _ in ranked])
            return reranked
        except Exception as exc:
            logger.warning("Reranker failed (%s) — returning pre-rerank order.", exc)
            return documents

    @staticmethod
    def _load_reranker(model_name: str) -> Any:
        """Load the BGE cross-encoder reranker."""
        try:
            from sentence_transformers import CrossEncoder  # noqa: PLC0415

            logger.info("Loading reranker: %s", model_name)
            return CrossEncoder(model_name)
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for reranking. "
                "Install it with: pip install sentence-transformers"
            ) from exc