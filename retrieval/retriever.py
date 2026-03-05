"""
Retriever module for harry_rag.
Supports similarity search, MMR search, and optional metadata filtering.
"""

import logging
from typing import Any

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

from ingestion.embed_store import EmbedStore

logger = logging.getLogger(__name__)

_DEFAULT_K = 5
_DEFAULT_FETCH_K = 20  # used by MMR before re-ranking


class RAGRetriever:
    """
    High-level retriever that delegates to an EmbedStore and supports
    similarity, MMR, and filtered searches.

    Args:
        embed_store: An initialised EmbedStore instance.
        k:           Number of documents to return.
        fetch_k:     Candidate pool size for MMR (ignored for similarity).
    """

    def __init__(
        self,
        embed_store: EmbedStore,
        k: int = _DEFAULT_K,
        fetch_k: int = _DEFAULT_FETCH_K,
    ) -> None:
        self.embed_store = embed_store
        self.k = k
        self.fetch_k = fetch_k

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def similarity_search(
        self,
        query: str,
        book_filter: str | None = None,
    ) -> list[Document]:
        """
        Retrieve the *k* most similar documents for *query*.

        Args:
            query:       User question.
            book_filter: If provided, restrict results to this book name.

        Returns:
            List of Documents ordered by descending relevance.
        """
        where = self._build_filter(book_filter)
        logger.info("Similarity search | query=%r filter=%s k=%d", query, where, self.k)
        try:
            results = self.embed_store.raw_vectorstore.similarity_search(
                query=query,
                k=self.k,
                filter=where,
            )
            logger.info("Similarity search returned %d docs.", len(results))
            return results
        except Exception as exc:
            logger.error("Similarity search failed: %s", exc)
            raise

    def mmr_search(
        self,
        query: str,
        book_filter: str | None = None,
        lambda_mult: float = 0.5,
    ) -> list[Document]:
        """
        Retrieve *k* documents using Maximal Marginal Relevance to balance
        relevance and diversity.

        Args:
            query:        User question.
            book_filter:  Optional book name filter.
            lambda_mult:  MMR diversity parameter in [0, 1].
                          0 → max diversity, 1 → max relevance.

        Returns:
            List of Documents.
        """
        where = self._build_filter(book_filter)
        logger.info(
            "MMR search | query=%r filter=%s k=%d fetch_k=%d lambda=%.2f",
            query, where, self.k, self.fetch_k, lambda_mult,
        )
        try:
            results = self.embed_store.raw_vectorstore.max_marginal_relevance_search(
                query=query,
                k=self.k,
                fetch_k=self.fetch_k,
                lambda_mult=lambda_mult,
                filter=where,
            )
            logger.info("MMR search returned %d docs.", len(results))
            return results
        except Exception as exc:
            logger.error("MMR search failed: %s", exc)
            raise

    def as_langchain_retriever(
        self,
        search_type: str = "similarity",
        book_filter: str | None = None,
    ) -> BaseRetriever:
        """
        Return a LangChain-compatible retriever (for use in LCEL chains).

        Args:
            search_type: ``"similarity"`` or ``"mmr"``.
            book_filter: Optional book name for metadata filtering.
        """
        search_kwargs: dict[str, Any] = {"k": self.k}
        if search_type == "mmr":
            search_kwargs["fetch_k"] = self.fetch_k
        if book_filter:
            search_kwargs["filter"] = self._build_filter(book_filter)

        return self.embed_store.raw_vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_filter(book_filter: str | None) -> dict[str, str] | None:
        if book_filter:
            return {"book": book_filter}
        return None