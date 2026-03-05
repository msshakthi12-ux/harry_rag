"""
Embedding and ChromaDB persistence layer for the harry_rag ingestion pipeline.
"""

import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
_COLLECTION_NAME = "harry_rag"
_CHROMA_DIR = "chroma_db"


class EmbedStore:
    """
    Wraps HuggingFace BGE embeddings and a persistent ChromaDB collection.

    Args:
        embedding_model: HuggingFace model name for embeddings.
        chroma_dir:      Local directory for ChromaDB persistence.
        collection_name: ChromaDB collection name.
        batch_size:      Number of documents to embed per batch.
    """

    def __init__(
        self,
        embedding_model: str = _EMBEDDING_MODEL,
        chroma_dir: str | Path = _CHROMA_DIR,
        collection_name: str = _COLLECTION_NAME,
        batch_size: int = 64,
    ) -> None:
        self.batch_size = batch_size
        self.collection_name = collection_name
        self.chroma_dir = str(chroma_dir)

        logger.info("Initialising BGE embeddings: %s", embedding_model)
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": self._detect_device()},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.vectorstore: Chroma = self._build_vectorstore()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add_documents(self, documents: list[Document]) -> None:
        """
        Embed and persist *documents* in batches.

        Args:
            documents: LangChain Document objects with page_content and metadata.
        """
        if not documents:
            logger.warning("add_documents called with empty list — nothing to store.")
            return

        total = len(documents)
        logger.info("Embedding %d documents in batches of %d …", total, self.batch_size)

        for start in range(0, total, self.batch_size):
            batch = documents[start : start + self.batch_size]
            try:
                self.vectorstore.add_documents(batch)
                logger.debug("Stored batch [%d:%d]", start, start + len(batch))
            except Exception as exc:
                logger.error("Failed to store batch starting at %d: %s", start, exc)
                raise

        logger.info("Successfully stored %d documents.", total)

    def get_retriever(self, search_type: str = "similarity", **kwargs: Any) -> Any:
        """
        Return a LangChain retriever backed by this vectorstore.

        Args:
            search_type: ``"similarity"`` or ``"mmr"``.
            **kwargs:    Passed through to ``as_retriever`` (e.g. ``k``, ``fetch_k``).
        """
        return self.vectorstore.as_retriever(search_type=search_type, search_kwargs=kwargs)

    @property
    def raw_vectorstore(self) -> Chroma:
        """Direct access to the underlying Chroma vectorstore."""
        return self.vectorstore

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_vectorstore(self) -> Chroma:
        """Create or open the persistent Chroma collection."""
        client = chromadb.PersistentClient(
            path=self.chroma_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        logger.info(
            "ChromaDB persistent client at '%s', collection='%s'",
            self.chroma_dir,
            self.collection_name,
        )
        return Chroma(
            client=client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
        )

    @staticmethod
    def _detect_device() -> str:
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                logger.info("GPU detected — using CUDA for embeddings.")
                return "cuda"
        except ImportError:
            pass
        logger.info("No GPU detected — using CPU for embeddings.")
        return "cpu"