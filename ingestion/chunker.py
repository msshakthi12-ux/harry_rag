"""
Token-aware text chunker for the harry_rag ingestion pipeline.
Uses LangChain's RecursiveCharacterTextSplitter with a HuggingFace tokenizer
to honour the 1 000-token / 200-token overlap budget.
"""

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"
_CHUNK_SIZE = 1000      # tokens
_CHUNK_OVERLAP = 200    # tokens


class Chunker:
    """
    Splits cleaned text into overlapping token-counted chunks and wraps each
    chunk in a LangChain Document with the supplied metadata.

    Args:
        tokenizer_name: HuggingFace model name used for token counting.
        chunk_size:     Maximum tokens per chunk.
        chunk_overlap:  Overlapping tokens between consecutive chunks.
    """

    def __init__(
        self,
        tokenizer_name: str = _DEFAULT_MODEL,
        chunk_size: int = _CHUNK_SIZE,
        chunk_overlap: int = _CHUNK_OVERLAP,
    ) -> None:
        logger.info("Loading tokenizer: %s", tokenizer_name)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        def _token_len(text: str) -> int:
            return len(tokenizer.encode(text, add_special_tokens=False))

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=_token_len,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        logger.info(
            "Chunker ready — size=%d tokens, overlap=%d tokens", chunk_size, chunk_overlap
        )

    def chunk(self, text: str, metadata: dict[str, Any]) -> list[Document]:
        """
        Split *text* into chunks and return a list of Documents.

        Args:
            text:     Cleaned page text.
            metadata: Dict with keys ``book``, ``chapter``, ``page``.

        Returns:
            List of LangChain Documents, each carrying a copy of *metadata*
            extended with a ``chunk_index`` field.
        """
        if not text.strip():
            logger.debug("Skipping empty text for metadata=%s", metadata)
            return []

        raw_chunks = self._splitter.split_text(text)
        documents: list[Document] = []

        for idx, chunk_text in enumerate(raw_chunks):
            doc_metadata = {**metadata, "chunk_index": idx}
            documents.append(Document(page_content=chunk_text, metadata=doc_metadata))

        logger.debug(
            "book=%r page=%s → %d chunks",
            metadata.get("book"),
            metadata.get("page"),
            len(documents),
        )
        return documents