"""
LLM generation layer for harry_rag.
Uses a locally running Ollama model (Mistral / Llama-3 Instruct) via
LangChain's ChatOllama integration.
"""

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough

from llm.prompt import RAG_PROMPT, format_context

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "mistral"
_DEFAULT_OLLAMA_BASE = "http://localhost:11434"

_OUT_OF_SCOPE_MARKERS = [
    "cannot answer this question based on the available documents",
    "outside the scope of the available documents",
]


def _load_chat_ollama(model_name: str, ollama_base: str, temperature: float, top_p: float):
    """
    Try multiple import paths for ChatOllama across LangChain versions.
    """
    # Try langchain-ollama package first (newest)
    try:
        from langchain_ollama import ChatOllama
        logger.info("Using langchain_ollama.ChatOllama")
        return ChatOllama(
            model=model_name,
            base_url=ollama_base,
            temperature=temperature,
        )
    except ImportError:
        pass

    # Fall back to langchain_community
    try:
        from langchain_community.chat_models import ChatOllama
        logger.info("Using langchain_community.chat_models.ChatOllama")
        return ChatOllama(
            model=model_name,
            base_url=ollama_base,
            temperature=temperature,
        )
    except ImportError:
        pass

    raise ImportError(
        "Could not import ChatOllama. "
        "Run: pip install langchain-ollama"
    )


class RAGGenerator:
    """
    Wraps a local Ollama LLM in a LangChain LCEL chain for RAG generation.

    Args:
        model_name:   Ollama model tag (e.g. 'mistral' or 'llama3').
        ollama_base:  Base URL of the locally running Ollama service.
        temperature:  Sampling temperature (0 = deterministic).
        top_p:        Nucleus sampling parameter.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        ollama_base: str = _DEFAULT_OLLAMA_BASE,
        temperature: float = 0.0,
        top_p: float = 0.9,
    ) -> None:
        logger.info(
            "Initialising Ollama LLM | model=%s base_url=%s", model_name, ollama_base
        )
        self._llm = _load_chat_ollama(model_name, ollama_base, temperature, top_p)
        self._chain: Runnable = self._build_chain()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(
        self,
        question: str,
        documents: list[Document],
    ) -> dict[str, Any]:
        """
        Generate an answer to *question* grounded in *documents*.

        Args:
            question:  The user's question.
            documents: Retrieved context documents.

        Returns:
            Dict with keys:
            - answer   (str): LLM response text.
            - sources  (list[dict]): Metadata for each source document.
            - grounded (bool): False if model signalled it cannot answer.
        """
        if not documents:
            logger.warning("generate() called with no documents for question=%r", question)
            return {
                "answer": "I cannot answer this question based on the available documents.",
                "sources": [],
                "grounded": False,
            }

        context_str = format_context(documents)
        logger.info(
            "Generating answer | question=%r | context_docs=%d", question, len(documents)
        )

        try:
            answer: str = self._chain.invoke(
                {"context": context_str, "question": question}
            )
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            raise

        grounded = not any(marker in answer.lower() for marker in _OUT_OF_SCOPE_MARKERS)
        sources = self._extract_sources(documents)

        logger.info(
            "Answer generated | grounded=%s | length=%d chars", grounded, len(answer)
        )
        return {"answer": answer, "sources": sources, "grounded": grounded}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_chain(self) -> Runnable:
        """Assemble the LCEL chain: prompt | llm | output_parser."""
        return (
            RunnablePassthrough()
            | RAG_PROMPT
            | self._llm
            | StrOutputParser()
        )

    @staticmethod
    def _extract_sources(documents: list[Document]) -> list[dict[str, Any]]:
        """Deduplicate and serialise source metadata from retrieved documents."""
        seen: set[tuple] = set()
        sources: list[dict[str, Any]] = []
        for doc in documents:
            meta = doc.metadata
            key = (meta.get("book"), meta.get("chapter"), meta.get("page"))
            if key not in seen:
                seen.add(key)
                sources.append(
                    {
                        "book": meta.get("book", "unknown"),
                        "chapter": meta.get("chapter", "unknown"),
                        "page": meta.get("page"),
                        "chunk_index": meta.get("chunk_index"),
                    }
                )
        return sources