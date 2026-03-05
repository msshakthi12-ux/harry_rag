"""
FastAPI application for harry_rag.
Exposes POST /query endpoint for question answering.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from ingestion.embed_store import EmbedStore
from llm.generator import RAGGenerator
from retrieval.retriever import RAGRetriever

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment-driven configuration
# ---------------------------------------------------------------------------

CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434")
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))
SEARCH_TYPE = os.getenv("SEARCH_TYPE", "mmr")  # "similarity" | "mmr"


# ---------------------------------------------------------------------------
# Shared application state
# ---------------------------------------------------------------------------

class AppState:
    embed_store: EmbedStore | None = None
    retriever: RAGRetriever | None = None
    generator: RAGGenerator | None = None


_state = AppState()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialise heavy components once at startup."""
    logger.info("Starting harry_rag API …")

    _state.embed_store = EmbedStore(
        embedding_model=EMBEDDING_MODEL,
        chroma_dir=CHROMA_DIR,
    )
    _state.retriever = RAGRetriever(embed_store=_state.embed_store, k=RETRIEVAL_K)
    _state.generator = RAGGenerator(
        model_name=OLLAMA_MODEL,
        ollama_base=OLLAMA_BASE,
    )

    logger.info("harry_rag API ready.")
    yield

    logger.info("Shutting down harry_rag API …")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="harry_rag API",
    description="Retrieval-Augmented Generation over local PDF documents.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000, description="User question")
    book_filter: str | None = Field(None, description="Restrict retrieval to this book name")
    search_type: str = Field("mmr", description="'similarity' or 'mmr'")


class SourceItem(BaseModel):
    book: str
    chapter: str
    page: int | None
    chunk_index: int | None


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    grounded: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Answer a question using RAG",
)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Retrieve relevant document chunks and generate a grounded answer.

    - **question**: Natural language question about the loaded documents.
    - **book_filter**: Optional book name to restrict retrieval scope.
    - **search_type**: Retrieval strategy — ``"similarity"`` or ``"mmr"``.
    """
    if _state.retriever is None or _state.generator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialised yet.",
        )

    search_fn = (
        _state.retriever.mmr_search
        if request.search_type == "mmr"
        else _state.retriever.similarity_search
    )

    try:
        docs = search_fn(query=request.question, book_filter=request.book_filter)
    except Exception as exc:
        logger.error("Retrieval error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval failed: {exc}",
        ) from exc

    try:
        result = _state.generator.generate(
            question=request.question,
            documents=docs,
        )
    except Exception as exc:
        logger.error("Generation error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {exc}",
        ) from exc

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        grounded=result["grounded"],
    )


@app.get("/health", summary="Health check")
async def health() -> dict[str, str]:
    """Returns ``{"status": "ok"}`` when the service is running."""
    return {"status": "ok"}