"""
Streamlit UI for harry_rag.
Run with: streamlit run app.py
"""

import logging
from pathlib import Path

import streamlit as st

from ingestion.embed_store import EmbedStore
from ingestion.pdf_loader import PDFLoader, PDFLoadError
from ingestion.text_cleaner import TextCleaner
from ingestion.chunker import Chunker
from llm.generator import RAGGenerator
from retrieval.retriever import RAGRetriever
from retrieval.hybrid_search import HybridSearcher
from evaluation.rag_eval import RAGEvaluator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PDF_DIR = Path("data/pdf_books")
CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
OLLAMA_MODEL = "mistral"
OLLAMA_BASE = "http://localhost:11434"

# ---------------------------------------------------------------------------
# Cached resource initialisers (run once per Streamlit session)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading embeddings and vector store …")
def get_embed_store() -> EmbedStore:
    return EmbedStore(embedding_model=EMBEDDING_MODEL, chroma_dir=CHROMA_DIR)


@st.cache_resource(show_spinner="Initialising retriever …")
def get_retriever(k: int = 5) -> RAGRetriever:
    return RAGRetriever(embed_store=get_embed_store(), k=k)


@st.cache_resource(show_spinner="Loading LLM …")
def get_generator() -> RAGGenerator:
    return RAGGenerator(model_name=OLLAMA_MODEL, ollama_base=OLLAMA_BASE)


# ---------------------------------------------------------------------------
# Helper: ingestion pipeline
# ---------------------------------------------------------------------------


def run_ingestion(pdf_dir: Path) -> int:
    """
    Run the full ingestion pipeline on *pdf_dir*.
    Returns the total number of chunks stored.
    """
    loader = PDFLoader(pdf_dir)
    cleaner = TextCleaner()
    chunker = Chunker()
    store = get_embed_store()

    all_docs = []
    try:
        for pdf_page in loader.load_all():
            cleaned = cleaner.clean(pdf_page.text)
            chunks = chunker.chunk(cleaned, pdf_page.metadata)
            all_docs.extend(chunks)
    except PDFLoadError as exc:
        st.error(f"PDF loading error: {exc}")
        return 0

    if all_docs:
        store.add_documents(all_docs)

    return len(all_docs)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="ChatBot - RAG",
        page_icon="📚",
        layout="wide",
    )
    st.title("📚 ChatBot for Potterheads - RAG")
    st.caption("Are you a Harry Potter fan? Ask questions about the books and get answers grounded in the text!")

    # -----------------------------------------------------------------------
    # Sidebar — Configuration & Ingestion
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Configuration")

        search_type = st.selectbox(
            "Search strategy",
            options=["MMR", "Similarity", "Hybrid"],
            index=0,
        )
        k = st.slider("Documents to retrieve (k)", min_value=1, max_value=10, value=10)
        book_filter = st.text_input("Filter by book name (optional)", value="")
        use_reranker = st.toggle("Enable reranker (bge-reranker-large)", value=False)
        show_eval = st.toggle("Show evaluation metrics", value=False)

        st.divider()
        st.header("📥 Ingest PDFs")
        if st.button("Run Ingestion Pipeline", use_container_width=True):
            if not PDF_DIR.exists():
                st.error(f"PDF directory not found: {PDF_DIR}")
            else:
                with st.spinner("Ingesting PDFs …"):
                    n = run_ingestion(PDF_DIR)
                st.success(f"Ingested {n} chunks into ChromaDB.")

    # -----------------------------------------------------------------------
    # Main area — Q&A
    # -----------------------------------------------------------------------
    st.header("💬 Ask a Question")

    question = st.text_area(
        "Your question",
        placeholder="e.g. What is the main theme of Chapter 3?",
        height=100,
    )

    if st.button("Ask", type="primary", use_container_width=True) and question.strip():
        retriever = get_retriever(k=k)
        generator = get_generator()
        evaluator = RAGEvaluator()

        with st.spinner("Retrieving relevant passages …"):
            try:
                if search_type == "hybrid":
                    # Build HybridSearcher on-the-fly (BM25 needs all docs)
                    # For production, cache the corpus separately
                    store = get_embed_store()
                    all_docs_cursor = store.raw_vectorstore.get(
                        include=["documents", "metadatas"]
                    )
                    from langchain_core.documents import Document as LC_Doc

                    corpus = [
                        LC_Doc(page_content=d, metadata=m)
                        for d, m in zip(
                            all_docs_cursor["documents"],
                            all_docs_cursor["metadatas"],
                        )
                    ]
                    hybrid = HybridSearcher(
                        documents=corpus,
                        embed_store=store,
                        k=k,
                        use_reranker=use_reranker,
                    )
                    docs = hybrid.search(
                        query=question,
                        book_filter=book_filter or None,
                    )
                elif search_type == "mmr":
                    docs = retriever.mmr_search(
                        query=question,
                        book_filter=book_filter or None,
                    )
                else:
                    docs = retriever.similarity_search(
                        query=question,
                        book_filter=book_filter or None,
                    )
            except Exception as exc:
                st.error(f"Retrieval failed: {exc}")
                logger.error("Retrieval error: %s", exc)
                return

        with st.spinner("Generating answer …"):
            try:
                result = generator.generate(question=question, documents=docs)
            except Exception as exc:
                st.error(f"Generation failed: {exc}")
                logger.error("Generation error: %s", exc)
                return

        # -----------------------------------------------------------------------
        # Display answer
        # -----------------------------------------------------------------------
        st.subheader("🤖 Answer")
        grounded_badge = "✅ Merlin Beard!" if result["grounded"] else "⚠️ Possibly Hallucinated"
        st.markdown(f"**{grounded_badge}**")
        st.markdown(result["answer"])

        # -----------------------------------------------------------------------
        # Sources
        # -----------------------------------------------------------------------
        if result["sources"]:
            st.subheader("📖 Sources")
            for src in result["sources"]:
                st.markdown(
                    f"- **{src.get('book', '?')}** · "
                    f"Chapter: {src.get('chapter', '?')} · "
                    f"Page: {src.get('page', '?')}"
                )

        # -----------------------------------------------------------------------
        # Retrieved chunks (expander)
        # -----------------------------------------------------------------------
        with st.expander("🔍 Retrieved Chunks", expanded=False):
            for i, doc in enumerate(docs, start=1):
                meta = doc.metadata
                st.markdown(
                    f"**[{i}] {meta.get('book', '?')} — page {meta.get('page', '?')}**"
                )
                st.text(doc.page_content[:600])
                st.divider()

        # -----------------------------------------------------------------------
        # Evaluation metrics (optional)
        # -----------------------------------------------------------------------
        if show_eval:
            eval_result = evaluator.evaluate(
                question=question,
                answer=result["answer"],
                retrieved_docs=docs,
            )
            st.subheader("📊 Evaluation Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Precision@K", f"{eval_result.precision_at_k:.2f}")
            col2.metric("Grounded", "Yes" if eval_result.is_grounded else "No")
            col3.metric("Flagged Sentences", len(eval_result.hallucination_flags))

            if eval_result.hallucination_flags:
                with st.expander("⚠️ Suspicious Sentences"):
                    for sentence in eval_result.hallucination_flags:
                        st.warning(sentence)


if __name__ == "__main__":
    main()