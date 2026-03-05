"""
Prompt templates for the harry_rag LLM layer.
Enforces strict anti-hallucination and context-only answering.
"""

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# ---------------------------------------------------------------------------
# System prompt — strict grounding
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a precise, trustworthy assistant whose ONLY knowledge source is the \
context passages provided below.

STRICT RULES you must follow at all times:
1. Answer ONLY using information explicitly present in the provided context.
2. If the answer is not found in the context, respond EXACTLY with:
   "I cannot answer this question based on the available documents."
3. Never invent facts, names, dates, or quotations.
4. Never use prior knowledge outside the provided context.
5. If the question is off-topic (not related to the document corpus), respond EXACTLY with:
   "This question is outside the scope of the available documents."
6. Always cite the source book and page number (book: ..., page: ...) at the end \
of your answer.
7. Be concise and factual. Do not add disclaimers beyond what these rules require.
"""

# ---------------------------------------------------------------------------
# Human-turn template
# ---------------------------------------------------------------------------

HUMAN_TEMPLATE = """\
CONTEXT:
{context}

QUESTION:
{question}

ANSWER (based strictly on the context above):"""

# ---------------------------------------------------------------------------
# Public prompt objects
# ---------------------------------------------------------------------------

RAG_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_TEMPLATE),
    ]
)

# A plain (non-chat) variant for models that only accept a single string prompt.
RAG_PROMPT_PLAIN: PromptTemplate = PromptTemplate(
    input_variables=["context", "question"],
    template=f"{SYSTEM_PROMPT}\n\n{HUMAN_TEMPLATE}",
)


def format_context(documents: list) -> str:
    """
    Render a list of LangChain Documents into a numbered context block.

    Each entry is formatted as::

        [1] (book: <book>, chapter: <chapter>, page: <page>)
        <text>

    Args:
        documents: List of ``langchain_core.schema.Document`` objects.

    Returns:
        A single string ready for injection into the prompt.
    """
    sections: list[str] = []
    for idx, doc in enumerate(documents, start=1):
        meta = doc.metadata
        header = (
            f"[{idx}] (book: {meta.get('book', 'unknown')}, "
            f"chapter: {meta.get('chapter', 'unknown')}, "
            f"page: {meta.get('page', '?')})"
        )
        sections.append(f"{header}\n{doc.page_content.strip()}")
    return "\n\n".join(sections)