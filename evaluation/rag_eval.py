"""
Evaluation utilities for harry_rag.
Provides retrieval precision calculation and basic hallucination detection.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Container for a single evaluation result."""

    question: str
    answer: str
    retrieved_docs: list[Document]
    precision_at_k: float = 0.0
    hallucination_flags: list[str] = field(default_factory=list)
    is_grounded: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "precision_at_k": self.precision_at_k,
            "hallucination_flags": self.hallucination_flags,
            "is_grounded": self.is_grounded,
            "num_docs_retrieved": len(self.retrieved_docs),
        }


class RAGEvaluator:
    """
    Evaluates retrieval quality and detects potential hallucinations.

    Args:
        relevant_keywords_threshold: Minimum fraction of answer sentences that
            must contain at least one token from the retrieved context for the
            answer to be considered grounded.
    """

    def __init__(self, relevant_keywords_threshold: float = 0.5) -> None:
        self.threshold = relevant_keywords_threshold

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate(
        self,
        question: str,
        answer: str,
        retrieved_docs: list[Document],
        relevant_doc_ids: list[str] | None = None,
    ) -> EvalResult:
        """
        Run all evaluation checks and return an EvalResult.

        Args:
            question:         The user's original question.
            answer:           The LLM-generated answer.
            retrieved_docs:   Documents returned by the retriever.
            relevant_doc_ids: Ground-truth relevant doc IDs (for precision calc).
                              If None, precision is estimated heuristically.

        Returns:
            EvalResult with precision and hallucination flags populated.
        """
        precision = self._compute_precision(retrieved_docs, relevant_doc_ids)
        flags, is_grounded = self._detect_hallucination(answer, retrieved_docs)

        result = EvalResult(
            question=question,
            answer=answer,
            retrieved_docs=retrieved_docs,
            precision_at_k=precision,
            hallucination_flags=flags,
            is_grounded=is_grounded,
        )

        logger.info(
            "Eval | precision=%.3f grounded=%s flags=%s",
            precision,
            is_grounded,
            flags,
        )
        return result

    def batch_evaluate(
        self,
        samples: list[dict[str, Any]],
    ) -> list[EvalResult]:
        """
        Evaluate a batch of QA samples.

        Each sample dict must have:
        - ``question`` (str)
        - ``answer`` (str)
        - ``retrieved_docs`` (list[Document])
        - ``relevant_doc_ids`` (list[str], optional)
        """
        results: list[EvalResult] = []
        for sample in samples:
            result = self.evaluate(
                question=sample["question"],
                answer=sample["answer"],
                retrieved_docs=sample["retrieved_docs"],
                relevant_doc_ids=sample.get("relevant_doc_ids"),
            )
            results.append(result)

        avg_precision = sum(r.precision_at_k for r in results) / max(len(results), 1)
        grounded_rate = sum(r.is_grounded for r in results) / max(len(results), 1)
        logger.info(
            "Batch eval complete | n=%d avg_precision=%.3f grounded_rate=%.3f",
            len(results),
            avg_precision,
            grounded_rate,
        )
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_precision(
        self,
        retrieved_docs: list[Document],
        relevant_doc_ids: list[str] | None,
    ) -> float:
        """
        Compute Precision@K.

        If ground-truth IDs are supplied, counts how many retrieved docs are
        in the relevant set.  Otherwise falls back to a heuristic that checks
        whether each retrieved doc has non-empty page_content.
        """
        if not retrieved_docs:
            return 0.0

        k = len(retrieved_docs)

        if relevant_doc_ids is not None:
            relevant_set = set(relevant_doc_ids)
            hits = sum(
                1
                for doc in retrieved_docs
                if doc.metadata.get("chunk_index") is not None
                and str(hash(doc.page_content)) in relevant_set
            )
            return hits / k

        # Heuristic: a doc is "relevant" if it has substantial content
        hits = sum(1 for doc in retrieved_docs if len(doc.page_content.strip()) > 50)
        return hits / k

    def _detect_hallucination(
        self,
        answer: str,
        retrieved_docs: list[Document],
    ) -> tuple[list[str], bool]:
        """
        Basic hallucination detection.

        Strategy:
        1. Split answer into sentences.
        2. For each sentence, check that at least one meaningful token
           (length ≥ 4) appears in the combined context.
        3. Sentences with no overlap with context are flagged.

        Returns:
            (flags, is_grounded) where flags is a list of suspicious sentences
            and is_grounded is True when the hallucination rate is below
            self.threshold.
        """
        if not answer.strip() or not retrieved_docs:
            return [], not retrieved_docs

        combined_context = " ".join(doc.page_content for doc in retrieved_docs).lower()
        context_tokens: set[str] = set(re.findall(r"\b\w{4,}\b", combined_context))

        sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
        flagged: list[str] = []

        for sentence in sentences:
            if not sentence.strip():
                continue
            # Skip boilerplate refusal sentences — they are expected
            if any(
                marker in sentence.lower()
                for marker in [
                    "cannot answer",
                    "outside the scope",
                    "based on the available",
                ]
            ):
                continue
            sentence_tokens = set(re.findall(r"\b\w{4,}\b", sentence.lower()))
            overlap = sentence_tokens & context_tokens
            if not overlap:
                flagged.append(sentence)

        hallucination_rate = len(flagged) / max(len(sentences), 1)
        is_grounded = hallucination_rate <= (1.0 - self.threshold)

        return flagged, is_grounded