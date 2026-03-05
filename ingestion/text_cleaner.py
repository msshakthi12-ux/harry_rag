"""
Text cleaning utilities for the harry_rag ingestion pipeline.
Normalises whitespace, removes artefacts, and prepares text for chunking.
"""

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)


class TextCleaner:
    """
    Cleans raw extracted text before chunking.

    Cleaning steps (applied in order):
    1. Unicode normalisation (NFKC).
    2. Strip non-printable / control characters.
    3. Collapse excessive whitespace / blank lines.
    4. Remove common PDF artefacts (page numbers, running headers).
    5. Fix hyphenated line breaks.
    """

    # Matches standalone page-number lines such as "- 42 -", "42", "Page 42"
    _PAGE_NUM_RE = re.compile(r"^\s*[-–]?\s*[Pp]age\s+\d+\s*[-–]?\s*$|^\s*\d{1,4}\s*$", re.MULTILINE)

    # Collapse 3+ consecutive blank lines into two
    _MULTI_BLANK_RE = re.compile(r"(\n\s*){3,}")

    # Hyphenated line-break: word-\nword → wordword
    _HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")

    # Replace non-breaking spaces and other odd space chars
    _WHITESPACE_RE = re.compile(r"[ \t\u00a0\u200b]+")

    def clean(self, text: str) -> str:
        """
        Apply all cleaning steps to *text* and return the cleaned string.

        Args:
            text: Raw text extracted from a PDF page.

        Returns:
            Cleaned text suitable for chunking.
        """
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}")

        text = self._normalise_unicode(text)
        text = self._strip_control_chars(text)
        text = self._fix_hyphen_breaks(text)
        text = self._remove_page_numbers(text)
        text = self._collapse_whitespace(text)
        text = text.strip()

        logger.debug("Cleaned text length: %d chars", len(text))
        return text

    # ------------------------------------------------------------------
    # Private steps
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_unicode(text: str) -> str:
        return unicodedata.normalize("NFKC", text)

    @staticmethod
    def _strip_control_chars(text: str) -> str:
        """Remove non-printable control characters except newlines and tabs."""
        return "".join(
            ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t"
        )

    def _fix_hyphen_breaks(self, text: str) -> str:
        return self._HYPHEN_BREAK_RE.sub(r"\1\2", text)

    def _remove_page_numbers(self, text: str) -> str:
        return self._PAGE_NUM_RE.sub("", text)

    def _collapse_whitespace(self, text: str) -> str:
        # Normalise horizontal whitespace first
        text = self._WHITESPACE_RE.sub(" ", text)
        # Then collapse excessive blank lines
        text = self._MULTI_BLANK_RE.sub("\n\n", text)
        return text