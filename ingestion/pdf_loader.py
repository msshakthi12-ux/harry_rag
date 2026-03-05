"""
PDF loader module for the harry_rag ingestion pipeline.
Extracts text and metadata from PDF files using PyMuPDF (fitz).
"""

import logging
import re
from pathlib import Path
from typing import Generator

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


class PDFLoadError(Exception):
    """Raised when a PDF cannot be loaded or parsed."""


class PDFPage:
    """Represents a single extracted page with its metadata."""

    def __init__(self, text: str, book: str, chapter: str, page: int) -> None:
        self.text = text
        self.metadata = {"book": book, "chapter": chapter, "page": page}

    def __repr__(self) -> str:
        return (
            f"PDFPage(book={self.metadata['book']!r}, "
            f"chapter={self.metadata['chapter']!r}, "
            f"page={self.metadata['page']})"
        )


class PDFLoader:
    """
    Loads PDF files from a directory or a single path and yields PDFPage objects.

    Args:
        pdf_dir: Path to directory containing PDF files, or a single PDF file.
        header_margin: Fraction of page height to treat as header zone (default 0.08).
        footer_margin: Fraction of page height to treat as footer zone (default 0.08).
    """

    CHAPTER_PATTERN = re.compile(
        r"^(chapter\s+\d+|part\s+\d+|section\s+\d+[\.\d]*)",
        re.IGNORECASE | re.MULTILINE,
    )

    def __init__(
        self,
        pdf_dir: str | Path,
        header_margin: float = 0.08,
        footer_margin: float = 0.08,
    ) -> None:
        self.pdf_dir = Path(pdf_dir)
        self.header_margin = header_margin
        self.footer_margin = footer_margin

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load_all(self) -> Generator[PDFPage, None, None]:
        """
        Iterate over every PDF in the configured directory (or single file)
        and yield PDFPage objects.
        """
        paths = (
            [self.pdf_dir]
            if self.pdf_dir.is_file()
            else sorted(self.pdf_dir.glob("*.pdf"))
        )

        if not paths:
            logger.warning("No PDF files found at %s", self.pdf_dir)
            return

        for pdf_path in paths:
            logger.info("Loading PDF: %s", pdf_path.name)
            yield from self._load_single(pdf_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_single(self, pdf_path: Path) -> Generator[PDFPage, None, None]:
        """Open one PDF and yield a PDFPage per page."""
        try:
            doc: fitz.Document = fitz.open(str(pdf_path))
        except Exception as exc:
            raise PDFLoadError(f"Cannot open {pdf_path}: {exc}") from exc

        book_name = pdf_path.stem
        current_chapter = "unknown"

        try:
            for page_index in range(len(doc)):
                page: fitz.Page = doc[page_index]
                page_height = page.rect.height

                text = self._extract_body_text(page, page_height)
                if not text.strip():
                    logger.debug("Skipping empty page %d in %s", page_index + 1, book_name)
                    continue

                detected = self._detect_chapter(text)
                if detected:
                    current_chapter = detected

                yield PDFPage(
                    text=text,
                    book=book_name,
                    chapter=current_chapter,
                    page=page_index + 1,
                )
        finally:
            doc.close()

    def _extract_body_text(self, page: fitz.Page, page_height: float) -> str:
        """
        Extract text blocks that fall outside the header/footer margin zones.
        Returns concatenated body text.
        """
        header_cutoff = page_height * self.header_margin
        footer_cutoff = page_height * (1.0 - self.footer_margin)

        blocks = page.get_text("blocks")  # list of (x0, y0, x1, y1, text, ...)
        body_lines: list[str] = []

        for block in blocks:
            x0, y0, x1, y1, text, *_ = block
            if y0 < header_cutoff or y1 > footer_cutoff:
                continue
            stripped = text.strip()
            if stripped:
                body_lines.append(stripped)

        return "\n".join(body_lines)

    def _detect_chapter(self, text: str) -> str | None:
        """
        Attempt to detect a chapter heading in the first 300 characters of text.
        Returns the heading string or None.
        """
        sample = text[:300]
        match = self.CHAPTER_PATTERN.search(sample)
        if match:
            return match.group(0).strip()
        return None