from __future__ import annotations

import re
import sys
import pymupdf as fitz  # PyMuPDF
from pathlib import Path


# Typographic ligatures (one glyph for "fi", "fl", ...) break regexes, words like "figshare"
# and URLs/DOIs. Replaced explicitly rather than with unicodedata NFKC, because NFKC also
# turns superscript citation markers into digits ("zenodo.123456¹" -> "zenodo.1234561").
_LIGATURES = str.maketrans({
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi",
    "\ufb04": "ffl", "\ufb05": "st", "\ufb06": "st",
})
# Invisible characters some publishers put inside URLs/DOIs (e.g. a zero-width space between
# every character), plus soft hyphens. They break URLs, DOIs and pattern matching.
_INVISIBLE = re.compile("[\u00ad\u200b\u200c\u200d\u2060\ufeff]")


def _clean(text: str) -> str:
    return _INVISIBLE.sub("", text.translate(_LIGATURES))


def _page_text(page, textpage=None) -> str:
    """Page text in reading order, aware of two-column layouts.

    Text blocks are grouped into horizontal bands separated by full-width blocks (titles,
    wide figures/tables, single-column text). Within a band the left column is read
    top-to-bottom before the right column. Unlike get_text(sort=True), this never merges
    lines from the two columns that happen to sit at the same height (which garbled
    headings like "Data Availability" and "References").
    """
    blocks = [b for b in page.get_text("blocks", textpage=textpage)
              if b[6] == 0 and b[4].strip()]          # b = (x0, y0, x1, y1, text, no, type)
    mid = page.rect.width / 2

    def column(b) -> int:
        if b[2] <= mid + 5:
            return 0          # left column
        if b[0] >= mid - 5:
            return 1          # right column
        return -1             # spans the middle: full width

    out: list[str] = []
    band: list = []

    def flush() -> None:
        for col in (0, 1):
            out.extend(b[4].strip() for b in sorted((b for b in band if column(b) == col),
                                                     key=lambda b: b[1]))
        band.clear()

    for b in sorted(blocks, key=lambda b: (b[1], b[0])):
        if column(b) == -1:
            flush()
            out.append(b[4].strip())
        else:
            band.append(b)
    flush()
    return "\n".join(out)


def extract_text(pdf_path: Path, ocr_fallback: bool = True,
                 ocr_threshold: int = 100, ocr_dpi: int = 300) -> tuple[str, list[int]]:
    """Extract text from a PDF, preserving reading order as best we can.

    If fewer than ocr_threshold characters are extracted, the PDF is likely
    image-based; with ocr_fallback=True we re-extract it via Tesseract OCR
    (requires `tesseract` to be installed and PyMuPDF >= 1.19).

    Returns (text, ocr_pages) where ocr_pages is a list of 1-based page
    numbers that were OCR-processed.
    """
    parts = []
    ocr_pages: list[int] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text = _page_text(page)
            if ocr_fallback and len(text.strip()) < ocr_threshold:
                try:
                    tp = page.get_textpage_ocr(language="eng", dpi=ocr_dpi, full=False)
                    ocr_text = _page_text(page, textpage=tp)
                    if len(ocr_text.strip()) > len(text.strip()):
                        print(f"[info] OCR used on page {page.number + 1} "
                              f"({len(ocr_text.strip())} chars)", file=sys.stderr)
                        ocr_pages.append(page.number + 1)
                        text = ocr_text
                except Exception as e:
                    print(f"[warn] OCR failed on page {page.number + 1}: {e}",
                          file=sys.stderr)
            parts.append(text)
    text = _clean("\n".join(parts))
    # Collapse "D A T A" style single-char artifacts without touching normal word spacings
    text = re.sub(r'(?<!\w)([A-Za-z] )+([A-Za-z])(?!\w)',
                  lambda m: m.group(0).replace(' ', ''), text)
    print(f"[info] Extracted {len(text)} chars from PDF "
          f"(OCR on {len(ocr_pages)} page(s))", file=sys.stderr)
    return text, ocr_pages
