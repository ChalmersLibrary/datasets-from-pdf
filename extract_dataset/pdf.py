from __future__ import annotations

import re
import sys
import fitz  # PyMuPDF
from pathlib import Path


def extract_text(pdf_path: Path, ocr_fallback: bool = True,
                 ocr_threshold: int = 100, ocr_dpi: int = 300) -> tuple[str, list[int]]:
    """Extract text from a PDF, preserving reading order as best we can.

    If fewer than ocr_threshold characters are extracted, the PDF is likely
    image-based; with ocr_fallback=True we re-extract it via Tesseract OCR
    (requires `tesseract` to be installed and PyMuPDF >= 1.19).

    Returns (text, ocr_pages) where ocr_pages is a list of 1-based page
    numbers that were OCR-processed.
    """
    doc = fitz.open(pdf_path)
    parts = []
    ocr_pages: list[int] = []
    for page in doc:
        text = page.get_text("text", sort=True)
        if ocr_fallback and len(text.strip()) < ocr_threshold:
            try:
                tp = page.get_textpage_ocr(language="eng", dpi=ocr_dpi, full=False)
                ocr_text = page.get_text("text", textpage=tp, sort=True)
                if len(ocr_text.strip()) > len(text.strip()):
                    print(f"[info] OCR used on page {page.number + 1} "
                          f"({len(ocr_text.strip())} chars)", file=sys.stderr)
                    ocr_pages.append(page.number + 1)
                    text = ocr_text
            except Exception as e:
                print(f"[warn] OCR failed on page {page.number + 1}: {e}",
                      file=sys.stderr)
        parts.append(text)
    doc.close()
    text = "\n".join(parts)
    # Collapse "D A T A" style single-char artifacts without touching normal word spacings
    text = re.sub(r'(?<!\w)([A-Za-z] )+([A-Za-z])(?!\w)',
                  lambda m: m.group(0).replace(' ', ''), text)
    print(f"[info] Extracted {len(text)} chars from PDF "
          f"(OCR on {len(ocr_pages)} page(s))", file=sys.stderr)
    return text, ocr_pages
