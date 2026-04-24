#!/usr/bin/env python3

"""
extract_datasets.py

Try and extract datasets created/used in a scientific article from a PDF full text version.

Focuses on:
  - the Data Availability Statement section (if present)
  - the References / Bibliography (for cited data repositories)

  1. Extract text from PDF with PyMuPDF, using OCR fallback for image-based pages.
  2. Locate the DAS and References sections with regex.
  3. Ask a local Ollama model (e.g. qwen2.5) to extract structured
     dataset records as JSON.
  4. Print / save the result.

Requirements:
    ollama (local LLM server, installed and running with a suitable model)
    pip install pymupdf requests
    tesseract (for OCR fallback, also requires PyMuPDF >= 1.19)
    ollama pull qwen2.5:14b      # or qwen2.5:7b for lighter hardware
    ollama serve                 
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import fitz  # PyMuPDF
import requests
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------
# 1. PDF to text
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# 2. Locate Data Availability and References sections
# --------------------------------------------------------------------------

# Heading patterns
DAS_HEADING_PATTERNS = [
    r"data\s+availability\s+statement",
    r"data\s+availability",
    r"availability\s+of\s+data(?:\s+and\s+materials?)?",
    r"data\s+and\s+code\s+availability",
    r"data\s+access(?:ibility)?",
    r"research\s+data",
]

# Headings post DAS
SECTION_END_PATTERNS = [
    r"acknowledg(e)?ments?",
    r"author\s+contributions?",
    r"competing\s+interests?",
    r"conflict\s+of\s+interest",
    r"funding",
    r"references?",
    r"bibliography",
    r"supplementary\s+(material|information)",
    r"appendix",
    r"ethics\s+(approval|statement)",
]

def _find_section(text: str, start_patterns: list[str],
                  end_patterns: list[str]) -> Optional[str]:
    """Return the substring from the first start heading to the next end heading."""
    joined_start = "|".join(start_patterns)
    # ^ anchored to line start, allow optional numbering like "5.", "V."
    start_re = re.compile(
        rf"(?im)^\s*(?:\d+\.?\s*|[ivxlcdm]+\.\s*)?({joined_start})\b[:.\s]*$",
        re.MULTILINE,
    )
    m = start_re.search(text)
    if not m:
        # Looser fallback: heading may be inline, not on its own line
        #start_re2 = re.compile(rf"(?i)\b({joined_start})\b[:.]", re.MULTILINE)
        start_re2 = re.compile(rf"(?i)\b({joined_start})\b", re.MULTILINE)
        m = start_re2.search(text)
        if not m:
            return None

    start_idx = m.end()
    joined_end = "|".join(end_patterns)
    end_re = re.compile(
        rf"(?im)^\s*(?:\d+\.?\s*|[ivxlcdm]+\.\s*)?({joined_end})\b[:.\s]*$",
        re.MULTILINE,
    )
    m_end = end_re.search(text, pos=start_idx)
    end_idx = m_end.start() if m_end else min(start_idx + 4000, len(text))

    return text[start_idx:end_idx].strip()


def find_data_availability(text: str) -> Optional[str]:
    return _find_section(text, DAS_HEADING_PATTERNS, SECTION_END_PATTERNS)


def find_references(text: str) -> Optional[str]:
    ref_heading = [r"references?", r"bibliography", r"works\s+cited"]
    # References run to end of document, so just grab from heading onward
    joined = "|".join(ref_heading)
    m = re.search(
        rf"(?im)^\s*(?:\d+\.?\s*)?({joined})\b[:.\s]*$",
        text,
        re.MULTILINE,
    )
    if not m:
        return None
    return text[m.end():].strip()


# --------------------------------------------------------------------------
# 3. Ask Ollama (prompt)
# --------------------------------------------------------------------------

ollama_url = "http://localhost:11434/api/generate"
model = "qwen2.5:14b"

prompt = """You extract information about research datasets from
excerpts of scientific articles. A "dataset" here means data that the
authors created, collected, generated, or deposited — or pre-existing
datasets they explicitly reused.

For each dataset you find, return an object with these fields:
  - name:        short descriptive name of the dataset (string, or null)
  - repository:  where it is hosted, e.g. "Zenodo", "Figshare", "GenBank",
                 "Dryad", "GitHub", "institutional repository" (string, or null)
  - identifier:  DOI, accession number, or similar persistent ID (string, or null)
  - url:         direct URL if given (string, or null)
  - created_by_authors:  true if the authors created/generated this dataset
                         in this study, false if they merely reused an
                         existing dataset, null if unclear
  - source_section:  "data_availability" or "references"
  - evidence:    short quote (max ~200 chars) from the text that supports
                 this entry

Return ONLY a JSON object of the form:
  { "datasets": [ ... ] }

If no datasets are mentioned, return { "datasets": [] }.
Do not include any prose, markdown, or code fences — JSON only.
"""


def query_ollama(model: str, section_name: str, section_text: str,
                 host: str = ollama_url, timeout: int = 300) -> dict:
    """Send a section to Ollama and parse the JSON response."""
    user_prompt = (
        f"Section: {section_name}\n\n"
        f"---BEGIN TEXT---\n{section_text}\n---END TEXT---\n\n"
        "Extract all datasets as specified."
    )

    payload = {
        "model": model,
        "system": prompt,
        "prompt": user_prompt,
        "stream": False,
        "format": "json",          # forces valid JSON output
        "options": {
            "temperature": 0.1,    # low temp for more deterministic output
            "num_ctx": 8192,
        },
    }

    r = requests.post(host, json=payload, timeout=timeout)
    r.raise_for_status()
    raw = r.json().get("response", "").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Salvage attempt: find the first {...} block
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise ValueError(f"Model did not return valid JSON:\n{raw}")

# --------------------------------------------------------------------------
# 4. Process and normalize results
# --------------------------------------------------------------------------

@dataclass
class Dataset:
    name: Optional[str]
    repository: Optional[str]
    identifier: Optional[str]
    url: Optional[str]
    created_by_authors: Optional[bool]
    source_section: str
    evidence: Optional[str]


def extract_datasets_from_pdf(pdf_path: Path, model: str = model,
                              include_references: bool = True,
                              ref_char_limit: int = 20000,
                              ocr_fallback: bool = True) -> tuple[list[Dataset], list[int]]:
    text, ocr_pages = extract_text(pdf_path, ocr_fallback=ocr_fallback)

    all_records: list[dict] = []

    das = find_data_availability(text)
    if das:
        print(f"[info] Found Data Availability section ({len(das)} chars)",
              file=sys.stderr)
        result = query_ollama(model, "data_availability", das)
        for d in result.get("datasets", []):
            d["source_section"] = "data_availability"
            all_records.append(d)
    else:
        print("[info] No Data Availability Statement found.", file=sys.stderr)

    if include_references:
        refs = find_references(text)
        if refs:
            # References can be very long; trim to keep context manageable.
            if len(refs) > ref_char_limit:
                print(f"[info] References truncated "
                      f"({len(refs)} → {ref_char_limit} chars)", file=sys.stderr)
                refs = refs[:ref_char_limit]
            print(f"[info] Scanning References section ({len(refs)} chars)",
                  file=sys.stderr)
            result = query_ollama(model, "references", refs)
            for d in result.get("datasets", []):
                d["source_section"] = "references"
                all_records.append(d)
        else:
            print("[info] No References section found.", file=sys.stderr)

    # Normalize to Dataset dataclass
    datasets = []
    for d in all_records:
        datasets.append(Dataset(
            name=d.get("name"),
            repository=d.get("repository"),
            identifier=d.get("identifier"),
            url=d.get("url"),
            created_by_authors=d.get("created_by_authors"),
            source_section=d.get("source_section", "unknown"),
            evidence=d.get("evidence"),
        ))
    return datasets, ocr_pages


# --------------------------------------------------------------------------
# CLI and output
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdf", type=Path, help="Path to the PDF file")
    ap.add_argument("--model", default=model,
                    help="Ollama model name (default: " + model + ")")
    ap.add_argument("--no-ocr", action="store_true",
                    help="Disable OCR fallback for image-based pages")
    ap.add_argument("--no-references", action="store_true",
                    help="Skip scanning the references section")
    ap.add_argument("--out", type=Path, default=None,
                    help="Write JSON to this file instead of stdout")
    args = ap.parse_args()

    if not args.pdf.exists():
        print(f"error: {args.pdf} not found", file=sys.stderr)
        return 1

    datasets, ocr_pages = extract_datasets_from_pdf(
        args.pdf,
        model=args.model,
        include_references=not args.no_references,
        ocr_fallback=not args.no_ocr,
    )

    output = {
        "pdf": str(args.pdf),
        "ocr_pages": ocr_pages,
        "datasets": [asdict(d) for d in datasets],
    }
    text = json.dumps(output, indent=2, ensure_ascii=False)

    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"[info] wrote {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0

if __name__ == "__main__":
    sys.exit(main())
