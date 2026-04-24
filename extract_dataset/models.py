from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .pdf import extract_text
from .sections import find_data_availability, find_references
from .ollama import query_ollama, default_model as DEFAULT_MODEL


@dataclass
class Dataset:
    name: Optional[str]
    repository: Optional[str]
    identifier: Optional[str]
    url: Optional[str]
    created_by_authors: Optional[bool]
    source_section: str
    evidence: Optional[str]


def extract_datasets_from_pdf(pdf_path: Path, model: str = DEFAULT_MODEL,
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

    datasets = [
        Dataset(
            name=d.get("name"),
            repository=d.get("repository"),
            identifier=d.get("identifier"),
            url=d.get("url"),
            created_by_authors=d.get("created_by_authors"),
            source_section=d.get("source_section", "unknown"),
            evidence=d.get("evidence"),
        )
        for d in all_records
    ]
    return datasets, ocr_pages
