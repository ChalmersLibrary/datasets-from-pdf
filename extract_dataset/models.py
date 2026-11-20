from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .pdf import extract_text
from .sections import find_data_availability, find_references, find_paper_doi
from . import ollama, vllm_api, usage
from .usage import LLMUsage
from .ollama import default_model as DEFAULT_MODEL
from .fetch import resolve_url, fetch_page_text


# ---------------------------------------------------------------------------
# Prompts (shared by all LLM backends)
# ---------------------------------------------------------------------------

EXTRACT_PROMPT = """You extract information about research datasets from
excerpts of scientific articles. A "dataset" here means data that the
authors created, collected, generated, or deposited — or pre-existing
datasets they explicitly reused. 

For each dataset you find, return an object with these fields:
  - das_found: true if a Data Availability Statement was found in the article, false otherwise
  - name:        short descriptive name of the dataset (string, or null). If there is a reference - eg. "the ABC dataset [49]" - you can try and find more information in the reference list or text.
  - repository:  where it is hosted, e.g. "Zenodo", "Figshare", "GenBank",
                 "Dryad", "GitHub", "institutional repository" (string, or null)
  - identifier:  DOI, accession number, or similar persistent ID, please omit the resolver URL e.g. "https://doi.org/" and only return actual ID, e.g. "10.5281/zenodo.123456" (string, or null)
  - url:         direct URL if given (string, or null)
  - license:     if a license is mentioned, e.g. "CC-BY-4.0" (string, or null)
  - created_by_authors:  true if the authors created/generated this dataset
                         in this study, false if they merely reused an
                         existing dataset, null if unclear
  - source_section:  "data_availability_statement" or "references"
  - is_open: true if the dataset seems to be openly accessible, false otherwise
  - is_code: true if the dataset seems to be software code, false otherwise
  - is_supplementary: true if the dataset seems to be supplementary material to the article, false otherwise
  - notes: short quote (max ~200 chars) from the text that support this entry. If there is a reference - eg. "the ABC dataset [49]" - you can try and find more information in the reference list or text.

Return ONLY a JSON object of the form:
  { "datasets": [ ... ] }

If no datasets are mentioned, return { "datasets": [] }.
Do not include any prose, markdown, or code fences — JSON only.
"""


ENRICH_PROMPT = """You are updating a dataset record that was extracted from a scientific article.

You have:
1. The current dataset record (JSON)
2. Text fetched from the dataset's external repository or landing page

Update the record using the fetched page as evidence (notes). Focus on:
- created_by_authors: check whether the creators listed on the page appear to be the
  same people as the article authors mentioned in the notes (evidence) field. Set true if they
  seem to match, false if clearly different, null if unclear.
- license: update if the page states a specific license (e.g. "CC-BY-4.0")
- is_open: update based on access or availability information on the page
- name: refine if the page gives a more precise dataset title
- repository: confirm or correct the repository name

Return ONLY the updated JSON object for this single dataset, using exactly the same
field names as the input. Do not add new fields. JSON only — no prose, no markdown.
"""


def _section_user_prompt(section_name: str, section_text: str) -> str:
    return (
        f"Section: {section_name}\n\n"
        f"---BEGIN TEXT---\n{section_text}\n---END TEXT---\n\n"
        "Extract all datasets as specified."
    )


def _enrich_user_prompt(record: dict, fetched_text: str) -> str:
    return (
        f"Current dataset record:\n{json.dumps(record, ensure_ascii=False)}\n\n"
        f"---BEGIN FETCHED PAGE---\n{fetched_text}\n---END FETCHED PAGE---\n\n"
        "Return the updated dataset record as JSON."
    )


def _parse_json_response(raw: str, fallback: dict | None = None) -> dict:
    stripped = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if not stripped:
        m = re.search(r"<think>(.*)</think>", raw, re.DOTALL)
        stripped = m.group(1).strip() if m else raw
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", stripped, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        if fallback is not None:
            return fallback
        raise ValueError(f"Model did not return valid JSON:\n{raw}")


# ---------------------------------------------------------------------------
# Backend-independent LLM calls. `complete(model, system, user) -> str` is the
# only thing a backend has to provide.
# ---------------------------------------------------------------------------

_BACKENDS = {
    "ollama": ollama.complete,
    "vllm": vllm_api.complete,
}


def query_section(complete, model: str, section_name: str, section_text: str) -> dict:
    """Ask the LLM to extract all datasets from one article section."""
    raw = complete(model, EXTRACT_PROMPT, _section_user_prompt(section_name, section_text))
    return _parse_json_response(raw)


def enrich_record(complete, model: str, record: dict, fetched_text: str) -> dict:
    """Ask the LLM to update a dataset record using text fetched from its landing page."""
    raw = complete(model, ENRICH_PROMPT, _enrich_user_prompt(record, fetched_text))
    enriched = _parse_json_response(raw, fallback=record)
    return {**record, **enriched}


@dataclass
class Dataset:
    name: Optional[str]
    repository: Optional[str]
    identifier: Optional[str]
    url: Optional[str]
    license: Optional[str]
    created_by_authors: Optional[bool]
    source_section: str
    notes: Optional[str]
    is_open: Optional[bool]
    is_code: Optional[bool]
    is_supplementary: Optional[bool]


def extract_datasets_from_pdf(pdf_path: Path, model: str = DEFAULT_MODEL,
                               include_references: bool = True,
                               ref_char_limit: int = 30000,
                               ocr_fallback: bool = True,
                               enrich_urls: bool = False,
                               backend: str = "ollama") -> tuple[list[Dataset], list[int], bool, str | None, LLMUsage]:
    if backend not in _BACKENDS:
        raise ValueError(f"unknown backend {backend!r} (choose from {', '.join(_BACKENDS)})")
    complete = _BACKENDS[backend]

    usage.reset()
    text, ocr_pages = extract_text(pdf_path, ocr_fallback=ocr_fallback)
    paper_doi = find_paper_doi(text)

    all_records: list[dict] = []

    das = find_data_availability(text)
    das_found = bool(das)
    if das:
        print(f"[info] Found Data Availability section ({len(das)} chars)",
              file=sys.stderr)
        result = query_section(complete, model, "data_availability_statement", das)
        for d in result.get("datasets", []):
            d["source_section"] = "data_availability_statement"
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
            result = query_section(complete, model, "references", refs)
            for d in result.get("datasets", []):
                d["source_section"] = "references"
                all_records.append(d)
        else:
            print("[info] No References section found.", file=sys.stderr)

    if enrich_urls:
        for i, record in enumerate(all_records):
            url = resolve_url(record)
            if not url:
                continue
            print(f"[info] Fetching {url}", file=sys.stderr)
            page_text = fetch_page_text(url)
            if page_text:
                all_records[i] = enrich_record(complete, model, record, page_text)

    datasets = [
        Dataset(
            name=d.get("name"),
            repository=d.get("repository"),
            identifier=d.get("identifier"),
            url=d.get("url"),
            license=d.get("license"),
            created_by_authors=d.get("created_by_authors"),
            source_section=d.get("source_section", "unknown"),
            notes=d.get("notes"),
            is_open=d.get("is_open"),
            is_code=d.get("is_code"),
            is_supplementary=d.get("is_supplementary"),
        )
        for d in all_records
    ]
    return datasets, ocr_pages, das_found, paper_doi, usage.current()
