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
from .llm_http import FatalLLMError
from .ollama import default_model as DEFAULT_MODEL
from .fetch import resolve_url, fetch_page_text


# ---------------------------------------------------------------------------
# Prompts (shared by all LLM backends)
# ---------------------------------------------------------------------------

EXTRACT_PROMPT = """You extract research datasets from ONE section of a scientific article
(either its Data Availability Statement or its reference list). You are also given the
article header (title, authors, affiliations) as context. Do NOT extract datasets from the
header; use it only to know who the article's authors are. Use only the text you are given.

INCLUDE
- Data the authors created, collected, generated or deposited in this study
  (measurements, survey responses, sequences, images, simulation output, a deposited data package).
- Specific existing datasets the authors explicitly used: a named database or data collection,
  or a record with its own DOI or accession number.
- Code or software the authors themselves wrote and shared for this study (is_code = true).
- Data stated to be available only on request or under restrictions (is_open = false).

DO NOT INCLUDE
- Software, tools, packages or libraries the authors only used (e.g. R, Python, SPSS, ImageJ,
  scikit-learn, QGIS), including links to their websites or GitHub pages.
- Ordinary journal articles, books, reports or method papers in a reference list, even if they
  contain data, unless the citation is clearly a dataset: a data repository DOI (Zenodo, Dryad,
  Figshare, PANGAEA, Mendeley Data, ...), an accession number, or a label such as "[Data set]".
- The article itself or its DOI, journals, publishers, funders, ethics approvals, trial
  registrations, protocols, and links to organisations or general web pages.
- The same dataset twice: merge all mentions of the same data into one entry.

created_by_authors
- true: the text says the data/code were generated, collected, produced or deposited by the
  authors for this study ("data generated in this study", "we deposited", "our code"), or the
  authors offer their own data on request.
- true: a dataset in the reference list whose creators include one or more of the article's
  authors (compare with the names in the article header).
- false: the authors obtained, downloaded, accessed or reused data made by others ("obtained
  from", "retrieved from", "provided by", "publicly available data from", "we used the X database"),
  or a cited dataset whose creators are clearly different people from the article's authors.
- null: when the text does not make it clear. Do not guess.

For each dataset return an object with exactly these fields, in this order:
  - notes: verbatim quote (max ~200 chars) from the text that supports this entry.
           Write this first and base all other fields on it.
  - name: short descriptive name of the dataset (string or null)
  - repository: where it is hosted, e.g. "Zenodo", "Figshare", "Dryad", "GenBank", "ENA",
                "PANGAEA", "institutional repository" (string or null)
  - identifier: DOI, accession number or similar persistent ID without resolver prefix,
                e.g. "10.5281/zenodo.123456", not "https://doi.org/10.5281/zenodo.123456" (string or null)
  - url: URL exactly as given in the text (string or null)
  - license: e.g. "CC-BY-4.0" if stated (string or null)
  - created_by_authors: true, false or null (see rules above)
  - is_open: true if openly accessible, false if on request/restricted/embargoed, null if unclear
  - is_code: true if this is software/code, otherwise false
  - is_supplementary: true if it is supplementary material published with the article, otherwise false

Never invent names, DOIs, accession numbers or URLs; use null for anything not stated.

EXAMPLE
Text: "The sequencing data generated in this study have been deposited in ENA under accession
PRJEB12345. Climate data were obtained from ERA5 (Hersbach et al., 2020). All analyses were run
in R 4.2 using the vegan package."
Answer:
{"datasets": [
 {"notes": "The sequencing data generated in this study have been deposited in ENA under accession PRJEB12345",
  "name": "Sequencing data", "repository": "ENA", "identifier": "PRJEB12345", "url": null,
  "license": null, "created_by_authors": true, "is_open": true, "is_code": false, "is_supplementary": false},
 {"notes": "Climate data were obtained from ERA5 (Hersbach et al., 2020)",
  "name": "ERA5 reanalysis", "repository": null, "identifier": null, "url": null,
  "license": null, "created_by_authors": false, "is_open": true, "is_code": false, "is_supplementary": false}
]}
(R and vegan are software the authors only used, so they are not included.)

Return ONLY a JSON object of the form { "datasets": [ ... ] }.
If there are no datasets, return { "datasets": [] }. No prose, markdown or code fences.
"""


ENRICH_PROMPT = """You are updating a dataset record that was extracted from a scientific article.

You have:
1. The article header (title, authors, affiliations) of the scientific article
2. The current dataset record (JSON)
3. Text fetched from the dataset's external repository or landing page

Update the record using the fetched page as evidence. Focus on:
- created_by_authors: compare the creators/authors listed on the page with the article's
  authors in the article header. Set true if one or more of them clearly match, false if
  they are clearly different people, and keep the current value if the page does not list
  creators or it is unclear.
- license: update if the page states a specific license (e.g. "CC-BY-4.0")
- is_open: update based on access or availability information on the page
- name: refine if the page gives a more precise dataset title
- repository: confirm or correct the repository name
- is_code: set true if the page shows this is software/code (e.g. a code repository)

Never invent values; keep the current value when the page gives no evidence.
Return ONLY the updated JSON object for this single dataset, using exactly the same
field names as the input. Do not add new fields. JSON only — no prose, no markdown.
"""

# How much of the start of the article is sent as context (title, authors, affiliations).
ARTICLE_HEADER_CHARS = 2000


def article_header(text: str, limit: int = ARTICLE_HEADER_CHARS) -> str:
    """The first part of the article text, which normally holds the title and author list."""
    return re.sub(r"\s+", " ", text[: limit * 2]).strip()[:limit]


def _header_block(header: str) -> str:
    if not header:
        return ""
    return (
        "Article header (context only, do not extract datasets from it):\n"
        f"---BEGIN HEADER---\n{header}\n---END HEADER---\n\n"
    )


def _section_user_prompt(section_name: str, section_text: str, header: str = "") -> str:
    return (
        _header_block(header)
        + f"Section: {section_name}\n\n"
        f"---BEGIN TEXT---\n{section_text}\n---END TEXT---\n\n"
        "Extract all datasets as specified."
    )


def _enrich_user_prompt(record: dict, fetched_text: str, header: str = "") -> str:
    return (
        _header_block(header)
        + f"Current dataset record:\n{json.dumps(record, ensure_ascii=False)}\n\n"
        f"---BEGIN FETCHED PAGE---\n{fetched_text}\n---END FETCHED PAGE---\n\n"
        "Return the updated dataset record as JSON."
    )


def _load_json(raw: str):
    """Parse the model's answer as JSON, tolerating <think> blocks and prose around the JSON.

    Returns the parsed value, or None if no valid JSON could be found.
    """
    stripped = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if not stripped:
        m = re.search(r"<think>(.*)</think>", raw, re.DOTALL)
        stripped = m.group(1).strip() if m else raw
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", stripped, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _short(raw: str, limit: int = 300) -> str:
    return raw[:limit] + ("…" if len(raw) > limit else "")


def _parse_json_response(raw: str, fallback: dict | None = None) -> dict:
    """Parse the model's answer as a JSON object; return fallback (if given) when that fails."""
    data = _load_json(raw)
    if isinstance(data, dict):
        return data
    if fallback is not None:
        print(f"[warn] model did not return a JSON object; keeping previous values. "
              f"Answer was: {_short(raw)}", file=sys.stderr)
        return fallback
    raise ValueError(f"Model did not return a valid JSON object:\n{_short(raw)}")


# ---------------------------------------------------------------------------
# Backend-independent LLM calls. `complete(model, system, user) -> str` is the
# only thing a backend has to provide.
# ---------------------------------------------------------------------------

_BACKENDS = {
    "ollama": ollama.complete,
    "vllm": vllm_api.complete,
}


def query_section(complete, model: str, section_name: str, section_text: str,
                  header: str = "") -> list[dict]:
    """Ask the LLM to extract all datasets from one article section; returns the dataset records."""
    raw = complete(model, EXTRACT_PROMPT, _section_user_prompt(section_name, section_text, header))
    data = _load_json(raw)
    if isinstance(data, list):              # model returned the list without the wrapper object
        items = data
    elif isinstance(data, dict):
        items = data.get("datasets") or []
    else:
        raise ValueError(f"Model did not return valid JSON for {section_name}:\n{_short(raw)}")
    if not isinstance(items, list):
        print(f"[warn] 'datasets' in model answer is not a list; ignoring it: {_short(raw)}",
              file=sys.stderr)
        return []
    records = [d for d in items if isinstance(d, dict)]
    if len(records) != len(items):
        print(f"[warn] skipped {len(items) - len(records)} malformed dataset entr(y/ies) "
              f"in {section_name}", file=sys.stderr)
    return records


def enrich_record(complete, model: str, record: dict, fetched_text: str,
                  header: str = "") -> dict:
    """Ask the LLM to update a dataset record using text fetched from its landing page.

    On a non-fatal failure the original record is kept, so one bad enrichment never loses the paper.
    """
    try:
        raw = complete(model, ENRICH_PROMPT, _enrich_user_prompt(record, fetched_text, header))
    except FatalLLMError:
        raise
    except Exception as exc:
        print(f"[warn] enrichment failed, keeping original record: {exc}", file=sys.stderr)
        return record
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
    header = article_header(text)

    all_records: list[dict] = []

    das = find_data_availability(text)
    das_found = bool(das)
    if das:
        print(f"[info] Found Data Availability section ({len(das)} chars)",
              file=sys.stderr)
        for d in query_section(complete, model, "data_availability_statement", das, header):
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
            for d in query_section(complete, model, "references", refs, header):
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
                all_records[i] = enrich_record(complete, model, record, page_text, header)

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
