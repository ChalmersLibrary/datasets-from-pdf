from __future__ import annotations

import json
import re
import requests


ollama_url = "http://localhost:11434/api/generate"
default_model = "qwen3.5" # or "qwen3.5-mini" for faster but less accurate results
#default_model = "qwen2.5:14b"

prompt = """You extract information about research datasets from
excerpts of scientific articles. A "dataset" here means data that the
authors created, collected, generated, or deposited — or pre-existing
datasets they explicitly reused.

For each dataset you find, return an object with these fields:
  - das_found: true if a Data Availability Statement was found in the article, false otherwise
  - name:        short descriptive name of the dataset (string, or null)
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
  - evidence: short quote (max ~200 chars) from the text that support this entry

Return ONLY a JSON object of the form:
  { "datasets": [ ... ] }

If no datasets are mentioned, return { "datasets": [] }.
Do not include any prose, markdown, or code fences — JSON only.
"""


enrich_prompt = """You are updating a dataset record that was extracted from a scientific article.

You have:
1. The current dataset record (JSON)
2. Text fetched from the dataset's external repository or landing page

Update the record using the fetched page as evidence. Focus on:
- created_by_authors: check whether the creators listed on the page appear to be the
  same people as the article authors mentioned in the evidence field. Set true if they
  seem to match, false if clearly different, null if unclear.
- license: update if the page states a specific license (e.g. "CC-BY-4.0")
- is_open: update based on access or availability information on the page
- name: refine if the page gives a more precise dataset title
- repository: confirm or correct the repository name

Return ONLY the updated JSON object for this single dataset, using exactly the same
field names as the input. Do not add new fields. JSON only — no prose, no markdown.
"""


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


def enrich_dataset_record(model: str, record: dict, fetched_text: str,
                           host: str = ollama_url, timeout: int = 300) -> dict:
    """Re-query Ollama with fetched page content to enrich a single dataset record."""
    user_prompt = (
        f"Current dataset record:\n{json.dumps(record, ensure_ascii=False)}\n\n"
        f"---BEGIN FETCHED PAGE---\n{fetched_text}\n---END FETCHED PAGE---\n\n"
        "Return the updated dataset record as JSON."
    )
    payload = {
        "model": model,
        "system": enrich_prompt,
        "prompt": user_prompt,
        "stream": False,
        "format": "json",
        # Comment the next line if using Qwen2.5x or other models that don't support the "think" tag.
        "think": False,
        "options": {"temperature": 0.1, "num_ctx": 8192},
    }
    r = requests.post(host, json=payload, timeout=timeout)
    r.raise_for_status()
    raw = r.json().get("response", "").strip()
    enriched = _parse_json_response(raw, fallback=record)
    return {**record, **enriched}


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
        "format": "json",
        # Comment the next line if using Qwen2.5x or other models that don't support the "think" tag.
        "think": False,
        "options": {
            "temperature": 0.1,
            "num_ctx": 8192,
        },
    }

    r = requests.post(host, json=payload, timeout=timeout)
    r.raise_for_status()
    raw = r.json().get("response", "").strip()
    return _parse_json_response(raw)
