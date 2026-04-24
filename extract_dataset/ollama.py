from __future__ import annotations

import json
import re
import requests


ollama_url = "http://localhost:11434/api/generate"
default_model = "qwen2.5:14b"

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
        "format": "json",
        "options": {
            "temperature": 0.1,
            "num_ctx": 8192,
        },
    }

    r = requests.post(host, json=payload, timeout=timeout)
    r.raise_for_status()
    raw = r.json().get("response", "").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise ValueError(f"Model did not return valid JSON:\n{raw}")
