from __future__ import annotations

import json
import re
import requests


ollama_url = "http://localhost:11434/api/generate"
#default_model = "qwen3.5" # or "qwen3.5-mini" for faster but less accurate results
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
  - source_section:  "data_availability_statement" or "references"
  - is_open: true if the dataset seems to be openly accessible, false otherwise
  - is_code: true if the dataset seems to be software code, false otherwise
  - is_supplementary: true if the dataset seems to be supplementary material, false otherwise
  - evidence: short quote (max ~200 chars) from the text that support this entry

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

    # Strip <think>…</think> blocks emitted by reasoning models (e.g. qwen3.x).
    # If nothing remains, the JSON was inside the think block — search there instead.
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
        raise ValueError(f"Model did not return valid JSON:\n{raw}")
