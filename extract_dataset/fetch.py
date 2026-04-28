from __future__ import annotations

import html
import re
import sys

import requests


def resolve_url(dataset: dict) -> str | None:
    """Return the best URL to fetch for a dataset record, or None."""
    if dataset.get("url"):
        return dataset["url"]
    identifier = dataset.get("identifier") or ""
    if re.match(r"10\.\d{4,}", identifier):
        return f"https://doi.org/{identifier}"
    return None


def fetch_page_text(url: str, char_limit: int = 4000, timeout: int = 15) -> str | None:
    """Fetch a URL and return stripped plain text, or None on failure."""
    try:
        r = requests.get(
            url, timeout=timeout,
            headers={"User-Agent": "datasets-from-pdf/1.0"},
            allow_redirects=True,
        )
        r.raise_for_status()
        raw = r.text
    except Exception as exc:
        print(f"[warn] fetch failed for {url}: {exc}", file=sys.stderr)
        return None

    text = re.sub(r"<[^>]+>", " ", raw)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:char_limit]
