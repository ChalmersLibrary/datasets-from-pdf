from __future__ import annotations

import re
from typing import Optional


DAS_HEADING_PATTERNS = [
    r"data\s+availability\s+statement",
    r"data\s+availability",
    r"availability\s+of\s+data(?:\s+and\s+materials?)?",
    r"data\s+and\s+code\s+availability",
    r"data\s+access(?:ibility)?",
    r"research\s+data",
]

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
    # Anchored to line start, allow optional numbering like "5.", "V."
    start_re = re.compile(
        rf"(?im)^\s*(?:\d+\.?\s*|[ivxlcdm]+\.\s*)?({joined_start})\b[:.\s]*$",
        re.MULTILINE,
    )
    m = start_re.search(text)
    if not m:
        # Looser fallback: heading may be inline, not on its own line
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
    joined = "|".join(ref_heading)
    m = re.search(
        rf"(?im)^\s*(?:\d+\.?\s*)?({joined})\b[:.\s]*$",
        text,
        re.MULTILINE,
    )
    if not m:
        return None
    return text[m.end():].strip()
