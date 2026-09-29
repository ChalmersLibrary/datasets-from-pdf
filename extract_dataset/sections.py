from __future__ import annotations

import re
from typing import Optional

# Building blocks. "avail-\nability" style hyphenation from PDF line breaks is tolerated.
_AVAIL = r"avail(?:-\s*)?abilit(?:y|ies)"
_THING = r"(?:data(?:\s*sets?)?|datasets?|codes?|software|materials?|scripts?|resources?)"
_THINGS = rf"{_THING}(?:\s*(?:,|&|/|and|,\s*and)\s*{_THING})*"   # "data, code and materials"

# Strong headings: specific enough to be trusted even when found mid-line
# (e.g. two-column layouts or "■ASSOCIATED CONTENT Data Availability Statement").
DAS_STRONG_PATTERNS = [
    rf"{_THINGS}\s+{_AVAIL}(?:\s+statement(?:s(?![a-z]))?)?+",                    # data (and code) availability
    rf"{_AVAIL}\s+of\s+(?:the\s+)?{_THINGS}",                       # availability of data and materials
    rf"{_THINGS}\s+(?:accessibility|sharing|access)\s+statements?",
    rf"{_THINGS}\s+{_AVAIL}\s+and\s+(?:access|sharing)",
    r"data\s+(?:deposition|archiving|sharing\s+and\s+accessibility)\s+(?:statement|policy)",
    r"accession\s+(?:codes?|numbers?)",
    r"open\s+research\s+statement",
    r"data\s+statement",
]

# Weak headings: generic phrases that also occur in body text, so they are only
# accepted when they start a line.
DAS_HEADING_PATTERNS = DAS_STRONG_PATTERNS + [
    rf"{_THINGS}\s+(?:accessibility|sharing|access|deposition|archiving)",
    r"support(?:ing)?\s+data",
    r"shared\s+data",
    r"research\s+data(?:\s+(?:management|policy))?",
    r"(?:code|data|software)\s+and\s+(?:code|data|software)",
    r"open\s+(?:data|research)",
    r"replication\s+(?:package|data|materials?)",
    r"reproducibility(?:\s+statement)?",
]

SECTION_END_PATTERNS = [
    r"acknowledg(?:e)?ments?",
    r"author\s+contributions?",
    r"credit\s+author(?:ship)?\s+contribution(?:s|\s+statement)?",
    r"author\s+information",
    r"corresponding\s+authors?",
    r"(?:declaration\s+of\s+)?competing\s+interests?",
    r"conflicts?\s+of\s+interests?",
    r"declarations?(?:\s+of\s+interests?)?",
    r"declaration\s+of\s+generative\s+ai.*",
    r"disclosure(?:\s+statement)?",
    r"funding(?:\s+(?:information|sources?|statement))?",
    r"financial\s+support",
    r"references?",
    r"bibliography",
    r"(?:literature|works)\s+cited",
    r"supplementary\s+(?:materials?|information|data)",
    r"supporting\s+information",
    r"associated\s+content",
    r"additional\s+information",
    r"appendix(?:\s+[a-z0-9]+)?",
    r"abbreviations",
    r"notes",
    r"orcid(?:\s+ids?)?",
    r"ethics(?:\s+(?:approval|statement|declarations?))?",
    r"consent\s+for\s+publication",
    r"publisher['’]?s\s+note",
    r"open\s+access",
    r"copyright",
    r"keywords",
]

# Optional line prefix before a heading: bullets/markers and numbering like "5.", "5.1", "V.", "A."
_PREFIX = r"(?:[■●•*#]+\s*)?(?:(?:\d+(?:\.\d+)*\.?|[ivxlcdm]+\.|[a-z]\.)\s*)?"
# Not glued to a preceding lowercase letter; uppercase ("ASSOCIATEDData") is allowed.
_LEFT = r"(?-i:(?<![a-z]))"

# A DAS rarely exceeds this; keeps runaway sections (no end heading found) from swamping the prompt.
MAX_SECTION_CHARS = 4000


def _find_section(text: str, start_patterns: list[str], end_patterns: list[str],
                  strong_patterns: list[str] | None = None,
                  max_chars: int = MAX_SECTION_CHARS,
                  min_chars: int = 20) -> Optional[str]:
    """Return the substring from the first start heading to the next end heading."""
    joined_start = "|".join(start_patterns)
    tiers = [
        # 1. Heading alone on its line
        rf"(?im)^\s*{_PREFIX}(?:{joined_start})\b[\s:.\-–—]*$",
        # 2. Heading starting a line, followed by text: "Data availability: The data ..."
        rf"(?im)^\s*{_PREFIX}(?:{joined_start})\b\s*[:\-–—]\s*",
    ]
    if strong_patterns:
        # 3. Capitalised strong heading anywhere in the text, even glued to the next
        #    word by the PDF extraction ("Data availabilityThe data ...")
        tiers.append(rf"(?i){_LEFT}(?-i:(?=[A-Z]))(?:{'|'.join(strong_patterns)})[\s:.\-–—]*")

    m = None
    for pattern in tiers:
        m = re.search(pattern, text)
        if m:
            break
    if not m:
        return None

    start_idx = m.end()
    joined_end = "|".join(end_patterns)
    # End heading alone on its line, or starting a line followed by ":" ("Funding: ...")
    end_re = re.compile(
        rf"(?im)^\s*{_PREFIX}(?:{joined_end})\b(?:[\s:.]*$|\s*:)",
    )
    end_idx = len(text)
    for m_end in end_re.finditer(text, pos=start_idx):
        # Two-column layouts can put another heading right after ours; skip it
        if len(text[start_idx:m_end.start()].strip()) >= min_chars:
            end_idx = m_end.start()
            break
    end_idx = min(end_idx, start_idx + max_chars)

    return text[start_idx:end_idx].strip()


def find_data_availability(text: str) -> Optional[str]:
    return _find_section(text, DAS_HEADING_PATTERNS, SECTION_END_PATTERNS,
                         strong_patterns=DAS_STRONG_PATTERNS)

# Parse and return the DOI of the paper itself, could and should probably be supplied as input instead.
def find_paper_doi(text: str) -> Optional[str]:
    """Return the first DOI that looks like the paper's own DOI (from title area)."""
    # Search the first 3000 chars first (header/title area), then full text
    for chunk in (text[:3000], text):
        m = re.search(r"\b(10\.\d{4,}/[^\s,;)\]\"\'<>]+)", chunk)
        if m:
            return m.group(1).rstrip(".")
    return None


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
