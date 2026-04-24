from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

from .models import extract_datasets_from_pdf
from .ollama import default_model


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description="Extract dataset references from a scientific article PDF."
    )
    ap.add_argument("pdf", type=Path, help="Path to the PDF file")
    ap.add_argument("--model", default=default_model,
                    help=f"Ollama model name (default: {default_model})")
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
