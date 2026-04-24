from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

from .models import extract_datasets_from_pdf
from .ollama import default_model


def _process_one(pdf: Path, args) -> int:
    if not pdf.exists():
        print(f"error: {pdf} not found", file=sys.stderr)
        return 1

    datasets, ocr_pages = extract_datasets_from_pdf(
        pdf,
        model=args.model,
        include_references=not args.no_references,
        ocr_fallback=not args.no_ocr,
    )

    output = {
        "pdf": str(pdf),
        "ocr_pages": ocr_pages,
        "datasets": [asdict(d) for d in datasets],
    }
    text = json.dumps(output, indent=2, ensure_ascii=False)

    if args.out:
        out_path = Path(args.out)
        if out_path.is_dir():
            out_path = out_path / (pdf.stem + ".json")
        out_path.write_text(text, encoding="utf-8")
        print(f"[info] wrote {out_path}", file=sys.stderr)
    else:
        print(text)
    return 0


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description="Extract dataset references from a scientific article PDF."
    )
    ap.add_argument("pdf", type=Path, nargs="?",
                    help="Path to the PDF file")
    ap.add_argument("--batch-dir", type=Path, default=None,
                    help="Process all PDFs in this directory")
    ap.add_argument("--model", default=default_model,
                    help=f"Ollama model name (default: {default_model})")
    ap.add_argument("--no-ocr", action="store_true",
                    help="Disable OCR fallback for image-based pages")
    ap.add_argument("--no-references", action="store_true",
                    help="Skip scanning the references section")
    ap.add_argument("--out", type=Path, default=None,
                    help="Write JSON to this file (single) or directory (batch) instead of stdout")
    args = ap.parse_args()

    if args.batch_dir and args.pdf:
        print("error: specify either a pdf or --batch-dir, not both", file=sys.stderr)
        return 1
    if not args.batch_dir and not args.pdf:
        ap.print_help()
        return 1

    if args.batch_dir:
        if not args.batch_dir.is_dir():
            print(f"error: {args.batch_dir} is not a directory", file=sys.stderr)
            return 1
        pdfs = sorted(args.batch_dir.glob("*.pdf"))
        if not pdfs:
            print(f"[info] no PDF files found in {args.batch_dir}", file=sys.stderr)
            return 0
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
        errors = 0
        for i, pdf in enumerate(pdfs, 1):
            print(f"[info] processing {i}/{len(pdfs)}: {pdf.name}", file=sys.stderr)
            try:
                rc = _process_one(pdf, args)
                if rc != 0:
                    errors += 1
            except Exception as exc:
                print(f"error: {pdf.name}: {exc}", file=sys.stderr)
                errors += 1
        return 0 if errors == 0 else 1

    return _process_one(args.pdf, args)
