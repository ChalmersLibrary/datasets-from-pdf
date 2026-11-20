from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import zipfile
from dataclasses import asdict, fields
from pathlib import Path

from .models import Dataset, extract_datasets_from_pdf
from .usage import LLMUsage
from .llm_http import FatalLLMError
from .ollama import default_model
from .vllm_api import default_model as vllm_default_model

_DATASET_FIELDS = [f.name for f in fields(Dataset)]
_USAGE_FIELDS = ["response_time", "prompt_tokens", "completion_tokens"]
_CSV_COLUMNS = ["file", "paper_doi", "das_found", "ocr_page_count"] + _DATASET_FIELDS + ["vlm"] + _USAGE_FIELDS


def _bools_to_ints(obj):
    """Recursively replace True/False with 1/0 (None and other values are left unchanged)."""
    if isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _bools_to_ints(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_bools_to_ints(v) for v in obj]
    return obj


def _result_rows(pdf: Path, paper_doi: str | None, das_found: bool,
                 ocr_pages: list[int], datasets: list[Dataset], vlm: str,
                 llm_usage: LLMUsage) -> list[dict]:
    """One row per dataset (or a single empty row if none were found).

    Usage columns are per-paper totals over all LLM calls, repeated on each row.
    """
    base = {"file": pdf.name, "paper_doi": paper_doi, "das_found": das_found, "ocr_page_count": len(ocr_pages)}
    tail = {"vlm": vlm, **llm_usage.as_dict()}
    if datasets:
        rows = [{**base, **asdict(d), **tail} for d in datasets]
    else:
        rows = [{**base, **{k: None for k in _DATASET_FIELDS}, **tail}]
    return _bools_to_ints(rows)


def _check_header(path: Path, header: list) -> None:
    if list(header) != _CSV_COLUMNS:
        raise RuntimeError(
            f"{path} has different columns than this version writes "
            f"(e.g. created before the response_time/token columns were added); "
            f"use a new output file")


def _append_csv_rows(csv_path: Path, rows: list[dict]) -> None:
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    if not write_header:
        with csv_path.open(newline="", encoding="utf-8") as f:
            _check_header(csv_path, next(csv.reader(f), []))
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _append_xlsx_rows(xlsx_path: Path, rows: list[dict]) -> None:
    try:
        from openpyxl import Workbook, load_workbook
    except ImportError:
        raise RuntimeError("--xlsx-output requires openpyxl (pip install openpyxl)")
    if xlsx_path.exists() and xlsx_path.stat().st_size > 0:
        wb = load_workbook(xlsx_path)
        ws = wb.active
        _check_header(xlsx_path, next(ws.iter_rows(min_row=1, max_row=1, values_only=True), ()))
    else:
        wb = Workbook()
        ws = wb.active
        ws.title = "datasets"
        ws.append(_CSV_COLUMNS)
        ws.freeze_panes = "A2"
    for row in rows:
        ws.append([row.get(c) for c in _CSV_COLUMNS])
    # Save to a temp file and rename it over the original, so a job killed mid-save
    # (e.g. at the SLURM time limit) never leaves a corrupted workbook behind.
    tmp = xlsx_path.with_name(f".{xlsx_path.name}.tmp")
    try:
        wb.save(tmp)
        os.replace(tmp, xlsx_path)
    finally:
        tmp.unlink(missing_ok=True)


def _extract_zip_pdfs(zip_path: Path, dest: Path) -> list[Path]:
    """Extract all PDFs from zip_path into dest and return their paths (sorted by name)."""
    pdfs = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename
            if info.is_dir() or not name.lower().endswith(".pdf"):
                continue
            if name.startswith("__MACOSX/") or Path(name).name.startswith("._"):
                continue
            try:
                # ZipFile.extract sanitises absolute paths and ".." components.
                pdfs.append(Path(zf.extract(info, dest)))
            except (zipfile.BadZipFile, RuntimeError, OSError, NotImplementedError) as exc:
                # RuntimeError: encrypted member; NotImplementedError: unsupported compression
                print(f"[warn] skipping {name} in ZIP: {exc}", file=sys.stderr)
    return sorted(pdfs, key=lambda p: p.name.lower())


def _process_one(pdf: Path, args) -> int:
    if not pdf.exists():
        print(f"error: {pdf} not found", file=sys.stderr)
        return 1

    datasets, ocr_pages, das_found, paper_doi, llm_usage = extract_datasets_from_pdf(
        pdf,
        model=args.model,
        include_references=not args.no_references,
        ocr_fallback=not args.no_ocr,
        enrich_urls=args.enrich_urls,
        backend=args.backend,
    )

    output = {
        "file": pdf.name,
        "paper_doi": paper_doi,
        "das_found": das_found,
        "ocr_pages": ocr_pages,
        **llm_usage.as_dict(),
        "datasets": [{**asdict(d), "vlm": args.model} for d in datasets],
    }
    text = json.dumps(_bools_to_ints(output), indent=2, ensure_ascii=False)

    if args.out:
        out_path = Path(args.out)
        if out_path.is_dir():
            out_path = out_path / (pdf.stem + ".json")
        out_path.write_text(text, encoding="utf-8")
        print(f"[info] wrote {out_path}", file=sys.stderr)
    elif not args.quiet:
        print(text)

    if args.csv_output or args.xlsx_output:
        rows = _result_rows(pdf, paper_doi, das_found, ocr_pages, datasets, args.model, llm_usage)
        if args.csv_output:
            _append_csv_rows(args.csv_output, rows)
            print(f"[info] appended to {args.csv_output}", file=sys.stderr)
        if args.xlsx_output:
            _append_xlsx_rows(args.xlsx_output, rows)
            print(f"[info] appended to {args.xlsx_output}", file=sys.stderr)

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
    ap.add_argument("--input-zip", type=Path, default=None,
                    help="Process all PDFs inside this ZIP file (incl. sub-folders)")
    ap.add_argument("--backend", choices=["ollama", "vllm"], default="ollama",
                    help="LLM backend: local Ollama or a vLLM (OpenAI-compatible) server (default: ollama)")
    ap.add_argument("--model", default=None,
                    help=f"Model name (default: {default_model} for ollama, "
                         f"{vllm_default_model} for vllm)")
    ap.add_argument("--no-ocr", action="store_true",
                    help="Disable OCR fallback for image-based pages")
    ap.add_argument("--no-references", action="store_true",
                    help="Skip scanning the references section")
    ap.add_argument("--enrich-urls", action="store_true",
                    help="Fetch external URLs/DOIs from results and re-query the model to enrich records")
    ap.add_argument("--out", type=Path, default=None,
                    help="Write JSON to this file (single) or directory (batch) instead of stdout")
    ap.add_argument("--csv-output", type=Path, default=None,
                    help="Append results to this CSV file (one row per dataset)")
    ap.add_argument("--xlsx-output", type=Path, default=None,
                    help="Append results to this Excel (.xlsx) file (one row per dataset)")
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="Do not print the JSON output to the screen")
    ap.add_argument("--max-consecutive-errors", type=int, default=5, metavar="N",
                    help="Stop a batch after N PDFs in a row have failed (default: 5, 0 = never stop)")
    args = ap.parse_args()
    if args.model is None:
        args.model = vllm_default_model if args.backend == "vllm" else default_model

    sources = [x for x in (args.pdf, args.batch_dir, args.input_zip) if x]
    if len(sources) > 1:
        print("error: specify only one of: a pdf, --batch-dir or --input-zip", file=sys.stderr)
        return 1
    if not sources:
        ap.print_help()
        return 1
    if args.xlsx_output and args.xlsx_output.suffix.lower() != ".xlsx":
        print("error: --xlsx-output file must have a .xlsx extension", file=sys.stderr)
        return 1

    # Fail early (before any LLM calls) on problems with the output files.
    try:
        for path in (args.csv_output, args.xlsx_output):
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
        if args.out and args.pdf:            # single PDF: --out may be a file in a new folder
            (args.out if args.out.suffix == "" else args.out.parent).mkdir(parents=True, exist_ok=True)
        if args.csv_output and args.csv_output.exists() and args.csv_output.stat().st_size > 0:
            with args.csv_output.open(newline="", encoding="utf-8") as f:
                _check_header(args.csv_output, next(csv.reader(f), []))
        if args.xlsx_output and args.xlsx_output.exists() and args.xlsx_output.stat().st_size > 0:
            from openpyxl import load_workbook
            wb = load_workbook(args.xlsx_output, read_only=True)
            _check_header(args.xlsx_output, next(wb.active.iter_rows(min_row=1, max_row=1, values_only=True), ()))
            wb.close()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ImportError:
        print("error: --xlsx-output requires openpyxl (pip install openpyxl)", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"error: cannot use output file: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if args.input_zip:
        if not args.input_zip.is_file():
            print(f"error: {args.input_zip} not found", file=sys.stderr)
            return 1
        if not zipfile.is_zipfile(args.input_zip):
            print(f"error: {args.input_zip} is not a valid ZIP file", file=sys.stderr)
            return 1
        with tempfile.TemporaryDirectory(prefix="extract_dataset_") as tmp:
            pdfs = _extract_zip_pdfs(args.input_zip, Path(tmp))
            if not pdfs:
                print(f"[info] no PDF files found in {args.input_zip}", file=sys.stderr)
                return 0
            return _process_batch(pdfs, args)

    if args.batch_dir:
        if not args.batch_dir.is_dir():
            print(f"error: {args.batch_dir} is not a directory", file=sys.stderr)
            return 1
        pdfs = sorted(args.batch_dir.glob("*.pdf"))
        if not pdfs:
            print(f"[info] no PDF files found in {args.batch_dir}", file=sys.stderr)
            return 0
        return _process_batch(pdfs, args)

    try:
        return _process_one(args.pdf, args)
    except FatalLLMError as exc:
        print(f"error: LLM server rejected the request: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: {args.pdf.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def _process_batch(pdfs: list[Path], args) -> int:
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    consecutive = 0
    rc = 0
    done = 0
    for i, pdf in enumerate(pdfs, 1):
        print(f"[info] processing {i}/{len(pdfs)}: {pdf.name}", file=sys.stderr)
        done = i
        try:
            ok = _process_one(pdf, args) == 0
        except FatalLLMError as exc:
            # Bad API key, unknown model, wrong URL … every remaining PDF would fail the same way.
            print(f"error: LLM server rejected the request, stopping batch: {exc}", file=sys.stderr)
            failed.append(pdf.name)
            rc = 2
            break
        except Exception as exc:
            print(f"error: {pdf.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            ok = False
        if ok:
            consecutive = 0
            continue
        failed.append(pdf.name)
        consecutive += 1
        if args.max_consecutive_errors and consecutive >= args.max_consecutive_errors:
            print(f"error: {consecutive} PDFs in a row failed, stopping batch "
                  f"(see errors above; change with --max-consecutive-errors)", file=sys.stderr)
            rc = 2
            break

    skipped = len(pdfs) - done
    print(f"[info] done: {done - len(failed)} succeeded, {len(failed)} failed"
          + (f", {skipped} not processed" if skipped else ""), file=sys.stderr)
    if failed:
        print("[info] failed files: " + ", ".join(failed), file=sys.stderr)
    if rc:
        return rc
    return 1 if failed else 0
