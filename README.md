# datasets-from-pdf
Python proof-of-concept app to ***try*** and extract data availability information from a published paper in PDF format, by using Ollama LLM and Qwen2.   

**Requirements**  
- Ollama LLM (latest version, installed and running locally or boxed)
- Model (tested): qwen2.5:14b (or qwen2.5:7b, faster but maybe less efficient)
- PyMuPDF (for data extraction)
- Tesseract (recommended, OCR fallback for image PDF:s, require PyMuPDF >= 1.19 - pip install pymupdf --upgrade)
     
**Usage**    

- single PDF    
python -m main.py paper.pdf

- batch, JSON files written alongside the PDFs    
python -m main.py --batch-dir /pdf_files

- batch, JSON files written to a separate output dir    
python -m main.py --batch-dir /pdf_files --out /results   

**Arguments**

| Argument | Description |
|---|---|
| `pdf` | Path to the PDF file |
| `--batch_dir` | Path to directory containing PDF:s for batch processing |
| `--out` | Path to output directory (if not current) |
| `--model MODEL` | Ollama model name (default: `qwen2.5:14b`) |
| `--no-ocr` | Disable OCR fallback for image-based pages |
| `--no-references` | Skip scanning the references section |
     
The app will try and extract the text from the PDF (or OCR if text extraction fails). It will try and look for data availability information in Data Availability Statement or References sections (see sections.py) and analyze the results. Response is returned as JSON.    

**Response (JSON)**

| Field | Description |
|---|---|
| name | short descriptive name of the dataset (string, or null) |
| repository | where it is hosted, e.g. "Zenodo", "Figshare", "GenBank", "Dryad", "GitHub", "institutional repository" (string, or null) |
| identifier | DOI, accession number, or similar persistent ID (string, or null) |
| url | URL (if present) (string, or null) |
| created_by_authors | true if the authors created/generated this dataset in this study, false if they merely reused an existing dataset, null if unclear |
| source_section | "data_availability" or "references" |
| is_open | true if data that seems to be openly accessible, false otherwise, null if unclear |
| is_code | true if dataset seems to be software code, false otherwise |
| is_supplementary | true if dataset seems to be spplement to the paper, false otherwise |
| evidence | short quote (max ~200 chars) from the text that supports this entry |

