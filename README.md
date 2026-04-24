# datasets-from-pdf
Python proof-of-concept app to try and extract data availability information from a published paper in PDF format, by using Ollama LLM and Qwen2.   

Currently only processes a single (1) pdf from the CLI.   

**Requirements**  
- Ollama LLM (latest version, installed and running locally or boxed)
- Model (recommended): qwen2.5:14b (or qwen2.5:7b if running too slow)
- PyMuPDF (for data extraction)
- Tesseract (recommended, OCR fallback for image PDF:s, require PyMuPDF >= 1.19 - pip install pymupdf --upgrade)
     
**Run**    
python main.py [PATH]/[PDF_FILE]
*Example:* python main.py pdf/test.pdf

**Arguments**

| Argument | Description |
|---|---|
| `pdf` | Path to the PDF file |
| `--model MODEL` | Ollama model name (default: `qwen2.5:14b`) |
| `--no-ocr` | Disable OCR fallback for image-based pages |
| `--no-references` | Skip scanning the references section |
| `--out FILE` | Write JSON to this file instead of stdout |
     
The app will try and extract the text from the PDF (or OCR if text extraction fails). It will try and look for data availability information in Data Availability Statement or References sections (see sections.py) and analyze the results. Response is returned as JSON.   

