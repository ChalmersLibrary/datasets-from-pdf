# datasets-from-pdf
Python proof-of-concept app to ***try*** and extract data availability information from a published paper in PDF format, by using Ollama LLM and Qwen3.   

**Requirements**  
- Ollama LLM (latest version, installed and running locally or boxed)
- Models (tested): qwen3.5 (default), qwen2.5:14b (or qwen2.5:7b, faster but maybe less efficient)
- PyMuPDF (for data extraction)
- Tesseract (recommended, OCR fallback for image PDF:s, require PyMuPDF >= 1.19).   

**Installation**    

Install Ollama (latest version for your o/s)   
 [download and installation instructions](https://ollama.com/download)  

Start Ollama    
```
ollama serve 
```   
Download the model   
```
ollama pull qwen3.5:latest
```    
Install python dependencies    
```
pip install pymupdf requests
```   

Install Tesseract (for OCR, recommended)    
[download and installation instructions](https://tesseract-ocr.github.io/tessdoc/Installation.html)    

Test
``` 
ollama run qwen3.5 "Please print Hello world! to the screen"
```    
       

**Usage**    

- single PDF    
```
python -m main.py paper.pdf
```

- batch, JSON files written alongside the PDFs    
```
python -m main.py --batch-dir /pdf_files
```

- batch, JSON files written to a separate output dir    
```
python -m main.py --batch-dir /pdf_files --out /results   
```

**Arguments**

| Argument | Description |
|---|---|
| `pdf` | Path to the PDF file |
| `--batch_dir` | Path to directory containing PDF:s for batch processing |
| `--out` | Path to output directory (if not current) |
| `--model MODEL` | Ollama model name (default: `qwen3.5`) |
| `--no-ocr` | Disable OCR fallback for image-based pages |
| `--no-references` | Skip scanning the references section |
     
The app will try and extract the text from the PDF (or OCR if text extraction fails). It will try and look for data availability information in Data Availability Statement or References sections (see sections.py) and analyze the results. Response is returned as JSON.    

**Response (dataset, JSON)**

| Field | Description |
|---|---|
| name | short descriptive name of the dataset (string, or null) |
| repository | where it is hosted, e.g. "Zenodo", "Figshare", "GenBank", "Dryad", "GitHub", "institutional repository" (string, or null) |
| identifier | DOI, accession number, or similar persistent ID (string, or null) |
| url | URL (if present) (string, or null) |    
| license | License information (if present) (string, or null) |
| created_by_authors | true if the authors created/generated this dataset in this study, false if they merely reused an existing dataset, null if unclear |
| source_section | "data_availability" or "references" |
| is_open | true if data seems to be openly accessible, false otherwise, null if unclear |
| is_code | true if dataset seems to be software code, false otherwise |
| is_supplementary | true if dataset seems to be a supplement to the paper, false otherwise |
| evidence | short quote (max ~200 chars) from the text that supports this entry |    

**Sample output**   

```
% python3 main.py "pdf_files/test.pdf"             
[info] Extracted 125749 chars from PDF (OCR on 0 page(s))
[info] Found Data Availability section (3737 chars)
[info] No References section found.
{
  "pdf": "pdf_files/test.pdf",
  "das_found": true,
  "ocr_pages": [],
  "datasets": [
    {
      "name": "Stimuli recordings and study data",
      "repository": "Zenodo",
      "identifier": "10.5281/zenodo.123456",
      "url": "https://doi.org/10.5281/zenodo.123456",
      "license": "CC-BY-4.0",
      "created_by_authors": true,
      "source_section": "data_availability_statement",
      "evidence": "All stimuli recordings and the data that support the ﬁndings of this study are openly available in Zenodo at https://doi.org/10.5281/zenodo.16901844",
      "is_open": true,
      "is_code": false,
      "is_supplementary": false
    },
    {
      "name": "R script for figure generation",
      "repository": "GitHub",
      "url": "https://github.com/MeaningOfLifeLab/First_project/",
      "license": "MIT",
      "created_by_authors": true,
      "source_section": "data_availability_statement",
      "evidence": "R script for ﬁgure generation is also provided in the github repository at https://github.com/MeaningOfLifeLab/First_project/",
      "is_open": true,
      "is_code": true,
      "is_supplementary": false
    }
  ]
}
```
 
