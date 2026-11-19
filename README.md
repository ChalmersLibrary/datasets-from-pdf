# datasets-from-pdf
Proof-of-concept app to try and extract data availability information from a published paper in PDF format, by using Python, [Ollama LLM](https://ollama.com/) and/or a vLLM - such as [LiteLLM API](https://docs.litellm.ai/docs/) - chatbot. Using another (external) API service would work, but might require some changes to the code.      

The app will try and extract the text from the PDF (or OCR if text extraction fails). It will then look for data and/or code information in Data Availability Statement (similair) or References sections (see *sections.py*) and analyze the results. The response is returned as JSON and can also be written to a CSV file (see below).   
The prompt can be modified in *ollama.py*.

**Requirements**  
- Ollama LLM (latest version, installed and running locally or boxed)   
*or*   
- Access to a vLLM service (hosting a suitable model)
- Models (tested): qwen3.5 (default), gemma-4-31B, Qwen3.6-35B (much better, but require a more resources) or qwen2.M5:7b (faster but probably less efficient)
- PyMuPDF (for data extraction)
- Tesseract (recommended, OCR fallback for image PDF:s, require PyMuPDF >= 1.19).   

**Installation (Ollama)**    

Install Ollama (latest version for your o/s, if not using vLLM)   
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
python -m main paper.pdf
```

- single PDF with vLLM   
```
python -m main --backend vllm pdfs/paper.pdf 
```   

- batch, JSON written as output    
```
python -m main --batch-dir ./pdf_files
```

- batch, output written to a CSV file (JSON to screen)       
```
python -m main --batch-dir ./pdf_files --csv-output ./results/output.csv
```    

- batch, output written to an Excel file, no JSON to screen
```
python -m main --batch-dir ./pdf_files --xlsx-output ./results/output.xlsx --quiet
```

- PDFs read from a ZIP file, output written to CSV and Excel
```
python -m main --input-zip ./zipdir/my_files.zip --csv-output ./results/output.csv --xlsx-output ./results/output.xlsx
```

- batch, JSON files written to a separate output dir    
```
python -m main --batch-dir ./pdf_files --out ./results   
``` 

- use the vLLM API     
```
python -m main paper.pdf --backend vllm      
```   

- batch vLLM, output written to a CSV file (JSON to screen)       
```
python -m main --batch-dir ./pdf_files --csv-output ./results/output.csv --backend vllm    
```    

- batch vLLM, files read from ZIP and output written to a MS Excel file (without printing JSON to screen)       
```
python -m main --input-zip ./pdf_files/files.zip --xlsx-output ./results/output.xlsx --backend vllm --quite       
``` 
    
**Arguments**

| Argument | Description |
|---|---|
| `pdf` | Path to the PDF file |
| `--backend` | default: ollama, use *vllm* to use the vLLM version |    
| `--batch-dir` | Path to directory containing PDF:s for batch processing |
| `--input-zip` | Path to a ZIP file containing PDF:s for batch processing (sub-folders included) |
| `--out` | Path to output directory (if not current) |
| `--csv-output` | Path to output CSV file (incl. directory + file name); appended to if it exists |
| `--xlsx-output` | Path to output Excel file (incl. directory + file name, `.xlsx`); appended to if it exists |
| `-q`, `--quiet` | Do not print JSON output to the screen |
| `--model MODEL` | Ollama model name (default: `qwen3.5`) |
| `--enrich-urls` | Try and also extract metadata from found data publications for more correct output (slower) |
| `--no-ocr` | Disable OCR fallback for image-based pages |
| `--no-references` | Skip scanning the references section |
     
**Response (dataset, JSON)**

| Field | Description |
|---|---|    
| file | PDF file name (string) |   
| name | short descriptive name of the dataset (string, or null) |
| repository | where it is hosted, e.g. "Zenodo", "Figshare", "GenBank", "Dryad", "GitHub", "institutional repository" (string, or null) |
| identifier | DOI, accession number, or similar persistent ID (string, or null) |
| url | URL (if present) (string, or null) |    
| license | License information (if present) (string, or null) |
| created_by_authors | true if the authors created/generated this dataset in this study, false if they reused an existing dataset, null if unclear |
| source_section | "data_availability" or "references" |
| is_open | true if data seems to be openly accessible, false otherwise, null if unclear |
| is_code | true if dataset seems to be software code, false otherwise |
| is_supplementary | true if dataset seems to be a supplement to the paper, false otherwise |
| notes | short quote (max ~200 chars) from the text that supports this entry |
| vlm | model used for the extraction, e.g. "qwen3.5" (string) |
| response_time | total seconds spent waiting for the LLM for this paper, summed over all calls (number) |
| prompt_tokens | total prompt (input) tokens for this paper, summed over all calls (integer, or null if not reported by the server) |
| completion_tokens | total completion (output) tokens for this paper, summed over all calls (integer, or null if not reported by the server) |    

**Sample output (json, paper)**   

```
% python -m main "pdf_files/test.pdf --enrich-urls"             
[info] Extracted 125749 chars from PDF (OCR on 0 page(s))
[info] Found Data Availability section (3737 chars)
[info] No References section found.
[info] Fetching https://doi.org/10.5281/zenodo.123456
[info] Fetching https://github.com/MeaningOfLifeLab/First_project/
{
  "pdf": "pdf_files/test.pdf",
  "paper_doi": "10.1016/j.jss.2022.xxx",
  "das_found": true,
  "ocr_pages": [],
  "response_time": "12.2",
  "prompt_tokens": 2417,
  "completion_tokens": 658,
  "datasets": [
    {
      "name": "Stimuli recordings and study data",
      "repository": "Zenodo",
      "identifier": "10.5281/zenodo.123456",
      "url": "https://doi.org/10.5281/zenodo.123456",
      "license": "CC-BY-4.0",
      "created_by_authors": true,
      "source_section": "data_availability_statement",
      "notes": "All stimuli recordings and the data that support the ﬁndings of this study are openly available in Zenodo at https://doi.org/10.5281/zenodo.123456",
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
      "notes": "R script for ﬁgure generation is also provided in the github repository at https://github.com/MeaningOfLifeLab/First_project/",
      "is_open": true,
      "is_code": true,
      "is_supplementary": false
    }
  ]
}
```
 
