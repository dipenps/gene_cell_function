# Gene-Cell Function Summarizer

A Streamlit application that searches PubMed for literature on gene function in specific cell types and generates AI-powered summaries using local LLMs via Ollama.

## Features

- **PubMed Search** - Query PubMed by gene name and cell type, retrieving up to 30 relevant articles
- **HGNC Gene Resolution** - Automatically resolves gene aliases to canonical HGNC symbols and displays alias/protein name tables
- **AI Summarization** - Uses Ollama (local LLMs) to synthesize findings across articles into integrated biological summaries with PMID citations
- **Multi-Gene Support** - Process multiple genes in a single run with brief per-gene summaries
- **Downloadable Reports** - Export results as Markdown or HTML reports

## Prerequisites

- Python 3.12+
- [Ollama](https://ollama.ai) installed and running (`ollama serve`)
- At least one Ollama model pulled (e.g., `ollama pull llama3.2`)

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies

- `streamlit` - Web UI
- `biopython` - PubMed/NCBI Entrez API access
- `ollama` - Local LLM inference
- `pandas` - Data display

## Usage

```bash
# Start Ollama (in a separate terminal)
ollama serve

# Run the app
streamlit run app.py
```

Then open the app in your browser (default: http://localhost:8501).

1. Select an Ollama model from the sidebar
2. Enter one or more gene names (comma, semicolon, or newline separated)
3. Enter a cell type (e.g., "T cells", "macrophages")
4. Click **Search & Summarize**

## Docker

```bash
# Build
docker build -t gene-cell-function .

# Run (Ollama must be running on the host)
docker run -p 8501:8501 gene-cell-function
```

The Docker container connects to Ollama on the host via `host.docker.internal:11434`.

## Project Structure

```
app.py           # Streamlit UI and report generation
pubmed.py        # PubMed search via NCBI Entrez API
summarizer.py    # LLM summarization prompts and Ollama calls
hgnc.py          # HGNC gene symbol/alias lookup
hgnc_2026.txt    # HGNC gene data (tab-delimited)
requirements.txt # Python dependencies
Dockerfile       # Container setup
```
