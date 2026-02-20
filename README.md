# Local RAG — PDF/DOCX Question Answering over a Local LLM

A lightweight, fully local retrieval-augmented generation (RAG) system. Drop in your PDF and DOCX files, run two commands, and ask questions through a browser UI. No cloud services, no API keys.

## Architecture

```
PDF / DOCX files
      |
      v
 Text extraction & cleaning (pypdf, python-docx)
      |
      v
 Chunking (overlapping, paragraph-aware)
      |
      v
 all-MiniLM-L6-v2 embeddings (384-dim, fast on CPU)
      |
      v
 Qdrant (local vector store)
      |
      v
 FastAPI retrieval API  <--  llama.cpp (local LLM, OpenAI-compatible)
      |
      v
 Browser UI
```

---

## Requirements

- Python 3.10+
- Docker (for Qdrant)
- Git, CMake, and a C++ compiler (to build llama.cpp)

On Ubuntu:
```bash
sudo apt install build-essential cmake git docker.io python3-venv
```

Any GGUF model works. For good CPU-friendly options try [Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF) or [Phi-3-mini](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf).

`run.sh` will automatically clone and build llama.cpp if it is not already installed. Qdrant is started automatically via Docker.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Configure

Edit `config.py` and set at minimum:
- `LLM_MODEL_PATH` — path to your `.gguf` model file
- `LLM_THREADS` — number of CPU cores to use
- `LLM_CONTEXT` — context window size (check your model's specs)

### 2. Add your documents

Place PDF and DOCX files in the `data_raw/` folder.

### 3. Start everything

```bash
chmod +x run.sh
./run.sh
```

This will automatically:
1. Create a Python virtual environment
2. Install all dependencies
3. Download the model from HuggingFace (if not already present)
4. Start Qdrant via Docker
5. Ingest all documents from `data_raw/`
6. Start the llama.cpp LLM server
7. Start the web UI at `http://localhost:8000`

Press `Ctrl+C` to stop all services.

### Useful flags

```bash
./run.sh --skip-ingest   # skip ingestion if documents haven't changed
./run.sh --skip-llm      # skip LLM server if it's already running
```

---

## Configuration

All settings are at the top of each file:

| Variable | File | Description |
|---|---|---|
| `EMBED_MODEL_NAME` | `ingest.py`, `rag_api.py` | Embedding model (default: MiniLM-L6-v2) |
| `MAX_CHARS` | `ingest.py` | Max chars per chunk (default: 1000) |
| `OVERLAP_CHARS` | `ingest.py` | Overlap between chunks (default: 100) |
| `LLM_URL` | `rag_api.py` | llama.cpp server endpoint |
| `TOKENS_PER_SECOND` | `rag_api.py` | Tune to your hardware for timeout estimation |
| `MAX_TOKENS` | `rag_api.py` | Max LLM output tokens (default: 500) |
| `COLLECTION` | both | Qdrant collection name |

---

## API Endpoints

### `POST /answer`
Retrieves relevant chunks and generates an answer with the LLM.

```bash
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"query": "what is X", "top_k": 5, "mode": "answer", "timeout": 60}'
```

### `POST /search`
Returns raw retrieved chunks without calling the LLM. Useful for debugging retrieval quality.

```bash
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"query": "what is X", "top_k": 5, "mode": "search"}'
```

### `GET /health`
Returns `{"ok": true}` if the API is running.

---

## Project Structure

```
.
|-- data_raw/          # Place your PDF and DOCX files here
|-- templates/
|   `-- index.html     # Browser UI
|-- config.py          # All settings — edit this first
|-- run.sh             # Entry point: creates venv, installs deps, runs start.py
|-- start.py           # Main launcher (Qdrant + LLM + web UI)
|-- ingest.py          # Parse, chunk, embed, and upsert into Qdrant
|-- rag_api.py         # FastAPI retrieval + LLM answer API
`-- requirements.txt
```

---

## Tips

- **Retrieval quality:** `top_k=5` is a good default. Increase if answers feel incomplete.
- **Speed:** MiniLM embeds ~1000 chunks/min on CPU. Ingestion is a one-time cost.
- **Chunk size:** `MAX_CHARS=1000` works well for most documents. Reduce to 600-700 for documents with many short definitions.
- **PDF cleaning:** `clean_pdf_text()` in `ingest.py` strips page numbers, TOC lines, and copyright watermarks automatically. Add custom rules for your document patterns.
- **Upgrading embeddings:** For better retrieval quality at the cost of needing a GPU, swap MiniLM for `BAAI/bge-m3` and enable hybrid dense+sparse search with Qdrant.