# ── config.py ─────────────────────────────────────────────────────────────────
# Central configuration for the local RAG system.
# Edit this file to tune the system for your hardware and documents.
# ──────────────────────────────────────────────────────────────────────────────


# ── Embedding model ───────────────────────────────────────────────────────────
# all-MiniLM-L6-v2 is fast on CPU and a good default.
# For better retrieval quality (requires GPU), try "BAAI/bge-large-en-v1.5".
# You can also set this to a local folder path, e.g. "/home/user/models/minilm"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ── Qdrant ────────────────────────────────────────────────────────────────────
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION  = "rag_docs"


# ── LLM (llama.cpp server) ────────────────────────────────────────────────────
LLM_URL = "http://localhost:8080/v1/chat/completions"

# Path where the GGUF model will be saved (or already exists)
# Example: "/home/user/models/qwen2.5-3b-instruct.gguf"
LLM_MODEL_PATH = "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"

# HuggingFace repo and filename to download if LLM_MODEL_PATH doesn't exist.
# Find GGUF models at https://huggingface.co/models?search=gguf
# Set LLM_MODEL_REPO to None to disable auto-download.
#
# Good CPU-friendly defaults:
#   Qwen2.5-3B  — fast, good quality for its size
#     LLM_MODEL_REPO = "Qwen/Qwen2.5-3B-Instruct-GGUF"
#     LLM_MODEL_FILE = "qwen2.5-3b-instruct-q4_k_m.gguf"
#
#   Phi-3-mini  — very fast on CPU, smaller context
#     LLM_MODEL_REPO = "microsoft/Phi-3-mini-4k-instruct-gguf"
#     LLM_MODEL_FILE = "Phi-3-mini-4k-instruct-q4.gguf"

LLM_MODEL_REPO = "bartowski/Llama-3.2-3B-Instruct-GGUF"
LLM_MODEL_FILE = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"

# llama.cpp server settings
# If llama-server is not on your PATH, set the full path here.
# Example: "/home/user/llama.cpp/build/bin/llama-server"
# Leave as None to auto-detect.
LLAMA_SERVER_BIN = None

LLM_HOST    = "0.0.0.0"
LLM_PORT    = 8080
LLM_CONTEXT = 4096   # context window size — increase if your model supports it (e.g. 8192, 32768)
LLM_THREADS = 8     # CPU threads for inference — set to your physical core count
LLM_GPU_LAYERS = 0   # layers to offload to GPU — 0 = CPU only, -1 = all layers

# Sampling parameters — tune to your model's recommendations
LLM_TEMPERATURE = 0.7
LLM_TOP_P       = 0.8
LLM_TOP_K       = 20
LLM_MIN_P       = 0.0


# ── Web UI ───────────────────────────────────────────────────────────────────
# Host and port for the web UI.
# Use "0.0.0.0" to accept connections from other machines on the network.
# Access it at http://<your-server-ip>:8000
API_HOST = "0.0.0.0"
API_PORT = 8000

# ── Output token limits ───────────────────────────────────────────────────────
# MAX_TOKENS: hard cap on LLM output length.
#   - Slow CPU (older hardware, <5 tok/s): 200–300
#   - Medium CPU (modern laptop/desktop): 300–500
#   - Fast CPU or GPU: 500–900
MAX_TOKENS = 500

# MIN_TOKENS: never generate fewer than this many tokens.
# Prevents the LLM from cutting off mid-sentence on tight timeouts.
MIN_TOKENS = 150


# ── Timeout & speed estimation ────────────────────────────────────────────────
# TOKENS_PER_SECOND: your LLM's generation speed.
# To measure it, run the test script: python test_speed.py
# Typical values:
#   - CPU only (4-8B model):  3–15 tok/s
#   - GPU (4-8B model):       30–80 tok/s
TOKENS_PER_SECOND = 6.4

# RETRIEVAL_OVERHEAD_S: time taken for embedding + Qdrant query.
# MiniLM on CPU is typically 0.5–1.5s. Increase if you have many chunks.
RETRIEVAL_OVERHEAD_S = 1.0


# ── Chunking (ingest.py) ──────────────────────────────────────────────────────
# MAX_CHARS: maximum characters per chunk.
#   - 600–800: better for documents with many short definitions
#   - 1000–1200: better for procedural or narrative content
MAX_CHARS = 1000

# OVERLAP_CHARS: characters of overlap between consecutive chunks.
# Helps avoid cutting context at chunk boundaries.
OVERLAP_CHARS = 100

# BATCH_SIZE: number of chunks to embed at once.
# Reduce if you run out of memory during ingestion.
BATCH_SIZE = 64


# ── Retrieval (rag_api.py) ────────────────────────────────────────────────────
# Maximum total characters of source context passed to the LLM.
# Prevents context overflow errors regardless of top_k setting.
# Rule of thumb: keep below (model context window in tokens) * 3
MAX_CONTEXT_CHARS = 6000