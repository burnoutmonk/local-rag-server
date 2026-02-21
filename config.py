# ── config.py ─────────────────────────────────────────────────────────────────
# Single source of truth for all settings.
#
# Native workflow (run.sh):   edit values directly in this file
# Docker workflow:            values are overridden by .env via environment variables
# ──────────────────────────────────────────────────────────────────────────────

import os


def env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))


def env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, default))


# ── Embedding model ───────────────────────────────────────────────────────────
# all-MiniLM-L6-v2 is fast on CPU and a good default.
# For better retrieval quality (requires GPU), try "BAAI/bge-large-en-v1.5".
EMBED_MODEL_NAME = env("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")


# ── Qdrant ────────────────────────────────────────────────────────────────────
QDRANT_HOST = env("QDRANT_HOST", "localhost")
QDRANT_PORT = env_int("QDRANT_PORT", 6333)
COLLECTION  = env("COLLECTION", "rag_docs")


# ── LLM (llama.cpp server) ────────────────────────────────────────────────────
LLM_URL = env("LLM_URL", "http://localhost:8080/v1/chat/completions")

# Path where the GGUF model will be saved (or already exists)
LLM_MODEL_PATH = env("LLM_MODEL_PATH", "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf")

# HuggingFace repo and filename to download if model doesn't exist
LLM_MODEL_REPO = "bartowski/Llama-3.2-3B-Instruct-GGUF"
LLM_MODEL_FILE = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"

# llama.cpp server settings
# If llama-server is not on your PATH, set the full path here.
LLAMA_SERVER_BIN = env("LLAMA_SERVER_BIN", "") or None

LLM_HOST       = env("LLM_HOST", "0.0.0.0")
LLM_PORT       = env_int("LLM_PORT", 8080)
LLM_CONTEXT    = env_int("LLM_CONTEXT", 4096)
LLM_THREADS    = env_int("LLM_THREADS", 8)
LLM_GPU_LAYERS = env_int("LLM_GPU_LAYERS", 0)

# Sampling parameters
LLM_TEMPERATURE = env_float("LLM_TEMPERATURE", 0.7)
LLM_TOP_P       = env_float("LLM_TOP_P", 0.8)
LLM_TOP_K       = env_int("LLM_TOP_K", 20)
LLM_MIN_P       = env_float("LLM_MIN_P", 0.0)


# ── Output token limits ───────────────────────────────────────────────────────
# MAX_TOKENS: hard cap on LLM output length.
#   - Slow CPU (older hardware, <5 tok/s): 200-300
#   - Medium CPU (modern laptop/desktop):  300-500
#   - Fast CPU or GPU:                     500-900
MAX_TOKENS = env_int("MAX_TOKENS", 500)
MIN_TOKENS = env_int("MIN_TOKENS", 150)


# ── Timeout & speed estimation ────────────────────────────────────────────────
# Run python test_speed.py to measure your actual tok/s
TOKENS_PER_SECOND    = env_float("TOKENS_PER_SECOND", 10.0)
RETRIEVAL_OVERHEAD_S = env_float("RETRIEVAL_OVERHEAD_S", 1.0)


# ── Chunking (ingest.py) ──────────────────────────────────────────────────────
MAX_CHARS     = env_int("MAX_CHARS", 1000)
OVERLAP_CHARS = env_int("OVERLAP_CHARS", 100)
BATCH_SIZE    = env_int("BATCH_SIZE", 64)


# ── Retrieval (rag_api.py) ────────────────────────────────────────────────────
MAX_CONTEXT_CHARS = env_int("MAX_CONTEXT_CHARS", 6000)


# ── Web UI ────────────────────────────────────────────────────────────────────
API_HOST = env("API_HOST", "0.0.0.0")
API_PORT = env_int("API_PORT", 8000)