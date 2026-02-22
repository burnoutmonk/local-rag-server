#!/bin/bash
# run.sh — Entry point for the local RAG system.
# - Creates a Python virtual environment and installs dependencies
# - Builds llama.cpp from source if llama-server is not found
# - Launches start.py
#
# Usage:
#   chmod +x run.sh
#   ./run.sh [--skip-ingest] [--skip-llm]

if [ -f .env ]; then
    # Removes comments and exports each line
    export $(grep -v '^#' .env | xargs)
    echo "✅ Environment variables loaded from .env"
else
    echo "❌ Error: .env file not found."
    exit 1
fi

Diagnostic check to ensure GPU is ready
if [ "$CUDA_AVAILABLE" = "true" ]; then
    echo "🚀 Launching with GPU Layers: $LLM_GPU_LAYERS"
fi

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
LLAMA_DIR="$SCRIPT_DIR/llama.cpp"
LLAMA_BIN="$LLAMA_DIR/build/bin/llama-server"

cd "$SCRIPT_DIR"

# ── Check for conflicting Docker Compose stack ────────────────────────────────
if docker ps --format "{{.Names}}" 2>/dev/null | grep -qE "rag_api|rag_llm|rag_ingest"; then
    echo "ERROR: Docker Compose stack appears to be already running."
    echo "  Stop it first with: docker compose down"
    echo "  Then run: ./run.sh"
    exit 1
fi

# ── Check for conflicting Qdrant container ────────────────────────────────────
if docker ps --format "{{.Names}}" 2>/dev/null | grep -q "^qdrant$"; then
    echo "WARNING: A Qdrant container is already running on port 6333."
    echo "  This may have been started by a previous Docker Compose run."
    echo "  Stop it with: docker stop qdrant && docker rm qdrant"
    echo "  Then run: ./run.sh"
    exit 1
fi

# ── Python venv ───────────────────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "Installing Python requirements..."
pip install -r requirements.txt --quiet
echo "  Done."
echo ""

# ── llama.cpp ─────────────────────────────────────────────────────────────────
if command -v llama-server &> /dev/null; then
    echo "llama-server found on PATH — skipping build."
elif [ -f "$LLAMA_BIN" ]; then
    echo "llama-server already built at $LLAMA_BIN — skipping build."
    export PATH="$LLAMA_DIR/build/bin:$PATH"
else
    echo "llama-server not found — building llama.cpp from source..."
    echo "  NOTE: This is a one-time setup and may take 10-20 minutes depending on your hardware."
    echo "  This will take a few minutes."
    echo ""

    # Check build dependencies
    for dep in git cmake make; do
        if ! command -v "$dep" &> /dev/null; then
            echo "ERROR: '$dep' is required to build llama.cpp."
            echo "  Run: sudo apt install build-essential cmake git"
            exit 1
        fi
    done

    if [ ! -d "$LLAMA_DIR" ]; then
        echo "  Cloning llama.cpp..."
        git clone https://github.com/ggerganov/llama.cpp "$LLAMA_DIR" --depth=1
    else
        echo "  Updating llama.cpp..."
        git -C "$LLAMA_DIR" pull --quiet
    fi

    echo "  Building..."
    # Check if CUDA is available and build with GPU support if so
    if command -v nvcc &> /dev/null; then
        echo "  CUDA found — building with GPU support..."
        cmake -B "$LLAMA_DIR/build" -S "$LLAMA_DIR" -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_CUDA=ON > /dev/null
    else
        echo "  CUDA not found — building CPU only..."
        cmake -B "$LLAMA_DIR/build" -S "$LLAMA_DIR" -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF > /dev/null
    fi
    cmake --build "$LLAMA_DIR/build" --config Release -j"$(nproc)" > /dev/null

    echo "  llama.cpp built successfully."
    export PATH="$LLAMA_DIR/build/bin:$PATH"
    echo ""
fi

# ── CUDA check ───────────────────────────────────────────────────────────────
if command -v nvcc &> /dev/null; then
    if "$LLAMA_DIR/build/bin/llama-server" --version 2>&1 | grep -q "CUDA\|cublas\|cuda"; then
        echo "GPU support: CUDA enabled"
    elif command -v llama-server &> /dev/null && llama-server --version 2>&1 | grep -q "CUDA\|cublas\|cuda"; then
        echo "GPU support: CUDA enabled"
    else
        echo ""
        echo "WARNING: CUDA is installed on your system but llama-server was built without GPU support."
        echo "  To enable GPU acceleration, delete the build and rerun:"
        echo "    rm -rf llama.cpp/build"
        echo "    ./run.sh"
        echo ""
    fi
fi

# ── Launch ────────────────────────────────────────────────────────────────────
# If using Docker Compose, use start.bat (Windows) or this script handles native
if [ -f "docker/docker-compose.yml" ] && [ "$1" = "--docker" ]; then
    # Use GPU compose override if CUDA_AVAILABLE is set
    if grep -q "CUDA_AVAILABLE=true" .env 2>/dev/null; then
        echo "  GPU mode enabled — using docker/docker-compose.gpu.yml"
        docker compose -f docker/docker-compose.yml -f docker/docker-compose.gpu.yml up -d --build
        docker compose -f docker/docker-compose.yml -f docker/docker-compose.gpu.yml --profile test build rag_test
    else
        docker compose -f docker/docker-compose.yml up -d --build
        docker compose -f docker/docker-compose.yml --profile test build rag_test
    fi
    echo "Waiting for services to be ready..."
    while ! docker inspect rag_ready --format "{{.State.Status}}" 2>/dev/null | grep -q "exited"; do
        sleep 2
    done
    echo "Opening browser..."
    xdg-open http://localhost:8000 2>/dev/null || open http://localhost:8000 2>/dev/null || echo "Open your browser at: http://localhost:8000"
else
    python start.py "$@"
fi