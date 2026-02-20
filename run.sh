#!/bin/bash
# run.sh — Entry point for the local RAG system.
# - Creates a Python virtual environment and installs dependencies
# - Builds llama.cpp from source if llama-server is not found
# - Launches start.py
#
# Usage:
#   chmod +x run.sh
#   ./run.sh [--skip-ingest] [--skip-llm]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
LLAMA_DIR="$SCRIPT_DIR/llama.cpp"
LLAMA_BIN="$LLAMA_DIR/build/bin/llama-server"

cd "$SCRIPT_DIR"

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
    cmake -B "$LLAMA_DIR/build" -S "$LLAMA_DIR" -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF > /dev/null
    cmake --build "$LLAMA_DIR/build" --config Release -j"$(nproc)" > /dev/null

    echo "  llama.cpp built successfully."
    export PATH="$LLAMA_DIR/build/bin:$PATH"
    echo ""
fi

# ── Launch ────────────────────────────────────────────────────────────────────
python start.py "$@"
