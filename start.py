"""
start.py — One-command startup for the local RAG system.

  0. Install Python requirements
  1. Download the model from HuggingFace (if not already present)
  2. Start Qdrant via Docker
  3. Ingest documents from data_raw/
  4. Start the llama.cpp LLM server
  5. Start the FastAPI web UI

Usage:
    python start.py [--skip-ingest] [--skip-llm]
"""

# ONLY stdlib at module level
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path


# ── Third-party imports ───────────────────────────────────────────────────────
from config import (
    LLM_MODEL_PATH, LLM_MODEL_REPO, LLM_MODEL_FILE,
    LLM_HOST, LLM_PORT, LLM_CONTEXT, LLM_THREADS, LLM_GPU_LAYERS,
    QDRANT_HOST, QDRANT_PORT, LLAMA_SERVER_BIN,
    API_HOST, API_PORT,
)

# ── Constants ─────────────────────────────────────────────────────────────────
QDRANT_TIMEOUT = 30
LLM_TIMEOUT    = 180  # 3 minutes — large models can take a while to load


# ── Helpers ───────────────────────────────────────────────────────────────────
def section(title: str) -> None:
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")


def require(binary: str, install_hint: str) -> str:
    path = shutil.which(binary)
    if path is None:
        print(f"ERROR: '{binary}' not found on PATH.")
        print(f"  {install_hint}")
        sys.exit(1)
    return path


def wait_for_http(url: str, label: str, timeout: int) -> None:
    import urllib.request
    import urllib.error
    print(f"  Waiting for {label} to be ready", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status < 500:
                    print(" ready!")
                    return
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(1)
    print()
    print(f"ERROR: {label} did not become ready within {timeout}s.")
    sys.exit(1)


# ── Steps ─────────────────────────────────────────────────────────────────────
def download_model() -> None:
    section("Step 1/5 — Checking model")
    model = Path(LLM_MODEL_PATH)

    if model.exists():
        print(f"  Model already exists: {model}")
        return

    if not LLM_MODEL_REPO:
        print(f"ERROR: Model not found at {LLM_MODEL_PATH}")
        print("  Set LLM_MODEL_PATH to an existing .gguf file, or set")
        print("  LLM_MODEL_REPO and LLM_MODEL_FILE in config.py to enable auto-download.")
        sys.exit(1)

    from huggingface_hub import hf_hub_download
    model.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {LLM_MODEL_FILE} from {LLM_MODEL_REPO} ...")
    print(f"  Saving to  {model}")
    print(f"  This may take a few minutes.\n")
    hf_hub_download(repo_id=LLM_MODEL_REPO, filename=LLM_MODEL_FILE, local_dir=str(model.parent))
    print(f"  Download complete: {model}")


def start_qdrant(procs: list) -> None:
    section("Step 2/5 — Starting Qdrant")
    require("docker", "Install Docker from https://docs.docker.com/get-docker/")

    running = subprocess.run(
        ["docker", "ps", "--filter", "name=qdrant", "--format", "{{.Names}}"],
        capture_output=True, text=True,
    )
    if "qdrant" in running.stdout:
        print("  Qdrant container already running — skipping.")
    else:
        stopped = subprocess.run(
            ["docker", "ps", "-a", "--filter", "name=qdrant", "--format", "{{.Names}}"],
            capture_output=True, text=True,
        )
        if "qdrant" in stopped.stdout:
            print("  Removing stopped Qdrant container...")
            subprocess.run(["docker", "rm", "qdrant"], capture_output=True)

        print("  Starting Qdrant Docker container...")
        subprocess.run([
            "docker", "run", "-d", "--name", "qdrant",
            "-p", f"{QDRANT_PORT}:6333",
            "-v", "qdrant_storage:/qdrant/storage",  # persist data across restarts
            "qdrant/qdrant",
        ], check=True)

    wait_for_http(f"http://{QDRANT_HOST}:{QDRANT_PORT}/healthz", "Qdrant", QDRANT_TIMEOUT)


def run_ingest() -> None:
    section("Step 3/5 — Ingesting documents")
    data_dir = Path("data_raw")
    if not data_dir.exists() or not any(data_dir.iterdir()):
        print("  WARNING: data_raw/ is empty or missing.")
        print("  Add PDF or DOCX files to data_raw/ and re-run, or use --skip-ingest.")
        return

    result = subprocess.run([sys.executable, "ingest.py"])
    if result.returncode != 0:
        print("ERROR: Ingestion failed. Fix the errors above and try again.")
        sys.exit(1)


def find_llama_server() -> str:
    """Find llama-server binary — checks config, PATH, and common install locations."""
    # Check config override first
    if LLAMA_SERVER_BIN:
        if Path(LLAMA_SERVER_BIN).exists():
            return LLAMA_SERVER_BIN
        print(f"ERROR: LLAMA_SERVER_BIN set in config.py but not found: {LLAMA_SERVER_BIN}")
        sys.exit(1)

    # Check PATH
    path = shutil.which("llama-server")
    if path:
        return path

    # Common locations people build llama.cpp into
    common = [
        Path.home() / "llama.cpp" / "build" / "bin" / "llama-server",
        Path.home() / "llama.cpp" / "llama-server",
        Path("/usr/local/bin/llama-server"),
        Path("/opt/llama.cpp/llama-server"),
    ]
    for p in common:
        if p.exists():
            return str(p)

    print("ERROR: llama-server not found.")
    print()
    print("  To install llama.cpp on Ubuntu:")
    print("    git clone https://github.com/ggerganov/llama.cpp")
    print("    cd llama.cpp")
    print("    cmake -B build")
    print("    cmake --build build --config Release -j$(nproc)")
    print()
    print("  Then either:")
    print("    a) Add it to PATH:  export PATH=$PATH:~/llama.cpp/build/bin")
    print("    b) Set the full path in start.py LLAMA_SERVER_BIN variable")
    sys.exit(1)


def start_llm(procs: list) -> None:
    section("Step 4/5 — Starting LLM server")
    binary = find_llama_server()

    model = Path(LLM_MODEL_PATH)
    if not model.exists():
        print(f"ERROR: Model file not found: {LLM_MODEL_PATH}")
        sys.exit(1)

    cmd = [
        binary,
        "--model",        str(model),
        "--host",         LLM_HOST,
        "--port",         str(LLM_PORT),
        "--ctx-size",     str(LLM_CONTEXT),
        "--threads",      str(LLM_THREADS),
        "--n-gpu-layers", str(LLM_GPU_LAYERS),
    ]
    print(f"  Model   : {model.name}")
    print(f"  Threads : {LLM_THREADS}  |  Context: {LLM_CONTEXT}  |  GPU layers: {LLM_GPU_LAYERS}")

    proc = subprocess.Popen(cmd)  # let output through so user can see loading progress
    procs.append(("LLM server", proc))
    wait_for_http(f"http://localhost:{LLM_PORT}/health", "LLM server", LLM_TIMEOUT)


def measure_token_speed() -> float:
    """Run a quick single inference pass to estimate tok/s."""
    import json
    import urllib.request
    from config import LLM_URL, MAX_TOKENS

    payload = json.dumps({
        "model": "local",
        "messages": [{"role": "user", "content": "Write a 100 word technical summary of CPU architecture."}],
        "max_tokens": 120,
        "temperature": 0.7,
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        LLM_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.time()
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read())
    elapsed = time.time() - start
    words = len(data["choices"][0]["message"]["content"].split())
    tok_s = round(words / elapsed, 1)
    return tok_s


def benchmark_llm() -> None:
    section("Step 4.5/5 — Measuring LLM speed")
    print("  Running a quick inference pass to estimate tok/s...")
    try:
        tok_s = measure_token_speed()
        print(f"  Estimated speed: {tok_s} tok/s")
        print(f"  Updating TOKENS_PER_SECOND in config.py...")

        config_path = Path(__file__).resolve().parent / "config.py"
        config_text = config_path.read_text()
        import re
        config_text = re.sub(
            r"TOKENS_PER_SECOND\s*=\s*[\d.]+",
            f"TOKENS_PER_SECOND = {tok_s}",
            config_text,
        )
        config_path.write_text(config_text)
        print(f"  config.py updated.")
    except Exception as e:
        print(f"  WARNING: Speed measurement failed ({e}) — keeping existing value.")


def start_api(procs: list) -> None:
    section("Step 5/5 — Starting web UI")
    cmd = [sys.executable, "-m", "uvicorn", "rag_api:app", "--host", API_HOST, "--port", str(API_PORT)]
    proc = subprocess.Popen(cmd)
    procs.append(("Web UI", proc))
    wait_for_http(f"http://localhost:{API_PORT}/health", "Web UI", 15)
    print(f"\n  Open your browser at: http://<your-server-ip>:{API_PORT}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Start the local RAG system.")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip document ingestion")
    parser.add_argument("--skip-llm",    action="store_true", help="Skip starting the LLM server")
    args = parser.parse_args()

    procs = []
    try:
        download_model()
        start_qdrant(procs)

        if not args.skip_ingest:
            run_ingest()
        else:
            print("\n  Skipping ingestion (--skip-ingest).")

        if not args.skip_llm:
            start_llm(procs)
        else:
            print("\n  Skipping LLM server (--skip-llm).")

        benchmark_llm()
        start_api(procs)

        print(f"\n{'='*50}")
        print("  All services running. Press Ctrl+C to stop.")
        print(f"{'='*50}\n")

        while True:
            time.sleep(1)
            for name, proc in procs:
                if proc.poll() is not None:
                    print(f"\nERROR: {name} exited unexpectedly (code {proc.returncode}).")
                    raise SystemExit(1)

    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        for name, proc in procs:
            if proc.poll() is None:
                print(f"  Stopping {name}...")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        print("  Done.")


if __name__ == "__main__":
    main()