"""
benchmark.py — Measures LLM generation speed and updates .env with the result.
Runs as a Docker service after the LLM server is ready.
"""

import json
import os
import re
import time
import urllib.request
from pathlib import Path

LLM_PORT    = int(os.environ.get("LLM_PORT", 8080))
LLM_URL     = os.environ.get("LLM_URL", f"http://rag_llm:{LLM_PORT}/v1/chat/completions")
ENV_FILE    = Path("/app/host_env/.env")

# Use separate marker files for CPU and GPU so switching modes triggers a re-benchmark
_gpu = int(os.environ.get("LLM_GPU_LAYERS", 0)) != 0
MARKER_FILE = Path("/app/host_env/.benchmarked_gpu" if _gpu else "/app/host_env/.benchmarked_cpu")
_mode_label = "GPU" if _gpu else "CPU"

PROMPT = "Give a short summary of uploaded sources."


def wait_for_llm(timeout: int = 600) -> None:
    print("Waiting for LLM server...", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://rag_llm:{LLM_PORT}/health", timeout=2) as r:
                if r.status == 200:
                    print(" ready!")
                    return
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(1)
    print("\nERROR: LLM server did not become ready in time.")
    raise SystemExit(1)


_max_tokens = 500 if _gpu else 50


def measure() -> float:
    payload = json.dumps({
        "model": "local",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": _max_tokens,
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
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.loads(r.read())
    elapsed = time.time() - start
    tokens = (data.get("usage") or {}).get("completion_tokens") or \
             len(data["choices"][0]["message"]["content"].split())
    return round(tokens / elapsed, 1)


def update_env(tok_s: float) -> None:
    if not ENV_FILE.exists():
        print(f"  WARNING: .env not found at {ENV_FILE} — skipping update.")
        return

    text = ENV_FILE.read_text()
    if "TOKENS_PER_SECOND" in text:
        text = re.sub(r"TOKENS_PER_SECOND=[\d.]+", f"TOKENS_PER_SECOND={tok_s}", text)
    else:
        text += f"\nTOKENS_PER_SECOND={tok_s}\n"
    ENV_FILE.write_text(text)
    print(f"  .env updated with TOKENS_PER_SECOND={tok_s}")


def main() -> None:
    print(f"\nLLM Speed Benchmark [{_mode_label}]")
    print("=" * 40)

    if MARKER_FILE.exists():
        cached = MARKER_FILE.read_text().strip()
        print(f"  Already benchmarked ({_mode_label}) — skipping.")
        print(f"  Cached result: {cached} tok/s")
        print(f"  (Delete {MARKER_FILE.name} from project root to re-run)")
        print("=" * 40)
        return

    wait_for_llm()

    print(f"Running inference pass ({_max_tokens} tokens)...", end="", flush=True)
    tok_s = measure()
    print(f" {tok_s} tok/s")

    update_env(tok_s)
    MARKER_FILE.write_text(str(tok_s))

    print("=" * 40)
    print(f"  Result: {tok_s} tok/s  [{_mode_label}]")
    print("=" * 40)


if __name__ == "__main__":
    main()