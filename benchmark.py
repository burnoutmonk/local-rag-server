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

LLM_PORT = int(os.environ.get("LLM_PORT", 8080))
LLM_URL  = os.environ.get("LLM_URL", f"http://rag_llm:{LLM_PORT}/v1/chat/completions")
ENV_FILE = Path("/app/host_env/.env")

PROMPT = "Write a detailed technical explanation of how a CPU processes instructions."


def wait_for_llm(timeout: int = 300) -> None:
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
        time.sleep(2)
    print("\nERROR: LLM server did not become ready in time.")
    raise SystemExit(1)


def measure() -> float:
    payload = json.dumps({
        "model": "local",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 150,
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
    words = len(data["choices"][0]["message"]["content"].split())
    return round(words / elapsed, 1)


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
    print("\nLLM Speed Benchmark")
    print("=" * 40)

    wait_for_llm()

    print("Running inference pass...", end="", flush=True)
    tok_s = measure()
    print(f" {tok_s} tok/s")

    update_env(tok_s)

    print("=" * 40)
    print(f"  Result: {tok_s} tok/s")
    print("=" * 40)


if __name__ == "__main__":
    main()
