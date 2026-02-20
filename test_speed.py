"""
test_speed.py — Measure your LLM's generation speed.

Run this after starting the LLM server to get an accurate
TOKENS_PER_SECOND value to set in config.py.

Usage:
    python test_speed.py
"""

import json
import time
import urllib.request
import urllib.error
from config import LLM_URL, LLM_PORT

TEST_PROMPT = (
    "Write a detailed technical explanation of how a CPU processes instructions, "
    "covering fetch, decode, execute, and writeback stages. Be thorough."
)
MAX_TOKENS = 300
RUNS       = 3  # average over multiple runs for accuracy


def check_server() -> bool:
    try:
        with urllib.request.urlopen(f"http://localhost:{LLM_PORT}/health", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def run_once() -> tuple[float, int]:
    """Returns (elapsed_seconds, word_count)."""
    payload = json.dumps({
        "model": "local",
        "messages": [{"role": "user", "content": TEST_PROMPT}],
        "max_tokens": MAX_TOKENS,
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

    text = data["choices"][0]["message"]["content"]
    words = len(text.split())
    return elapsed, words, text


def main() -> None:
    print("\nLLM Speed Test")
    print("=" * 40)

    if not check_server():
        print(f"ERROR: LLM server not responding at port {LLM_PORT}.")
        print("  Start the server first with: ./run.sh --skip-ingest")
        return

    print(f"Prompt    : {TEST_PROMPT[:60]}...")
    print(f"Max tokens: {MAX_TOKENS}")
    print(f"Runs      : {RUNS}")
    print()

    results = []
    for i in range(1, RUNS + 1):
        print(f"Run {i}/{RUNS}... ", end="", flush=True)
        elapsed, words, text = run_once()
        tok_s = words / elapsed
        results.append(tok_s)
        print(f"{words} words in {elapsed:.1f}s = {tok_s:.1f} tok/s")

    avg = sum(results) / len(results)

    print()
    print("=" * 40)
    print(f"  Average  : {avg:.1f} tok/s")
    print(f"  Min      : {min(results):.1f} tok/s")
    print(f"  Max      : {max(results):.1f} tok/s")
    print()
    print(f"  Set this in config.py:")
    print(f"    TOKENS_PER_SECOND = {avg:.1f}")
    print("=" * 40)
    print()
    print("Last response preview:")
    print("-" * 40)
    print(text[:300] + ("..." if len(text) > 300 else ""))


if __name__ == "__main__":
    main()