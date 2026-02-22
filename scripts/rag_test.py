"""
RAG Accuracy Test — evaluates retrieval + LLM quality on actual documents.

Generates questions from random chunks, answers them via RAG, and scores both
retrieval and answer accuracy. Reports results per search mode (Dense/Hybrid/Rerank).

Usage:
    python scripts/rag_test.py [--questions 20] [--top-k 5] [--api-url http://localhost:8000]
"""

import argparse
import json
import random
import time
import urllib.request
from pathlib import Path
from typing import Any


# ── Config ────────────────────────────────────────────────────────────────────
import os

DEFAULT_API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000")
DEFAULT_LLM_URL = os.environ.get("LLM_URL", "http://rag_llm:8080/v1/chat/completions")
DEFAULT_NUM_QUESTIONS = 20
DEFAULT_TOP_K = 5
DEFAULT_OUTPUT = "test_results.json"

# Qdrant settings (same as config.py, but use test collection if available)
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
COLLECTION = os.environ.get("QDRANT_COLLECTION", "rag_docs")


# ── HTTP Helpers ──────────────────────────────────────────────────────────────
def http_post(url: str, data: dict, timeout: int = 60) -> dict:
    """POST JSON to URL, return parsed response."""
    payload = json.dumps(data).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as e:
        print(f"ERROR: HTTP request failed: {e}")
        raise


# ── Phase 1: Question Generation ───────────────────────────────────────────────
def generate_questions(num_questions: int) -> list[dict]:
    """
    Sample random chunks from Qdrant and generate QA pairs using the LLM.

    Returns:
        List of {"question": str, "answer": str, "source_file": str, "source_chunk": str}
    """
    print(f"\n[Phase 1] Generating {num_questions} questions from Qdrant chunks...")

    from qdrant_client import QdrantClient

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Get total collection size
    collection_info = client.get_collection(COLLECTION)
    total_points = collection_info.points_count or 0

    if total_points == 0:
        raise SystemExit("ERROR: Qdrant collection is empty. Ingest documents first.")

    print(f"  Collection has {total_points} points. Sampling {num_questions} random chunks...")

    # Scroll to get actual points (IDs are UUIDs, not sequential integers)
    # Oversample then randomly pick to get a representative spread
    oversample = min(num_questions * 5, total_points)
    print(f"  Scrolling {oversample} points from collection...")
    all_points, _ = client.scroll(
        collection_name=COLLECTION,
        limit=oversample,
        with_payload=True,
        with_vectors=False,
    )
    print(f"  Scroll returned {len(all_points)} points.")

    if not all_points:
        raise SystemExit("ERROR: Scroll returned 0 points despite non-empty collection.")

    sampled_points = random.sample(all_points, min(num_questions, len(all_points)))
    print(f"  Sampled {len(sampled_points)} points for question generation.")
    print(f"  LLM URL: {DEFAULT_LLM_URL}")

    qa_pairs = []
    t0 = time.time()

    for i, point in enumerate(sampled_points, 1):
        payload = point.payload or {}
        chunk_text = payload.get("text", "")
        source_file = payload.get("source_file", "unknown")

        if not chunk_text:
            print(f"  [{i}/{len(sampled_points)}] Skipping point with empty text (source: {source_file})")
            continue

        # Ask LLM to generate a question
        prompt = f"""Given this text, generate ONE factual question that can be answered ONLY from this text.
Also provide the correct answer based on the text.

Text:
{chunk_text}

Respond ONLY with valid JSON (no markdown, no extra text):
{{"question": "...", "answer": "..."}}"""

        payload_llm = {
            "model": "local",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
            "temperature": 0.5,
            "stream": False,
        }

        try:
            response = http_post(DEFAULT_LLM_URL, payload_llm, timeout=60)
            content = response["choices"][0]["message"]["content"].strip()

            # Parse JSON response — handle markdown code fences
            if "```" in content:
                content = content.split("```")[1].lstrip("json").strip()

            qa = json.loads(content)
            qa_pairs.append({
                "question": qa["question"],
                "answer": qa["answer"],
                "source_file": source_file,
                "source_chunk": chunk_text[:500],
            })
            print(f"  [{i}/{len(sampled_points)}] Generated question from {source_file}")
        except Exception as e:
            print(f"  [{i}/{len(sampled_points)}] FAILED to generate question: {e}")
            continue

    elapsed = time.time() - t0
    print(f"  Generated {len(qa_pairs)} QA pairs in {elapsed:.1f}s")
    return qa_pairs


# ── Phase 2: RAG Answering ────────────────────────────────────────────────────
def test_retrieval_and_answer(questions: list[dict], api_url: str, top_k: int) -> list[dict]:
    """
    For each question, retrieve answers via all 3 search modes.

    Returns:
        List of test results with query, answers, timings per mode.
    """
    print(f"\n[Phase 2] Testing retrieval + answering ({len(questions)} questions, {len(['Dense', 'Hybrid', 'Hybrid+Rerank'])} modes)...")

    results = []
    modes = [
        ("Dense", {"use_bm25": False, "use_reranker": False}),
        ("Hybrid", {"use_bm25": True, "use_reranker": False}),
        ("Hybrid+Rerank", {"use_bm25": True, "use_reranker": True}),
    ]

    for q_idx, qa in enumerate(questions, 1):
        question = qa["question"]
        ground_truth_answer = qa["answer"]
        source_file = qa["source_file"]

        result = {
            "question": question,
            "ground_truth": ground_truth_answer,
            "expected_source_file": source_file,
            "modes": {},
        }

        for mode_name, search_flags in modes:
            t0 = time.time()
            try:
                answer_payload = {
                    "query": question,
                    "top_k": top_k,
                    "mode": "answer",
                    "timeout": 60,
                    **search_flags,
                }
                response = http_post(f"{api_url}/answer", answer_payload, timeout=90)
                elapsed = time.time() - t0

                answer_text = response.get("answer", "")
                citations = response.get("citations", [])
                usage = response.get("usage", {})

                # Extract source files from citations
                cited_files = set(c.get("source_file") for c in citations if c.get("source_file"))

                result["modes"][mode_name] = {
                    "answer": answer_text[:300],  # Store first 300 chars
                    "citations": [c.get("source_file") for c in citations],
                    "response_time_s": round(elapsed, 2),
                    "usage": {
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                    },
                    "retrieved_correct_source": source_file in cited_files,
                }
            except Exception as e:
                print(f"    ERROR during {mode_name} for question {q_idx}: {e}")
                result["modes"][mode_name] = {
                    "error": str(e),
                    "response_time_s": 0,
                }

        results.append(result)
        print(f"  [{q_idx}/{len(questions)}] Tested: {question[:60]}...")

    return results


# ── Phase 3: Scoring ──────────────────────────────────────────────────────────
def score_answers(results: list[dict]) -> dict:
    """
    Evaluate retrieval and answer accuracy using LLM as judge.

    Returns:
        Aggregated scores per search mode.
    """
    print(f"\n[Phase 3] Scoring answers ({len(results)} questions)...")

    scores = {
        "Dense": {"retrieval": [], "answer": [], "times": []},
        "Hybrid": {"retrieval": [], "answer": [], "times": []},
        "Hybrid+Rerank": {"retrieval": [], "answer": [], "times": []},
    }

    for q_idx, result in enumerate(results, 1):
        question = result["question"]
        ground_truth = result["ground_truth"]
        expected_source = result["expected_source_file"]

        for mode_name, mode_result in result["modes"].items():
            if "error" in mode_result:
                continue

            # 1) Retrieval accuracy (simple: did it find the right source file?)
            retrieval_correct = mode_result.get("retrieved_correct_source", False)
            scores[mode_name]["retrieval"].append(1.0 if retrieval_correct else 0.0)

            # 2) Answer accuracy (LLM as judge)
            answer_text = mode_result.get("answer", "")
            if answer_text:
                judge_prompt = f"""Given a question, ground-truth answer, and RAG-generated answer,
rate if the RAG answer is correct and helpful.

Question: {question}
Ground truth: {ground_truth}
RAG answer: {answer_text}

Reply with ONLY valid JSON (no markdown):
{{"correct": true/false, "reason": "brief explanation"}}"""

                try:
                    judge_payload = {
                        "model": "local",
                        "messages": [{"role": "user", "content": judge_prompt}],
                        "max_tokens": 100,
                        "temperature": 0.3,
                        "stream": False,
                    }
                    judge_response = http_post(DEFAULT_LLM_URL, judge_payload, timeout=30)
                    judge_content = judge_response["choices"][0]["message"]["content"].strip()

                    if judge_content.startswith("```"):
                        judge_content = judge_content.split("```")[1].lstrip("json").strip()

                    judge_result = json.loads(judge_content)
                    scores[mode_name]["answer"].append(1.0 if judge_result.get("correct", False) else 0.0)
                except Exception as e:
                    print(f"    WARNING: judge failed for question {q_idx} mode {mode_name}: {e}")
                    scores[mode_name]["answer"].append(0.0)
            else:
                scores[mode_name]["answer"].append(0.0)

            # 3) Response time
            response_time = mode_result.get("response_time_s", 0)
            scores[mode_name]["times"].append(response_time)

        print(f"  [{q_idx}/{len(results)}] Scored question {q_idx}")

    return scores


# ── Phase 4: Report ───────────────────────────────────────────────────────────
def print_report(results: list[dict], scores: dict, elapsed_total: float) -> None:
    """Print formatted report to stdout."""
    print("\n" + "=" * 70)
    print("RAG Accuracy Test Report")
    print("=" * 70)
    print(f"Questions tested:    {len(results)}")

    # Count unique source files
    source_files = set(r["expected_source_file"] for r in results)
    print(f"Unique source files: {len(source_files)}")
    print()

    # Results table
    print(f"{'Search Mode':<20} {'Retrieval %':<15} {'Answer %':<15} {'Avg Time (s)':<15}")
    print("-" * 70)

    for mode_name in ["Dense", "Hybrid", "Hybrid+Rerank"]:
        mode_scores = scores[mode_name]

        retrieval_pct = (
            100.0 * sum(mode_scores["retrieval"]) / len(mode_scores["retrieval"])
            if mode_scores["retrieval"]
            else 0.0
        )
        answer_pct = (
            100.0 * sum(mode_scores["answer"]) / len(mode_scores["answer"])
            if mode_scores["answer"]
            else 0.0
        )
        avg_time = (
            sum(mode_scores["times"]) / len(mode_scores["times"])
            if mode_scores["times"]
            else 0.0
        )

        print(f"{mode_name:<20} {retrieval_pct:>6.1f}%       {answer_pct:>6.1f}%       {avg_time:>8.2f}")

    print("=" * 70)
    print(f"\nTotal test time: {elapsed_total:.1f}s")
    print()


def main():
    parser = argparse.ArgumentParser(description="RAG Accuracy Test")
    parser.add_argument("--questions", type=int, default=DEFAULT_NUM_QUESTIONS, help="Number of questions to generate")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Top-k for retrieval")
    parser.add_argument("--api-url", type=str, default=DEFAULT_API_URL, help="API base URL")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Output JSON file")
    args = parser.parse_args()

    t_start = time.time()

    try:
        # Phase 1
        questions = generate_questions(args.questions)
        if not questions:
            raise SystemExit("No questions generated. Check document ingestion.")

        # Phase 2
        results = test_retrieval_and_answer(questions, args.api_url, args.top_k)

        # Phase 3
        scores = score_answers(results)

        # Phase 4
        elapsed_total = time.time() - t_start
        print_report(results, scores, elapsed_total)

        # Save results
        output_data = {
            "timestamp": time.time(),
            "config": {
                "num_questions": len(questions),
                "top_k": args.top_k,
                "api_url": args.api_url,
            },
            "results": results,
            "scores": {k: {sk: list(sv) for sk, sv in v.items()} for k, v in scores.items()},
            "elapsed_seconds": round(elapsed_total, 1),
        }
        output_path = Path(args.output)
        output_path.write_text(json.dumps(output_data, indent=2))
        print(f"Full results saved to: {output_path.absolute()}\n")

    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
