from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import httpx
import psutil
import subprocess
import threading
import time

psutil.cpu_percent()  # prime — discard first meaningless reading

from config import (
    QDRANT_HOST, QDRANT_PORT, COLLECTION,
    EMBED_MODEL_NAME,
    LLM_URL, LLM_MODEL_FILE, LLM_GPU_LAYERS,
    LLM_CONTEXT, LLM_THREADS,
    LLM_TEMPERATURE, LLM_TOP_P, LLM_TOP_K, LLM_MIN_P,
    TOKENS_PER_SECOND, RETRIEVAL_OVERHEAD_S,
    MIN_TOKENS, MAX_TOKENS,
    MAX_CONTEXT_CHARS, MAX_CHARS, OVERLAP_CHARS,
    API_HOST, API_PORT,
    CHAT_MEMORY_TURNS,
    RERANKER_MODEL, RERANKER_ENABLED, BM25_WEIGHT, RETRIEVAL_MULTIPLIER,
)

LLM_LOCK = threading.Lock()

# ── Analytics store ───────────────────────────────────────────────────────────
from collections import deque
metrics_history: deque = deque(maxlen=100)  # keep last 100 queries

# ── Session memory store ───────────────────────────────────────────────────────
# key: session_id (str), value: {"messages": deque[dict], "last_seen": float}
SESSION_TTL_S = 7200  # 2 hours
sessions: dict[str, dict] = {}

# ── Lazy reranker ─────────────────────────────────────────────────────────────
_reranker = None
_reranker_lock = threading.Lock()


def get_reranker():
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                from sentence_transformers import CrossEncoder
                _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


app = FastAPI(title="Local RAG API")
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
embedder = SentenceTransformer(EMBED_MODEL_NAME)
templates = Jinja2Templates(directory="templates")


def estimate_max_tokens(timeout_s: int | None, top_k: int) -> int:
    if timeout_s is None:
        return MAX_TOKENS
    prefill_penalty = (top_k - 1) * 0.3
    available_s = timeout_s - RETRIEVAL_OVERHEAD_S - prefill_penalty
    tokens = int(available_s * TOKENS_PER_SECOND)
    return max(MIN_TOKENS, min(tokens, MAX_TOKENS))


def build_context(points, max_total_chars: int = MAX_CONTEXT_CHARS) -> str:
    MAX_CHUNK_CHARS = 1000
    blocks = []
    total = 0
    for i, pt in enumerate(points, 1):
        p = pt.payload or {}
        cite = f"[S{i}] {p.get('source_file','?')} -- {p.get('section','?')} (chunk {p.get('chunk_index','?')})"
        text = (p.get("text") or "").strip()
        if len(text) > MAX_CHUNK_CHARS:
            text = text[:MAX_CHUNK_CHARS].rsplit(" ", 1)[0] + "..."
        block = f"{cite}\n{text}"
        if total + len(block) > max_total_chars:
            break
        blocks.append(block)
        total += len(block)
    return "\n\n".join(blocks)


def hybrid_retrieve(query: str, top_k: int, use_bm25: bool, use_reranker: bool):
    """Retrieve top_k points using dense search, optionally fused with BM25 and reranked."""
    candidate_limit = top_k * RETRIEVAL_MULTIPLIER if (use_bm25 or use_reranker) else top_k

    vec = embedder.encode([query], normalize_embeddings=True)[0].tolist()
    res = client.query_points(
        collection_name=COLLECTION,
        query=vec,
        limit=candidate_limit,
        with_payload=True,
        with_vectors=False,
    )
    points = res.points

    if not points:
        return points

    if use_bm25:
        from rank_bm25 import BM25Okapi
        corpus = [(p.payload or {}).get("text", "") for p in points]
        tokenized = [doc.lower().split() for doc in corpus]
        bm25 = BM25Okapi(tokenized)
        bm25_scores = bm25.get_scores(query.lower().split())

        # Normalise dense scores to [0,1]
        dense_scores = [p.score for p in points]
        d_min, d_max = min(dense_scores), max(dense_scores)
        d_range = d_max - d_min or 1.0
        norm_dense = [(s - d_min) / d_range for s in dense_scores]

        # Normalise BM25 scores to [0,1]
        b_min, b_max = min(bm25_scores), max(bm25_scores)
        b_range = b_max - b_min or 1.0
        norm_bm25 = [(s - b_min) / b_range for s in bm25_scores]

        dense_w = 1.0 - BM25_WEIGHT
        fused = [dense_w * d + BM25_WEIGHT * b for d, b in zip(norm_dense, norm_bm25)]
        points = [p for _, p in sorted(zip(fused, points), key=lambda x: x[0], reverse=True)]

    if use_reranker and RERANKER_ENABLED:
        reranker = get_reranker()
        texts = [(p.payload or {}).get("text", "") for p in points]
        pairs = [[query, t] for t in texts]
        scores = reranker.predict(pairs)
        points = [p for _, p in sorted(zip(scores, points), key=lambda x: x[0], reverse=True)]

    return points[:top_k]


def _cleanup_sessions() -> None:
    now = time.time()
    expired = [sid for sid, s in sessions.items() if now - s["last_seen"] > SESSION_TTL_S]
    for sid in expired:
        del sessions[sid]


class QueryIn(BaseModel):
    query: str
    top_k: int = 5
    mode: str = "answer"   # "answer" or "search"
    timeout: int | None = 30
    max_tokens: int | None = None  # None = use config default
    temperature: float | None = None
    top_p: float | None = None
    llm_top_k: int | None = None
    min_p: float | None = None
    session_id: str | None = None
    use_bm25: bool = True
    use_reranker: bool = False


class CompleteIn(BaseModel):
    messages: list[dict]
    max_tokens: int = 200
    temperature: float = 0.5


@app.post("/complete")
def complete(req: CompleteIn):
    """Direct LLM completion through the LLM_LOCK — for internal tooling (e.g. rag_test)."""
    payload = {
        "model": "local",
        "messages": req.messages,
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "stream": False,
    }
    if not LLM_LOCK.acquire(blocking=True, timeout=300):
        raise HTTPException(status_code=503, detail="LLM lock timeout")
    try:
        with httpx.Client(timeout=httpx.Timeout(300, connect=10.0)) as h:
            r = h.post(LLM_URL, json=payload)
            r.raise_for_status()
            return r.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LLM error: {repr(e)}")
    finally:
        LLM_LOCK.release()


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/metrics")
def metrics():
    return {"history": list(metrics_history)}


def _read_tok_s() -> float:
    """Read the actual benchmark result from the marker file, falling back to config."""
    from pathlib import Path
    marker = Path("/app/host_env/.benchmarked_gpu" if LLM_GPU_LAYERS != 0 else "/app/host_env/.benchmarked_cpu")
    try:
        val = float(marker.read_text().strip())
        if val > 0:
            return val
    except Exception:
        pass
    return TOKENS_PER_SECOND


@app.get("/health")
def health():
    import os
    gpu_enabled = LLM_GPU_LAYERS != 0
    cuda_available = (
        os.environ.get("CUDA_AVAILABLE", "").lower() == "true"
        or os.path.exists("/usr/local/cuda/lib64/libcudart.so")
        or os.path.exists("/usr/local/cuda/lib64/libcudart.so.12")
    )

    # CPU & RAM
    cpu_percent = psutil.cpu_percent()
    mem = psutil.virtual_memory()

    # GPU stats via nvidia-smi (only available in the CUDA image)
    gpu_stats = None
    if cuda_available:
        try:
            result = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2,
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(",")
                gpu_stats = {
                    "util_percent":    int(parts[0].strip()),
                    "memory_used_mb":  int(parts[1].strip()),
                    "memory_total_mb": int(parts[2].strip()),
                }
        except Exception:
            pass

    return {
        "ok": True,
        "gpu_enabled": gpu_enabled,
        "cuda_available": cuda_available,
        "gpu_layers": LLM_GPU_LAYERS,
        "model": LLM_MODEL_FILE,
        "embed_model": EMBED_MODEL_NAME,
        "tokens_per_second": _read_tok_s(),
        "cpu_percent": cpu_percent,
        "ram_used_gb": round(mem.used / 1024 ** 3, 1),
        "ram_total_gb": round(mem.total / 1024 ** 3, 1),
        "ram_percent": round(mem.percent, 1),
        "gpu": gpu_stats,
        # Server admin config
        "llm_context": LLM_CONTEXT,
        "llm_threads": LLM_THREADS,
        "max_tokens_config": MAX_TOKENS,
        "chunk_size": MAX_CHARS,
        "overlap": OVERLAP_CHARS,
        "collection": COLLECTION,
        "reranker_available": RERANKER_ENABLED,
        "reranker_model": RERANKER_MODEL if RERANKER_ENABLED else None,
        "min_tokens_config": MIN_TOKENS,
        "bm25_weight": BM25_WEIGHT,
        "retrieval_multiplier": RETRIEVAL_MULTIPLIER,
        "chat_memory_turns": CHAT_MEMORY_TURNS,
    }


@app.post("/search")
def search(q: QueryIn):
    points = hybrid_retrieve(q.query, q.top_k, q.use_bm25, q.use_reranker)
    results = []
    for p in points:
        payload = p.payload or {}
        results.append({
            "score": p.score,
            "source_file": payload.get("source_file"),
            "section": payload.get("section"),
            "doc_id": payload.get("doc_id"),
            "chunk_index": payload.get("chunk_index"),
            "text": payload.get("text"),
        })
    return {"query": q.query, "results": results}


@app.post("/answer")
def answer(q: QueryIn):
    _cleanup_sessions()

    # 1) Retrieve
    points = hybrid_retrieve(q.query, q.top_k, q.use_bm25, q.use_reranker)

    if not points:
        return {"query": q.query, "answer": "No relevant sources found.", "citations": []}

    # 2) Build citations
    citations = []
    for i, pt in enumerate(points, 1):
        p = pt.payload or {}
        citations.append({
            "tag": f"S{i}",
            "source_file": p.get("source_file"),
            "section": p.get("section"),
            "chunk_index": p.get("chunk_index"),
        })

    # 3) SEARCH mode -- skip LLM, return raw passages
    if q.mode == "search":
        passages = []
        for i, pt in enumerate(points, 1):
            p = pt.payload or {}
            text = (p.get("text") or "").strip()
            passages.append(f"[S{i}] {p.get('source_file','?')} -- {p.get('section','?')}\n{text}")
        return {"query": q.query, "answer": "\n\n".join(passages), "citations": citations}

    # 4) ANSWER mode -- pass to LLM
    context = build_context(points)

    system = (
        "You are a helpful assistant. Answer using the provided SOURCES. "
        "Cite sources inline using [S1], [S2], etc. "
        "If the sources do not contain the answer, say so but comment as best you can."
    )
    user = f"QUESTION:\n{q.query}\n\nSOURCES:\n{context}"

    # 5) Build message list, prepending session history if available
    history_messages: list[dict] = []
    if q.session_id and CHAT_MEMORY_TURNS > 0:
        if q.session_id not in sessions:
            sessions[q.session_id] = {"messages": deque(maxlen=CHAT_MEMORY_TURNS * 2), "last_seen": time.time()}
        session = sessions[q.session_id]
        session["last_seen"] = time.time()
        history_messages = list(session["messages"])

    llm_messages = [{"role": "system", "content": system}]
    llm_messages.extend(history_messages)
    llm_messages.append({"role": "user", "content": user})

    payload = {
        "model": "local",
        "messages": llm_messages,
        "temperature": q.temperature if q.temperature is not None else LLM_TEMPERATURE,
        "top_p": q.top_p if q.top_p is not None else LLM_TOP_P,
        "top_k": q.llm_top_k if q.llm_top_k is not None else LLM_TOP_K,
        "min_p": q.min_p if q.min_p is not None else LLM_MIN_P,
        "max_tokens": q.max_tokens if q.max_tokens else estimate_max_tokens(q.timeout, q.top_k),
    }

    _t_start = time.time()
    if not LLM_LOCK.acquire(blocking=False):
        return {"query": q.query, "answer": "Server busy -- please try again shortly.", "citations": []}
    try:
        llm_timeout = httpx.Timeout(q.timeout, connect=10.0) if q.timeout is not None else httpx.Timeout(None, connect=10.0)
        with httpx.Client(timeout=llm_timeout) as h:
            r = h.post(LLM_URL, json=payload)
            r.raise_for_status()
            data = r.json()
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail=f"LLM timed out after {q.timeout}s")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LLM error: {repr(e)}")
    finally:
        LLM_LOCK.release()

    answer_text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    elapsed = time.time() - _t_start
    tps = round(completion_tokens / elapsed, 1) if completion_tokens and elapsed > 0 else TOKENS_PER_SECOND
    metrics_history.append({
        "timestamp": time.time(),
        "query": q.query[:80],
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "tokens_per_second": tps,
        "response_time": round(elapsed, 2),
    })

    # 6) Store exchange in session memory
    if q.session_id and CHAT_MEMORY_TURNS > 0 and q.session_id in sessions:
        session = sessions[q.session_id]
        session["messages"].append({"role": "user", "content": q.query})
        session["messages"].append({"role": "assistant", "content": answer_text})

    return {"query": q.query, "answer": answer_text, "citations": citations, "usage": usage}
