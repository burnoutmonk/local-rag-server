from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import httpx
import threading

from config import (
    QDRANT_HOST, QDRANT_PORT, COLLECTION,
    EMBED_MODEL_NAME,
    LLM_URL,
    LLM_TEMPERATURE, LLM_TOP_P, LLM_TOP_K, LLM_MIN_P,
    TOKENS_PER_SECOND, RETRIEVAL_OVERHEAD_S,
    MIN_TOKENS, MAX_TOKENS,
    MAX_CONTEXT_CHARS,
    API_HOST, API_PORT,
)

LLM_LOCK = threading.Lock()

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


class QueryIn(BaseModel):
    query: str
    top_k: int = 5
    mode: str = "answer"   # "answer" or "search"
    timeout: int | None = 30


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/search")
def search(q: QueryIn):
    vec = embedder.encode([q.query], normalize_embeddings=True)[0].tolist()
    res = client.query_points(
        collection_name=COLLECTION,
        query=vec,
        limit=q.top_k,
        with_payload=True,
        with_vectors=False,
    )
    results = []
    for p in res.points:
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
    # 1) Retrieve
    vec = embedder.encode([q.query], normalize_embeddings=True)[0].tolist()
    res = client.query_points(
        collection_name=COLLECTION,
        query=vec,
        limit=q.top_k,
        with_payload=True,
        with_vectors=False,
    )

    if not res.points:
        return {"query": q.query, "answer": "No relevant sources found.", "citations": []}

    # 2) Build citations
    citations = []
    for i, pt in enumerate(res.points, 1):
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
        for i, pt in enumerate(res.points, 1):
            p = pt.payload or {}
            text = (p.get("text") or "").strip()
            passages.append(f"[S{i}] {p.get('source_file','?')} -- {p.get('section','?')}\n{text}")
        return {"query": q.query, "answer": "\n\n".join(passages), "citations": citations}

    # 4) ANSWER mode -- pass to LLM
    context = build_context(res.points)

    system = (
        "You are a helpful assistant. Answer using the provided SOURCES. "
        "Cite sources inline using [S1], [S2], etc. "
        "If the sources do not contain the answer, say so but comment as best you can."
    )
    user = f"QUESTION:\n{q.query}\n\nSOURCES:\n{context}"

    payload = {
        "model": "local",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": LLM_TEMPERATURE,
        "top_p": LLM_TOP_P,
        "top_k": LLM_TOP_K,
        "min_p": LLM_MIN_P,
        "max_tokens": estimate_max_tokens(q.timeout, q.top_k),
    }

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

    text = data["choices"][0]["message"]["content"]
    return {"query": q.query, "answer": text, "citations": citations}