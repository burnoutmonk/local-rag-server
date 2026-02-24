# v2.4 Plan — Observability, Better Testing, RAGAS

---

## 1. Better Question Generation in `rag_test.py`

**Problem:** The local 3B model generates vague, decontextualized, or trivial questions.
Examples of bad output: "What is the temperature deep in the layer?", "up to the dispersal by photoevaporation".
These produce unfairly low answer scores and don't test the RAG system meaningfully.

**Root cause:** Small local model doesn't have the instruction-following quality needed
for reliable benchmark question generation.

**Options (pick one):**

### Option A — Better system prompt with few-shot examples
Add 2–3 concrete examples of good vs bad questions directly in the prompt.
Show the model exactly what "specific, answerable, self-contained" means.
Cost: zero. Quality improvement: moderate.

### Option B — Use Claude API for question generation only
The test script could call `anthropic.messages.create()` if an `ANTHROPIC_API_KEY`
is set in `.env`. Falls back to local LLM if key not present.
Cost: ~$0.01–0.05 per test run (Haiku pricing). Quality improvement: dramatic.
Implementation: add optional `ANTHROPIC_API_KEY` to `.env.example`,
update `generate_questions()` to try Claude Haiku first.

### Option C — Pre-filter + regenerate loop
After generating, score each question with the local LLM ("Is this question specific
and answerable? yes/no") and discard/regenerate poor ones.
Cost: 2x LLM calls per question. Quality improvement: moderate.

**Recommendation:** Option B with Option A as fallback when no API key present.

---

## 2. Better Embedders and Rerankers

**Problem:** 78% retrieval (Hybrid), 85% (Hybrid+Rerank) and ~50% answer accuracy.
Retrieval ceiling is blocking answer quality — wrong chunks = wrong answers.

**Current:**
- Embedder: `sentence-transformers/all-MiniLM-L6-v2` (384-dim, 22M params, fast)
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` (fast but basic)

**Candidates to test:**

| Model | Dims | Size | Expected retrieval gain |
|---|---|---|---|
| `BAAI/bge-small-en-v1.5` | 384 | 33M | +3–5% over MiniLM |
| `BAAI/bge-base-en-v1.5` | 768 | 109M | +6–10% — best quality/speed tradeoff |
| `intfloat/e5-small-v2` | 384 | 33M | Similar to bge-small |
| `BAAI/bge-reranker-base` | — | 278M | Better reranker than ms-marco-MiniLM |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | — | larger | Drop-in upgrade to current reranker |

**Implementation:**
- `EMBED_MODEL_NAME` and `RERANKER_MODEL` already configurable via `.env`
- Changing embedder requires full re-ingest (`docker volume rm local_rag_ingest_state`)
- Add to `rag_test.py`: run test before and after model swap, compare scores
- Add model benchmark table to README

---

## 3. Chunk-Level Retrieval Accuracy in `rag_test.py`

**Problem:** Currently checking `source_file in cited_files` — coarse.
A document with 200 chunks could have 199 wrong chunks from the right file.

**What to store at question generation time:**
```python
qa_pairs.append({
    "question": ...,
    "answer": ...,
    "source_file": ...,
    "source_chunk": chunk_text[:500],      # already stored
    "chunk_index": payload.get("chunk_index"),   # ADD THIS
    "section": payload.get("section"),           # ADD THIS
})
```

**Scoring change in Phase 3:**
```python
# Current (file-level):
retrieved_correct_source = source_file in cited_files

# New (chunk-level):
cited_chunks = [(c.get("source_file"), c.get("chunk_index")) for c in citations]
chunk_hit = (source_file, chunk_index) in cited_chunks
```

**Result:** Two separate retrieval scores per mode — file-level % and chunk-level %.
Chunk-level will be lower and more meaningful (e.g. 85% file → maybe 60% chunk).

**Note:** Requires that the `/answer` endpoint returns `chunk_index` in citations,
which it already does via the Qdrant payload.

---

## 4. RAGAS Integration

**What RAGAS is:** Open-source framework for RAG evaluation.
Metrics: faithfulness, answer relevancy, context precision, context recall.
Uses LLM-as-judge internally — can work with local models via a custom LLM wrapper.

**The problem:** RAGAS was built for OpenAI/HuggingFace pipeline APIs.
Running it with a llama.cpp server requires a custom `LangchainLLM` wrapper.

**Integration plan:**

### Step 1 — Install RAGAS
```
ragas
langchain
langchain-community
```

### Step 2 — Adapter for local LLM
```python
from langchain_community.llms import LlamaCpp  # or custom HTTP wrapper
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

ragas_llm = LangchainLLMWrapper(your_local_llm)
ragas_embeddings = LangchainEmbeddingsWrapper(your_embedder)
```

### Step 3 — Wrap existing test results
RAGAS takes a `Dataset` with columns: `question`, `answer`, `contexts`, `ground_truth`.
Our `rag_test.py` already collects all of these — just needs formatting.

### Step 4 — Metrics to run
- `faithfulness` — does the answer stick to retrieved chunks?
- `answer_relevancy` — is the answer on-topic?
- `context_precision` — are the retrieved chunks relevant to the question?
- `context_recall` — did retrieval find all necessary context?

**Recommendation:** Add RAGAS as an optional `--ragas` flag to `rag_test.py`.
Adds ~3–5 minutes to test run. Requires `ANTHROPIC_API_KEY` or tolerates slow local eval.

**Caveat:** RAGAS with a 3B local model will give noisy scores. Use Claude Haiku as the
judge LLM for RAGAS if possible (same `ANTHROPIC_API_KEY` as item 1).

---

## 5. Request Tracing ("Where did it go?" view)

**Goal:** For any bad answer, see: embedding time, exact Qdrant chunks returned,
final prompt sent to llama.cpp, generation time.

**Tooling options:**

| Tool | Self-hosted | Effort | Notes |
|---|---|---|---|
| **Arize Phoenix** | Yes (Docker) | Low | Best UI, native LLM tracing, OpenTelemetry |
| **Langfuse** | Yes (Docker) | Medium | More features, REST API for custom logging |
| **OpenTelemetry + Jaeger** | Yes (Docker) | High | Generic, requires custom spans |

**Recommendation: Arize Phoenix**
- Single Docker container: `docker run -p 6006:6006 arizephoenix/phoenix`
- Python SDK: `pip install arize-phoenix-otel opentelemetry-sdk`
- Instruments FastAPI + custom spans with minimal code changes

**What to instrument in `rag_api.py`:**
```python
from opentelemetry import trace
tracer = trace.get_tracer("rag_api")

@app.post("/answer")
def answer(q: QueryIn):
    with tracer.start_as_current_span("rag_answer") as span:
        span.set_attribute("query", q.query)
        span.set_attribute("top_k", q.top_k)

        with tracer.start_as_current_span("embedding"):
            # embed query

        with tracer.start_as_current_span("qdrant_retrieval"):
            points = hybrid_retrieve(...)
            span.set_attribute("chunks_retrieved", len(points))

        with tracer.start_as_current_span("llm_generation"):
            # call llama.cpp
            span.set_attribute("prompt_tokens", ...)
            span.set_attribute("completion_tokens", ...)
```

**Docker service to add:**
```yaml
phoenix:
  image: arizephoenix/phoenix:latest
  ports: ["6006:6006"]
  volumes: [phoenix_data:/phoenix]
  restart: unless-stopped
```

Access at `http://<server-ip>:6006` — timeline view of every request.

---

## 6. Metrics Dashboard (Grafana + Prometheus)

**Goal:** Time-series view of tok/s, context window usage, retrieval latency,
Qdrant collection size over time. Alerts when GPU degrades or Qdrant slows down.

**Current state:** `rag_api.py` already tracks `metrics_history` (last 100 queries)
in memory via `/metrics` endpoint. This is ephemeral — lost on restart.

**Target stack:**
```
rag_api → Prometheus metrics endpoint (/metrics/prometheus)
              ↓
          Prometheus (scrapes every 15s, stores time-series)
              ↓
          Grafana (dashboard, alerting)
```

**Implementation:**

### Step 1 — Add Prometheus metrics to `rag_api.py`
```python
from prometheus_client import Histogram, Gauge, Counter, make_asgi_app

REQUEST_LATENCY   = Histogram("rag_request_latency_seconds", "End-to-end request time")
TOKENS_PER_SEC    = Gauge("rag_tokens_per_second", "Last measured tok/s")
PROMPT_TOKENS     = Counter("rag_prompt_tokens_total", "Total prompt tokens consumed")
COMPLETION_TOKENS = Counter("rag_completion_tokens_total", "Total completion tokens")
RETRIEVAL_LATENCY = Histogram("rag_retrieval_latency_seconds", "Qdrant query time")
CONTEXT_USAGE     = Histogram("rag_context_chars", "Characters sent to LLM")
```

Expose via `/metrics/prometheus` using `prometheus_client.make_asgi_app()`.

### Step 2 — Docker services
```yaml
prometheus:
  image: prom/prometheus
  volumes:
    - ./docker/prometheus.yml:/etc/prometheus/prometheus.yml
    - prometheus_data:/prometheus
  ports: ["9090:9090"]

grafana:
  image: grafana/grafana
  volumes: [grafana_data:/var/lib/grafana]
  ports: ["3000:3000"]
  environment:
    - GF_AUTH_ANONYMOUS_ENABLED=true
    - GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer
```

### Step 3 — Prometheus config
```yaml
# docker/prometheus.yml
scrape_configs:
  - job_name: 'rag_api'
    scrape_interval: 15s
    static_configs:
      - targets: ['rag_api:8000']
    metrics_path: '/metrics/prometheus'
```

### Step 4 — Pre-built Grafana dashboard
Import dashboard JSON (to be created) with panels:
- **tok/s over time** — is GPU performance degrading?
- **Request latency histogram** — p50/p95/p99
- **Context window usage** — are you hitting the 4096 limit?
- **Retrieval latency** — is Qdrant slowing as collection grows?
- **Token consumption rate** — usage monitoring for multi-user deployments

**Grafana access:** `http://<server-ip>:3000` (anonymous viewer access, no login required
for read-only — matches the no-auth philosophy of this project).

**Optional:** Add Prometheus + Grafana behind a profile flag so they don't start by default:
```yaml
prometheus:
  profiles: ["monitoring"]
grafana:
  profiles: ["monitoring"]
```
Start with: `docker compose --profile monitoring up -d`

---

## Implementation Order

| Priority | Item | Effort | Impact |
|---|---|---|---|
| 1 | **Chunk-level retrieval scoring** (#3) | Small | Accurate baseline measurement |
| 2 | **Better question generation** (#1, Option A) | Small | Better test signal immediately |
| 3 | **Better question generation** (#1, Option B) | Medium | Best test quality |
| 4 | **Arize Phoenix tracing** (#5) | Medium | Diagnose bad answers |
| 5 | **RAGAS** (#4) | Medium | Industry-standard eval metrics |
| 6 | **Prometheus + Grafana** (#6) | Large | Production monitoring |
| 7 | **Embedder/reranker swap** (#2) | Large | Requires re-ingest + re-test |
