# v2.3 Implementation Plan

## Feature 1: RAG Accuracy Test Script (`scripts/rag_test.py`)

A CLI script that admins run to evaluate retrieval + LLM quality against their actual documents. The LLM generates questions from the ingested documents, then answers them through the full RAG pipeline, and a judge evaluates correctness.

### How it works

**Phase 1 — Question Generation**
- Script connects to Qdrant, samples N random chunks (configurable, default 20)
- For each chunk, calls the LLM (`/v1/chat/completions` directly, same as benchmark.py) with a prompt like:
  > "Given this text, generate 1 factual question that can be answered ONLY from this text. Also provide the correct answer. Respond as JSON: {\"question\": \"...\", \"answer\": \"...\", \"source_chunk\": \"...\"}"
- Stores the generated QA pairs with their source chunk text as ground truth

**Phase 2 — RAG Answering**
- For each generated question, calls `POST /answer` on the running API (http://localhost:8000/answer) with all three search modes:
  - Dense only (`use_bm25=false, use_reranker=false`)
  - Hybrid (`use_bm25=true, use_reranker=false`)
  - Hybrid + Rerank (`use_bm25=true, use_reranker=true`)
- Records: answer text, response time, tok/s, which source files were cited

**Phase 3 — Scoring**
- **Retrieval accuracy**: Did the returned citations include the original source chunk's file? (binary hit/miss per question)
- **Answer accuracy**: Send (question, ground_truth_answer, rag_answer) to the LLM as a judge:
  > "Rate if the RAG answer is correct given the ground truth. Reply JSON: {\"correct\": true/false, \"reason\": \"...\"}"
- Aggregate scores per search mode

**Phase 4 — Report**
- Print a table to stdout:
  ```
  ══════════════════════════════════════════════════════════
  RAG Accuracy Test Report
  ══════════════════════════════════════════════════════════
  Questions generated:  20
  Documents sampled:    8

  Search Mode       Retrieval %   Answer %   Avg Time (s)
  ─────────────────────────────────────────────────────────
  Dense               85.0%        75.0%       1.2
  Hybrid              90.0%        80.0%       1.5
  Hybrid + Rerank     95.0%        90.0%       3.1
  ══════════════════════════════════════════════════════════

  Timing Breakdown:
    Question generation:  45.2s
    Embedding (avg):       0.03s
    LLM (avg):             2.1s
    Total test time:      189.3s
  ```
- Also save full results as `test_results.json` in project root (questions, answers, scores, timings — useful for debugging which questions fail)

### Config / CLI args

```
python scripts/rag_test.py [--questions 20] [--top-k 5] [--api-url http://localhost:8000] [--output test_results.json]
```

No new env vars needed — the script talks to the already-running API and LLM. It uses `urllib.request` (like benchmark.py) to avoid adding deps, or `httpx` since it's already in requirements.

### Docker integration

Add a `rag_test` profile service in docker-compose.yml that can be run on-demand:
```yaml
rag_test:
  profiles: ["test"]
  # ...runs scripts/rag_test.py
```

Admin runs: `docker compose --profile test run rag_test`

---

## Feature 2: Expand Supported File Types in Ingest

Currently `ingest.py` handles `.pdf` and `.docx`. Add support for:

| Format | Library | Notes |
|--------|---------|-------|
| `.txt` | built-in `open()` | Read as UTF-8, chunk directly |
| `.json` | built-in `json` | Flatten to text (stringify values recursively), chunk |
| `.xlsx` | `openpyxl` | Read each sheet → rows as text, one section per sheet |
| `.xls` | `xlrd` | Legacy Excel format |
| `.pptx` | `python-pptx` | Extract text from each slide, one section per slide |
| `.csv` | built-in `csv` | Read rows, convert to text lines, chunk |
| `.md` | built-in `open()` | Same as .txt (markdown is already plain text) |

### Implementation approach

1. **Add reader functions** in `ingest.py` following the existing pattern — each returns `List[Tuple[str, str]]` (section_title, section_text):
   - `read_txt(path)` — single section "Document", full text
   - `read_json(path)` — recursively flatten JSON to key-value text lines
   - `read_excel(path)` — one section per sheet, rows as tab-separated lines
   - `read_pptx(path)` — one section per slide ("Slide 1", "Slide 2", ...)
   - `read_csv(path)` — single section, rows as text lines

2. **Update `process_file_for_chunks()`** — extend the if/elif chain to dispatch by suffix:
   ```python
   ext = path.suffix.lower()
   if ext == ".pdf":
       sections = read_pdf_sections(path)
   elif ext == ".docx":
       sections = read_docx_sections(path)
   elif ext == ".pptx":
       sections = read_pptx(path)
   elif ext in (".xlsx", ".xls"):
       sections = read_excel(path)
   elif ext == ".json":
       sections = read_json(path)
   elif ext in (".txt", ".md", ".csv"):
       sections = read_txt(path)  # csv handled as plain text lines
   ```

3. **Update file discovery** — change the `files` glob filter from `[".pdf", ".docx"]` to include all new extensions.

4. **Add dependencies** to `requirements.txt`:
   ```
   openpyxl
   xlrd
   python-pptx
   ```

5. **Update Dockerfile** — no changes needed (pip install handles it via requirements.txt).

6. **CSV special handling**: For CSV files, instead of raw text dump, format each row as a readable line. For very wide CSVs (>20 columns), use the header + value format:
   ```
   Row 1: Name=John, Age=30, City=NYC
   Row 2: Name=Jane, Age=25, City=LA
   ```

7. **JSON special handling**: Recursively walk the JSON tree and produce "path: value" lines:
   ```
   employees[0].name: John
   employees[0].age: 30
   employees[0].department: Engineering
   ```
   This makes JSON content searchable while preserving structure context.

---

## Implementation Order

**Day 1 — File type expansion (Feature 2)**
1. Add `openpyxl`, `xlrd`, `python-pptx` to requirements.txt
2. Write reader functions: `read_txt`, `read_json`, `read_excel`, `read_pptx`, `read_csv`
3. Update `process_file_for_chunks()` dispatch
4. Update file discovery filter in `main()`
5. Test with sample files of each type
6. Rebuild Docker image

**Day 2 — RAG test script (Feature 1)**
1. Create `scripts/rag_test.py` with the 4-phase pipeline
2. Add `rag_test` profile service to docker-compose.yml
3. Test against the running stack
4. Update README and CLAUDE.md

**Final — Commit as v2.3**
