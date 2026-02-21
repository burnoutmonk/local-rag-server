from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm
from pypdf import PdfReader
import docx

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from config import (
    QDRANT_HOST, QDRANT_PORT, COLLECTION,
    EMBED_MODEL_NAME,
    MAX_CHARS, OVERLAP_CHARS, BATCH_SIZE,
)

DATA_DIR  = Path(__file__).resolve().parent / "data_raw"
# Store hash file in /app/state if it exists (Docker volume), otherwise next to ingest.py
_STATE_DIR = Path("/app/state") if Path("/app/state").exists() else Path(__file__).resolve().parent
HASH_FILE = _STATE_DIR / ".ingest_hashes.json"


# ── File hashing ──────────────────────────────────────────────────────────────
def file_hash(path: Path) -> str:
    """MD5 hash of file contents — fast enough for large PDFs."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_hashes() -> dict:
    try:
        if HASH_FILE.exists() and HASH_FILE.is_file():
            text = HASH_FILE.read_text().strip()
            if text:
                return json.loads(text)
    except Exception:
        pass
    return {}


def save_hashes(hashes: dict) -> None:
    HASH_FILE.write_text(json.dumps(hashes, indent=2))


# ── Text processing ───────────────────────────────────────────────────────────
def normalize_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = s.replace("\f", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def clean_pdf_text(text: str) -> str:
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append("")
            continue
        if re.fullmatch(r"[-\u2013]?\s*\d+\s*[-\u2013]?", stripped):
            continue
        if re.fullmatch(r"[Pp]age\s+\d+(\s+of\s+\d+)?", stripped):
            continue
        if re.fullmatch(r"\d+\s*/\s*\d+", stripped):
            continue
        if re.search(r"\xa9|copyright|\bconfidential\b|\ball rights reserved\b|\bproprietary\b", stripped, re.IGNORECASE):
            continue
        if re.search(r"\.{4,}\s*\d+\s*$", stripped):
            continue
        if re.fullmatch(r"[.\-_\s]{5,}", stripped):
            continue
        cleaned.append(line)

    result = "\n".join(cleaned)
    result = re.sub(r"(\w)-\n(\w)", r"\1\2", result)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def chunk_text(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP_CHARS) -> List[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    buf = ""

    def flush() -> None:
        nonlocal buf
        if buf.strip():
            chunks.append(buf.strip())
        buf = ""

    def hard_split(prefix: str, long_para: str) -> None:
        start = 0
        while start < len(long_para):
            head = (prefix + "\n\n") if (prefix and start == 0) else ""
            available = max_chars - len(head)
            end = min(start + available, len(long_para))
            chunks.append((head + long_para[start:end]).strip())
            next_start = end - overlap
            if next_start <= start:
                next_start = end
            start = next_start

    for p in paras:
        if len(buf) + len(p) + 2 <= max_chars:
            buf = (buf + "\n\n" + p).strip() if buf else p
        else:
            overlap_prefix = buf[-overlap:] if (buf and overlap > 0) else ""
            flush()
            if len(p) <= max_chars:
                buf = (overlap_prefix + "\n\n" + p).strip() if overlap_prefix else p
            else:
                hard_split(overlap_prefix, p)

    flush()
    return chunks


def read_pdf_sections(path: Path) -> List[Tuple[str, str]]:
    try:
        reader = PdfReader(str(path), strict=False)
    except Exception as exc:
        print(f"    WARNING: could not open PDF ({exc}) -- skipping entire file")
        return []

    total_pages = len(reader.pages)
    out: List[Tuple[str, str]] = []
    for i, page in enumerate(reader.pages):
        print(f"    reading page {i+1}/{total_pages} ...", end="\r", flush=True)
        try:
            raw = normalize_text(page.extract_text() or "")
            text = clean_pdf_text(raw)
        except Exception as exc:
            print(f"    WARNING: page {i+1} failed ({exc}) -- skipping")
            continue
        if text:
            out.append((f"Page {i+1}", text))
    print(f"    -> {len(out)}/{total_pages} pages with text    ")
    return out


def read_docx_sections(path: Path) -> List[Tuple[str, str]]:
    d = docx.Document(str(path))
    sections: List[Tuple[str, List[str]]] = []
    current_title = "Document"
    current_lines: List[str] = []

    for p in d.paragraphs:
        style = (p.style.name or "").lower()
        try:
            txt = normalize_text(p.text)
        except Exception as exc:
            print(f"    WARNING: paragraph parse failed ({exc}) -- skipping")
            continue
        if not txt:
            continue
        if "heading" in style:
            if current_lines:
                sections.append((current_title, current_lines))
                current_lines = []
            current_title = txt
        else:
            current_lines.append(txt)

    if current_lines:
        sections.append((current_title, current_lines))

    result = [(title, "\n".join(lines)) for title, lines in sections]
    print(f"    -> {len(result)} sections found")
    return result


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    total_start = time.time()

    if not DATA_DIR.exists():
        raise SystemExit(f"Missing data folder: {DATA_DIR}")

    files = [p for p in DATA_DIR.iterdir() if p.suffix.lower() in [".pdf", ".docx"]]
    if not files:
        raise SystemExit(f"No .pdf/.docx files found in {DATA_DIR}")

    print(f"\n{'='*50}")
    print(f"  Found {len(files)} file(s) in {DATA_DIR}")
    for f in files:
        print(f"    * {f.name}  ({f.stat().st_size/1024:.1f} KB)")
    print(f"{'='*50}\n")

    # Step 1: Load embedder
    print(f"[1/4] Loading embedding model: {EMBED_MODEL_NAME} ...")
    t0 = time.time()
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    dim = embedder.get_sentence_embedding_dimension()
    print(f"      Model loaded in {time.time()-t0:.1f}s  (dim={dim})\n")

    # Step 2: Connect to Qdrant
    print(f"[2/4] Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT} ...")
    t0 = time.time()
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    if not client.collection_exists(COLLECTION):
        print(f"      Collection '{COLLECTION}' not found -- creating fresh.")
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
    else:
        count = client.get_collection(COLLECTION).points_count or 0
        print(f"      Collection '{COLLECTION}' exists with {count} points.")
    print(f"      Collection ready in {time.time()-t0:.1f}s\n")

    # Step 3: Check hashes — skip unchanged files
    print(f"[3/4] Checking which files need ingestion ...")
    stored_hashes = load_hashes()
    new_hashes = {}
    files_to_ingest = []

    for f in files:
        h = file_hash(f)
        new_hashes[f.name] = h
        if stored_hashes.get(f.name) == h:
            print(f"    SKIP  {f.name} (unchanged)")
        else:
            status = "NEW" if f.name not in stored_hashes else "CHANGED"
            print(f"    {status:7s} {f.name}")
            files_to_ingest.append(f)

    # Check for deleted files — remove their chunks from Qdrant
    deleted = [name for name in stored_hashes if name not in new_hashes]
    for name in deleted:
        print(f"    DELETED {name} -- removing from Qdrant...")
        client.delete(
            collection_name=COLLECTION,
            points_selector=Filter(
                must=[FieldCondition(key="source_file", match=MatchValue(value=name))]
            ),
        )

    if not files_to_ingest:
        print("\n  All files up to date -- nothing to ingest.")
        save_hashes(new_hashes)
        return

    print(f"\n  {len(files_to_ingest)} file(s) to ingest.\n")

    # Step 4: Parse, chunk & embed changed files
    print(f"[4/4] Parsing, chunking and embedding ...")
    all_payloads = []
    all_texts = []
    t0 = time.time()

    for file_idx, f in enumerate(files_to_ingest, 1):
        file_start = time.time()
        print(f"\n  [{file_idx}/{len(files_to_ingest)}] {f.name}")

        # Remove old chunks for this file before re-adding
        client.delete(
            collection_name=COLLECTION,
            points_selector=Filter(
                must=[FieldCondition(key="source_file", match=MatchValue(value=f.name))]
            ),
        )

        sections = read_pdf_sections(f) if f.suffix.lower() == ".pdf" else read_docx_sections(f)
        doc_id = f.stem
        file_chunks = 0

        for s_idx, (section_title, section_text) in enumerate(sections):
            print(f"    chunking section {s_idx+1}/{len(sections)}: '{section_title[:50]}' ...", end="\r", flush=True)
            chunks = chunk_text(section_text)
            file_chunks += len(chunks)
            for idx, chunk in enumerate(chunks):
                all_payloads.append({
                    "doc_id": doc_id,
                    "source_file": f.name,
                    "section": section_title,
                    "chunk_index": idx,
                    "text": chunk,
                })
                all_texts.append(chunk)

        print(f"    -> {len(sections)} section(s), {file_chunks} chunk(s)  ({time.time()-file_start:.1f}s)    ")

    total_chars = sum(len(t) for t in all_texts)
    print(f"\n  Chunking done in {time.time()-t0:.1f}s")
    print(f"  Total: {len(all_texts)} chunks  |  ~{total_chars/1000:.0f}k chars  |  avg {total_chars//max(len(all_texts),1)} chars/chunk\n")

    print(f"  Embedding {len(all_texts)} chunks ...")
    t0 = time.time()
    vectors = embedder.encode(
        all_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    print(f"  Embedded in {time.time()-t0:.1f}s\n  Upserting ...")
    t0 = time.time()
    points = []
    batches_sent = 0

    for i in tqdm(range(len(all_payloads)), desc="  Upserting"):
        points.append(PointStruct(
            id=uuid.uuid4(),
            vector=vectors[i].tolist(),
            payload=all_payloads[i],
        ))
        if len(points) >= 256:
            client.upsert(collection_name=COLLECTION, points=points)
            batches_sent += 1
            points = []

    if points:
        client.upsert(collection_name=COLLECTION, points=points)
        batches_sent += 1

    print(f"  Upserted in {time.time()-t0:.1f}s  ({batches_sent} batch(es))\n")

    # Save updated hashes
    save_hashes(new_hashes)

    final_count = client.get_collection(COLLECTION).points_count or 0
    print(f"{'='*50}")
    print(f"  All done in {time.time()-total_start:.1f}s")
    print(f"  Collection : '{COLLECTION}'")
    print(f"  Points     : {final_count}")
    print(f"  Ingested   : {len(files_to_ingest)} file(s)")
    print(f"  Skipped    : {len(files) - len(files_to_ingest)} file(s)")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()