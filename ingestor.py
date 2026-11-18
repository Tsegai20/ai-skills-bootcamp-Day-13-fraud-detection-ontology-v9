import json, re, time, urllib.request
from pathlib import Path
from agents.groq_agent import call_groq_json, call_groq

EXTRACT_PROMPT = """Extract entities and relationships from the research text below.
Return ONLY valid JSON, no markdown fences, no extra text:
{{
  "nodes": [
    {{"id": "snake_case_id", "label": "Human Label",
      "type": "concept|method|technology|organization|person|application|challenge",
      "desc": "one sentence"}}
  ],
  "edges": [
    {{"source": "node_id", "target": "node_id", "relation": "verb_phrase"}}
  ]
}}
Rules: 5–15 nodes, 5–20 edges per chunk. snake_case ids, 2–5 words.

TEXT:
{text}"""


def extract_text_from_pdf(path: str) -> str:
    """Extract text from PDF — tries pdfminer first, falls back to pypdf."""
    # Try pdfminer.six (best quality)
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(path) or ""
        if text.strip():
            return text
    except ImportError:
        pass
    except Exception:
        pass
    # Fallback: pypdf
    try:
        from pypdf import PdfReader
        pages = PdfReader(path).pages
        return "\n".join(p.extract_text() or "" for p in pages)
    except ImportError:
        pass
    except Exception:
        pass
    return ""


def chunk_text(text: str, chunk_size: int = 2500, overlap: int = 200):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, buf = [], ""
    for s in sentences:
        if len(buf) + len(s) > chunk_size and buf:
            chunks.append(buf.strip())
            buf = buf[-overlap:] + " " + s
        else:
            buf += " " + s
    if buf.strip():
        chunks.append(buf.strip())
    return [c for c in chunks if len(c) > 80]


def extract_chunk(chunk: str, doc_name: str) -> dict:
    try:
        raw = call_groq(EXTRACT_PROMPT.format(text=chunk[:2000])).strip()
        raw = re.sub(r'^```json\s*', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'^```\s*',     '', raw, flags=re.MULTILINE)
        raw = re.sub(r'\s*```$',     '', raw, flags=re.MULTILINE).strip()
        s = raw.find('{'); e2 = raw.rfind('}')
        if s != -1 and e2 != -1:
            raw = raw[s:e2+1]
        return json.loads(raw)
    except Exception as e:
        print(f'[ingestor] chunk failed: {e}')
        return {'nodes': [], 'edges': []}


def ingest_document(path: str, doc_name: str, max_chunks: int = 40, progress_cb=None) -> dict:
    if path.lower().endswith(".pdf"):
        text = extract_text_from_pdf(path)
    else:
        text = Path(path).read_text(errors="replace")

    if not text.strip():
        print(f"[ingestor] WARNING: no text extracted from {path}")
        return {"nodes": [], "edges": [], "doc_name": doc_name}

    chunks = chunk_text(text)[:max_chunks]
    print(f"[ingestor] {doc_name}: {len(text)} chars → {len(chunks)} chunks")

    all_nodes, all_edges = [], []
    for i, chunk in enumerate(chunks):
        if progress_cb:
            progress_cb(i, len(chunks))
        r = extract_chunk(chunk, doc_name)
        all_nodes.extend(r.get("nodes", []))
        all_edges.extend(r.get("edges", []))
        if i < len(chunks) - 1:
            time.sleep(0.4)

    return {"nodes": all_nodes, "edges": all_edges, "doc_name": doc_name}


def ingest_text(text: str, doc_name: str, max_chunks: int = 20) -> dict:
    chunks = chunk_text(text)[:max_chunks]
    all_nodes, all_edges = [], []
    for chunk in chunks:
        r = extract_chunk(chunk, doc_name)
        all_nodes.extend(r.get("nodes", []))
        all_edges.extend(r.get("edges", []))
        time.sleep(0.4)
    return {"nodes": all_nodes, "edges": all_edges, "doc_name": doc_name}


def ingest_url(url: str) -> dict:
    doc_name = url.rstrip("/").split("/")[-1][:40] or url[:40]
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        for _tag in ["script","style","nav","header","footer","aside"]:
            html = re.sub(rf"<{_tag}[^>]*>.*?</{_tag}>", " ", html, flags=re.DOTALL|re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
    except Exception as e:
        return {"nodes": [], "edges": [], "doc_name": doc_name}
    return ingest_text(text[:4000], doc_name, max_chunks=2)
