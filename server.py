
import os
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from agents.graph_store import store
from agents.ingestor import ingest_document, ingest_text, ingest_url
from agents.fraud_detector import FraudDetector
from agents.zero_shot_classifier import ZeroShotFraudDetector

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
_progress: dict = {}

app = FastAPI(title="Ontology Engine")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.middleware("http")
async def ngrok_skip(request, call_next):
    response = await call_next(request)
    response.headers["ngrok-skip-browser-warning"] = "true"
    return response

class TextIngest(BaseModel):
    text: str
    source_name: str = "Pasted Text"

class UrlIngest(BaseModel):
    url: str

class QueryRequest(BaseModel):
    q: str

class LLMQueryRequest(BaseModel):
    q: str
    use_llm: bool = True

class FraudAnalyzeRequest(BaseModel):
    text: str

def _bg_file(path: str, doc_name: str):
    try:
        _progress[doc_name] = {"done": 0, "total": 1, "status": "running"}
        def cb(done, total):
            _progress[doc_name] = {"done": done+1, "total": total, "status": "running"}
        store.add_document(ingest_document(path, doc_name, progress_cb=cb))
        _progress[doc_name]["status"] = "done"
    except Exception as e:
        _progress[doc_name] = {"done": 0, "total": 1, "status": f"error: {e}"}

def _bg_text(text: str, doc_name: str):
    try:
        _progress[doc_name] = {"done": 0, "total": 1, "status": "running"}
        store.add_document(ingest_text(text, doc_name))
        _progress[doc_name]["status"] = "done"
    except Exception as e:
        _progress[doc_name] = {"done": 0, "total": 1, "status": f"error: {e}"}

def _bg_url(url: str, doc_name: str = None):
    import subprocess, json
    if not doc_name:
        doc_name = url.rstrip("/").split("/")[-1][:40] or url[:40]
    _progress[doc_name] = {"done": 0, "total": 1, "status": "running"}
    # Resolve worker path relative to this file — works in any directory
    _worker = str(Path(__file__).parent / "ingest_worker.py")
    try:
        result = subprocess.run(
            ["python", "-I", _worker, url],
            capture_output=True, text=True, timeout=25,
            env={**os.environ}
        )
        if result.returncode == 0 and result.stdout.strip():
            doc = json.loads(result.stdout.strip())
            if "error" not in doc:
                store.add_document(doc)
                _progress[doc_name]["status"] = "done"
            else:
                _progress[doc_name]["status"] = f"error: {doc['error']}"
        else:
            stderr_tail = result.stderr.strip().splitlines()[-3:] if result.stderr else []
            err_msg = " | ".join(stderr_tail) if stderr_tail else "worker exited non-zero"
            print(f"[ingest/url] worker failed for {url}: {err_msg}", flush=True)
            _progress[doc_name]["status"] = f"error: {err_msg[:120]}"
    except subprocess.TimeoutExpired:
        _progress[doc_name] = {"done": 0, "total": 1, "status": "error: timeout"}
        print(f"[ingest/url] TIMEOUT: {url}")
    except Exception as e:
        _progress[doc_name] = {"done": 0, "total": 1, "status": f"error: {e}"}


@app.get("/health")
def health():
    return {"status": "ok", "nodes": len(store.nodes),
            "edges": len(store.edges), "docs": len(store.docs)}

@app.post("/ingest/file")
async def ingest_file_ep(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    doc_name = file.filename or "uploaded_file"
    save_path = UPLOAD_DIR / doc_name
    save_path.write_bytes(await file.read())
    _progress[doc_name] = {"done": 0, "total": 1, "status": "queued"}
    background_tasks.add_task(_bg_file, str(save_path), doc_name)
    return {"status": "queued", "doc_name": doc_name}

@app.post("/ingest/text")
async def ingest_text_ep(req: TextIngest, background_tasks: BackgroundTasks):
    _progress[req.source_name] = {"done": 0, "total": 1, "status": "queued"}
    background_tasks.add_task(_bg_text, req.text, req.source_name)
    return {"status": "queued", "doc_name": req.source_name}

@app.post("/ingest/url")
async def ingest_url_ep(req: UrlIngest, background_tasks: BackgroundTasks):
    doc_name = req.url.rstrip("/").split("/")[-1][:40] or req.url[:40]
    _progress[doc_name] = {"done": 0, "total": 1, "status": "queued"}
    background_tasks.add_task(_bg_url, req.url, doc_name)
    return {"status": "queued", "doc_name": doc_name}

@app.get("/progress/{doc_name}")
def get_progress(doc_name: str):
    return _progress.get(doc_name, {"status": "unknown"})

@app.get("/progress")
def get_all_progress():
    return _progress

@app.get("/graph")
def get_graph():
    return JSONResponse(store.to_ui_json())

@app.get("/sources")
def list_sources():
    return {"sources": store.docs}

@app.post("/query")
def query_graph(req: QueryRequest):
    if not store.nodes:
        return {"answer": "Graph empty — ingest documents first.", "path": [], "matches": []}
    return store.query(req.q)

@app.post("/query/llm")
def query_llm(req: LLMQueryRequest):
    if not store.nodes:
        return {"answer": "Graph empty — ingest documents first.", "path": [], "matches": []}
    graph_result = store.query(req.q)
    if not req.use_llm:
        return graph_result
    match_ids = (graph_result.get("matches") or [])[:20]
    if not match_ids:
        match_ids = list(store.nodes.keys())[:30]
    context = "\n".join(
        f"- {store.nodes[nid]['label']} ({store.nodes[nid]['type']}): {store.nodes[nid].get('desc','')}"
        for nid in match_ids if nid in store.nodes
    ) or "No matching nodes."
    from agents.groq_agent import call_groq as call_llm
    prompt = f"""You are a research knowledge-graph assistant. Answer using ONLY the graph context below.
Be concise (2-4 sentences). Say if the answer is not in context.

Question: {req.q}

Context:
{context}

Answer:"""
    try:
        answer = call_llm(prompt, temperature=0.3).strip()
    except Exception as e:
        print(f"[query/llm] LLM call failed: {e}")
        answer = graph_result["answer"]
    return {"answer": answer,
            "path": graph_result.get("path", []),
            "path_rels": graph_result.get("path_rels", []),
            "matches": graph_result.get("matches", [])}

@app.post("/clear")
def clear_graph():
    store.clear(); _progress.clear()
    return {"status": "cleared"}


class PlanRequest(BaseModel):
    topic: str
    question: str = ""

@app.post("/plan")
def generate_plan(req: PlanRequest):
    from agents.groq_agent import call_groq_json as call_gemini_json
    prompt = f"""You are a research assistant helping build a knowledge graph about: {req.topic}

Research question: {req.question or "General overview"}

Generate a focused research plan with 4-5 Wikipedia articles that cover this topic well.
Return ONLY valid JSON, no markdown fences:
{{
  "steps": [
    {{"title": "step title", "desc": "one sentence why relevant", "url": "https://en.wikipedia.org/wiki/Article_Name"}}
  ],
  "summary": "2 sentence research strategy"
}}

Rules:
- URLs must be real en.wikipedia.org/wiki/ links
- Cover different angles: core concept, technology, regulation, applications
- Topic: {req.topic}
"""
    try:
        result = call_gemini_json(prompt, temperature=0.3)
        return {"status": "ok", "plan": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ── Fraud / Scam Detection ───────────────────────────────────────────────────

_fraud_detector = None      # legacy DS detector (kept for /analyze/fraud/debug)
_zs_detector    = None      # zero-shot ontology-grounded detector

def get_fraud_detector():
    """Legacy DS detector — used only for debug/attribute endpoint."""
    global _fraud_detector
    if _fraud_detector is None:
        _fraud_detector = FraudDetector(store)
    return _fraud_detector

def get_zs_detector():
    """Zero-shot ontology-grounded detector — used by main /analyze/fraud endpoint."""
    global _zs_detector
    if _zs_detector is None:
        _zs_detector = ZeroShotFraudDetector(store)
    return _zs_detector

@app.post("/analyze/fraud")
def analyze_fraud(req: FraudAnalyzeRequest):
    """
    Ontology-guided fraud/scam detection.

    Runs input text through:
      1. AttributeExtractor (LLM scores 8 fraud attributes + graph context)
      2. BayesianFraudClassifier (conditional dependency model)
      3. CrossAttributeAttention (non-linear co-occurrence amplification)

    Attributes scored (from transferable attribute learning research):
      urgency, anonymity, asymmetric_benefit, authority_impersonation,
      isolation_tactic, information_harvesting, payment_irregularity,
      too_good_to_be_true

    Example request:
        POST /analyze/fraud
        {"text": "URGENT: Your account is suspended. Send $200 in gift cards to restore access."}

    Example response:
        {
          "fraud_score": 0.9412,
          "risk_level": "CRITICAL",
          "confidence": 0.88,
          "top_signals": [
            {"attribute": "payment_irregularity", "score": 0.95, ...},
            {"attribute": "urgency", "score": 0.90, ...},
            {"attribute": "authority_impersonation", "score": 0.80, ...}
          ],
          "attribute_scores": { ... },
          "attention_weights": { ... },
          "bayesian_evidence": { ... },
          "graph_boost": 0.031,
          "llm_reasoning": "...",
          "graph_context": { "fraud_related_nodes": 12, ... },
          "explanation": "Risk level: CRITICAL ..."
        }
    """
    if not req.text or not req.text.strip():
        return JSONResponse(
            {"error": "text field is required and must not be empty"},
            status_code=400
        )
    try:
        detector = get_zs_detector()
        result = detector.analyze(req.text)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": f"Analysis failed: {str(e)}"}, status_code=500)


@app.post("/analyze/fraud/build-prototypes")
def build_fraud_prototypes():
    """
    Rebuild fraud TYPE prototypes from the current ontology graph.

    Call this after seeding the ontology (or any time the graph grows).
    Prototypes are saved to data/prototypes.json and loaded automatically
    on the next /analyze/fraud call.

    Returns the prototype vectors built for each fraud type.
    """
    if len(store.nodes) == 0:
        return JSONResponse(
            {"error": "Graph is empty — seed the ontology first via seed_fraud_ontology.py"},
            status_code=400
        )
    try:
        detector   = get_zs_detector()
        prototypes = detector.build_prototypes()
        summary    = {}
        for ft, proto in prototypes.items():
            if ft == "_meta":
                continue
            summary[ft] = {
                "node_count":  proto.get("node_count", 0),
                "confidence":  round(proto.get("confidence", 0), 3),
                "is_fallback": proto.get("is_fallback", False),
                "reasoning":   proto.get("reasoning", ""),
            }
        return JSONResponse({
            "status":     "built",
            "graph_nodes": len(store.nodes),
            "prototypes":  summary,
        })
    except Exception as e:
        return JSONResponse({"error": f"Prototype build failed: {str(e)}"}, status_code=500)


@app.post("/analyze/fraud/debug")
def analyze_fraud_debug(req: FraudAnalyzeRequest):
    """Returns raw LLM attribute scores before Bayesian classification — for tuning."""
    from agents.fraud_detector import AttributeExtractor, ATTRIBUTES
    extractor = AttributeExtractor(store)
    scores, reasoning, graph_ctx = extractor.extract(req.text)
    return {
        "text_preview": req.text[:150],
        "attribute_scores": {k: round(v, 3) for k, v in scores.items()},
        "llm_reasoning": reasoning,
        "graph_fraud_nodes": graph_ctx.get("fraud_related_nodes", 0),
        "active_signals": {k: v for k, v in scores.items() if v >= 0.5},
    }

@app.get("/analyze/fraud/attributes")
def fraud_attribute_taxonomy():
    """Return the fraud attribute taxonomy — names, descriptions, and research priors."""
    from agents.fraud_detector import ATTRIBUTES, ATTRIBUTE_DESCRIPTIONS, FRAUD_PRIORS
    return {
        "attributes": [
            {
                "name": attr,
                "description": ATTRIBUTE_DESCRIPTIONS[attr],
                "fraud_prior": FRAUD_PRIORS[attr],
            }
            for attr in ATTRIBUTES
        ],
        "methodology": (
            "Three-source Dempster-Shafer fusion: "
            "(1) Zero-shot prototype cosine match to ontology-derived fraud type prototypes, "
            "(2) Bayesian attribute priors P(fraud|attribute) from research literature, "
            "(3) Graph path BFS evidence from ontology topology. "
            "Cross-attribute attention amplifies co-occurring signals non-linearly. "
            "All three sources combined via Dempster's combination rule — not weighted sum."
        ),
    }


# ── serve index.html at root (explicit route beats mount for /)
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    p = Path("static/index.html")
    if p.exists():
        return HTMLResponse(
            content=p.read_text(),
            headers={"ngrok-skip-browser-warning": "true"}
        )
    return HTMLResponse("<h2>UI not found — run Cell 4</h2>", status_code=404)

# ── serve other static assets (css, js if any)
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
