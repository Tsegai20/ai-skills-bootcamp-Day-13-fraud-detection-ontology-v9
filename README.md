# Fraud Detection Ontology Engine V9

Dempster-Shafer evidential fusion over a live knowledge graph for real-time fraud detection.

## Architecture
- **3-source fusion**: Prototype cosine similarity + Bayesian attribute priors + Knowledge graph BFS
- **8 fraud attributes**: urgency_pressure, authority_claims, payment_demand, credential_harvesting, reward_promise, isolation_tactic, domain_spoofing, fee_extraction  
- **6 fraud prototypes**: advance_fee_fraud, impersonation_scam, phishing, investment_fraud, romance_scam, identity_theft
- **316 base nodes / 369 edges** — expands via live ingest (PDF, text, URL)
- **FastAPI server** with ngrok tunnel support
- **Groq LLM** (llama3-70b) for natural language explanations

## Results (5/5 CRITICAL test cases)
| Case | Type | Score |
|------|------|-------|
| IRS gift card threat | Impersonation Scam | 0.992 |
| Apple ID phishing | Phishing | 0.989 |
| Crypto 300% returns | Investment Fraud | 0.984 |
| Nigerian estate fee | Advance Fee | 0.978 |
| Military romance | Romance Scam | 0.971 |

## Quick Start (Google Colab)

```python
import os, shutil, subprocess, time, requests

ENGINE_DIR = "/content/ontology_engine_v9"
os.makedirs(f"{ENGINE_DIR}/data/uploads", exist_ok=True)
os.makedirs(f"{ENGINE_DIR}/static", exist_ok=True)

os.environ["GEMINI_API_KEY"] = "your_gemini_key"
os.environ["GROQ_API_KEY"] = "your_groq_key"

server = subprocess.Popen(
    ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"],
    cwd=ENGINE_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    env={**os.environ}
)
time.sleep(5)
print(requests.get("http://localhost:8000/health").json())
```

## Ingest Sources
- `POST /ingest/file` — PDF upload
- `POST /ingest/text` — raw text
- `POST /ingest/url`  — web URL (use en.m.wikipedia.org for Wikipedia)

## Fraud Detection
```python
r = requests.post("http://localhost:8000/analyze/fraud",
                  json={"text": "Your text here"}, timeout=120)
print(r.json())
```

## Security
- Never hardcode API keys — use environment variables
- GEMINI_API_KEY and GROQ_API_KEY must be set before launching server

## Tech Stack
Python · FastAPI · NetworkX · Dempster-Shafer · Groq · ngrok
