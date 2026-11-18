"""
seed_fraud_ontology.py — Parallel fraud ontology seeder (V8 final)

Ingests fraud knowledge from Wikipedia (MediaWiki API) directly into the graph.
FTC URLs dropped — they 403 on cloud VMs. Wikipedia API used instead of raw HTML.

Run once after starting the server:
    python seed_fraud_ontology.py
"""

import time, json, sys, concurrent.futures
import urllib.request, urllib.parse
import requests

SERVER = "http://localhost:8000"

WIKIPEDIA_URLS = [
    ("Advance-fee fraud",      "https://en.wikipedia.org/wiki/Advance-fee_scam"),
    ("Phishing",               "https://en.wikipedia.org/wiki/Phishing"),
    ("Social engineering",     "https://en.wikipedia.org/wiki/Social_engineering_(security)"),
    ("Identity theft",         "https://en.wikipedia.org/wiki/Identity_theft"),
    ("Romance scam",           "https://en.wikipedia.org/wiki/Romance_scam"),
    ("Lottery scam",           "https://en.wikipedia.org/wiki/Lottery_scam"),
    ("Tech support scam",      "https://en.wikipedia.org/wiki/Technical_support_scam"),
    ("Ponzi scheme",           "https://en.wikipedia.org/wiki/Ponzi_scheme"),
    ("Pyramid scheme",         "https://en.wikipedia.org/wiki/Pyramid_scheme"),
    ("Investment fraud",       "https://en.wikipedia.org/wiki/Investment_fraud"),
    ("Email fraud",            "https://en.wikipedia.org/wiki/Email_fraud"),
    ("IRS impersonation scam", "https://en.wikipedia.org/wiki/IRS_impersonation_scam"),
    ("Confidence trick",       "https://en.wikipedia.org/wiki/Confidence_trick"),
    ("Money mule",             "https://en.wikipedia.org/wiki/Money_mule"),
    ("Pig butchering scam",    "https://en.wikipedia.org/wiki/Pig_butchering_scam"),
]


def _wiki_text(wiki_url: str) -> str:
    """Fetch clean Wikipedia article text via MediaWiki API."""
    title = wiki_url.rstrip("/").split("/wiki/")[-1]
    api = (
        "https://en.wikipedia.org/w/api.php"
        f"?action=query&prop=extracts&explaintext=1&exintro=1"
        f"&titles={urllib.parse.quote(title)}&format=json"
    )
    try:
        req = urllib.request.Request(api, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=12) as r:
            data = json.loads(r.read())
        pages = data.get("query", {}).get("pages", {})
        for p in pages.values():
            return p.get("extract", "")[:4000]
    except Exception:
        pass
    return ""


def ingest_one(name: str, url: str) -> bool:
    """Fetch Wikipedia text and POST directly to /ingest/text."""
    try:
        text = _wiki_text(url)
        if not text.strip():
            print(f"  ⚠  Empty text: {name}")
            return False

        r = requests.post(
            f"{SERVER}/ingest/text",
            json={"text": text, "source_name": name},
            timeout=10
        )
        if r.status_code != 200:
            print(f"  ✗ {name}: HTTP {r.status_code}")
            return False

        doc_name = r.json().get("doc_name", name)
        # Poll up to 60s
        for _ in range(30):
            time.sleep(2)
            pr = requests.get(f"{SERVER}/progress/{urllib.parse.quote(doc_name)}", timeout=5)
            status = pr.json().get("status", "unknown")
            if status == "done":
                print(f"  ✓  {name}")
                return True
            elif status.startswith("error"):
                print(f"  ✗  {name} → {status}")
                return False
        print(f"  ⚠  Timeout: {name}")
        return False
    except Exception as e:
        print(f"  ✗  Exception: {name} → {e}")
        return False


def check_server():
    try:
        d = requests.get(f"{SERVER}/health", timeout=5).json()
        print(f"Server OK — {d['nodes']} nodes, {d['edges']} edges, {d['docs']} docs")
        return True
    except Exception as e:
        print(f"Server not reachable: {e}")
        return False


def run_seed(max_workers: int = 4):
    if not check_server():
        return
    print(f"\nSeeding {len(WIKIPEDIA_URLS)} sources (parallel, {max_workers} workers)...\n")
    success = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(ingest_one, name, url): name for name, url in WIKIPEDIA_URLS}
        for fut in concurrent.futures.as_completed(futures):
            if fut.result():
                success += 1
    d = requests.get(f"{SERVER}/health", timeout=5).json()
    print(f"\n{'='*50}")
    print(f"Seed complete: {success}/{len(WIKIPEDIA_URLS)} sources")
    print(f"Graph: {d['nodes']} nodes, {d['edges']} edges, {d['docs']} docs")
    print(f"{'='*50}")


if __name__ == "__main__":
    run_seed()
