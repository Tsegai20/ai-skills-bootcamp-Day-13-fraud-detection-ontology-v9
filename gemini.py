
import os, json, re, time

GEMINI_KEY   = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-3.8-flash"
FALLBACK_MODELS = ["gemini-3.5-flash", "gemini-flash-latest"]

def call_gemini(prompt: str, temperature: float = 0.2, retries: int = 5) -> str:
    if not GEMINI_KEY:
        raise RuntimeError("GEMINI_API_KEY not set in environment")
    import google.genai as genai
    client = genai.Client(api_key=GEMINI_KEY)
    models_to_try = [GEMINI_MODEL] + FALLBACK_MODELS
    for model in models_to_try:
        for attempt in range(retries):
            try:
                r = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config={"temperature": temperature, "max_output_tokens": 8192}
                )
                return r.text
            except Exception as e:
                err = str(e)
                if "503" in err or "UNAVAILABLE" in err:
                    wait = 2 ** attempt
                    print(f"[gemini] {model} busy, waiting {wait}s (attempt {attempt+1}/{retries})...")
                    time.sleep(wait)
                elif "404" in err or "NOT_FOUND" in err:
                    print(f"[gemini] {model} not available, trying next...")
                    break  # try next model
                else:
                    if attempt < retries - 1:
                        time.sleep(2)
                    else:
                        raise
    raise RuntimeError("All Gemini models unavailable")

def call_gemini_json(prompt: str, temperature: float = 0.1) -> dict:
    raw = call_gemini(prompt, temperature).strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$",          "", raw, flags=re.MULTILINE)
    start = raw.find("{"); end = raw.rfind("}") + 1
    if start >= 0 and end > start:
        raw = raw[start:end]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        nodes = [json.loads(m) for m in re.findall(r'\{[^{}]*"label"[^{}]*\}', raw)
                 if "label" in m]
        return {"nodes": nodes, "edges": []}
