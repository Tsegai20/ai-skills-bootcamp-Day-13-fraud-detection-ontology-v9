import os, json, re
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])

def call_groq(prompt: str, temperature: float = 0.2) -> str:
    resp = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=800
    )
    return resp.choices[0].message.content

def call_groq_json(prompt: str, temperature: float = 0.2) -> dict:
    raw = call_groq(prompt, temperature).strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)
