import json, sys, urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:11434"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3-vl:8b"

def get(path):
    with urllib.request.urlopen(BASE.rstrip("/") + path, timeout=10) as r:
        return json.loads(r.read().decode())

print("Base:", BASE)
version = get("/api/version")
print("Ollama version:", version.get("version"))

tags = get("/api/tags")
models = [m.get("name") for m in tags.get("models", [])]
print("Models:", models)
if MODEL not in models:
    raise SystemExit(f"Model '{MODEL}' not found. Run: ollama pull {MODEL}")

payload = json.dumps({
    "model": MODEL,
    "messages": [{"role": "user", "content": "Reply with exactly: H3_OK"}],
    "stream": False,
    "think": False,
    "format": "json",
    "options": {"temperature": 0, "num_predict": 32},
}).encode()

req = urllib.request.Request(
    BASE.rstrip("/") + "/api/chat",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=120) as r:
    obj = json.loads(r.read().decode())

print("Raw response content:", obj.get("message", {}).get("content"))
print("Diagnosis: Ollama chat endpoint is responding.")
