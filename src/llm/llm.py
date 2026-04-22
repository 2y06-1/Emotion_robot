import requests

OLLAMA_API_URL = "http://localhost:11434/api/chat"

payload = {
    "model": "qwen3:0.6b",
    "messages": [{"role": "user", "content": "你好"}]
}

response = requests.post(OLLAMA_API_URL, json=payload)
print(response.json()["response"]["content"])