from __future__ import annotations
import os
import time
import requests

from . import usage

ollama_url = "http://localhost:11434/api/generate"
default_model = os.environ.get("VLLM_MODEL", "qwen3.5:latest")


def _generate(host: str, payload: dict, timeout: int) -> str:
    """POST to Ollama's /api/generate, record timing/token usage and return the response text."""
    start = time.perf_counter()
    r = requests.post(host, json=payload, timeout=timeout)
    elapsed = time.perf_counter() - start
    r.raise_for_status()
    data = r.json()
    usage.record(elapsed, data.get("prompt_eval_count"), data.get("eval_count"))
    return (data.get("response") or "").strip()


def complete(model: str, system: str, user: str,
             host: str = ollama_url, timeout: int = 300) -> str:
    """Send a system + user prompt to Ollama (JSON mode) and return the raw response text."""
    payload = {
        "model": model,
        "system": system,
        "prompt": user,
        "stream": False,
        "format": "json",
        # Comment the next line if using Qwen2.5x or other models that don't support the "think" tag.
        "think": False,
        "options": {
            "temperature": 0.1,
            "num_ctx": 8192,
        },
    }
    return _generate(host, payload, timeout)
