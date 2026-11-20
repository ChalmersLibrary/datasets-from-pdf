from __future__ import annotations
import os
import sys

from . import usage
from .llm_http import post_json

ollama_url = "http://localhost:11434/api/generate"
default_model = os.environ.get("VLLM_MODEL", "qwen3.5:latest")


def _generate(host: str, payload: dict, timeout: int) -> str:
    """POST to Ollama's /api/generate, record timing/token usage and return the response text."""
    data, elapsed = post_json(host, payload=payload, timeout=timeout)
    usage.record(elapsed, data.get("prompt_eval_count"), data.get("eval_count"))
    if data.get("done_reason") == "length":
        print("[warn] LLM response was cut off at the output token limit; "
              "the JSON is probably incomplete", file=sys.stderr)
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
