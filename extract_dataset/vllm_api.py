from __future__ import annotations
import os
import time
from pathlib import Path
import requests

from . import usage


def _load_env_file(path: Path) -> None:
    """Load KEY=VALUE lines from a .env file without overriding variables already set."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ.setdefault(key, value)

# Look in the working directory first, then the project root.
_load_env_file(Path.cwd() / ".env")
_load_env_file(Path(__file__).resolve().parent.parent / ".env")

vllm_base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:4000").rstrip("/")
vllm_api_key = os.environ.get("VLLM_API_KEY")
vllm_timeout = int(os.environ.get("VLLM_TIMEOUT", "300"))
# Set VLLM_ENABLE_THINKING=1 to let Qwen3-style models think before answering.
vllm_enable_thinking = os.environ.get("VLLM_ENABLE_THINKING", "0").lower() in ("1", "true", "yes")
default_model = os.environ.get("VLLM_MODEL", "qwen3.5")


def chat_completion(model: str, prompt: str, image_url: str | None = None,
                    response_format: dict | None = None, system: str | None = None,
                    temperature: float = 0.1) -> str:
    """Call the vLLM OpenAI-compatible /v1/chat/completions endpoint and return the message text."""
    content = [{"type": "text", "text": prompt}]
    if image_url:
        content.append({"type": "image_url", "image_url": {"url": image_url}})

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": vllm_enable_thinking},
    }
    if response_format:
        payload["response_format"] = response_format

    headers = {"Content-Type": "application/json"}
    if vllm_api_key:
        headers["Authorization"] = f"Bearer {vllm_api_key}"

    start = time.perf_counter()
    response = requests.post(
        f"{vllm_base_url}/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=vllm_timeout
    )
    elapsed = time.perf_counter() - start
    response.raise_for_status()
    data = response.json()
    tokens = data.get("usage") or {}
    usage.record(elapsed, tokens.get("prompt_tokens"), tokens.get("completion_tokens"))
    return (data["choices"][0]["message"]["content"] or "").strip()


def complete(model: str, system: str, user: str) -> str:
    """Send a system + user prompt to the vLLM/LiteLLM server (JSON mode) and return the raw response text."""
    return chat_completion(model, user, system=system,
                           response_format={"type": "json_object"})
