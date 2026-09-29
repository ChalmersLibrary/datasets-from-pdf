from __future__ import annotations
import json
import os
from pathlib import Path
import requests

# Prompts and JSON parsing are shared with the Ollama backend so both behave the same.
from .ollama import prompt, enrich_prompt, _parse_json_response


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

litellm_base_url = os.environ.get("LITELLM_BASE_URL", "http://localhost:4000").rstrip("/")
litellm_api_key = os.environ.get("LITELLM_API_KEY")
litellm_timeout = int(os.environ.get("LITELLM_TIMEOUT", "300"))
# Set LITELLM_ENABLE_THINKING=1 to let Qwen3-style models think before answering.
litellm_enable_thinking = os.environ.get("LITELLM_ENABLE_THINKING", "0").lower() in ("1", "true", "yes")
default_model = os.environ.get("VLM_MODEL", "qwen3.5")


def chat_completion(model: str, prompt: str, image_url: str | None = None,
                    response_format: dict | None = None, system: str | None = None,
                    temperature: float = 0.1) -> str:
    """Call the LiteLLM OpenAI-compatible /v1/chat/completions endpoint and return the message text."""
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
        "chat_template_kwargs": {"enable_thinking": litellm_enable_thinking},
    }
    if response_format:
        payload["response_format"] = response_format

    headers = {"Content-Type": "application/json"}
    if litellm_api_key:
        headers["Authorization"] = f"Bearer {litellm_api_key}"

    response = requests.post(
        f"{litellm_base_url}/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=litellm_timeout
    )
    response.raise_for_status()
    return (response.json()["choices"][0]["message"]["content"] or "").strip()


def enrich_dataset_record(model: str, record: dict, fetched_text: str) -> dict:
    """Re-query LiteLLM with fetched page content to enrich a single dataset record."""
    user_prompt = (
        f"Current dataset record:\n{json.dumps(record, ensure_ascii=False)}\n\n"
        f"---BEGIN FETCHED PAGE---\n{fetched_text}\n---END FETCHED PAGE---\n\n"
        "Return the updated dataset record as JSON."
    )
    raw = chat_completion(model, user_prompt, system=enrich_prompt,
                          response_format={"type": "json_object"})
    enriched = _parse_json_response(raw, fallback=record)
    return {**record, **enriched}


def query_litellm(model: str, section_name: str, section_text: str) -> dict:
    """Send a section to LiteLLM and parse the JSON response."""
    user_prompt = (
        f"Section: {section_name}\n\n"
        f"---BEGIN TEXT---\n{section_text}\n---END TEXT---\n\n"
        "Extract all datasets as specified."
    )
    raw = chat_completion(model, user_prompt, system=prompt,
                          response_format={"type": "json_object"})
    return _parse_json_response(raw)
