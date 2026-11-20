"""HTTP helper shared by the LLM backends: retries on transient errors, clear messages on fatal ones."""
from __future__ import annotations

import os
import sys
import time

import requests


class FatalLLMError(RuntimeError):
    """An error that will fail every request (bad API key, unknown model/endpoint), so a batch should stop."""


# Worth retrying: timeouts, rate limiting, overloaded/restarting server or proxy.
_RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
# Will not get better by retrying or by moving on to the next PDF.
_FATAL_STATUS = {401, 403, 404, 405}


def env_int(name: str, default: int) -> int:
    """Read an integer environment variable, warning (instead of crashing) on a bad value."""
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        print(f"[warn] {name}={value!r} is not an integer; using {default}", file=sys.stderr)
        return default


max_retries = env_int("VLLM_RETRIES", 4)


def _body(response: requests.Response, limit: int = 500) -> str:
    text = (response.text or "").strip().replace("\n", " ")
    return text[:limit] + ("…" if len(text) > limit else "")


def _wait(attempt: int, response: requests.Response | None = None) -> float:
    """Seconds to wait before the next attempt: Retry-After if given, else 2, 4, 8, 16 … (max 60)."""
    if response is not None:
        try:
            return min(float(response.headers.get("Retry-After", "")), 120.0)
        except ValueError:
            pass
    return min(2.0 * 2 ** attempt, 60.0)


def post_json(url: str, *, payload: dict, headers: dict | None = None,
              timeout: float = 300) -> tuple[dict, float]:
    """POST a JSON payload and return (parsed JSON response, seconds taken by the successful attempt).

    Retries timeouts, connection errors and transient HTTP statuses up to n times.
    Raises FatalLLMError for auth/not-found errors, requests.HTTPError (with the server's
    error message) for other HTTP errors, and ValueError if the response is not JSON.
    """
    for attempt in range(max_retries + 1):
        last = attempt == max_retries
        start = time.perf_counter()
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        except (requests.Timeout, requests.ConnectionError) as exc:
            if last:
                raise
            wait = _wait(attempt)
            print(f"[warn] LLM request failed ({type(exc).__name__}); "
                  f"retry {attempt + 1}/{max_retries} in {wait:.0f}s", file=sys.stderr)
            time.sleep(wait)
            continue
        elapsed = time.perf_counter() - start

        status = response.status_code
        if status in _RETRY_STATUS and not last:
            wait = _wait(attempt, response)
            print(f"[warn] LLM server returned {status}; "
                  f"retry {attempt + 1}/{max_retries} in {wait:.0f}s", file=sys.stderr)
            time.sleep(wait)
            continue
        if status in _FATAL_STATUS:
            raise FatalLLMError(f"{status} {response.reason} from {url}: {_body(response)}")
        if not response.ok:
            raise requests.HTTPError(f"{status} {response.reason} from {url}: {_body(response)}",
                                     response=response)
        try:
            return response.json(), elapsed
        except ValueError:
            raise ValueError(f"LLM server returned non-JSON response from {url}: {_body(response)}")

    raise AssertionError("unreachable")
