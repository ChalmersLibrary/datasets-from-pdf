"""Accumulates LLM response time and token usage across all calls made for one PDF."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LLMUsage:
    response_time: float = 0.0             # total wall-clock seconds spent waiting for the LLM
    prompt_tokens: int | None = None       # None if the server never reported token counts
    completion_tokens: int | None = None
    calls: int = 0

    def as_dict(self) -> dict:
        return {
            "response_time": round(self.response_time, 2),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


_current = LLMUsage()


def reset() -> None:
    global _current
    _current = LLMUsage()


def current() -> LLMUsage:
    return _current


def record(elapsed: float, prompt_tokens: int | None, completion_tokens: int | None) -> None:
    _current.calls += 1
    _current.response_time += elapsed
    if prompt_tokens is not None:
        _current.prompt_tokens = (_current.prompt_tokens or 0) + prompt_tokens
    if completion_tokens is not None:
        _current.completion_tokens = (_current.completion_tokens or 0) + completion_tokens
