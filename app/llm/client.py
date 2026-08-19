"""Pluggable LLM client abstraction.

Providers:
- ``ollama``: local Ollama server (default), OpenAI-free, fully offline.
- ``openai``: any OpenAI-compatible chat-completions API (base URL + key).
- ``none``:   no LLM; every call returns an empty string and callers must
  degrade gracefully (pipeline keeps working without corrections).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import httpx

from ..config import Settings


class LLMClient(ABC):
    """Minimal chat interface used by both agents."""

    @abstractmethod
    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        """Return the assistant message content for a system+user prompt."""


class OllamaClient(LLMClient):
    def __init__(self, base_url: str, model: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": 0.1},
        }
        if json_mode:
            payload["format"] = "json"
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
        return (data.get("message") or {}).get("content", "")


class OpenAICompatClient(LLMClient):
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.base_url}/chat/completions", json=payload, headers=headers
            )
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"]


class NullClient(LLMClient):
    """Used when LLM_PROVIDER=none; agents must treat '' as 'skip'."""

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        return ""


def get_llm_client(settings: Settings) -> LLMClient:
    """Factory: build the client selected by configuration."""
    provider = settings.llm_provider.lower().strip()
    if provider == "ollama":
        return OllamaClient(
            settings.ollama_base_url, settings.ollama_model, settings.llm_timeout_seconds
        )
    if provider == "openai":
        return OpenAICompatClient(
            settings.openai_base_url,
            settings.openai_api_key,
            settings.openai_model,
            settings.llm_timeout_seconds,
        )
    if provider == "none":
        return NullClient()
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r}")
