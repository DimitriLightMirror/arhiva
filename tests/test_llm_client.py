"""Tests for the LLM provider abstraction (ollama / openai / none)."""

import httpx
import pytest

from app.config import Settings
from app.llm.client import NullClient, OllamaClient, OpenAICompatClient, get_llm_client


class MockTransportClient:
    """Helper patching httpx.Client with a scripted transport."""

    def __init__(self, handler):
        self.handler = handler

    def __enter__(self):
        transport = httpx.MockTransport(self.handler)
        self._real = httpx.Client
        httpx.Client = lambda *a, **k: self._real(transport=transport)
        return self

    def __exit__(self, *exc):
        httpx.Client = self._real


def test_factory_returns_ollama_by_default():
    settings = Settings(llm_provider="ollama")
    client = get_llm_client(settings)
    assert isinstance(client, OllamaClient)
    assert client.model == settings.ollama_model


def test_factory_returns_openai_client():
    settings = Settings(llm_provider="openai", openai_api_key="k", openai_model="m")
    client = get_llm_client(settings)
    assert isinstance(client, OpenAICompatClient)
    assert client.model == "m"


def test_factory_returns_null_client():
    settings = Settings(llm_provider="none")
    assert isinstance(get_llm_client(settings), NullClient)


def test_factory_rejects_unknown_provider():
    with pytest.raises(ValueError):
        get_llm_client(Settings(llm_provider="bogus"))


def test_null_client_returns_empty_string():
    assert NullClient().complete("sys", "user") == ""


def test_ollama_complete_parses_response():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        return httpx.Response(200, json={"message": {"content": "text corectat"}})

    with MockTransportClient(handler):
        client = OllamaClient("http://localhost:11434", "qwen2.5:7b")
        assert client.complete("sys", "user") == "text corectat"


def test_openai_complete_parses_response():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer secret"
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "rezultat"}}]}
        )

    with MockTransportClient(handler):
        client = OpenAICompatClient("http://llm.local/v1", "secret", "model-x")
        assert client.complete("sys", "user", json_mode=True) == "rezultat"
