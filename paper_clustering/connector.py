"""Interchangeable chat-completion connectors for llama.cpp and Perplexity."""

import os
from abc import ABC, abstractmethod

import requests
from dotenv import load_dotenv


class Connector(ABC):
    """Non-streaming chat completions in the choices/message/content format."""

    @abstractmethod
    def complete(self, messages: list[dict[str, str]], grammar: str | None = None,
                 **kwargs) -> dict:
        """Return the provider response; unsupported grammars must raise."""

    def count_prompt_tokens(self, messages: list[dict[str, str]]) -> int | None:
        """Return an exact preflight count, or None when unavailable."""
        return None

    @abstractmethod
    def close(self) -> None:
        """Release the connector's HTTP resources."""


class _HTTPConnector(Connector):
    """Shared session lifecycle, timeout validation, and HTTP error handling."""

    def __init__(self, server_url: str, request_timeout: float,
                 session: requests.Session | None = None):
        if request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        self.server_url = server_url.rstrip("/")
        self.request_timeout = request_timeout
        self.session = session if session is not None else requests.Session()
        self._api_key = None

    def close(self) -> None:
        self.session.close()

    def _post_json(self, endpoint: str, payload: dict) -> dict:
        http_response = None
        try:
            http_response = self.session.post(
                self.server_url + endpoint, json=payload, timeout=self.request_timeout,
            )
            http_response.raise_for_status()
            return http_response.json()
        except (requests.RequestException, ValueError) as exc:
            detail = http_response.text if http_response is not None else ""
            if self._api_key:
                detail = detail.replace(self._api_key, "[redacted]")
            raise RuntimeError(
                f"Failed to query {self.server_url}: {detail[:500]}"
            ) from exc


class LlamaConnector(_HTTPConnector):
    """Reuse an HTTP session for llama.cpp chat completion requests."""

    def __init__(
        self,
        server_url: str | None = None,
        request_timeout: float = 300.0,
        *,
        session: requests.Session | None = None,
    ):
        load_dotenv()
        url = (
            server_url or os.getenv("LLAMA_CPP_SERVER_URL") or "http://127.0.0.1:8080"
        ).rstrip("/")
        super().__init__(url, request_timeout, session)

    def complete(
        self, messages: list[dict[str, str]], grammar: str | None = None, **kwargs
    ) -> dict:
        """Return a JSON completion with caller-supplied request parameters."""
        payload = {"messages": messages, **kwargs}
        if grammar is not None:
            payload["grammar"] = grammar
        if payload.get("stream"):
            raise ValueError("Connector.complete does not support streaming")
        return self._post_json("/v1/chat/completions", payload)

    def count_prompt_tokens(self, messages: list[dict[str, str]]) -> int:
        rendered = self._post_json("/apply-template", {"messages": messages})
        tokens = self._post_json("/tokenize", {
            "content": rendered["prompt"], "add_special": True, "parse_special": True,
        })
        return len(tokens["tokens"])


class PerplexityConnector(_HTTPConnector):
    """Direct model access through Perplexity's Orchestrator chat API.

    Reads PERPLEXITY_API_KEY and optionally PERPLEXITY_MODEL from .env.
    No search tools are enabled. llama.cpp-only cache/template/top-k options
    are not forwarded; grammar-constrained generation is unsupported.
    https://docs.perplexity.ai/docs/orchestrator/quickstart
    """

    def __init__(self, api_key: str | None = None, model: str | None = None,
                 request_timeout: float = 300.0, *,
                 session: requests.Session | None = None):
        load_dotenv()
        key = (api_key or os.getenv("PERPLEXITY_API_KEY") or "").strip()
        if not key:
            raise ValueError("Set PERPLEXITY_API_KEY in .env or pass api_key")
        self.model = model or os.getenv("PERPLEXITY_MODEL") or "google/gemini-3.8-flash"
        super().__init__("https://api.perplexity.ai/orchestrator/v1", request_timeout, session)
        self._api_key = key
        self.session.headers.update({"Authorization": f"Bearer {key}"})

    def complete(self, messages: list[dict[str, str]], grammar: str | None = None,
                 **kwargs) -> dict:
        if grammar is not None:
            raise ValueError("Perplexity does not support llama.cpp grammars")
        if kwargs.get("stream"):
            raise ValueError("Connector.complete does not support streaming")
        for option in ("cache_prompt", "chat_template_kwargs", "top_k"):
            kwargs.pop(option, None)
        payload = {"model": self.model, "messages": messages, **kwargs}
        return self._post_json("/chat/completions", payload)
