"""HTTP connector for llama.cpp chat completions."""

import os

import requests
from dotenv import load_dotenv


class LlamaConnector:
    """Reuse an HTTP session for chat completion requests."""

    def __init__(
        self,
        server_url: str | None = None,
        request_timeout: float = 300.0,
    ):
        load_dotenv()
        self.server_url = (
            server_url or os.getenv("LLAMA_CPP_SERVER_URL") or "http://127.0.0.1:8080"
        ).rstrip("/")
        if request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        self.request_timeout = request_timeout
        self.session = requests.Session()

    def complete(
        self, messages: list[dict[str, str]], grammar: str | None = None, **kwargs
    ) -> dict:
        """Return a JSON completion with caller-supplied request parameters."""
        payload = {"messages": messages, **kwargs}
        if grammar is not None:
            payload["grammar"] = grammar
        http_response = None
        try:
            http_response = self.session.post(
                f"{self.server_url}/v1/chat/completions",
                json=payload,
                timeout=self.request_timeout,
            )
            http_response.raise_for_status()
            return http_response.json()
        except (requests.RequestException, ValueError) as exc:
            detail = ""
            if http_response is not None:
                detail = f": {http_response.text[:500]}"
            raise RuntimeError(
                f"Failed to query llama.cpp at {self.server_url}{detail}"
            ) from exc
