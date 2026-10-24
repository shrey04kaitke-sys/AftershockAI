"""Minimal client for a local Ollama server running Gemma 4."""
from __future__ import annotations

import os
from typing import Any

import requests

DEFAULT_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = "gemma4:e2b"


class GemmaError(RuntimeError):
    """Any failure talking to the local model (connection, timeout, missing model, bad reply)."""


class GemmaClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL,
                 timeout: float = 180.0, session: Any | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.http = session or requests

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            r = self.http.post(f"{self.base_url}{path}", json=payload, timeout=self.timeout)
        except requests.ConnectionError as exc:
            raise GemmaError(f"Cannot reach Ollama at {self.base_url}. Is it running? (start it with `ollama serve`)") from exc
        except requests.Timeout as exc:
            raise GemmaError(f"Ollama did not answer within {self.timeout:.0f}s. Try again or use a shorter description.") from exc
        except requests.RequestException as exc:
            raise GemmaError(f"Request to Ollama failed: {exc}") from exc
        if r.status_code == 404:
            raise GemmaError(f"Model '{self.model}' not found. Run `ollama pull {self.model}`.")
        if r.status_code != 200:
            raise GemmaError(f"Ollama returned HTTP {r.status_code}: {r.text[:200]}")
        try:
            return r.json()
        except ValueError as exc:
            raise GemmaError("Ollama returned a non-JSON HTTP body.") from exc

    def status(self) -> tuple[bool, str]:
        """(ok, message) - checks server reachability and model availability."""
        try:
            r = self.http.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            names = [m.get("name", "") for m in r.json().get("models", [])]
        except requests.ConnectionError:
            return False, f"Ollama not reachable at {self.base_url}"
        except (requests.RequestException, ValueError) as exc:
            return False, f"Ollama check failed: {exc}"
        if self.model in names or f"{self.model}:latest" in names:
            return True, f"Connected - {self.model} available"
        return False, f"Ollama is running but '{self.model}' is not pulled (run `ollama pull {self.model}`)"

    def _chat(self, messages: list[dict[str, str]], fmt: Any | None) -> str:
        payload: dict[str, Any] = {
            "model": self.model, "messages": messages, "stream": False,
            "think": False, "options": {"temperature": 0, "seed": 7},
        }
        if fmt is not None:
            payload["format"] = fmt
        data = self._post("/api/chat", payload)
        content = (data.get("message") or {}).get("content", "")
        if not content.strip():
            raise GemmaError("Model returned an empty response.")
        if data.get("done_reason") == "length":
            raise GemmaError("Model response was cut off (length limit); try a shorter description.")
        return content

    def chat_json(self, messages: list[dict[str, str]], schema: dict[str, Any] | None = None) -> str:
        return self._chat(messages, schema or "json")

    def chat_text(self, messages: list[dict[str, str]]) -> str:
        return self._chat(messages, None)
