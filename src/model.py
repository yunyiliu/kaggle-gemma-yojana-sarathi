"""Model access, kept behind one small interface.

Everything else in this project takes a `chat_fn(messages) -> str`.  That is deliberate:
the perception and explanation steps should be testable without a model running, and the
engine - the part that decides entitlement - should be testable without one existing.

The default backend is a local Ollama server, because the whole argument for an open
model here is that the thing runs where the person is: at a Common Service Centre with
intermittent connectivity, on a caseworker's laptop, on a phone. A household's income,
bereavement, and caste details are not data to post to someone's API.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

DEFAULT_MODEL = os.environ.get("GEMMA_MODEL", "gemma3:4b")
DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")


class ModelError(RuntimeError):
    pass


def ollama_chat(messages: list[dict[str, str]], model: str = DEFAULT_MODEL,
                host: str = DEFAULT_HOST, temperature: float = 0.0,
                timeout: int = 180) -> str:
    """One chat completion from a local Ollama server.

    Temperature defaults to 0: the perception step is filling in a form, and there is no
    upside to sampling a different reading of the same sentence on a second run.
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    req = urllib.request.Request(
        f"{host}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.loads(r.read())
    except urllib.error.URLError as exc:
        raise ModelError(
            f"cannot reach Ollama at {host} ({exc}). Start it with `ollama serve` "
            f"and `ollama pull {model}`.") from exc
    return body.get("message", {}).get("content", "")


def scripted_chat(replies: list[str]):
    """A chat_fn that returns canned replies, for tests and for offline demos."""
    queue = list(replies)

    def _fn(_messages: list[dict[str, str]]) -> str:
        if not queue:
            raise ModelError("scripted_chat ran out of replies")
        return queue.pop(0)

    return _fn


def available(host: str = DEFAULT_HOST, timeout: int = 5) -> bool:
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout):
            return True
    except urllib.error.URLError:
        return False
