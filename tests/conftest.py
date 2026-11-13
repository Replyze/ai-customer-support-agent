"""Shared test helpers: a fake Anthropic client and the example knowledge base."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from support_agent import ClaudeDecider, KnowledgeBase, SupportAgent

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "examples" / "shop-docs"


class FakeClient:
    """Stands in for anthropic.Anthropic(): records each request and returns scripted replies."""

    def __init__(self, *replies: Any, stop_reason: str = "end_turn"):
        self.messages = self
        self.requests: list[dict[str, Any]] = []
        self._replies = list(replies)
        self._stop_reason = stop_reason

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.requests.append(kwargs)
        reply = self._replies.pop(0) if self._replies else {}
        if isinstance(reply, Exception):
            raise reply
        text = reply if isinstance(reply, str) else json.dumps(reply)
        return SimpleNamespace(
            stop_reason=self._stop_reason,
            content=[
                SimpleNamespace(type="thinking", thinking="", signature="sig"),
                SimpleNamespace(type="text", text=text),
            ],
            usage=SimpleNamespace(
                input_tokens=100, output_tokens=20, cache_creation_input_tokens=0, cache_read_input_tokens=900
            ),
        )


def decision(action: str, reason: str, reply: str, citations: list[str] | None = None, note: str = "") -> dict:
    return {"action": action, "reason": reason, "citations": citations or [], "reply": reply, "note": note}


@pytest.fixture(scope="session")
def kb() -> KnowledgeBase:
    return KnowledgeBase.from_folder(DOCS)


@pytest.fixture
def make_agent(kb, monkeypatch):
    """make_agent(*replies) -> (agent, fake client). Clears the env vars that change the model."""
    monkeypatch.delenv("SUPPORT_AGENT_MODEL", raising=False)
    monkeypatch.delenv("SUPPORT_AGENT_EFFORT", raising=False)

    def make(*replies: Any, stop_reason: str = "end_turn") -> tuple[SupportAgent, FakeClient]:
        client = FakeClient(*replies, stop_reason=stop_reason)
        return SupportAgent(kb, ClaudeDecider(client=client), company="Larkspur Outdoor"), client

    return make
