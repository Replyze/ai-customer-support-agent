"""A stand-in for anthropic.Anthropic() that needs no API key and no network.

It is not a language model. It reads the request the agent built, and:
- answers with one sentence from the retrieved article that covers most of
  the visitor's words, and cites that article;
- hands off (no_source) when no article covers at least half of them;
- asks what the visitor needs when the message is too short to search.

The rules and guards around it are the real ones, so `--offline` is a fair
test of retrieval, the handoff rules and the guards. It says nothing about
how well Claude writes or judges; run the evals without `--offline` for that.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace
from typing import Any

from .knowledge import tokenize

_ARTICLE = re.compile(r'<article id="([^"]+)" title="([^"]*)">\n(.*?)\n</article>', re.DOTALL)
_VISITOR = re.compile(r"<visitor_message>\n(.*?)\n</visitor_message>", re.DOTALL)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
MIN_COVERAGE = 0.5


class OfflineClient:
    """Has the one method the agent uses: client.messages.create(**kwargs)."""

    def __init__(self) -> None:
        self.messages = self
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.requests.append(kwargs)
        texts = [_text(m["content"]) for m in kwargs["messages"] if m["role"] == "user"]
        visitor = _visitor(texts[-1])
        previous = _visitor(texts[-2]) if len(texts) > 1 else ""
        decision = _decide(visitor, previous, _ARTICLE.findall(texts[-1]))
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text=json.dumps(decision))],
            usage=SimpleNamespace(input_tokens=0, output_tokens=0),
        )


def _text(content: Any) -> str:
    return content if isinstance(content, str) else "\n".join(b["text"] for b in content)


def _visitor(text: str) -> str:
    found = _VISITOR.findall(text)
    return found[-1] if found else ""


def _decide(visitor: str, previous: str, articles: list[tuple[str, str, str]]) -> dict[str, Any]:
    asked = set(tokenize(visitor))
    if len(asked) < 2:
        return _json("ask", "clarify", "Could you tell me a bit more about what you need?")
    best, coverage = None, 0.0
    for article in articles:  # best-ranked first, so it wins a tie
        _id, title, body = article
        share = len(asked & set(tokenize(f"{title} {body}"))) / max(len(asked), 1)
        if share > coverage:
            best, coverage = article, share
    if best is None or coverage < MIN_COVERAGE:
        return _json("handoff", "no_source", "I don't have information about that, so I'm passing it to our team.")
    article_id, _title, body = best
    return _json("answer", "answered", _best_sentence(body, visitor, previous), [article_id])


def _best_sentence(body: str, question: str, previous: str = "") -> str:
    asked, before = set(tokenize(question)), set(tokenize(previous))
    lines = [re.sub(r"^\s*(?:[-*]|\d+\.)\s+", "", line) for line in body.splitlines()]
    sentences = [s.strip() for line in lines for s in _SENTENCE_END.split(line) if s.strip()]
    if not sentences:
        return body.strip()

    def score(i: int) -> tuple[float, int]:
        words = set(tokenize(sentences[i]))
        return (len(asked & words) + 0.5 * len(before & words), -i)  # earlier wins a tie

    return sentences[max(range(len(sentences)), key=score)]


def _json(action: str, reason: str, reply: str, citations: list[str] | None = None) -> dict[str, Any]:
    return {"action": action, "reason": reason, "citations": citations or [], "reply": reply, "note": ""}
