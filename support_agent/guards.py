"""Checks that run on the model's decision before the visitor sees it.

The prompt tells Claude not to make things up; these checks make sure of the
parts code can verify:

- an answer must cite at least one article that was actually retrieved;
- every number, link and email address in a reply must appear in those
  articles or in what the visitor wrote;
- the agent may ask for clarification only so many times in a row.
"""

from __future__ import annotations

import re
from typing import Iterable, Mapping

from .knowledge import Section

_LIST_MARKER = re.compile(r"(?m)^\s*\d+[.)]\s+")
_NUMBER = re.compile(r"\d+(?:[.,:]\d+)*")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL = re.compile(r"https?://[^\s<>()\"']+")


def _number(raw: str) -> str:
    n = raw.replace(",", "")
    return re.sub(r"[.:]0+$", "", n)  # "75.00" -> "75", "1:00" -> "1"


def unsupported_details(reply: str, sources: str) -> list[str]:
    """Numbers, links and email addresses in `reply` that `sources` doesn't contain."""
    reply = _LIST_MARKER.sub("", reply)  # "1. Go to ..." is a list, not a fact
    lowered = sources.lower()
    known_numbers = {_number(n) for n in _NUMBER.findall(sources)}
    missing = []
    for url in _URL.findall(reply):
        url = url.rstrip(".,;:!?")
        if url.lower() not in lowered:
            missing.append(url)
    without_links = _URL.sub(" ", reply)
    for email in _EMAIL.findall(without_links):
        if email.lower() not in lowered:
            missing.append(email)
    for number in _NUMBER.findall(_EMAIL.sub(" ", without_links)):
        if _number(number) not in known_numbers:
            missing.append(number)
    return missing


def clarifying_streak(history: Iterable[Mapping]) -> int:
    """How many of the agent's latest replies in a row were clarifying questions."""
    streak = 0
    for turn in reversed(list(history)):
        if turn.get("role") == "visitor":
            continue
        if turn.get("role") == "assistant" and turn.get("action") == "ask":
            streak += 1
        else:
            break
    return streak


def source_text(sections: Iterable[Section], visitor_texts: Iterable[str]) -> str:
    """Everything a reply may take a detail from."""
    parts = [f"{s.article}\n{s.heading}\n{s.text}" for s in sections]
    parts.extend(visitor_texts)
    return "\n".join(parts)
