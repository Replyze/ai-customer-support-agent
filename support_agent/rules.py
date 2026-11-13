"""Checks that run before the model: cases that always go to a person.

They are plain regular expressions, so they are cheap, predictable and easy to
test. They are deliberately narrow: a pattern that fires on a harmless message
hands a chat to your team for nothing. Claude gets the same rules in its
prompt and catches the wordings these patterns miss.

Tune the patterns with your own eval cases (see evals/cases.json).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping

from .knowledge import tokenize


def _patterns(*regexes: str) -> list[re.Pattern[str]]:
    return [re.compile(r, re.IGNORECASE) for r in regexes]


# Checked in this order; the first reason that matches is the one reported.
HANDOFF_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "account_security": _patterns(
        r"\b(hacked|hacker|compromised|phishing)\b",
        r"\bsomeone\b.{0,30}\b(into my account|logged in|signed in|my password)\b",
        r"\bunauthori[sz]ed (access|log ?in|sign[- ]?in|login)\b",
        r"\b(lost|lose|new phone)\b.{0,30}\b(2fa|two[- ]factor|authenticator)\b",
        r"\blocked out of my account\b",
    ),
    "legal": _patterns(
        r"\b(lawyer|attorney|solicitor|lawsuit|legal action|small claims|cease and desist)\b",
        r"\b(i will|i'll|going to|gonna)\s+(sue|take you to court)\b",
        r"\b(consumer protection|trading standards|better business bureau)\b",
        r"\b(gdpr|ccpa|data protection|subject access request)\b",
        r"\bdelete (all )?(of )?my (personal )?data\b",
    ),
    "billing_dispute": _patterns(
        r"\b(charged|billed) (me )?(twice|two times|2 times|double)\b",
        r"\bdouble[- ]?(charged|charge|billed)\b",
        r"\b(wrong|incorrect|extra|unknown|unauthori[sz]ed) (charge|payment|transaction)s?\b",
        r"\b(overcharged|chargeback)\b",
        r"\bdispute (the|a|this|my) (charge|payment|transaction)\b",
    ),
    # A refund the visitor wants for themselves. "What is your refund policy?"
    # is a question the articles answer, so it doesn't match.
    "refund_request": _patterns(
        r"\b(i want|i'd like|i would like|i need|i demand|give me)\b.{0,30}"
        r"\b(refund|my money back)\b(?!\s*(policy|policies|window|process|time|take))",
        r"\brefund (me|my order|my money|this order)\b",
        r"\b(want|need|get|demand)\b.{0,15}\bmoney back\b",
        r"\bi('m| am) (asking|requesting) (for )?a refund\b",
    ),
    "angry": _patterns(
        r"\b(wtf|fuck\w*|shit\w*|bullshit|pissed|furious|outraged|livid|disgusted)\b",
        r"\b(scam|scammers|rip[- ]?off|ridiculous|unacceptable|pathetic|incompetent)\b",
        r"\bworst (service|company|shop|store|experience|support)\b",
        r"\bnever (shop|order|buy)\w* (here|from you|with you) again\b",
        r"\b(fed up|sick of this|had enough)\b",
    ),
    "human_requested": _patterns(
        r"\b(talk|speak|chat|connect me|put me through)\b.{0,25}"
        r"\b(human|person|someone|somebody|agent|representative|operator|manager)\b",
        r"\b(real|live|actual) (human|person|agent)\b",
        r"^\W*(human|agent|operator|representative|a person)\W*$",
    ),
    "repeated_failure": _patterns(
        r"\b(that|this|it|you) (doesn't|does not|didn't|did not) (help|answer)",
        r"\bnot (helpful|what i (asked|meant))\b",
        r"\byou('re| are) not (helping|listening|understanding)\b",
        r"\bi (already|just) (told|said|asked)\b",
        r"\bwrong answer\b",
    ),
}


@dataclass(frozen=True)
class Trigger:
    reason: str  # one of the keys above
    note: str  # for the team: what matched


def is_shouting(text: str) -> bool:
    """Mostly capital letters in a message long enough to mean it."""
    letters = [c for c in text if c.isalpha()]
    return len(letters) >= 12 and sum(c.isupper() for c in letters) / len(letters) >= 0.7


def similar(a: str, b: str, threshold: float = 0.6) -> bool:
    """Whether two messages ask the same thing: the share of words they have in common."""
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if len(ta) < 2 or len(tb) < 2:
        return False
    return len(ta & tb) / len(ta | tb) >= threshold


def asked_again(message: str, history: Iterable[Mapping]) -> bool:
    """The visitor repeats a question the agent already answered: the answer didn't help."""
    turns = list(history)
    for i, turn in enumerate(turns):
        if turn.get("role") != "visitor" or not similar(message, turn.get("text", "")):
            continue
        reply = next((t for t in turns[i + 1 :] if t.get("role") != "visitor"), None)
        if reply is not None and reply.get("action") == "answer":
            return True
    return False


def check(message: str, history: Iterable[Mapping] = ()) -> Trigger | None:
    """The reason this message must go to a person, or None to let the agent answer."""
    matched = [r for r, pats in HANDOFF_PATTERNS.items() if any(p.search(message) for p in pats)]
    if is_shouting(message) and "angry" not in matched:
        matched.append("angry")
    if asked_again(message, history) and "repeated_failure" not in matched:
        matched.append("repeated_failure")
    if not matched:
        return None
    order = list(HANDOFF_PATTERNS)
    matched.sort(key=order.index)
    return Trigger(matched[0], "Rules matched: " + ", ".join(matched) + ".")
