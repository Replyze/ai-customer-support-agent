"""The support agent: one call per visitor message, one decision back.

    rules (before the model)  ->  BM25 search  ->  Claude  ->  guards (after the model)

Every path ends in a Decision: answer, ask a clarifying question, or hand off
to a person, with a reason. Nothing here keeps state between calls; the
caller passes the conversation so far.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

from . import guards, rules
from .claude import ClaudeDecider, ModelError
from .knowledge import Hit, KnowledgeBase
from .prompt import build_messages, system_prompt

log = logging.getLogger(__name__)

MAX_MESSAGE_CHARS = 4000

REASONS = {
    "answer": ("answered",),
    "ask": ("clarify", "off_topic"),
    "handoff": (
        "no_source",
        "human_requested",
        "angry",
        "refund_request",
        "billing_dispute",
        "legal",
        "account_security",
        "repeated_failure",
        "unsupported_claim",
        "agent_error",
    ),
}

# What the visitor sees when the code, not the model, decides to hand off.
# They don't promise a reply time: only you know when your team is in.
HANDOFF_REPLIES = {
    "human_requested": "Of course. I'm passing this conversation to our team, and a person will reply here.",
    "angry": "I'm sorry this has been frustrating. I'm passing this to a person on our team so they can sort it out.",
    "refund_request": "Our team handles refunds, so I'm passing this conversation to them. They'll reply here.",
    "billing_dispute": "Our team needs to look at your payment, so I'm passing this conversation to them.",
    "legal": "I'm passing this to our team so the right person can respond.",
    "account_security": "I'm passing this to our team right away. Please don't share passwords or codes in this chat.",
    "no_source": "I don't have information about that, so I'm passing your question to our team. They'll reply here.",
    "repeated_failure": "Sorry I couldn't help with that. I'm passing this conversation to a person on our team.",
    "unsupported_claim": "I don't have reliable information about that, so I'm passing your question to our team.",
    "agent_error": "Sorry, something went wrong on my side. I'm passing this conversation to our team.",
}


@dataclass
class Decision:
    action: str  # "answer", "ask" or "handoff"
    reason: str  # see REASONS
    reply: str  # the message for the visitor
    citations: list[str] = field(default_factory=list)  # section ids, e.g. "shipping.md#delivery-times"
    note: str = ""  # for the team, never shown to the visitor
    decided_by: str = "model"  # "rules", "model" or "guard"
    usage: dict[str, int] = field(default_factory=dict)  # tokens, when Claude was called

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def handoff(reason: str, note: str = "", decided_by: str = "guard", reply: str = "") -> Decision:
    return Decision("handoff", reason, reply or HANDOFF_REPLIES[reason], [], note, decided_by)


class SupportAgent:
    def __init__(
        self,
        knowledge: KnowledgeBase,
        decider: ClaudeDecider | None = None,
        company: str | None = None,
        max_clarifying_questions: int = 2,
        history_limit: int = 12,
    ):
        self.knowledge = knowledge
        self.decider = decider or ClaudeDecider()
        self.system = system_prompt(company)
        self.max_clarifying_questions = max_clarifying_questions
        self.history_limit = history_limit

    def reply(self, message: str, history: Iterable[Mapping] = ()) -> Decision:
        """Decide what to do with the visitor's `message`.

        history: the conversation so far, oldest first, as dicts with "role"
        ("visitor", "assistant" or "team"), "text", and for the agent's own
        turns the "action" it took.
        """
        message = message.strip()[:MAX_MESSAGE_CHARS]
        history = list(history)[-self.history_limit :]
        if not message:
            return Decision("ask", "clarify", "What can I help you with?", decided_by="rules")

        # 1. Cases that always go to a person, without asking the model.
        trigger = rules.check(message, history)
        if trigger:
            return handoff(trigger.reason, trigger.note, decided_by="rules")

        # 2. The articles for this message. The visitor's previous message
        #    helps with follow-ups ("and how long does that take?").
        previous = next((t.get("text", "") for t in reversed(history) if t.get("role") == "visitor"), "")
        hits = self.knowledge.search(message, context=previous)

        # 3. One call to Claude.
        try:
            result = self.decider.decide(self.system, build_messages(history, message, hits))
        except ModelError as exc:
            log.warning("model call failed: %s", exc)
            return handoff("agent_error", f"The model call failed: {exc}")

        decision = _decision_from(result.data)
        decision.usage = result.usage

        # 4. Check what the model decided before the visitor sees it.
        return self._review(decision, hits, message, history)

    def _review(self, decision: Decision, hits: list[Hit], message: str, history: list[Mapping]) -> Decision:
        retrieved = {h.section.id: h.section for h in hits}
        decision.citations = [c for c in decision.citations if c in retrieved]
        visitor_texts = [message] + [str(t.get("text", "")) for t in history if t.get("role") == "visitor"]

        if decision.action == "answer" and not decision.citations:
            return _with_usage(handoff("no_source", "The model answered without citing a retrieved article."), decision)

        # An answer may only use details from the articles it cites; other
        # replies from any article that was retrieved.
        allowed = [retrieved[c] for c in decision.citations] if decision.action == "answer" else retrieved.values()
        missing = guards.unsupported_details(decision.reply, guards.source_text(allowed, visitor_texts))
        if missing:
            note = f"The model's reply had details that aren't in the articles: {', '.join(missing)}."
            if decision.action == "handoff":
                # Keep the handoff, but send the plain message instead.
                return _with_usage(handoff(decision.reason, f"{decision.note} {note}".strip()), decision)
            return _with_usage(handoff("unsupported_claim", note), decision)

        if decision.action == "ask" and guards.clarifying_streak(history) >= self.max_clarifying_questions:
            note = f"The agent already asked {self.max_clarifying_questions} clarifying questions in a row."
            return _with_usage(handoff("repeated_failure", note), decision)

        if not decision.reply:
            if decision.action != "handoff":
                return _with_usage(handoff("agent_error", "The model returned an empty reply."), decision)
            decision.reply = HANDOFF_REPLIES[decision.reason]
        return decision


def _decision_from(data: Mapping[str, Any]) -> Decision:
    """The model's JSON as a Decision, with anything out of place set to a safe value."""
    action = data.get("action")
    if action not in REASONS:
        return handoff("agent_error", f"The model returned an unknown action: {action!r}.")
    reason = data.get("reason")
    if reason not in REASONS[action]:
        reason = REASONS[action][0]
    citations = [c for c in data.get("citations") or [] if isinstance(c, str)]
    return Decision(
        action=action,
        reason=reason,
        reply=str(data.get("reply") or "").strip(),
        citations=citations,
        note=str(data.get("note") or "").strip(),
        decided_by="model",
    )


def _with_usage(new: Decision, old: Decision) -> Decision:
    new.usage = old.usage
    return new
