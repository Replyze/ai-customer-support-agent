"""The one call to Claude per visitor message.

Structured outputs (`output_config.format`) make the reply a JSON object that
matches DECISION_SCHEMA, so there is no free text to parse. The system prompt
carries a cache breakpoint: it is the same for every visitor, so after the
first request it is read from the prompt cache.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import anthropic

DEFAULT_MODEL = "claude-haiku-5-5"
DEFAULT_EFFORT = "medium"

# What the model may choose. "unsupported_claim" and "agent_error" are only set
# by the code (guards.py and agent.py).
MODEL_REASONS = [
    "answered",
    "clarify",
    "off_topic",
    "human_requested",
    "angry",
    "refund_request",
    "billing_dispute",
    "legal",
    "account_security",
    "no_source",
    "repeated_failure",
]

DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["answer", "ask", "handoff"]},
        "reason": {"type": "string", "enum": MODEL_REASONS},
        "citations": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Ids of the articles the reply is based on, exactly as given.",
        },
        "reply": {"type": "string", "description": "The message for the visitor."},
        "note": {"type": "string", "description": "One sentence for the team; may be empty."},
    },
    "required": ["action", "reason", "citations", "reply", "note"],
    "additionalProperties": False,
}


class ModelError(Exception):
    """The reply can't be used: an API error, a refusal, a cut-off or malformed reply."""


@dataclass
class ModelReply:
    data: dict[str, Any]
    usage: dict[str, int] = field(default_factory=dict)


class ClaudeDecider:
    """Asks Claude for a decision.

    model:  SUPPORT_AGENT_MODEL, default claude-haiku-5-5.
    effort: SUPPORT_AGENT_EFFORT (low, medium, high, ...), default medium. Set it
            to an empty string for a model that doesn't take `effort`.
    client: an anthropic.Anthropic(); by default one is made from the
            environment (ANTHROPIC_API_KEY). Tests pass a fake one.
    """

    def __init__(
        self,
        client: Any = None,
        model: str | None = None,
        effort: str | None = None,
        max_tokens: int = 16000,
    ):
        self._client = client
        self.model = model or os.environ.get("SUPPORT_AGENT_MODEL") or DEFAULT_MODEL
        if effort is None:
            effort = os.environ.get("SUPPORT_AGENT_EFFORT", DEFAULT_EFFORT)
        self.effort = effort.strip() or None
        # Haiku 5.5 thinks before it answers, and thinking counts toward this
        # limit, so it is far above the size of the JSON reply itself.
        self.max_tokens = max_tokens

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = anthropic.Anthropic()
        return self._client

    def decide(self, system: str, messages: list[dict]) -> ModelReply:
        output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": DECISION_SCHEMA}}
        if self.effort:
            output_config["effort"] = self.effort
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=messages,
                output_config=output_config,
            )
        except anthropic.APIError as exc:  # status errors, timeouts, connection errors
            raise ModelError(f"{type(exc).__name__}: {exc}") from exc

        # Check why it stopped before reading the content.
        if response.stop_reason == "refusal":
            raise ModelError("Claude declined to answer (stop_reason: refusal)")
        if response.stop_reason != "end_turn":
            raise ModelError(f"unexpected stop_reason: {response.stop_reason}")

        # The reply can start with thinking blocks; the JSON is in the text block.
        text = next((block.text for block in response.content if block.type == "text"), "")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ModelError("the reply was not valid JSON") from exc
        if not isinstance(data, dict):
            raise ModelError("the reply was not a JSON object")
        return ModelReply(data, _usage(response))


def _usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    keys = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    return {k: getattr(usage, k) or 0 for k in keys if usage is not None and hasattr(usage, k)}
