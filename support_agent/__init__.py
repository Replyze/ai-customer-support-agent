"""An AI customer support agent that answers only from your docs and hands off to a person when it should."""

from .agent import Decision, SupportAgent
from .claude import ClaudeDecider
from .knowledge import KnowledgeBase

__all__ = ["ClaudeDecider", "Decision", "KnowledgeBase", "SupportAgent"]
__version__ = "0.1.0"
