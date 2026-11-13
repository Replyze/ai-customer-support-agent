"""Run the agent on a set of cases and score what it decided.

A case is a visitor message, the conversation before it, and what must be
true of the decision:

    {
      "id": "shipping-time-us",
      "history": [],
      "message": "How long does shipping take?",
      "expect": {
        "action": "answer",                 # or a list: ["ask", "handoff"]
        "reason": "answered",               # optional, also a string or a list
        "must_cite": ["shipping.md#delivery-times"],
        "must_say": ["3 to 5"],             # every one, case-insensitive
        "must_not_say": ["week"]            # none of them
      },
      "needs_model": false                  # true: skipped with --offline
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from .agent import Decision, SupportAgent


def _list(value: Any) -> list:
    return value if isinstance(value, list) else [value]


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = [c["id"] for c in cases]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"duplicate case ids: {sorted(duplicates)}")
    return cases


def check(expect: dict[str, Any], decision: Decision) -> list[str]:
    """What is wrong with `decision`; empty when it passes."""
    problems = []
    if "action" in expect and decision.action not in _list(expect["action"]):
        problems.append(f"expected {'/'.join(_list(expect['action']))}, got {decision.action}")
    if "reason" in expect and decision.reason not in _list(expect["reason"]):
        problems.append(f"expected reason {'/'.join(_list(expect['reason']))}, got {decision.reason}")
    for cid in expect.get("must_cite", []):
        if cid not in decision.citations:
            problems.append(f"did not cite {cid}")
    reply = decision.reply.lower()
    for text in expect.get("must_say", []):
        if text.lower() not in reply:
            problems.append(f"reply lacks {text!r}")
    for text in expect.get("must_not_say", []):
        if text.lower() in reply:
            problems.append(f"reply says {text!r}")
    return problems


@dataclass
class Result:
    case: dict[str, Any]
    decision: Decision | None = None
    problems: list[str] = field(default_factory=list)
    skipped: bool = False

    @property
    def passed(self) -> bool:
        return not self.skipped and not self.problems


def run(agent: SupportAgent, cases: list[dict[str, Any]], offline: bool = False) -> list[Result]:
    results = []
    for case in cases:
        if offline and case.get("needs_model"):
            results.append(Result(case, skipped=True))
            continue
        decision = agent.reply(case["message"], case.get("history", []))
        results.append(Result(case, decision, check(case.get("expect", {}), decision)))
    return results


def report(results: list[Result], out: TextIO, verbose: bool = False) -> float:
    """Print one line per case and the score; returns the share of cases that passed."""
    for r in results:
        cid = r.case["id"]
        if r.skipped:
            print(f"SKIP  {cid:<28} needs the model (run without --offline)", file=out)
            continue
        d = r.decision
        status = "PASS" if r.passed else "FAIL"
        cited = ", ".join(d.citations) or "-"
        print(f"{status}  {cid:<28} {d.action:<8} {d.reason:<18} {cited}", file=out)
        for problem in r.problems:
            print(f"        - {problem}", file=out)
        if verbose or r.problems:
            print(f"        reply: {d.reply}", file=out)
    ran = [r for r in results if not r.skipped]
    passed = sum(r.passed for r in ran)
    skipped = len(results) - len(ran)
    score = passed / len(ran) if ran else 0.0
    line = f"score: {passed}/{len(ran)} passed ({score:.0%})"
    if skipped:
        line += f", {skipped} skipped"
    print(line, file=out)
    return score
