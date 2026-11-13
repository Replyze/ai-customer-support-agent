import io

from conftest import ROOT
from support_agent import ClaudeDecider, SupportAgent
from support_agent.agent import Decision
from support_agent.evals import check, load_cases, report, run
from support_agent.offline import OfflineClient


def test_offline_eval_passes_every_case_it_runs(kb):
    agent = SupportAgent(kb, ClaudeDecider(client=OfflineClient()))
    cases = load_cases(ROOT / "evals" / "cases.json")
    results = run(agent, cases, offline=True)
    failed = [(r.case["id"], r.problems) for r in results if not r.skipped and not r.passed]
    assert failed == []
    assert any(r.skipped for r in results)  # the cases that need Claude

    out = io.StringIO()
    score = report(results, out)
    assert score == 1.0 and "score:" in out.getvalue()


def test_every_case_has_an_expectation():
    for case in load_cases(ROOT / "evals" / "cases.json"):
        assert case["message"].strip(), case["id"]
        assert case["expect"].get("action"), case["id"]


def test_check_reports_each_problem():
    d = Decision("answer", "answered", "It takes about a week.", ["shipping.md#where-we-ship"])
    problems = check(
        {
            "action": "answer",
            "must_cite": ["shipping.md#delivery-times"],
            "must_say": ["3 to 5"],
            "must_not_say": ["week"],
        },
        d,
    )
    assert len(problems) == 3


def test_check_accepts_a_list_of_actions():
    d = Decision("ask", "off_topic", "I can help with orders and shipping.")
    assert check({"action": ["ask", "handoff"]}, d) == []
