"""Command line: `support-agent chat`, `support-agent serve`, `support-agent eval`."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from .agent import Decision, SupportAgent
from .claude import ClaudeDecider
from .knowledge import KnowledgeBase
from .offline import OfflineClient


def load_dotenv(path: str = ".env") -> None:
    """Read KEY=value lines from .env into the environment, without overriding what is set."""
    file = Path(path)
    if not file.is_file():
        return
    for line in file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.removeprefix("export ").strip(), value.strip().strip("'\"")
        if value:  # an empty ANTHROPIC_API_KEY= would hide other credentials
            os.environ.setdefault(key, value)


def build_agent(args: argparse.Namespace) -> SupportAgent:
    knowledge = KnowledgeBase.from_folder(args.docs)
    decider = ClaudeDecider(client=OfflineClient()) if args.offline else ClaudeDecider()
    return SupportAgent(knowledge, decider, company=args.company)


def describe(agent: SupportAgent, args: argparse.Namespace) -> str:
    model = "offline stand-in (no API key, not a language model)" if args.offline else agent.decider.model
    return f"docs: {args.docs} ({len(agent.knowledge.sections)} sections) | model: {model}"


def show(decision: Decision, verbose: bool) -> None:
    print(f"agent> {decision.reply}")
    tags = [decision.action, decision.reason, decision.decided_by]
    print(f"       [{' | '.join(tags)}]" + (f" sources: {', '.join(decision.citations)}" if decision.citations else ""))
    if decision.note:
        print(f"       note for the team: {decision.note}")
    if verbose and decision.usage:
        print("       usage: " + ", ".join(f"{k}={v}" for k, v in decision.usage.items()))


def chat(args: argparse.Namespace) -> int:
    agent = build_agent(args)
    print(describe(agent, args))
    print("Type a message. /reset starts over, /quit leaves.\n")
    history: list[dict] = []
    while True:
        try:
            message = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if message in ("/quit", "/exit"):
            return 0
        if message == "/reset":
            history = []
            print("(new conversation)\n")
            continue
        if not message:
            continue
        try:
            decision = agent.reply(message, history)
        except TypeError as exc:  # the SDK found no credentials
            print(f"Could not call Claude: {exc}", file=sys.stderr)
            print("Set ANTHROPIC_API_KEY in .env or your shell, or try --offline.", file=sys.stderr)
            return 2
        show(decision, args.verbose)
        if decision.action == "handoff":
            print("       (Handed off: in production the AI goes quiet here and a person takes over.")
            print("        Starting a new conversation.)\n")
            history = []
            continue
        history.append({"role": "visitor", "text": message})
        history.append({"role": "assistant", "text": decision.reply, "action": decision.action})
        print()


def serve(args: argparse.Namespace) -> int:
    from .server import make_server

    agent = build_agent(args)
    server = make_server(agent, args.host, args.port)
    print(describe(agent, args))
    print(f"Listening on http://{args.host}:{server.server_port}  (POST /chat, GET /health). Ctrl+C stops.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def evaluate(args: argparse.Namespace) -> int:
    from . import evals

    agent = build_agent(args)
    cases = evals.load_cases(args.cases)
    print(describe(agent, args))
    if args.offline:
        print("Offline: this measures retrieval, rules and guards, not Claude.")
    print()
    try:
        results = evals.run(agent, cases, offline=args.offline)
    except TypeError as exc:
        print(f"Could not call Claude: {exc}", file=sys.stderr)
        print("Set ANTHROPIC_API_KEY in .env or your shell, or run with --offline.", file=sys.stderr)
        return 2
    score = evals.report(results, sys.stdout, verbose=args.verbose)
    return 0 if score >= args.fail_under else 1


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    logging.basicConfig(level=os.environ.get("SUPPORT_AGENT_LOG", "WARNING"))

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--docs", default=os.environ.get("SUPPORT_AGENT_DOCS", "examples/shop-docs"),
                        help="folder of Markdown articles (default: examples/shop-docs)")
    common.add_argument("--company", default=os.environ.get("SUPPORT_AGENT_COMPANY"),
                        help="the business name used in the prompt")
    common.add_argument("--offline", action="store_true",
                        help="use the offline stand-in instead of Claude (no API key needed)")
    common.add_argument("-v", "--verbose", action="store_true")

    parser = argparse.ArgumentParser(prog="support-agent", description="An AI customer support agent that answers from your docs.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("chat", parents=[common], help="chat with the agent in the terminal")
    server = commands.add_parser("serve", parents=[common], help="run the HTTP endpoint")
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8000)
    ev = commands.add_parser("eval", parents=[common], help="run the eval set and print a score")
    ev.add_argument("--cases", default="evals/cases.json")
    ev.add_argument("--fail-under", type=float, default=1.0,
                    help="exit with 1 when the share of passed cases is lower (default: 1.0)")

    args = parser.parse_args(argv)
    try:
        return {"chat": chat, "serve": serve, "eval": evaluate}[args.command](args)
    except FileNotFoundError as exc:
        print(f"{exc}. Run from the repository folder, or pass --docs / --cases.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
