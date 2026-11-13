"""A minimal HTTP endpoint, standard library only.

    POST /chat    {"message": "...", "history": [{"role": "visitor", "text": "..."}, ...]}
              ->  {"action": "answer", "reason": "answered", "reply": "...", "citations": [...], ...}
    GET  /health  {"status": "ok"}

It has no authentication: call it from your own backend, not from a browser,
and keep it on localhost or behind your own auth.
"""

from __future__ import annotations

import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .agent import MAX_MESSAGE_CHARS, SupportAgent

log = logging.getLogger(__name__)

MAX_BODY_BYTES = 256_000
ROLES = {"visitor", "assistant", "team"}


def parse_request(body: bytes) -> tuple[str, list[dict[str, Any]]]:
    """The message and history from a /chat body. Raises ValueError with a message for the caller."""
    try:
        data = json.loads(body or b"null")
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise ValueError("the body must be JSON") from None
    if not isinstance(data, dict):
        raise ValueError("the body must be a JSON object")
    message = data.get("message")
    if not isinstance(message, str) or not message.strip():
        raise ValueError('"message" must be a non-empty string')
    if len(message) > MAX_MESSAGE_CHARS:
        raise ValueError(f'"message" is longer than {MAX_MESSAGE_CHARS} characters')
    history = data.get("history", [])
    if not isinstance(history, list):
        raise ValueError('"history" must be a list')
    turns = []
    for turn in history:
        if not isinstance(turn, dict) or turn.get("role") not in ROLES or not isinstance(turn.get("text"), str):
            raise ValueError('each "history" item needs "role" (visitor, assistant or team) and "text"')
        clean = {"role": turn["role"], "text": turn["text"]}
        if isinstance(turn.get("action"), str):
            clean["action"] = turn["action"]
        turns.append(clean)
    return message, turns


def make_server(agent: SupportAgent, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(HTTPStatus.OK, {"status": "ok"})
            else:
                self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/chat":
                self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if length < 0 or length > MAX_BODY_BYTES:
                self._send(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "body too large"})
                return
            try:
                message, history = parse_request(self.rfile.read(length))
            except ValueError as exc:
                self._send(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            try:
                decision = agent.reply(message, history)
            except Exception:  # never leave the chat without an answer
                log.exception("the agent failed")
                self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "the agent failed"})
                return
            self._send(HTTPStatus.OK, decision.to_dict())

        def _send(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return ThreadingHTTPServer((host, port), Handler)
