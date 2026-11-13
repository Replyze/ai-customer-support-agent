"""The real anthropic SDK against a local stub of the Messages API.

No API key and no network: this checks the HTTP request the live path sends
and that the SDK's response objects are read correctly.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import anthropic
import pytest

from conftest import decision
from support_agent import ClaudeDecider, SupportAgent


class StubAPI:
    def __init__(self, status: int = 200, stop_reason: str = "end_turn", text: str = ""):
        self.requests: list[dict] = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                stub.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
                if status != 200:
                    payload = {"type": "error", "error": {"type": "api_error", "message": "boom"}}
                else:
                    payload = {
                        "id": "msg_test",
                        "type": "message",
                        "role": "assistant",
                        "model": body["model"],
                        "content": [
                            {"type": "thinking", "thinking": "", "signature": "sig"},
                            {"type": "text", "text": text},
                        ],
                        "stop_reason": stop_reason,
                        "stop_sequence": None,
                        "usage": {
                            "input_tokens": 50,
                            "output_tokens": 40,
                            "cache_creation_input_tokens": 0,
                            "cache_read_input_tokens": 1100,
                        },
                    }
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def stub_agent(kb, monkeypatch):
    monkeypatch.delenv("SUPPORT_AGENT_MODEL", raising=False)
    monkeypatch.delenv("SUPPORT_AGENT_EFFORT", raising=False)
    stubs = []

    def make(**kwargs):
        stub = StubAPI(**kwargs)
        stubs.append(stub)
        client = anthropic.Anthropic(api_key="test-key", base_url=stub.url, max_retries=0)
        return SupportAgent(kb, ClaudeDecider(client=client), company="Larkspur Outdoor"), stub

    yield make
    for stub in stubs:
        stub.close()


def test_live_request_body(stub_agent):
    reply = decision("answer", "answered", "Express shipping in the US costs $14.95.", ["shipping.md#shipping-costs"])
    agent, stub = stub_agent(text=json.dumps(reply))
    result = agent.reply("How much is express shipping?")

    assert result.action == "answer" and result.citations == ["shipping.md#shipping-costs"]
    assert result.usage["cache_read_input_tokens"] == 1100

    sent = stub.requests[0]
    assert sent["path"] == "/v1/messages"
    body = sent["body"]
    assert body["model"] == "claude-haiku-5-5"
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["output_config"]["format"]["schema"]["additionalProperties"] is False
    assert body["output_config"]["effort"] == "medium"
    assert body["messages"][-1]["role"] == "user"
    for removed in ("temperature", "top_p", "top_k", "thinking"):
        assert removed not in body
    assert "anthropic-version" in {k.lower() for k in sent["headers"]}


def test_live_refusal_is_a_handoff(stub_agent):
    agent, _ = stub_agent(stop_reason="refusal", text="")
    result = agent.reply("How much is express shipping?")
    assert (result.action, result.reason) == ("handoff", "agent_error")


def test_live_server_error_is_a_handoff(stub_agent):
    agent, _ = stub_agent(status=500)
    result = agent.reply("How much is express shipping?")
    assert (result.action, result.reason) == ("handoff", "agent_error")
    assert "InternalServerError" in result.note
