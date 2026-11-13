import json
import threading
import urllib.error
import urllib.request

import pytest

from support_agent import ClaudeDecider, SupportAgent
from support_agent.offline import OfflineClient
from support_agent.server import make_server, parse_request


@pytest.fixture
def base_url(kb):
    server = make_server(SupportAgent(kb, ClaudeDecider(client=OfflineClient())), port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


def post(url: str, payload) -> tuple[int, dict]:
    data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_health(base_url):
    with urllib.request.urlopen(f"{base_url}/health", timeout=10) as response:
        assert json.loads(response.read()) == {"status": "ok"}


def test_chat_answers(base_url):
    status, body = post(f"{base_url}/chat", {"message": "Do you ship to Canada?"})
    assert status == 200
    assert body["action"] == "answer"
    assert body["citations"] == ["shipping.md#where-we-ship"]


def test_chat_with_history_hands_off(base_url):
    history = [
        {"role": "visitor", "text": "How do I return an item I bought?"},
        {"role": "assistant", "text": "Go to the returns page.", "action": "answer"},
    ]
    status, body = post(f"{base_url}/chat", {"message": "How do I return an item I bought??", "history": history})
    assert status == 200
    assert (body["action"], body["reason"]) == ("handoff", "repeated_failure")


@pytest.mark.parametrize(
    "payload",
    [b"not json", [], {}, {"message": ""}, {"message": "hi", "history": "no"}, {"message": "hi", "history": [{"role": "x"}]}],
)
def test_bad_requests(base_url, payload):
    status, body = post(f"{base_url}/chat", payload)
    assert status == 400 and "error" in body


def test_unknown_path(base_url):
    status, _ = post(f"{base_url}/nope", {"message": "hi"})
    assert status == 404


def test_parse_request_keeps_only_known_fields():
    message, history = parse_request(
        json.dumps({"message": "hi", "history": [{"role": "visitor", "text": "a", "extra": 1}]}).encode()
    )
    assert message == "hi" and history == [{"role": "visitor", "text": "a"}]
