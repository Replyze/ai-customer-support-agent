import pytest

from support_agent import guards, rules


@pytest.mark.parametrize(
    "message, reason",
    [
        ("Can I talk to a real person?", "human_requested"),
        ("I'd like to speak with someone from your team", "human_requested"),
        ("agent", "human_requested"),
        ("I want a refund for my order", "refund_request"),
        ("Just give me my money back", "refund_request"),
        ("You charged me twice!", "billing_dispute"),
        ("There is an unknown charge on my card", "billing_dispute"),
        ("I'm going to sue you", "legal"),
        ("My lawyer will contact you", "legal"),
        ("Please delete all my personal data", "legal"),
        ("I think my account was hacked", "account_security"),
        ("Someone logged in to my account from another country", "account_security"),
        ("This is a scam", "angry"),
        ("WHERE IS MY ORDER, I PAID TWO WEEKS AGO", "angry"),
        ("That doesn't help at all", "repeated_failure"),
    ],
)
def test_messages_that_go_to_a_person(message, reason):
    trigger = rules.check(message)
    assert trigger is not None and trigger.reason == reason


@pytest.mark.parametrize(
    "message",
    [
        "What is your refund policy?",
        "I need to know your refund policy",
        "How long does a refund take?",
        "How do I reset my password?",
        "Do you ship to Canada?",
        "My name is Sue and I have a question about sizes",
        "Thanks, that helps!",
        "Is the jacket waterproof? I'm going hiking!",
    ],
)
def test_ordinary_questions_are_left_to_the_agent(message):
    assert rules.check(message) is None


def test_a_specific_topic_wins_over_a_general_request():
    trigger = rules.check("I want to talk to a human, you charged me twice")
    assert trigger.reason == "billing_dispute"
    assert "human_requested" in trigger.note


def test_repeating_an_answered_question_is_a_failure():
    history = [
        {"role": "visitor", "text": "How do I return an item I bought?"},
        {"role": "assistant", "text": "Go to the returns page.", "action": "answer"},
    ]
    assert rules.check("how do I return an item I bought??", history).reason == "repeated_failure"


def test_repeating_after_a_clarifying_question_is_not_a_failure():
    history = [
        {"role": "visitor", "text": "How do I return an item I bought?"},
        {"role": "assistant", "text": "Which order is it?", "action": "ask"},
    ]
    assert rules.check("How do I return an item I bought?", history) is None


def test_clarifying_streak_counts_the_latest_questions_only():
    history = [
        {"role": "assistant", "text": "a", "action": "ask"},
        {"role": "assistant", "text": "b", "action": "answer"},
        {"role": "visitor", "text": "c"},
        {"role": "assistant", "text": "d", "action": "ask"},
        {"role": "visitor", "text": "e"},
        {"role": "assistant", "text": "f", "action": "ask"},
        {"role": "visitor", "text": "g"},
    ]
    assert guards.clarifying_streak(history) == 2


def test_unsupported_details_finds_invented_figures_links_and_emails():
    sources = "Express shipping costs $14.95. Write to help@larkspur.example or see https://larkspur.example/track."
    assert guards.unsupported_details("Express costs $14.95.", sources) == []
    assert guards.unsupported_details("Express costs $14.95, or $9.99 for members.", sources) == ["9.99"]
    assert guards.unsupported_details("Email sales@larkspur.example", sources) == ["sales@larkspur.example"]
    assert guards.unsupported_details("See https://larkspur.example/track.", sources) == []
    assert guards.unsupported_details("See https://larkspur.example/faq", sources) == ["https://larkspur.example/faq"]


def test_unsupported_details_ignores_list_numbers_and_formatting():
    sources = "Orders before 1 pm ship the same day. Free on orders of $75 or more. Up to 1,000 items."
    reply = "1. Order before 1:00 pm.\n2. Orders of $75.00 ship free.\n3. Up to 1000 items."
    assert guards.unsupported_details(reply, sources) == []
