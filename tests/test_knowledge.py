from support_agent.knowledge import KnowledgeBase, parse_article, slugify, stem, tokenize


def test_stem_matches_word_forms():
    assert stem("returns") == stem("returned") == stem("returning") == stem("return")
    assert stem("shipping") == stem("shipped") == stem("ships") == "ship"
    assert stem("deliveries") == "delivery"
    assert stem("charges") == stem("charged") == stem("charge")


def test_tokenize_drops_stopwords_and_possessives():
    assert tokenize("What's the cost of shipping?") == ["cost", "ship"]


def test_slugify_matches_github_anchors():
    assert slugify("Delivery times") == "delivery-times"
    assert slugify("Out-of-stock items") == "out-of-stock-items"
    assert slugify("What's covered?") == "whats-covered"


def test_parse_article_splits_on_level_two_headings():
    md = "# Returns\n\nIntro text.\n\n## Return window\n\n30 days.\n\n## Refunds\n\nWithin 5 days.\n"
    sections = parse_article("returns.md", md)
    assert [s.id for s in sections] == ["returns.md", "returns.md#return-window", "returns.md#refunds"]
    assert sections[1].article == "Returns"
    assert sections[1].heading == "Return window"
    assert sections[1].text == "30 days."


def test_parse_article_numbers_repeated_headings():
    md = "# FAQ\n\n## Notes\n\nOne.\n\n## Notes\n\nTwo.\n"
    assert [s.id for s in parse_article("faq.md", md)] == ["faq.md#notes", "faq.md#notes-1"]


def test_folder_skips_readme(kb):
    ids = [s.id for s in kb.sections]
    assert "shipping.md#delivery-times" in ids
    assert not any(i.lower().startswith("readme") for i in ids)


def test_search_ranks_the_matching_section_first(kb):
    cases = {
        "How much is express shipping?": "shipping.md#shipping-costs",
        "How do I start a return?": "returns.md#how-to-start-a-return",
        "I forgot my password": "account.md#reset-your-password",
        "Can I use two promo codes?": "payments.md#promo-codes",
    }
    for query, expected in cases.items():
        assert kb.search(query)[0].section.id == expected, query


def test_search_uses_context_for_follow_ups(kb):
    hits = kb.search("How long does it take?", context="Do you ship to Canada?")
    assert hits[0].section.id == "shipping.md#delivery-times"


def test_search_returns_nothing_for_unknown_topics(kb):
    assert kb.search("When will the winter collection be released?") == []
    assert kb.search("hi") == []


def test_empty_knowledge_base_is_an_error(tmp_path):
    try:
        KnowledgeBase.from_folder(tmp_path)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
