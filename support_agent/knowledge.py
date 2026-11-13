"""A folder of Markdown help articles, split into sections and searched with BM25.

No vector database and no embeddings: every `##` section of every `.md` file is
one searchable unit, scored by BM25 (the ranking classic full-text search uses).
For a help center of a few hundred articles this is fast, needs no index on
disk, and you can see exactly why a section matched.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# Words that say nothing about the topic. Kept short on purpose: "not", "no",
# "can't" and the like stay, because they change what a question means.
STOPWORDS = frozenset(
    """a about after again all also am an and any are as at be been before being
    both but by can could did do does doing for from had has have having he her
    here hers him his how i i'd i'm if in into is it it's its just me more most
    my myself of on once only or other our ours out over own same she should so
    some such than that the their them then there these they this those through
    to too until up very was we were what when where which while who whom why
    will with would you your yours hi hello hey please thanks thank""".split()
)

_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def stem(word: str) -> str:
    """A tiny suffix stripper, so "returns", "returned" and "returning" all match "return"."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 5 and word.endswith("ing"):
        word = word[:-3]
    elif len(word) > 4 and word.endswith("ed"):
        word = word[:-2]
    elif len(word) > 4 and word.endswith(("sses", "xes", "shes", "ches")):
        word = word[:-2]
    elif len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        word = word[:-1]
    if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "aeiou":
        word = word[:-1]  # shipp(ed) -> ship
    if len(word) > 4 and word.endswith("e"):
        word = word[:-1]  # charge / charged -> charg
    return word


def tokenize(text: str) -> list[str]:
    """Lowercase words without stopwords, stemmed."""
    words = _WORD.findall(text.lower().replace("’", "'"))
    words = [w[:-2] if w.endswith("'s") else w for w in words]  # "what's" -> "what"
    return [stem(w) for w in words if w not in STOPWORDS]


def slugify(heading: str) -> str:
    """The anchor GitHub gives a heading: "Delivery times" -> "delivery-times"."""
    slug = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return re.sub(r"\s", "-", slug)


@dataclass(frozen=True)
class Section:
    """One `##` section of an article: the unit that is searched and cited."""

    id: str  # "shipping.md#delivery-times", the citation the agent returns
    article: str  # the article's `#` title
    heading: str  # the section's `##` heading
    text: str  # the section's Markdown, without the heading


@dataclass(frozen=True)
class Hit:
    section: Section
    score: float


@dataclass
class _Indexed:
    section: Section
    terms: Counter = field(default_factory=Counter)
    length: int = 0


def parse_article(rel_path: str, markdown: str) -> list[Section]:
    """Split one article into sections. Text before the first `##` gets the article's own id."""
    title = Path(rel_path).stem.replace("-", " ").replace("_", " ").capitalize()
    sections: list[Section] = []
    used: Counter = Counter()
    heading, lines = None, []

    def flush() -> None:
        body = "\n".join(lines).strip()
        if not body:
            return
        if heading is None:
            sections.append(Section(rel_path, title, title, body))
            return
        slug = slugify(heading)
        anchor = slug if used[slug] == 0 else f"{slug}-{used[slug]}"
        used[slug] += 1
        sections.append(Section(f"{rel_path}#{anchor}", title, heading, body))

    for line in markdown.splitlines():
        if line.startswith("# ") and not sections and heading is None and not any(lines):
            title = line[2:].strip()
        elif line.startswith("## "):
            flush()
            heading, lines = line[3:].strip(), []
        else:
            lines.append(line)
    flush()
    return sections


class KnowledgeBase:
    """The sections of a folder of Markdown files, ranked with BM25."""

    def __init__(self, sections: list[Section], k1: float = 1.5, b: float = 0.75):
        if not sections:
            raise ValueError("the knowledge base is empty")
        self.k1, self.b = k1, b
        self._docs: list[_Indexed] = []
        for section in sections:
            # The heading says what a section is about, so its words count double.
            terms = tokenize(section.article) + tokenize(section.heading) * 2 + tokenize(section.text)
            self._docs.append(_Indexed(section, Counter(terms), len(terms)))
        self._by_id = {d.section.id: d.section for d in self._docs}
        self._avg_len = sum(d.length for d in self._docs) / len(self._docs)
        doc_freq = Counter(term for d in self._docs for term in d.terms)
        n = len(self._docs)
        self._idf = {t: math.log(1 + (n - df + 0.5) / (df + 0.5)) for t, df in doc_freq.items()}

    @classmethod
    def from_folder(cls, folder: str | Path) -> "KnowledgeBase":
        """Every `.md` file under `folder`, except README.md files."""
        root = Path(folder)
        if not root.is_dir():
            raise FileNotFoundError(f"no such folder: {root}")
        sections: list[Section] = []
        for path in sorted(root.rglob("*.md")):
            if path.name.lower() == "readme.md":
                continue
            rel = path.relative_to(root).as_posix()
            sections.extend(parse_article(rel, path.read_text(encoding="utf-8")))
        return cls(sections)

    @property
    def sections(self) -> list[Section]:
        return [d.section for d in self._docs]

    def get(self, section_id: str) -> Section | None:
        return self._by_id.get(section_id)

    def search(self, query: str, context: str = "", limit: int = 3, min_score: float = 1.5) -> list[Hit]:
        """The best sections for `query`, best first.

        `context` (for example the visitor's previous message) counts at half
        weight, so a follow-up like "how long does it take?" still finds the
        topic of the conversation. Sections scoring below `min_score` are left
        out: one shared common word is not a match.
        """
        weights: dict[str, float] = {}
        for term in tokenize(context):
            weights[term] = 0.5
        for term in tokenize(query):
            weights[term] = 1.0

        hits = []
        for doc in self._docs:
            score = 0.0
            for term, weight in weights.items():
                tf = doc.terms.get(term)
                if tf:
                    norm = tf + self.k1 * (1 - self.b + self.b * doc.length / self._avg_len)
                    score += weight * self._idf[term] * tf * (self.k1 + 1) / norm
            if score >= min_score:
                hits.append(Hit(doc.section, round(score, 3)))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:limit]
