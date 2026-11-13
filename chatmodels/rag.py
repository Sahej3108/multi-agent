"""
rag.py - Lightweight Retrieval-Augmented Generation for CodeMind.

Loads a plain-text knowledge base (coding standards, error-detection criteria,
security guidelines, complexity rules, optimization principles, refactoring
best practices), splits it into chunks, and retrieves the most relevant
chunks for a given agent task using BM25 ranking.

Pure Python, no extra dependencies, no embedding API needed.

Knowledge-base format (all optional, the loader is forgiving):
    ## Section Title        <- markdown heading, [Section Title], or an ALL CAPS line
    - rule one
    - rule two

    Paragraphs separated by blank lines are kept together where possible.

Usage:
    kb = KnowledgeBase.from_file("data/knowledge_base.txt")
    hits = kb.retrieve_for(focus="null pointer and runtime errors", code=code, lang="Python", k=4)
    prompt_block = format_guidelines(hits)
"""
from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass

# -----------------------------
# Tokenization
# -----------------------------
STOPWORDS = frozenset(
    "a an and are as at be by from has have is it its of on or that the this to was were will with you your".split()
)
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _stem(word: str) -> str:
    """Very light suffix stripping; only needs to be consistent between query and documents."""
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ing") and len(word) > 5:
        return word[:-3]
    if word.endswith("ed") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    """Split text/code into lowercase stemmed tokens (handles snake_case and camelCase)."""
    tokens: list[str] = []
    for raw in _IDENT.findall(text):
        for piece in raw.split("_"):
            for part in _CAMEL.sub(" ", piece).split():
                word = part.lower()
                if len(word) > 1 and word not in STOPWORDS:
                    tokens.append(_stem(word))
    return tokens


# -----------------------------
# Chunking
# -----------------------------
_HEADING = re.compile(
    r"^\s*(?:#{1,6}\s+(?P<md>.+?)|\[(?P<br>[^\]]+)\]|(?P<caps>[A-Z][A-Z0-9 &/,\-\u2013\u2014]{3,}?):?)\s*$"
)
_RULE_LINE = re.compile(r"^\s*[=\-_*~]{3,}\s*$")


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Return [(heading, body)] using headings; falls back to a single 'General' section."""
    sections: list[tuple[str, list[str]]] = [("General", [])]
    for line in text.splitlines():
        if _RULE_LINE.match(line):
            continue
        m = _HEADING.match(line)
        if m:
            heading = (m.group("md") or m.group("br") or m.group("caps")).strip()
            sections.append((heading, []))
        else:
            sections[-1][1].append(line)
    return [(h, "\n".join(b).strip()) for h, b in sections if "\n".join(b).strip()]


def _pack(body: str, max_chars: int) -> list[str]:
    """Pack paragraphs/lines into chunks no larger than max_chars where possible."""
    if max_chars < 1:
        raise ValueError("max_chars must be at least 1")

    units: list[str] = []
    for para in re.split(r"\n\s*\n", body):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            units.append(para)
        else:  # long block with no blank lines (e.g. a big bullet list)
            for line in para.splitlines():
                line = line.strip()
                if not line:
                    continue
                # A single very long line must also be split.
                units.extend(
                    line[start:start + max_chars]
                    for start in range(0, len(line), max_chars)
                )
    pieces, current = [], ""
    for unit in units:
        if current and len(current) + len(unit) + 1 > max_chars:
            pieces.append(current)
            current = unit
        else:
            current = f"{current}\n{unit}" if current else unit
    if current:
        pieces.append(current)
    return pieces


@dataclass
class Chunk:
    id: str
    heading: str
    text: str
    tf: Counter
    length: int


@dataclass
class Hit:
    id: str
    heading: str
    text: str
    score: float


# -----------------------------
# Knowledge base + BM25 retrieval
# -----------------------------
class KnowledgeBase:
    K1 = 1.5
    B = 0.75

    def __init__(self, text: str, source: str = "", max_chars: int = 900):
        self.source = source
        self.chunks: list[Chunk] = []
        for heading, body in _split_sections(text):
            for piece in _pack(body, max_chars):
                tokens = tokenize(f"{heading} {piece}")
                if not tokens:
                    continue
                self.chunks.append(
                    Chunk(id=f"KB-{len(self.chunks) + 1}", heading=heading, text=piece,
                          tf=Counter(tokens), length=len(tokens))
                )
        n = len(self.chunks)
        self.avg_len = (sum(c.length for c in self.chunks) / n) if n else 0.0
        df: Counter = Counter()
        for c in self.chunks:
            df.update(c.tf.keys())
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    # ---- constructors ----
    @classmethod
    def from_file(cls, path: str, **kw) -> "KnowledgeBase":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return cls(f.read(), source=os.path.basename(path), **kw)

    @classmethod
    def from_text(cls, text: str, **kw) -> "KnowledgeBase":
        return cls(text, source="<text>", **kw)

    @property
    def is_empty(self) -> bool:
        return not self.chunks

    def __len__(self) -> int:
        return len(self.chunks)

    # ---- query building ----
    @staticmethod
    def build_query(focus: str, code: str, lang: str) -> Counter:
        """Weighted query: the task focus counts most, then language, then the code's own terms."""
        q: Counter = Counter()
        for t in tokenize(focus):
            q[t] += 3
        for t in tokenize(lang):
            q[t] += 2
        for t, _ in Counter(tokenize(code)).most_common(80):
            q[t] += 1
        return q

    # ---- retrieval ----
    def retrieve(self, query: Counter | str, k: int = 4, boost_headings: tuple[str, ...] = ()) -> list[Hit]:
        if k < 0:
            raise ValueError("k must be non-negative")
        if k == 0 or self.is_empty:
            return []
        q = query if isinstance(query, Counter) else Counter(tokenize(query))
        boosts = tuple(b.lower() for b in boost_headings)
        scored: list[tuple[float, Chunk]] = []
        for c in self.chunks:
            score = 0.0
            norm = self.K1 * (1 - self.B + self.B * c.length / self.avg_len)
            for term, qw in q.items():
                tf = c.tf.get(term)
                if tf:
                    score += min(qw, 3) * self.idf.get(term, 0.0) * tf * (self.K1 + 1) / (tf + norm)
            if score > 0 and boosts and any(b in c.heading.lower() for b in boosts):
                score *= 1.4  # soft preference for sections of the right category
            if score > 0:
                scored.append((score, c))
        scored.sort(key=lambda x: -x[0])
        hits = [Hit(c.id, c.heading, c.text, s) for s, c in scored[:k]]
        if not hits and boosts:  # fallback: nothing matched lexically, use category sections
            hits = [Hit(c.id, c.heading, c.text, 0.0) for c in self.chunks
                    if any(b in c.heading.lower() for b in boosts)][:k]
        return hits

    def retrieve_for(self, focus: str, code: str, lang: str, k: int = 4,
                     boost_headings: tuple[str, ...] = ()) -> list[Hit]:
        return self.retrieve(self.build_query(focus, code, lang), k=k, boost_headings=boost_headings)


# -----------------------------
# Prompt formatting
# -----------------------------
def format_guidelines(hits: list[Hit], max_chars: int = 900) -> str:
    """Render retrieved chunks as a prompt block ('' when nothing was retrieved)."""
    if not hits:
        return ""
    body = "\n\n".join(f"[{h.id}] ({h.heading})\n{h.text[:max_chars]}" for h in hits)
    return (
        "\nReference guidelines retrieved from the CodeMind knowledge base:\n"
        f"{body}\n\n"
        "Ground your analysis in these guidelines where they apply and cite them as [KB-n]. "
        "Do not invent guidelines. If none apply to a point, rely on general best practice and say so.\n"
    )