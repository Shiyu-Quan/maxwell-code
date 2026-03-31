#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import os
import re
import sys
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path


SECTION_PATHS = {
    "overview": ("index.html", "glossary.html"),
    "python": (
        "section_api/subsections/api_python.html",
        "tutorial/python_api_tutorial.html",
        "examples/python/",
    ),
    "cpp": (
        "section_api/subsections/api_cpp.html",
        "tutorial/closed_loop_tutorial.html",
        "examples/cpp/",
    ),
    "tutorial": ("tutorial/", "tutorial_index.html"),
    "faq": ("section_faq/",),
    "examples": ("examples/",),
    "all": (),
}


class TextExtractor(HTMLParser):
    BLOCK_TAGS = {
        "p",
        "div",
        "section",
        "article",
        "li",
        "ul",
        "ol",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "pre",
        "code",
        "table",
        "tr",
        "td",
        "th",
        "br",
    }

    def __init__(self) -> None:
        super().__init__()
        self.in_title = False
        self.title_parts: list[str] = []
        self.body_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "title":
            self.in_title = True
        if tag in self.BLOCK_TAGS:
            self.body_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self.in_title = False
        if tag in self.BLOCK_TAGS:
            self.body_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title_parts.append(data)
        self.body_parts.append(data)

    def result(self) -> tuple[str, str]:
        title = " ".join("".join(self.title_parts).split())
        body = html.unescape("".join(self.body_parts))
        body = re.sub(r"\n{3,}", "\n\n", body)
        body = re.sub(r"[ \t]+", " ", body)
        body = re.sub(r" *\n *", "\n", body)
        return title, body.strip()


@dataclass
class Hit:
    score: int
    path: Path
    title: str
    excerpt: str


def default_docs_root() -> Path | None:
    candidates = []
    env = os.getenv("MAXONE_DOCS_ROOT")
    if env:
        candidates.append(Path(env).expanduser())
    candidates.append(Path.cwd() / "docs")
    candidates.append(Path(__file__).resolve().parents[1] / "docs")
    for candidate in candidates:
        if candidate.exists() and any(candidate.rglob("*.html")):
            return candidate
    return None


def iter_pages(docs_root: Path):
    yield from sorted(docs_root.rglob("*.html"))


def matches_section(relative_path: str, section: str) -> bool:
    prefixes = SECTION_PATHS[section]
    if not prefixes:
        return True
    return any(relative_path == prefix or relative_path.startswith(prefix) for prefix in prefixes)


def load_page(path: Path) -> tuple[str, str]:
    parser = TextExtractor()
    parser.feed(path.read_text(encoding="utf-8", errors="ignore"))
    return parser.result()


def excerpt_for_query(body: str, query_lower: str, query_terms: list[str], width: int = 220) -> str:
    body_lower = body.lower()
    positions = [pos for pos in (body_lower.find(query_lower),) if pos >= 0]
    for term in query_terms:
        pos = body_lower.find(term)
        if pos >= 0:
            positions.append(pos)
    start = min(positions) if positions else 0
    snippet_start = max(0, start - width // 2)
    snippet_end = min(len(body), start + width)
    snippet = body[snippet_start:snippet_end].replace("\n", " ")
    snippet = re.sub(r"\s+", " ", snippet).strip()
    if snippet_start > 0:
        snippet = "..." + snippet
    if snippet_end < len(body):
        snippet = snippet + "..."
    return snippet


def score_page(title: str, body: str, query_lower: str, query_terms: list[str]) -> int:
    haystack_title = title.lower()
    haystack_body = body.lower()
    score = 0
    if query_lower in haystack_title:
        score += 40
    score += haystack_body.count(query_lower) * 15
    for term in query_terms:
        score += haystack_title.count(term) * 10
        score += haystack_body.count(term) * 3
    return score


def search(docs_root: Path, query: str, section: str, limit: int) -> list[Hit]:
    query_lower = query.lower().strip()
    query_terms = [term for term in re.split(r"\s+", query_lower) if term]
    hits: list[Hit] = []
    for path in iter_pages(docs_root):
        rel = path.relative_to(docs_root).as_posix()
        if not matches_section(rel, section):
            continue
        title, body = load_page(path)
        score = score_page(title, body, query_lower, query_terms)
        if score <= 0:
            continue
        hits.append(
            Hit(
                score=score,
                path=path,
                title=title or rel,
                excerpt=excerpt_for_query(body, query_lower, query_terms),
            )
        )
    hits.sort(key=lambda hit: (-hit.score, str(hit.path)))
    return hits[:limit]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search the local MaxOne docs mirror.")
    parser.add_argument("--query", required=True, help="Case-insensitive query string.")
    parser.add_argument(
        "--section",
        choices=sorted(SECTION_PATHS),
        default="all",
        help="Limit search to one document area.",
    )
    parser.add_argument("--limit", type=int, default=8, help="Maximum number of hits to print.")
    parser.add_argument(
        "--docs-root",
        type=Path,
        help="Override docs root. Defaults to MAXONE_DOCS_ROOT, ./docs, or the bundled skill mirror.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    docs_root = args.docs_root or default_docs_root()
    if docs_root is None or not docs_root.exists():
        print("Docs root not found. Set --docs-root or MAXONE_DOCS_ROOT, provide a workspace docs/, or use a skill install with bundled docs.", file=sys.stderr)
        return 1

    hits = search(docs_root.resolve(), args.query, args.section, args.limit)
    if not hits:
        print(f"No hits for query: {args.query}")
        print(f"Searched section: {args.section}")
        print(f"Docs root: {docs_root}")
        return 0

    print(f"Query: {args.query}")
    print(f"Section: {args.section}")
    print(f"Docs root: {docs_root.resolve()}")
    print()
    for index, hit in enumerate(hits, start=1):
        rel = hit.path.resolve().relative_to(docs_root.resolve()).as_posix()
        print(f"{index}. [{hit.score}] {hit.title}")
        print(f"   Path: {rel}")
        print(f"   Excerpt: {hit.excerpt}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
