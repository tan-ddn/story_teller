"""Split a cleaned book body into chapters, and chapters into scenes."""

import math
import re
from dataclasses import dataclass

# Heading styles, most specific first. See split_chapters for how one is chosen.
_ROMAN = r"[IVXLCDM]+"
HEADING_PATTERNS = [
    re.compile(rf"^(?:CHAPTER|Chapter)\s+(?:{_ROMAN}|\d+|[A-Za-z]+(?:-[A-Za-z]+)?)\b.*$"),
    re.compile(rf"^{_ROMAN}\.\s+[A-Z].*$"),
    re.compile(r"^\d+\.?\s+[A-Z].*$"),
    re.compile(rf"^{_ROMAN}\.?$"),
    re.compile(r"^\d+\.?$"),
]
MIN_CHAPTERS = 3
TOC_MAX_WORDS = 50  # a "chapter" shorter than this is a table-of-contents entry
TOC_WINDOW_LINES = 4  # a heading followed this closely by another is a table-of-contents entry
MIN_COVERAGE = 0.8  # chapters must hold at least this share of the book's words
SUBTITLE_MAX_CHARS = 80

SECTION_BREAK = re.compile(r"^\s*(?:(?:\*\s*){3,}|(?:[-—_~#=]\s*){3,})$")
MAX_SCENE_WORDS = 1500
MIN_SCENE_WORDS = 20  # shorter pieces are merged into the previous scene


@dataclass
class Chapter:
    title: str
    body: str


@dataclass
class ScenePiece:
    text: str
    split_reason: str


def word_count(text: str) -> int:
    return len(text.split())


def unwrap(text: str) -> str:
    """Join hard-wrapped lines into one line per paragraph; paragraphs separated by a blank line."""
    paragraphs = re.split(r"\n\s*\n", text.strip())
    return "\n\n".join(" ".join(line.strip() for line in p.splitlines()) for p in paragraphs)


def _heading_hits(lines: list[str], pattern: re.Pattern[str]) -> list[int]:
    """Lines that start a chapter: match the pattern, follow a blank line, and are not part of a
    table of contents (a dense run of lines in the same heading style)."""
    matches = [i for i, line in enumerate(lines) if pattern.match(line.strip())]
    hits = []
    for n, i in enumerate(matches):
        in_toc = n + 1 < len(matches) and matches[n + 1] - i <= TOC_WINDOW_LINES
        if not in_toc and (i == 0 or not lines[i - 1].strip()):
            hits.append(i)
    return hits


def _is_heading(line: str) -> bool:
    return any(p.match(line.strip()) for p in HEADING_PATTERNS)


def _build_chapters(lines: list[str], heads: list[int]) -> list[Chapter]:
    chapters = []
    for n, start in enumerate(heads):
        end = heads[n + 1] if n + 1 < len(heads) else len(lines)
        title = lines[start].strip()
        rest = lines[start + 1 : end]
        while rest and not rest[0].strip():
            rest = rest[1:]
        # "CHAPTER I." followed by a short subtitle line: fold the subtitle into the title.
        if (
            rest
            and len(rest[0].strip()) <= SUBTITLE_MAX_CHARS
            and not _is_heading(rest[0])
            and (len(rest) < 2 or not rest[1].strip())
        ):
            title = f"{title} {rest[0].strip()}"
            rest = rest[1:]
        text = "\n".join(rest).strip()
        if word_count(text) >= TOC_MAX_WORDS:
            chapters.append(Chapter(title=" ".join(title.split()), body=text))
    return chapters


def split_chapters(body: str) -> list[Chapter]:
    """Use the most specific heading style that splits the book into chapters covering most of it.

    Specific styles come first, so story titles ("I. A SCANDAL IN BOHEMIA") win over the bare
    section numerals ("I.") inside each story.
    """
    lines = body.splitlines()
    total = word_count(body)
    for pattern in HEADING_PATTERNS:
        heads = _heading_hits(lines, pattern)
        if len(heads) < MIN_CHAPTERS:
            continue
        chapters = _build_chapters(lines, heads)
        covered = sum(word_count(c.body) for c in chapters)
        if len(chapters) >= MIN_CHAPTERS and covered >= MIN_COVERAGE * total:
            return chapters
    return [Chapter(title="Full text", body=body.strip())]


def _split_long_paragraph(paragraph: str) -> list[str]:
    """Split a paragraph longer than MAX_SCENE_WORDS into runs of whole sentences."""
    if word_count(paragraph) <= MAX_SCENE_WORDS:
        return [paragraph]
    parts: list[str] = []
    current: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
        if current and word_count(" ".join([*current, sentence])) > MAX_SCENE_WORDS:
            parts.append(" ".join(current))
            current = []
        current.append(sentence)
    parts.append(" ".join(current))
    return parts


def _chunk_by_length(text: str) -> list[str]:
    paragraphs = [q for p in text.split("\n\n") for q in _split_long_paragraph(p)]
    n_chunks = math.ceil(word_count(text) / MAX_SCENE_WORDS)
    target = word_count(text) / n_chunks
    chunks: list[str] = []
    current: list[str] = []
    for p in paragraphs:
        # Close the chunk early rather than let one more paragraph push it past the cap.
        if current and word_count(" ".join([*current, p])) > MAX_SCENE_WORDS:
            chunks.append("\n\n".join(current))
            current = []
        current.append(p)
        if word_count(" ".join(current)) >= target and len(chunks) < n_chunks - 1:
            chunks.append("\n\n".join(current))
            current = []
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def split_scenes(chapter_body: str) -> list[ScenePiece]:
    sections: list[list[str]] = [[]]
    for line in chapter_body.splitlines():
        if SECTION_BREAK.match(line):
            sections.append([])
        else:
            sections[-1].append(line)
    texts = [unwrap("\n".join(s)) for s in sections]
    texts = [t for t in texts if t]
    has_breaks = len(texts) > 1

    pieces: list[ScenePiece] = []
    for text in texts:
        if word_count(text) > MAX_SCENE_WORDS:
            pieces.extend(ScenePiece(c, "length") for c in _chunk_by_length(text))
        else:
            pieces.append(ScenePiece(text, "section_break" if has_breaks else "whole_chapter"))

    merged: list[ScenePiece] = []
    for piece in pieces:
        if merged and word_count(piece.text) < MIN_SCENE_WORDS:
            merged[-1].text = f"{merged[-1].text}\n\n{piece.text}"
        else:
            merged.append(piece)
    return merged
