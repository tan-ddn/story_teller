"""Records produced by the ingest pipeline."""

import hashlib

from pydantic import BaseModel


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Scene(BaseModel):
    scene_id: str  # "<book_id>-c<chapter>-s<scene>", stable across runs
    book_id: int
    chapter_index: int  # 1-based position of the chapter in the book
    scene_index: int  # 1-based position of the scene in the chapter
    chapter_title: str
    split_reason: str  # "section_break" | "length" | "whole_chapter"
    text: str
    word_count: int
    sha256: str


class BookManifest(BaseModel):
    book_id: int
    title: str
    source_url: str
    raw_sha256: str
    chapter_count: int
    scene_count: int
    word_count: int
