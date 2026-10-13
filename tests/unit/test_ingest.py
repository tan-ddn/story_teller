import pytest

from pipelines.ingest.clean import BoilerplateError, extract_title, strip_boilerplate
from pipelines.ingest.split import (
    MAX_SCENE_WORDS,
    split_chapters,
    split_scenes,
    unwrap,
    word_count,
)


def para(n_words: int, word: str = "word") -> str:
    """A hard-wrapped paragraph of n words, like Gutenberg text."""
    words = [word] * n_words
    return "\n".join(" ".join(words[i : i + 10]) for i in range(0, n_words, 10))


RAW = (
    "Title: A Test Book\r\nAuthor: Nobody\r\n\r\n"
    "*** START OF THE PROJECT GUTENBERG EBOOK A TEST BOOK ***\r\n"
    "Once upon a time.\r\n"
    "*** END OF THE PROJECT GUTENBERG EBOOK A TEST BOOK ***\r\n"
    "License text.\r\n"
)


def test_strip_boilerplate_keeps_only_body() -> None:
    assert strip_boilerplate(RAW) == "Once upon a time.\n"
    assert extract_title(RAW) == "A Test Book"


def test_strip_boilerplate_requires_markers() -> None:
    with pytest.raises(BoilerplateError):
        strip_boilerplate("no markers here")


def test_unwrap_joins_lines_within_paragraphs() -> None:
    assert unwrap("one two\nthree\n\nfour\nfive") == "one two three\n\nfour five"


def book(*chapters: str, toc: str = "") -> str:
    return toc + "\n\n" + "\n\n".join(chapters)


def test_chapters_with_subtitles_and_toc() -> None:
    toc = "Contents\n\n CHAPTER I.  Start\n CHAPTER II.  Middle\n CHAPTER III.  End"
    body = book(
        f"CHAPTER I.\nStart\n\n{para(100)}",
        f"CHAPTER II.\nMiddle\n\n{para(100)}",
        f"CHAPTER III.\nEnd\n\n{para(100)}",
        toc=toc,
    )
    chapters = split_chapters(body)
    titles = [c.title for c in chapters]
    assert titles == ["CHAPTER I. Start", "CHAPTER II. Middle", "CHAPTER III. End"]
    assert all(word_count(c.body) == 100 for c in chapters)


def test_bare_roman_headings_beat_toc_style() -> None:
    toc = "    I.  THE START . . . 1\n    II.  THE MIDDLE . . . 9\n    III.  THE END . . . 20"
    body = book(*(f"{n}\n\n{para(100)}" for n in ("I", "II", "III", "IV")), toc=toc)
    assert len(split_chapters(body)) == 4


@pytest.mark.parametrize("heading", ["{n}\n\nA Title", "0{n} A Title"])
def test_arabic_numeral_headings(heading: str) -> None:
    body = book(*(heading.format(n=n) + f"\n\n{para(100)}" for n in (1, 2, 3)))
    assert len(split_chapters(body)) == 3


def test_section_numerals_are_not_folded_into_titles() -> None:
    stories = [f"{n}. A STORY\n\nI.\n\n{para(80)}\n\nII.\n\n{para(80)}" for n in ("I", "II", "III")]
    chapters = split_chapters(book(*stories))
    assert [c.title for c in chapters] == ["I. A STORY", "II. A STORY", "III. A STORY"]


def test_no_headings_gives_one_chapter() -> None:
    assert [c.title for c in split_chapters(para(200))] == ["Full text"]


def test_scenes_split_on_section_breaks() -> None:
    scenes = split_scenes(f"{para(100)}\n\n* * *\n\n{para(100)}")
    assert [s.split_reason for s in scenes] == ["section_break", "section_break"]


def test_long_chapter_is_chunked_at_paragraphs() -> None:
    text = "\n\n".join(para(100) for _ in range(40))  # 4000 words
    scenes = split_scenes(text)
    assert len(scenes) == 3
    assert all(s.split_reason == "length" for s in scenes)
    assert sum(word_count(s.text) for s in scenes) == 4000
    assert all(word_count(s.text) <= MAX_SCENE_WORDS for s in scenes)


def test_tiny_trailing_piece_is_merged() -> None:
    scenes = split_scenes(f"{para(100)}\n\n* * *\n\nTHE END")
    assert len(scenes) == 1
    assert scenes[0].text.endswith("THE END")


def test_chunks_never_exceed_max_when_paragraphs_overshoot_target() -> None:
    scenes = split_scenes(f"{para(700)}\n\n{para(850)}\n\n{para(50)}")
    assert all(word_count(s.text) <= MAX_SCENE_WORDS for s in scenes)
    assert sum(word_count(s.text) for s in scenes) == 1600


def test_single_huge_paragraph_is_split_at_sentences() -> None:
    sentence = " ".join(["word"] * 19) + " end."
    text = " ".join([sentence] * 100)  # one 2000-word paragraph
    scenes = split_scenes(text)
    assert len(scenes) == 2
    assert all(word_count(s.text) <= MAX_SCENE_WORDS for s in scenes)
    assert all(s.text.endswith("end.") for s in scenes)
