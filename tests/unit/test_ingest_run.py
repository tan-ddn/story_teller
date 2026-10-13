import io
import json
from pathlib import Path

import pytest

from pipelines.ingest import download as dl
from pipelines.ingest import run


def gutenberg_text(title: str) -> str:
    body = "\n".join(" ".join(["word"] * 10) for _ in range(10))  # hard-wrapped, 100 words
    chapters = "\n\n".join(f"CHAPTER {n}.\n\n{body}" for n in ("I", "II", "III"))
    return (
        f"Title: {title}\n\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\n\n"
        f"{chapters}\n\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\n"
    )


@pytest.fixture
def fake_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[int, str]:
    books: dict[int, str] = {}

    def download(book_id: int, raw_dir: Path) -> Path:
        raw_dir.mkdir(parents=True, exist_ok=True)
        path = raw_dir / f"pg{book_id}.txt"
        path.write_text(books[book_id], encoding="utf-8")
        return path

    monkeypatch.setattr(run, "download", download)
    return books


def test_process_and_build_corpus(fake_download: dict[int, str], tmp_path: Path) -> None:
    fake_download[120] = gutenberg_text("Treasure Island")
    fake_download[11] = gutenberg_text("Alice's Adventures in Wonderland")
    for book_id in (120, 11):
        manifest = run.process_book(book_id, tmp_path)
        assert manifest.chapter_count == 3
        assert manifest.word_count == 300

    corpus = run.build_corpus([120, 11], tmp_path)
    scenes = [json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines()]
    assert [s["book_id"] for s in scenes] == [11] * 3 + [120] * 3
    assert scenes[0]["scene_id"] == "11-c001-s01"
    manifests = json.loads((corpus.parent / "manifest.json").read_text(encoding="utf-8"))
    assert [m["book_id"] for m in manifests] == [11, 120]


def test_title_mismatch_fails(fake_download: dict[int, str], tmp_path: Path) -> None:
    fake_download[120] = gutenberg_text("Some Other Book")
    with pytest.raises(run.TitleMismatchError):
        run.process_book(120, tmp_path)


def test_download_caches_and_retries(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = []

    def urlopen(request: object, timeout: int) -> io.BytesIO:
        calls.append(request)
        if len(calls) == 1:
            raise OSError("flaky network")
        return io.BytesIO(b"book text")

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr("time.sleep", lambda _: None)

    path = dl.download(120, tmp_path)
    assert path.read_bytes() == b"book text"
    assert len(calls) == 2
    assert not path.with_suffix(".part").exists()

    dl.download(120, tmp_path)  # cached: no new request
    assert len(calls) == 2


def test_download_gives_up_after_retries(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def urlopen(request: object, timeout: int) -> io.BytesIO:
        raise OSError("down")

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr("time.sleep", lambda _: None)
    with pytest.raises(OSError):
        dl.download(120, tmp_path)
    assert not (tmp_path / "pg120.txt").exists()
