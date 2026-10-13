"""Ingest steps as plain functions, shared by the Airflow DAG and the CLI.

uv run python -m pipelines.ingest.run            # whole catalog
uv run python -m pipelines.ingest.run 120 11     # selected book IDs
"""

import argparse
import json
from pathlib import Path

from pipelines.ingest.catalog import CATALOG
from pipelines.ingest.clean import extract_title, strip_boilerplate
from pipelines.ingest.download import download, source_url
from pipelines.ingest.models import BookManifest, Scene, sha256
from pipelines.ingest.split import split_chapters, split_scenes, word_count

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw" / "gutenberg"
INTERIM_DIR = DATA_DIR / "interim" / "gutenberg"
PROCESSED_DIR = DATA_DIR / "processed"


class TitleMismatchError(ValueError):
    pass


def process_book(book_id: int, data_dir: Path = DATA_DIR) -> BookManifest:
    """Download, clean and split one book; write its scenes and manifest to interim/."""
    raw_path = download(book_id, data_dir / RAW_DIR.relative_to(DATA_DIR))
    raw = raw_path.read_text(encoding="utf-8")
    title = extract_title(raw)
    expected = CATALOG.get(book_id)
    if expected and expected.lower() not in title.lower():
        raise TitleMismatchError(f"book {book_id}: expected {expected!r}, got {title!r}")

    scenes: list[Scene] = []
    chapters = split_chapters(strip_boilerplate(raw))
    for c_idx, chapter in enumerate(chapters, start=1):
        for s_idx, piece in enumerate(split_scenes(chapter.body), start=1):
            scenes.append(
                Scene(
                    scene_id=f"{book_id}-c{c_idx:03d}-s{s_idx:02d}",
                    book_id=book_id,
                    chapter_index=c_idx,
                    scene_index=s_idx,
                    chapter_title=chapter.title,
                    split_reason=piece.split_reason,
                    text=piece.text,
                    word_count=word_count(piece.text),
                    sha256=sha256(piece.text),
                )
            )

    manifest = BookManifest(
        book_id=book_id,
        title=title,
        source_url=source_url(book_id),
        raw_sha256=sha256(raw),
        chapter_count=len(chapters),
        scene_count=len(scenes),
        word_count=sum(s.word_count for s in scenes),
    )
    out = data_dir / INTERIM_DIR.relative_to(DATA_DIR)
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"{book_id}.scenes.jsonl").open("w", encoding="utf-8") as f:
        f.writelines(s.model_dump_json() + "\n" for s in scenes)
    (out / f"{book_id}.manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def build_corpus(book_ids: list[int], data_dir: Path = DATA_DIR) -> Path:
    """Concatenate per-book scenes into processed/scenes.jsonl with a corpus manifest."""
    interim = data_dir / INTERIM_DIR.relative_to(DATA_DIR)
    processed = data_dir / PROCESSED_DIR.relative_to(DATA_DIR)
    processed.mkdir(parents=True, exist_ok=True)
    manifests = []
    corpus = processed / "scenes.jsonl"
    with corpus.open("w", encoding="utf-8") as out:
        for book_id in sorted(book_ids):
            out.write((interim / f"{book_id}.scenes.jsonl").read_text(encoding="utf-8"))
            manifests.append(
                json.loads((interim / f"{book_id}.manifest.json").read_text(encoding="utf-8"))
            )
    (processed / "manifest.json").write_text(
        json.dumps(manifests, indent=2) + "\n", encoding="utf-8"
    )
    return corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Gutenberg books into scenes.")
    parser.add_argument("book_ids", nargs="*", type=int, help="defaults to the whole catalog")
    args = parser.parse_args()
    book_ids = args.book_ids or sorted(CATALOG)
    for book_id in book_ids:
        m = process_book(book_id)
        print(f"{book_id:>6} {m.title[:40]:<40} {m.chapter_count:>3} ch {m.scene_count:>4} scenes")
    print(f"corpus: {build_corpus(book_ids)}")


if __name__ == "__main__":
    main()
