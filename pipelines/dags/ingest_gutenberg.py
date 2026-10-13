"""Airflow DAG: ingest the Gutenberg catalog into scenes. Logic lives in pipelines.ingest.run."""

import os
from datetime import datetime
from pathlib import Path

from airflow.sdk import dag, task

from pipelines.ingest.catalog import CATALOG
from pipelines.ingest.run import build_corpus, process_book

DATA_DIR = Path(os.environ.get("STORY_DATA_DIR", "data"))


@dag(
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_tasks=1,  # one download at a time, to be polite to Gutenberg
    tags=["ingest"],
)
def ingest_gutenberg() -> None:
    @task(retries=2)
    def ingest_book(book_id: int) -> int:
        process_book(book_id, DATA_DIR)
        return book_id

    @task
    def corpus(book_ids: list[int]) -> str:
        return str(build_corpus(list(book_ids), DATA_DIR))

    corpus(ingest_book.expand(book_id=sorted(CATALOG)))


ingest_gutenberg()
