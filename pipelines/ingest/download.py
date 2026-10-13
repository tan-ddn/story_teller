"""Download plain-text books from Project Gutenberg, politely and with a local cache."""

import time
import urllib.request
from pathlib import Path

USER_AGENT = "story_teller-ingest (+https://github.com/tan-ddn/story_teller)"
URL_TEMPLATE = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
RETRIES = 4
DELAY_SECONDS = 2.0  # between requests, per Gutenberg's guidance for automated access


def source_url(book_id: int) -> str:
    return URL_TEMPLATE.format(id=book_id)


def download(book_id: int, raw_dir: Path) -> Path:
    """Return the path to the raw text, downloading it unless it is already cached."""
    path = raw_dir / f"pg{book_id}.txt"
    if path.exists():
        return path
    raw_dir.mkdir(parents=True, exist_ok=True)
    url = source_url(book_id)  # fixed https URL, never user input
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310  # nosec B310
                data = response.read()
            break
        except OSError:
            if attempt == RETRIES:
                raise
            time.sleep(DELAY_SECONDS * 2**attempt)
    tmp = path.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.rename(path)
    time.sleep(DELAY_SECONDS)
    return path
