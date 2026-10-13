"""Strip Project Gutenberg boilerplate and normalise text."""

import re
import unicodedata

_START = re.compile(r"^\*{3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*$", re.I | re.M)
_END = re.compile(r"^\*{3}\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*$", re.I | re.M)
_TITLE = re.compile(r"^Title:\s*(.+)$", re.M)


class BoilerplateError(ValueError):
    pass


def extract_title(raw: str) -> str:
    match = _TITLE.search(raw)
    return match.group(1).strip() if match else "Unknown"


def strip_boilerplate(raw: str) -> str:
    """Return only the book body between the Gutenberg START and END markers."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    start, end = _START.search(text), _END.search(text)
    if not start or not end or end.start() <= start.end():
        raise BoilerplateError("Gutenberg START/END markers not found")
    body = text[start.end() : end.start()]
    body = unicodedata.normalize("NFC", body).replace("﻿", "")
    return body.strip("\n") + "\n"
