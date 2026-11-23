"""Scene extraction with schema and evidence validation."""

import json
from collections.abc import Callable
from pathlib import Path

from services.extractor.schema import SceneExtraction, check_evidence
from services.llm import parse_call

GUIDELINES = Path(__file__).resolve().parents[2] / "pipelines/labeling/guidelines.md"
Call = Callable[[str, str], tuple[SceneExtraction, int, int]]


class ExtractionError(ValueError):
    """The model could not produce a grounded scene extraction."""


class Archivist:
    def __init__(self, call: Call, attempts: int = 2) -> None:
        if attempts < 1:
            raise ValueError("attempts must be positive")
        self.call = call
        self.attempts = attempts
        self.system = GUIDELINES.read_text(encoding="utf-8")

    def extract(self, text: str, known: list[str]) -> SceneExtraction:
        if not text.strip():
            raise ValueError("scene must not be empty")
        user = (
            "Known character names (not IDs): "
            + json.dumps(known, ensure_ascii=False)
            + "\nDo not invent known_id values; leave them null.\n<scene>\n"
            + text
            + "\n</scene>"
        )
        last_error = ""
        for _ in range(self.attempts):
            try:
                extraction, _, _ = self.call(self.system, user)
                bad = check_evidence(extraction, text)
                if not bad:
                    return extraction
                last_error = "evidence not found in scene"
                user += "\nSome evidence was not verbatim. Copy every quote exactly from the scene."
            except ValueError:
                last_error = "invalid structured extraction"
                user += "\nReturn a valid extraction matching the required schema."
        raise ExtractionError(last_error)


def make_archivist(model: str) -> Archivist:
    """Construct an Archivist using the existing provider interface."""
    return Archivist(parse_call(model, SceneExtraction, name="archivist"))
