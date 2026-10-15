"""What the Writers' Room needs from the phase-1 services.

Archivist, Scripty and the Story Bible store are built elsewhere (Censor is guardrails.censor).
Showrunner only depends on these protocols, so it can be tested with fakes and wired to the real
services later.
"""

from dataclasses import dataclass
from typing import Protocol

from guardrails.censor import Verdict
from services.extractor.schema import SceneExtraction
from writers_room.models import Rating


@dataclass(frozen=True)
class Flag:
    """A continuity problem Scripty found in a new chapter."""

    message: str  # "Kael's eyes were grey in ch. 3, blue here"
    severity: str = "high"  # high | low ("possible, timeline unclear")


class Censor(Protocol):
    def check(self, text: str, rating: Rating) -> Verdict: ...


class Archivist(Protocol):
    def extract(self, text: str, known: list[str]) -> SceneExtraction: ...


class Scripty(Protocol):
    def check(self, chapter: int, scenes: list[SceneExtraction]) -> list[Flag]:
        """Flags for the new chapter's scenes, in order, against the written Bible so far."""
        ...


class Bible(Protocol):
    def facts_about(self, names: list[str]) -> list[str]:
        """Current facts for these entities, one line each, latest story time last."""
        ...

    def commit(self, chapter: int, scenes: list[SceneExtraction]) -> None:
        """Add an accepted chapter's facts, scene by scene, to the written Bible."""
        ...
