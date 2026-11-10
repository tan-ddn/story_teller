from pathlib import Path

import pytest

from guardrails.censor import Verdict
from writers_room.models import StoryPlan
from writers_room.showrunner.run import NotApproved
from writers_room.smoke import draft


def plan(approved: bool = True) -> StoryPlan:
    return StoryPlan.model_validate({
        "title": "A Test Story",
        "premise": "Mira finds a lantern.",
        "rating": "teen",
        "chapters": [{
            "number": 1, "title": "The Lantern", "pov": "Mira", "story_time": "evening",
            "beats": ["Mira finds a lantern."],
        }],
        "characters": [{"name": "Mira", "attributes": {"occupation": "repairer"}, "arc": "helps"}],
        "style": {"tense": "past", "pov_rules": "close third", "tone": "warm"},
        "approved": approved,
    })


class Censor:
    def __init__(self, allowed: bool) -> None:
        self.allowed = allowed
        self.seen: list[str] = []

    def check(self, text: str, rating: str) -> Verdict:
        self.seen.append(text)
        return Verdict(self.allowed, () if self.allowed else ("blocked",))


def test_smoke_saves_only_screened_draft_and_records_no_continuity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompts: list[str] = []

    def call(system: str, user: str) -> tuple[str, int, int]:
        prompts.append(user)
        return "Mira found the lantern.", 50, 10

    monkeypatch.setattr("writers_room.smoke.writer_in_child", call)
    monkeypatch.setattr("writers_room.smoke.require_memory", lambda reserve, headroom: 60.0)
    censor = Censor(True)
    result = draft(plan(), censor, tmp_path / "output")
    assert result["status"] == "draft"
    assert result["continuity_checked"] is False
    assert "Mira: occupation = repairer" in prompts[0]
    assert censor.seen == ["Mira found the lantern."]
    assert (tmp_path / "output" / "ch01.md").read_text() == censor.seen[0]


def test_smoke_discards_blocked_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "writers_room.smoke.writer_in_child", lambda system, user: ("blocked prose", 1, 1)
    )
    monkeypatch.setattr("writers_room.smoke.require_memory", lambda reserve, headroom: 60.0)
    out = tmp_path / "output"
    result = draft(plan(), Censor(False), out)
    assert result["status"] == "censored"
    assert not (out / "ch01.md").exists()
    assert "blocked prose" not in (out / "result.json").read_text()


def test_smoke_requires_approval_and_empty_output(tmp_path: Path) -> None:
    with pytest.raises(NotApproved):
        draft(plan(False), Censor(True), tmp_path / "output")
    (tmp_path / "existing").write_text("preserve")
    with pytest.raises(FileExistsError):
        draft(plan(), Censor(True), tmp_path)


def test_low_memory_stops_before_calling_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def low_memory(reserve: float, headroom: float) -> float:
        raise RuntimeError("low memory")

    monkeypatch.setattr("writers_room.smoke.require_memory", low_memory)
    with pytest.raises(RuntimeError, match="low memory"):
        draft(plan(), Censor(True), tmp_path / "output")
    assert not (tmp_path / "output").exists()
