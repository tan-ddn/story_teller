from pathlib import Path

import pytest

from guardrails.censor import Verdict
from services.checker.scripty import Scripty
from services.extractor.archivist import Archivist, ExtractionError
from services.extractor.schema import Event, SceneExtraction, TimeCue
from store.bible import SQLiteBible
from writers_room.models import StoryPlan
from writers_room.showrunner.run import Assignment, Services, Settings, Showrunner


def extraction(evidence: str = "Mira has a red coat.") -> SceneExtraction:
    return SceneExtraction.model_validate(
        {
            "entities": [{"id": "e1", "type": "character", "name": "Mira", "evidence": "Mira"}],
            "attributes": [
                {"entity": "e1", "key": "appearance", "value": "red coat", "evidence": evidence},
                {
                    "entity": "e1",
                    "key": "trait",
                    "value": "brave",
                    "hedged": True,
                    "evidence": "Mira",
                },
                {
                    "entity": "e1",
                    "key": "occupation",
                    "value": "sailor",
                    "mode": "flashback",
                    "evidence": "Mira",
                },
            ],
            "events": [],
            "relations": [],
            "time": {"mode": "main"},
        }
    )


def test_bible_persists_and_keeps_uncertain_facts_out_of_current_context(tmp_path: Path) -> None:
    path = tmp_path / "story.sqlite"
    bible = SQLiteBible(path)
    bible.commit(1, [extraction()])
    bible.commit(1, [extraction()])  # retry is idempotent
    assert SQLiteBible(path).facts_about(["MIRA"]) == ["Mira: appearance = red coat (chapter 1)"]
    assert SQLiteBible(path).facts_about(["Someone else"]) == []
    with pytest.raises(ValueError, match="different committed facts"):
        bible.commit(1, [extraction("different evidence")])


def test_archivist_retries_bad_evidence_and_rejects_repeated_failure() -> None:
    calls: list[str] = []

    def call(system: str, user: str) -> tuple[SceneExtraction, int, int]:
        calls.append(user)
        return extraction("not in scene" if len(calls) == 1 else "Mira has a red coat."), 1, 1

    result = Archivist(call).extract("Mira has a red coat.", ["Mira"])
    assert result.attributes[0].value == "red coat"
    assert len(calls) == 2 and "not verbatim" in calls[1]
    with pytest.raises(ExtractionError):
        Archivist(lambda system, user: (extraction("not in scene"), 1, 1)).extract(
            "Mira has a red coat.", ["Mira"]
        )


def test_showrunner_feeds_committed_first_chapter_into_second(tmp_path: Path) -> None:
    plan = StoryPlan.model_validate(
        {
            "title": "Coat",
            "premise": "Mira travels.",
            "rating": "teen",
            "approved": True,
            "chapters": [
                {
                    "number": n,
                    "title": f"Part {n}",
                    "pov": "Mira",
                    "story_time": "today",
                    "beats": ["Mira travels."],
                }
                for n in (1, 2)
            ],
            "characters": [{"name": "Mira", "arc": "travels"}],
            "style": {"tense": "past", "pov_rules": "third person", "tone": "warm"},
        }
    )
    prompts: list[str] = []

    def writer(system: str, user: str) -> tuple[str, int, int]:
        prompts.append(user)
        return "Mira has a red coat.", 1, 1

    class Censor:
        def check(self, text: str, rating: str) -> Verdict:
            return Verdict(True)

    class Scripty:
        def check(self, chapter: int, scenes: list[SceneExtraction]) -> list:
            return []

    bible_path = tmp_path / "bible.sqlite"
    services = Services(
        Censor(),
        Archivist(lambda system, user: (extraction(), 1, 1)),
        Scripty(),
        SQLiteBible(bible_path),
    )
    runner = Showrunner(
        plan,
        Assignment("local:test"),
        services,
        Settings(100),
        tmp_path / "chapters",
        lambda model: writer,
    )
    assert [run.status for run in runner.run().runs] == ["accepted", "accepted"]
    assert "Mira: appearance = red coat (chapter 1)" in prompts[1]
    assert SQLiteBible(bible_path).facts_about(["Mira"])[0].endswith("(chapter 1)")
    chapter_one = tmp_path / "chapters" / "ch01.md"
    chapter_one.unlink()
    assert runner.run().runs == []
    assert chapter_one.read_text() == "Mira has a red coat."


def test_scripty_reports_possible_change_and_exempts_dreams(tmp_path: Path) -> None:
    bible = SQLiteBible(tmp_path / "bible.sqlite")
    bible.commit(1, [extraction()])
    other = extraction().model_copy(deep=True)
    other.attributes[0].value = "blue coat"
    flags = Scripty(bible).check(2, [other])
    assert len(flags) == 1 and flags[0].severity == "low"
    assert "red coat" in flags[0].message and "blue coat" in flags[0].message
    other.time.mode = "dream"
    assert Scripty(bible).check(2, [other]) == []


def test_scripty_flags_possible_event_after_death_without_assuming_timeline(tmp_path: Path) -> None:
    before = extraction().model_copy(deep=True)
    before.attributes[0].key = "life_status"
    before.attributes[0].value = "dead"
    bible = SQLiteBible(tmp_path / "bible.sqlite")
    bible.commit(1, [before])
    after = extraction().model_copy(deep=True)
    after.events = [Event(description="Mira arrives.", participants=["e1"], evidence="Mira")]
    flags = Scripty(bible).check(2, [after])
    assert any("after a death claim" in flag.message and flag.severity == "low" for flag in flags)


def test_scripty_does_not_flag_event_before_explicit_death_date(tmp_path: Path) -> None:
    death = extraction().model_copy(deep=True)
    death.attributes[0].key = "life_status"
    death.attributes[0].value = "dead"
    death.time.cues = [TimeCue(kind="absolute", value="2025-02-01", evidence="Mira")]
    bible = SQLiteBible(tmp_path / "bible.sqlite")
    bible.commit(1, [death])
    earlier = extraction().model_copy(deep=True)
    earlier.attributes = []
    earlier.events = [Event(description="Mira arrives.", participants=["e1"], evidence="Mira")]
    earlier.time.cues = [TimeCue(kind="absolute", value="2025-01-01", evidence="Mira")]
    assert Scripty(bible).check(2, [earlier]) == []


def test_atomic_chapter_record_and_plan_binding(tmp_path: Path) -> None:
    bible = SQLiteBible(tmp_path / "bible.sqlite")
    bible.bind_plan('{"title": "one"}')
    with pytest.raises(ValueError, match="different plan"):
        bible.bind_plan('{"title": "two"}')
    bible.commit_chapter(1, [extraction()], "Mira has a red coat.")
    assert SQLiteBible(bible.path).chapter_texts() == {1: "Mira has a red coat."}
    with pytest.raises(ValueError, match="different committed text"):
        bible.commit_chapter(1, [extraction()], "changed")


def test_unambiguous_alias_lookup(tmp_path: Path) -> None:
    scene = extraction()
    scene.entities[0].aliases = ["M"]
    bible = SQLiteBible(tmp_path / "bible.sqlite")
    bible.commit(1, [scene])
    assert bible.facts_about(["M"]) == ["Mira: appearance = red coat (chapter 1)"]
