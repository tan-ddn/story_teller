from pathlib import Path
from typing import Any

import pytest

from guardrails.censor import Verdict
from services.extractor.schema import SceneExtraction, SceneTime
from services.llm import TextCall
from writers_room.brainstormer.plan import PlanRejected, brainstorm
from writers_room.context.builder import build
from writers_room.models import StoryPlan
from writers_room.ports import Flag
from writers_room.showrunner.run import (
    Assignment,
    NotApproved,
    Services,
    Settings,
    Showrunner,
)

EMPTY = SceneExtraction(
    entities=[], attributes=[], events=[], relations=[], time=SceneTime(mode="main")
)


def plan_dict(chapters: int = 2) -> dict[str, Any]:
    return {
        "title": "The Ashborn",
        "premise": "A boy finds a burning feather.",
        "rating": "teen",
        "chapters": [
            {
                "number": i,
                "title": f"Chapter {i}",
                "pov": "Kael" if i % 2 else "Brenna",
                "story_time": "spring",
                "beats": [f"Kael meets Brenna ({i})"],
            }
            for i in range(1, chapters + 1)
        ],
        "characters": [
            {
                "name": "Kael",
                "aliases": ["the Ashborn"],
                "attributes": {"appearance": "grey eyes"},
                "arc": "a",
            },
            {"name": "Brenna", "arc": "b"},
            {"name": "Orrin", "arc": "c"},
        ],
        "style": {
            "tense": "past",
            "pov_rules": "close third",
            "tone": "warm",
            "voices": {"Kael": "terse"},
        },
    }


def make_plan(chapters: int = 2, approved: bool = True) -> StoryPlan:
    return StoryPlan.model_validate({**plan_dict(chapters), "approved": approved})


class FakeCensor:
    def __init__(self, banned: str = "BLOOD") -> None:
        self.banned = banned

    def check(self, text: str, rating: str) -> Verdict:
        if self.banned in text:
            return Verdict(False, ("violence 3 > teen limit 2",))
        return Verdict(True)


class FakeArchivist:
    def extract(self, text: str, known: list[str]) -> SceneExtraction:
        return EMPTY


class FakeScripty:
    def __init__(self, flag_if: str = "BLUE EYES") -> None:
        self.flag_if = flag_if
        self.text = ""  # every scene Archivist saw for the current chapter
        self.scene_counts: list[int] = []

    def check(self, chapter: int, scenes: list[SceneExtraction]) -> list[Flag]:
        self.scene_counts.append(len(scenes))
        flagged, self.text = self.flag_if in self.text, ""
        return [Flag("Kael's eyes were grey")] if flagged else []


class Archivist(FakeArchivist):
    """Passes the chapter text through to FakeScripty, which only sees extractions."""

    def __init__(self, scripty: FakeScripty) -> None:
        self.scripty = scripty

    def extract(self, text: str, known: list[str]) -> SceneExtraction:
        self.scripty.text += text
        return EMPTY


class FakeBible:
    def __init__(self, facts: list[str] | None = None) -> None:
        self.facts = facts or []
        self.committed: list[int] = []

    def facts_about(self, names: list[str]) -> list[str]:
        return self.facts

    def commit(self, chapter: int, scenes: list[SceneExtraction]) -> None:
        self.committed.append(chapter)


# --- models ------------------------------------------------------------------------------


def test_plan_rejects_unknown_pov_and_gaps() -> None:
    bad = plan_dict()
    bad["chapters"][0]["pov"] = "Nobody"
    with pytest.raises(ValueError, match="POV"):
        StoryPlan.model_validate(bad)
    gap = plan_dict()
    gap["chapters"][1]["number"] = 3
    with pytest.raises(ValueError, match="numbered"):
        StoryPlan.model_validate(gap)


# --- brainstormer ------------------------------------------------------------------------


def test_brainstorm_retries_with_problems_and_never_self_approves() -> None:
    prompts: list[str] = []
    answers = [make_plan(chapters=1), make_plan(chapters=2, approved=True)]

    def call(system: str, user: str) -> tuple[StoryPlan, int, int]:
        prompts.append(user)
        return answers[len(prompts) - 1], 1000, 2000

    result = brainstorm(call, "claude-opus-5", "premise", 2, "teen", FakeCensor())
    assert "has 1 chapters, not 2" in prompts[1]
    assert result.plan.approved is False
    assert (result.input_tokens, result.output_tokens) == (2000, 4000)


def test_brainstorm_gives_up_on_beats_over_the_rating() -> None:
    def call(system: str, user: str) -> tuple[StoryPlan, int, int]:
        return make_plan(), 1, 1

    with pytest.raises(PlanRejected, match="not allowed"):
        brainstorm(call, "m", "premise", 2, "teen", FakeCensor(banned="Kael meets"))


# --- context builder ---------------------------------------------------------------------


def test_context_includes_named_characters_facts_and_previous_tail() -> None:
    plan = make_plan()
    bible = FakeBible(["Kael: eyes grey (ch. 1)"])
    prompt = build(plan, plan.chapters[0], bible, "x" * 100 + "THE END", budget=4000)
    assert "Kael: eyes grey" in prompt
    assert "Kael: terse" in prompt
    assert prompt.rstrip().endswith("THE END\n</previous>")


def test_no_memory_strategy_leaves_out_bible() -> None:
    plan = make_plan()
    prompt = build(plan, plan.chapters[0], FakeBible(["Kael: eyes grey"]), "", 4000, "no_memory")
    assert "Story Bible" not in prompt


def test_context_respects_budget() -> None:
    plan = make_plan()
    bible = FakeBible([f"fact {i} " + "y" * 200 for i in range(100)])
    prompt = build(plan, plan.chapters[0], bible, "z" * 50_000, budget=1000)
    assert len(prompt) // 4 <= 1000


# --- showrunner --------------------------------------------------------------------------


def scripted(*texts: str) -> tuple[TextCall, list[str]]:
    """A model that returns `texts` in turn and records the user prompts."""
    prompts: list[str] = []

    def call(system: str, user: str) -> tuple[str, int, int]:
        prompts.append(user)
        return texts[min(len(prompts), len(texts)) - 1], 100, 1000

    return call, prompts


def showrunner(
    tmp_path: Path,
    call: TextCall,
    budget: float = 100.0,
    plan: StoryPlan | None = None,
    censor: Any = None,
    gate: bool = False,
) -> tuple[Showrunner, FakeBible, FakeScripty]:
    scripty = FakeScripty()
    bible = FakeBible()
    services = Services(censor or FakeCensor(), Archivist(scripty), scripty, bible)
    sr = Showrunner(
        plan or make_plan(),
        Assignment(default="claude-opus-5"),
        services,
        Settings(budget_usd=budget, chapter_gate=gate),
        tmp_path,
        lambda model: call,
    )
    return sr, bible, scripty


def statuses(runs: list[Any]) -> list[str]:
    return [r.status for r in runs]


def test_unapproved_plan_is_refused(tmp_path: Path) -> None:
    call, _ = scripted("text")
    with pytest.raises(NotApproved):
        showrunner(tmp_path, call, plan=make_plan(approved=False))


def test_clean_chapters_are_accepted_and_committed(tmp_path: Path) -> None:
    call, _ = scripted("A quiet morning.")
    sr, bible, _ = showrunner(tmp_path, call)
    assert statuses(sr.run().runs) == ["accepted", "accepted"]
    assert bible.committed == [1, 2]
    assert (tmp_path / "ch02.md").read_text() == "A quiet morning."
    assert len((tmp_path / "ledger.jsonl").read_text().splitlines()) == 2


def test_chapters_are_checked_scene_by_scene(tmp_path: Path) -> None:
    scene = " ".join(["word"] * 30)
    call, _ = scripted(f"{scene}\n\n* * *\n\n{scene}\n\n* * *\n\n{scene}")
    sr, _, scripty = showrunner(tmp_path, call, plan=make_plan(chapters=1))
    [run] = sr.run().runs
    assert run.scenes == 3 and scripty.scene_counts == [3]


def test_long_novel_does_not_hit_the_graph_step_limit(tmp_path: Path) -> None:
    call, _ = scripted("Fine.")
    sr, _, _ = showrunner(tmp_path, call, plan=make_plan(chapters=30))
    assert statuses(sr.run().runs) == ["accepted"] * 30


def test_flags_go_to_the_editor_then_accept(tmp_path: Path) -> None:
    call, prompts = scripted("He had BLUE EYES.", "He had grey eyes.")
    sr, _, _ = showrunner(tmp_path, call, plan=make_plan(chapters=1))
    [run] = sr.run().runs
    assert run.status == "accepted" and run.revisions == 1
    assert "- Kael's eyes were grey" in prompts[1]
    assert (tmp_path / "ch01.md").read_text() == "He had grey eyes."


def test_persistent_flags_escalate_without_storing(tmp_path: Path) -> None:
    call, _ = scripted("He had BLUE EYES.")
    sr, bible, _ = showrunner(tmp_path, call)
    runs = sr.run().runs
    assert statuses(runs) == ["flagged"]  # stops at chapter 1
    assert runs[0].revisions == 2
    assert bible.committed == [] and not (tmp_path / "ch01.md").exists()


def test_censor_tone_down_once_then_discard(tmp_path: Path) -> None:
    call, prompts = scripted("BLOOD everywhere.", "Still BLOOD.")
    sr, bible, _ = showrunner(tmp_path, call)
    [run] = sr.run().runs
    assert run.status == "censored"
    assert "Tone down for a teen audience" in prompts[1]
    assert len(prompts) == 2 and bible.committed == []
    assert not (tmp_path / "ch01.md").exists()


def test_unavailable_censor_fails_closed_without_editing(tmp_path: Path) -> None:
    class Down:
        def check(self, text: str, rating: str) -> Verdict:
            return Verdict(False, ("censor unavailable: TimeoutError",), unavailable=True)

    call, prompts = scripted("A quiet morning.")
    sr, _, _ = showrunner(tmp_path, call, censor=Down())
    assert statuses(sr.run().runs) == ["censored"]
    assert len(prompts) == 1


def test_chapter_gate_interrupts_and_resumes(tmp_path: Path) -> None:
    call, _ = scripted("A quiet morning.")
    sr, bible, _ = showrunner(tmp_path, call, gate=True)
    first = sr.run()
    assert first.waiting_for == {"chapter": 1, "text": "A quiet morning."}
    assert bible.committed == []
    second = sr.resume(True)
    assert second.waiting_for is not None and second.waiting_for["chapter"] == 2
    final = sr.resume(False)
    assert statuses(final.runs) == ["accepted", "rejected_by_author"]
    assert bible.committed == [1]


def test_budget_stops_the_run_and_rerun_resumes(tmp_path: Path) -> None:
    call, _ = scripted("Fine.")
    one_chapter = (100 * 5 + 1000 * 25) / 1_000_000  # claude-opus-5 prices
    sr, _, _ = showrunner(tmp_path, call, budget=one_chapter)
    assert statuses(sr.run().runs) == ["accepted", "over_budget"]

    sr2, bible, _ = showrunner(tmp_path, call, budget=one_chapter * 3)
    assert sr2.spent == pytest.approx(one_chapter, abs=1e-4)
    assert statuses(sr2.run().runs) == ["accepted"]
    assert bible.committed == [2]


def test_writer_assignment_by_pov() -> None:
    a = Assignment(default="claude-opus-5", by_pov={"Brenna": "gpt-6.1-sol"})
    plan = make_plan()
    assert [a.writer(c) for c in plan.chapters] == ["claude-opus-5", "gpt-6.1-sol"]
