"""Showrunner: a LangGraph state machine that writes an approved plan chapter by chapter.

    pick -> draft -> check -> gate -> commit -> record -> pick ...
                       ^  |
                       revise

- draft: Context Builder prompt -> writer -> screen. revise: Editor -> screen.
- screen (inside draft/revise): Censor checks each scene of the chapter separately, the unit the
  rubric is tuned on, so a failure names the scene. If a scene is over the rating, the Editor gets
  one "tone down" edit for the scenes named. A hard ban, an unavailable Censor or a second failure
  discards the text and the run stops ("censored"). Text only enters graph state after Censor
  allowed every scene, so it is never checkpointed otherwise.
- check: the chapter is split into scenes (same splitter as ingest), each scene goes through
  Archivist, then Scripty checks the chapter's scenes. High-severity flags go to revise, up to
  `max_revisions` times, then the run stops ("flagged"). Low-severity flags are only recorded.
- gate: with `chapter_gate` on, the graph interrupts with the chapter for the author; resume
  with True to accept or False to stop ("rejected_by_author").
- The run stops when the cost budget is used up ("over_budget").

Accepted chapters are written to `out_dir/chNN.md`, and one ChapterRun per chapter attempt is
appended to `out_dir/ledger.jsonl` (no story text). A new run skips chapters already written,
so after an escalation the author fixes the plan or chapter and runs again.
"""

import operator
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel

from guardrails.censor import Verdict
from pipelines.ingest.split import split_scenes
from services.extractor.schema import SceneExtraction
from services.llm import TextCall, cost
from writers_room.context.builder import Strategy, build, system_prompt
from writers_room.editor.revise import revise
from writers_room.models import ChapterPlan, StoryPlan
from writers_room.ports import Archivist, Bible, Censor, Scripty

CENSOR_WORKERS = 4  # parallel Censor calls per chapter

Status = Literal["accepted", "censored", "flagged", "rejected_by_author", "over_budget"]


class NotApproved(Exception):
    pass


@dataclass
class Assignment:
    """Which model writes which chapter: by POV character, else the default."""

    default: str
    by_pov: dict[str, str] = field(default_factory=dict)
    editor: str | None = None  # defaults to the chapter's writer

    def writer(self, chapter: ChapterPlan) -> str:
        return self.by_pov.get(chapter.pov, self.default)


@dataclass
class Services:
    censor: Censor
    archivist: Archivist
    scripty: Scripty
    bible: Bible


@dataclass
class Settings:
    budget_usd: float
    max_revisions: int = 2
    context_tokens: int = 8000
    strategy: Strategy = "bible_v1"
    chapter_gate: bool = False  # interrupt for the author's approval of every chapter


class ChapterRun(BaseModel):
    chapter: int
    writer: str
    editor: str
    status: Status | None = None
    revisions: int = 0
    scenes: int = 0
    flags: list[str] = []  # every flag seen, including low-severity ones
    censor_reasons: list[str] = []
    warnings: list[str] = []  # Censor warning tags: above the usual level for the rating
    labels: list[str] = []  # Censor content warnings for chapters and exports
    words: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0


class State(TypedDict, total=False):
    run: ChapterRun | None  # the chapter in progress; None when nothing is left to write
    text: str  # current chapter text, only ever Censor-approved
    toned_down: bool
    extractions: list[SceneExtraction]
    blocking: list[str]
    started: float
    runs: Annotated[list[ChapterRun], operator.add]  # finished chapter attempts


def _current(state: State) -> ChapterRun:
    run = state["run"]
    if run is None:  # routing only enters these nodes with a chapter in progress
        raise RuntimeError("no chapter in progress")
    return run


@dataclass
class Outcome:
    runs: list[ChapterRun]
    waiting_for: dict[str, Any] | None = None  # set when paused at the chapter gate


class Showrunner:
    def __init__(
        self,
        plan: StoryPlan,
        assignment: Assignment,
        services: Services,
        settings: Settings,
        out_dir: Path,
        calls: Callable[[str], TextCall],
        checkpointer: BaseCheckpointSaver[Any] | None = None,
    ) -> None:
        if not plan.approved:
            raise NotApproved("the author has not approved this plan")
        self.plan, self.assignment, self.services = plan, assignment, services
        self.settings, self.out_dir = settings, out_dir
        self._calls = calls
        self._cache: dict[str, TextCall] = {}
        self.spent = self._spent_so_far()
        self.graph = self._graph().compile(checkpointer=checkpointer or InMemorySaver())
        self.thread = ""

    # --- running -------------------------------------------------------------------------

    def run(self) -> Outcome:
        """Write every chapter not yet written; stop at the first one that is not accepted."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.thread = uuid.uuid4().hex
        return self._invoke({"runs": []})

    def resume(self, approved: bool) -> Outcome:
        """Answer the chapter gate the last run paused at."""
        return self._invoke(Command(resume=approved))

    def _invoke(self, payload: Any) -> Outcome:
        # Each chapter takes up to 5 steps plus 2 per revision; LangGraph's default limit is 25.
        steps = len(self.plan.chapters) * (5 + 2 * self.settings.max_revisions) + 5
        config: Any = {"configurable": {"thread_id": self.thread}, "recursion_limit": steps}
        state = self.graph.invoke(payload, config)
        pending = state.get("__interrupt__")
        return Outcome(state.get("runs", []), pending[0].value if pending else None)

    # --- helpers -------------------------------------------------------------------------

    def call(self, model: str) -> TextCall:
        if model not in self._cache:
            self._cache[model] = self._calls(model)
        return self._cache[model]

    def chapter_path(self, number: int) -> Path:
        return self.out_dir / f"ch{number:02d}.md"

    def _spent_so_far(self) -> float:
        ledger = self.out_dir / "ledger.jsonl"
        if not ledger.exists():
            return 0.0
        lines = ledger.read_text(encoding="utf-8").splitlines()
        return sum(ChapterRun.model_validate_json(line).cost_usd for line in lines)

    def _over_budget(self) -> bool:
        return self.spent >= self.settings.budget_usd

    def _spend(self, run: ChapterRun, model: str, tokens_in: int, tokens_out: int) -> None:
        run.input_tokens += tokens_in
        run.output_tokens += tokens_out
        usd = cost(model, tokens_in, tokens_out)
        run.cost_usd = round(run.cost_usd + usd, 4)
        self.spent += usd

    def _chapter(self, run: ChapterRun) -> ChapterPlan:
        return self.plan.chapters[run.chapter - 1]

    def _previous(self, number: int) -> str:
        path = self.chapter_path(number - 1)
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def _censor_scenes(self, text: str) -> list[Verdict]:
        """One verdict per scene, in order (scenes are checked in parallel)."""
        scenes = [piece.text for piece in split_scenes(text)] or [text]
        rating = self.plan.rating
        with ThreadPoolExecutor(min(len(scenes), CENSOR_WORKERS)) as pool:
            return list(pool.map(lambda scene: self.services.censor.check(scene, rating), scenes))

    def _screen(self, run: ChapterRun, text: str, toned_down: bool) -> State:
        """Censor every scene, toning the failing ones down once per chapter; discard the
        chapter if any scene still fails."""
        rating = self.plan.rating
        while True:
            verdicts = self._censor_scenes(text)
            failed = [(n, v) for n, v in enumerate(verdicts, 1) if not v.allowed]
            if not failed:
                run.warnings = [
                    f"Scene {n}: {w.message}" for n, v in enumerate(verdicts, 1) for w in v.warnings
                ]
                run.labels = sorted({label for v in verdicts for label in v.labels})
                return {"run": run, "text": text, "toned_down": toned_down}
            run.censor_reasons += [f"Scene {n}: {r}" for n, v in failed for r in v.reasons]
            stop = any(v.hard_ban or v.unavailable for _, v in failed)
            if stop or toned_down or self._over_budget():
                run.status = "censored"  # the text is dropped here, never stored
                return {"run": run, "text": "", "toned_down": toned_down}
            toned_down = True
            issues = [
                f"Scene {n}: tone down for a {rating} audience. {r}"
                for n, v in failed
                for r in v.reasons
            ]
            text, t_in, t_out = revise(self.call(run.editor), rating, text, issues)
            self._spend(run, run.editor, t_in, t_out)

    # --- nodes ---------------------------------------------------------------------------

    def pick(self, state: State) -> State:
        todo = (c for c in self.plan.chapters if not self.chapter_path(c.number).exists())
        chapter = next(todo, None)
        if chapter is None:
            return {"run": None}
        writer = self.assignment.writer(chapter)
        editor = self.assignment.editor or writer
        run = ChapterRun(chapter=chapter.number, writer=writer, editor=editor)
        if self._over_budget():
            run.status = "over_budget"
        return {"run": run, "text": "", "toned_down": False, "started": time.monotonic()}

    def draft(self, state: State) -> State:
        run = _current(state)
        chapter = self._chapter(run)
        user = build(
            self.plan,
            chapter,
            self.services.bible,
            self._previous(chapter.number),
            self.settings.context_tokens,
            self.settings.strategy,
        )
        text, t_in, t_out = self.call(run.writer)(system_prompt(self.plan), user)
        self._spend(run, run.writer, t_in, t_out)
        return self._screen(run, text, toned_down=False)

    def revise(self, state: State) -> State:
        run = _current(state)
        run.revisions += 1
        rating = self.plan.rating
        text, t_in, t_out = revise(self.call(run.editor), rating, state["text"], state["blocking"])
        self._spend(run, run.editor, t_in, t_out)
        return self._screen(run, text, state["toned_down"])

    def check(self, state: State) -> State:
        run = _current(state)
        s = self.services
        known = [c.name for c in self.plan.characters]
        scenes = [piece.text for piece in split_scenes(state["text"])]
        extractions = [s.archivist.extract(scene, known) for scene in scenes]
        flags = s.scripty.check(run.chapter, extractions)
        run.scenes = len(scenes)
        run.flags += [f.message for f in flags]
        blocking = [f.message for f in flags if f.severity == "high"]
        if blocking and run.revisions >= self.settings.max_revisions:
            run.status = "flagged"
        elif blocking and self._over_budget():
            run.status = "over_budget"
        return {"run": run, "extractions": extractions, "blocking": blocking}

    def gate(self, state: State) -> State:
        run = _current(state)
        if self.settings.chapter_gate:
            approved = interrupt({"chapter": run.chapter, "text": state["text"]})
            if not approved:
                run.status = "rejected_by_author"
        return {"run": run}

    def commit(self, state: State) -> State:
        run = _current(state)
        self.services.bible.commit(run.chapter, state["extractions"])
        self.chapter_path(run.chapter).write_text(state["text"], encoding="utf-8")
        run.words = len(state["text"].split())
        run.status = "accepted"
        return {"run": run}

    def record(self, state: State) -> State:
        run = _current(state)
        run.seconds = round(time.monotonic() - state["started"], 1)
        with (self.out_dir / "ledger.jsonl").open("a", encoding="utf-8") as f:
            f.write(run.model_dump_json() + "\n")
        return {"runs": [run]}

    # --- graph ---------------------------------------------------------------------------

    def _graph(self) -> StateGraph[State]:
        def stopped(state: State) -> bool:
            run = state["run"]
            return run is not None and run.status is not None

        def after_pick(state: State) -> str:
            if state["run"] is None:
                return END
            return "record" if stopped(state) else "draft"

        def after_screen(state: State) -> str:
            return "record" if stopped(state) else "check"

        def after_check(state: State) -> str:
            if stopped(state):
                return "record"
            return "revise" if state["blocking"] else "gate"

        def after_gate(state: State) -> str:
            return "record" if stopped(state) else "commit"

        def after_record(state: State) -> str:
            run = state["run"]
            return "pick" if run is not None and run.status == "accepted" else END

        g = StateGraph(State)
        for name in ("pick", "draft", "revise", "check", "gate", "commit", "record"):
            g.add_node(name, getattr(self, name))
        g.add_edge(START, "pick")
        g.add_conditional_edges("pick", after_pick, ["draft", "record", END])
        g.add_conditional_edges("draft", after_screen, ["check", "record"])
        g.add_conditional_edges("revise", after_screen, ["check", "record"])
        g.add_conditional_edges("check", after_check, ["revise", "gate", "record"])
        g.add_conditional_edges("gate", after_gate, ["commit", "record"])
        g.add_edge("commit", "record")
        g.add_conditional_edges("record", after_record, ["pick", END])
        return g
