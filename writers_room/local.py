"""Run an approved Writers' Room plan locally, isolating each MLX model call."""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from guardrails.censor import DEFAULT_MODEL, Verdict, Warning, make_censor, overlapping_windows
from services.checker.scripty import Scripty
from services.extractor.archivist import ExtractionError, make_archivist
from services.extractor.schema import SceneExtraction
from store.bible import SQLiteBible
from writers_room.models import Rating, StoryPlan
from writers_room.ports import Censor, WorkerFailure
from writers_room.showrunner.run import Assignment, Services, Settings, Showrunner
from writers_room.smoke import WRITER, require_memory, watch_free_reserve

EXTRACTOR = WRITER  # baseline; not yet trained or evaluated as an Archivist
MODEL_HEADROOM_GIB = 8.0
RESERVE_GIB = 24.0
CENSOR_WINDOW_CHARS = 0


def screen_windows(censor: Censor, text: str, rating: Rating, size: int) -> Verdict:
    windows = overlapping_windows(text, size, size // 5) if size else [text]
    verdicts = [censor.check(window, rating) for window in windows]
    return Verdict(
        allowed=all(verdict.allowed for verdict in verdicts),
        reasons=tuple(reason for verdict in verdicts for reason in verdict.reasons),
        hard_ban=any(verdict.hard_ban for verdict in verdicts),
        unavailable=any(verdict.unavailable for verdict in verdicts),
        warnings=tuple(warning for verdict in verdicts for warning in verdict.warnings),
        labels=tuple(sorted({label for verdict in verdicts for label in verdict.labels})),
    )


def worker() -> None:
    from services.mlx_guard import memory_guard

    memory_guard(max_gb=36, min_free_percent=30)
    watch_free_reserve(float(os.environ.get("WRITERS_ROOM_RESERVE_GIB", "24")))
    request = json.load(sys.stdin)
    kind = request["kind"]
    if kind == "writer":
        from services.llm import text_call

        result: Any = text_call(WRITER)(request["system"], request["user"])
    elif kind == "extract":
        try:
            extraction = make_archivist(EXTRACTOR).extract(request["text"], request["known"])
            result = extraction.model_dump()
        except ExtractionError as exc:
            result = {"_error": str(exc)}
    elif kind == "censor":
        censor = make_censor(DEFAULT_MODEL)
        size = request.get("window_chars", 0)
        verdict = screen_windows(censor, request["text"], request["rating"], size)
        result = {
            "allowed": verdict.allowed,
            "reasons": list(verdict.reasons),
            "hard_ban": verdict.hard_ban,
            "unavailable": verdict.unavailable,
            "warnings": [vars(warning) for warning in verdict.warnings],
            "labels": list(verdict.labels),
        }
    else:
        raise ValueError("unknown worker kind")
    print(json.dumps(result))


def call_worker(kind: str, **kwargs: Any) -> Any:
    require_memory(RESERVE_GIB, MODEL_HEADROOM_GIB)
    try:
        process = subprocess.run(  # noqa: S603
            [sys.executable, "-m", "writers_room.local", "--worker"],
            input=json.dumps({"kind": kind, **kwargs}),
            capture_output=True,
            text=True,
            check=True,
            timeout=2100,
            env={**os.environ, "WRITERS_ROOM_RESERVE_GIB": str(RESERVE_GIB)},
        )
    except subprocess.CalledProcessError as exc:
        # Avoid logging model output or scene content from the child's traceback.
        reason = "worker exited"
        if exc.returncode == 3:
            reason = "memory guard stopped worker"
            diagnostics = (exc.stderr or "") + "\n" + (exc.stdout or "")
            patterns = (
                r"free-page guard: free memory [0-9.]+ GiB is below [0-9.]+ GiB required",
                r"free-page guard: cannot determine macOS free memory",
                r"memory guard: MLX [0-9.]+ GB > [0-9.]+ GB, stopping",
                r"memory guard: system free memory [0-9]+% < [0-9]+%, stopping",
            )
            for pattern in patterns:
                if match := re.search(pattern, diagnostics):
                    reason = match.group(0)
                    break
        raise WorkerFailure(kind, f"{reason} (status {exc.returncode})") from exc
    except subprocess.TimeoutExpired as exc:
        raise WorkerFailure(kind, "worker timed out") from exc
    result = json.loads(process.stdout)
    if isinstance(result, dict) and "_error" in result:
        raise WorkerFailure(kind, "extraction validation failed")
    return result


class LocalCensor:
    def __init__(self) -> None:
        self.lock = threading.Lock()

    def check(self, text: str, rating: Rating) -> Verdict:
        try:
            with self.lock:  # Showrunner screens scenes in parallel; run one local model at a time.
                result = call_worker(
                    "censor", text=text, rating=rating, window_chars=CENSOR_WINDOW_CHARS
                )
            return Verdict(
                allowed=result["allowed"],
                reasons=tuple(result["reasons"]),
                hard_ban=result["hard_ban"],
                unavailable=result["unavailable"],
                warnings=tuple(Warning(**item) for item in result["warnings"]),
                labels=tuple(result["labels"]),
            )
        except WorkerFailure as exc:
            return Verdict(False, (f"local Censor unavailable: {exc}",), unavailable=True)
        except (subprocess.SubprocessError, OSError, ValueError, KeyError, RuntimeError):
            return Verdict(False, ("local Censor unavailable",), unavailable=True)


class LocalArchivist:
    def extract(self, text: str, known: list[str]) -> SceneExtraction:
        return SceneExtraction.model_validate(call_worker("extract", text=text, known=known))


def main() -> None:
    global CENSOR_WINDOW_CHARS, MODEL_HEADROOM_GIB, RESERVE_GIB
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--chapter-gate", action="store_true")
    parser.add_argument("--model-headroom-gib", type=float, default=8)
    parser.add_argument("--reserve-gib", type=float, default=24)
    parser.add_argument(
        "--censor-window-chars",
        type=int,
        default=0,
        help="experimental overlapping windows; 0 screens full scenes",
    )
    args = parser.parse_args()
    if args.model_headroom_gib < 0 or args.reserve_gib < 0:
        parser.error("memory limits must be non-negative")
    if args.censor_window_chars < 0:
        parser.error("window size must be non-negative")
    MODEL_HEADROOM_GIB = args.model_headroom_gib
    RESERVE_GIB = args.reserve_gib
    CENSOR_WINDOW_CHARS = args.censor_window_chars
    plan = StoryPlan.model_validate_json(args.plan.read_text(encoding="utf-8"))
    if not plan.approved:
        raise ValueError("the author has not approved this plan")
    bible = SQLiteBible(args.out_dir / "bible.sqlite")
    bible.bind_plan(plan.model_dump_json())
    committed = bible.chapter_texts()
    for chapter in plan.chapters:
        path = args.out_dir / f"ch{chapter.number:02d}.md"
        if path.exists() and chapter.number not in committed:
            raise ValueError(f"{path} exists without a matching committed Bible chapter")
    services = Services(LocalCensor(), LocalArchivist(), Scripty(bible), bible)

    def writer(system: str, user: str) -> tuple[str, int, int]:
        text, tokens_in, tokens_out = call_worker("writer", system=system, user=user)
        return str(text), int(tokens_in), int(tokens_out)

    runner = Showrunner(
        plan,
        Assignment(WRITER),
        services,
        Settings(budget_usd=1, chapter_gate=args.chapter_gate),
        args.out_dir,
        lambda model: writer,
    )
    outcome = runner.run()
    while outcome.waiting_for:
        pending = outcome.waiting_for
        print(f"\nChapter {pending['chapter']} (Censor checked):\n{pending['text']}")
        print("\nExtracted claims (check their evidence before accepting):")
        print(json.dumps(pending["extractions"], ensure_ascii=False, indent=2))
        print("\nContinuity flags:")
        print(json.dumps(pending["flags"], ensure_ascii=False, indent=2))
        approved = input("Accept chapter? [y/N] ").strip().lower() == "y"
        outcome = runner.resume(approved)
    print(json.dumps([run.model_dump() for run in outcome.runs], indent=2))
    if any(run.status != "accepted" for run in outcome.runs):
        raise SystemExit(1)


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        worker()
    else:
        main()
