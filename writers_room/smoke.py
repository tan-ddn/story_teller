"""Draft one chapter with local models; this is not a continuity-checked Showrunner run."""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from guardrails.censor import DEFAULT_MODEL, Verdict, make_censor
from pipelines.ingest.split import split_scenes
from services.extractor.schema import SceneExtraction
from services.llm import text_call
from writers_room.context.builder import build, system_prompt
from writers_room.models import Rating, StoryPlan
from writers_room.ports import Censor
from writers_room.showrunner.run import NotApproved

WRITER = "mlx:Qwen3.6-27B-4bit"
GIB = 1024**3


def require_memory(reserve_gib: float, model_headroom_gib: float) -> float:
    """Fail closed unless macOS reports enough free pages for reserve and model headroom."""
    if reserve_gib < 0 or model_headroom_gib < 0:
        raise ValueError("memory limits must be non-negative")
    try:
        output = subprocess.check_output(["vm_stat"], text=True)  # noqa: S603,S607
        page = re.search(r"page size of (\d+) bytes", output)
        free = re.search(r"Pages free:\s+(\d+)", output)
        if page is None or free is None:
            raise ValueError("missing free-page statistics")
        available = int(page.group(1)) * int(free.group(1)) / GIB
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise RuntimeError("cannot determine macOS free memory; refusing to start") from exc
    required = reserve_gib + model_headroom_gib
    if available < required:
        raise RuntimeError(f"free memory {available:.1f} GiB is below {required:.1f} GiB required")
    return available


def writer_in_child(system: str, prompt: str) -> tuple[str, int, int]:
    """Transfer unscreened prose in a pipe and release writer memory when the child exits."""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "writers_room.smoke", "--writer-worker"],
        input=json.dumps({"system": system, "prompt": prompt}),
        capture_output=True,
        text=True,
        timeout=2100,
        check=True,
    )
    answer = json.loads(result.stdout)
    return str(answer[0]), int(answer[1]), int(answer[2])


def writer_worker() -> None:
    from services.mlx_guard import memory_guard

    memory_guard(max_gb=36, min_free_percent=30)
    request = json.load(sys.stdin)
    result = text_call(WRITER)(request["system"], request["prompt"])
    print(json.dumps(result))


class PlannedFacts:
    """Planned character attributes only; no written-story memory is available yet."""

    def __init__(self, plan: StoryPlan) -> None:
        self.plan = plan

    def facts_about(self, names: list[str]) -> list[str]:
        return [
            f"{character.name}: {key} = {value}"
            for character in self.plan.characters
            if character.name in names
            for key, value in character.attributes.items()
        ]

    def commit(self, chapter: int, scenes: list[SceneExtraction]) -> None:
        raise NotImplementedError("the smoke test has no written Story Bible")


class LazyCensor:
    """Construct the MLX Censor only after the preflight memory check."""

    def __init__(self) -> None:
        self.censor: Censor | None = None

    def check(self, text: str, rating: Rating) -> Verdict:
        if self.censor is None:
            self.censor = make_censor(DEFAULT_MODEL)
        return self.censor.check(text, rating)


def draft(
    plan: StoryPlan, censor: Censor, out_dir: Path, reserve_gib: float = 24,
    model_headroom_gib: float = 24,
) -> dict[str, object]:
    """Write chapter one once; persist prose only after every scene passes the Censor."""
    if not plan.approved:
        raise NotApproved("the author has not approved this plan")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {out_dir}")

    writer_free_gib = require_memory(reserve_gib, model_headroom_gib)
    chapter = plan.chapters[0]
    prompt = build(plan, chapter, PlannedFacts(plan), "", 8000)
    start = time.monotonic()
    text, tokens_in, tokens_out = writer_in_child(system_prompt(plan), prompt)
    elapsed = round(time.monotonic() - start, 2)
    if not text.strip():
        raise ValueError("writer returned an empty chapter")
    censor_free_gib = require_memory(reserve_gib, model_headroom_gib)
    scenes = [piece.text for piece in split_scenes(text)] or [text]
    verdicts = [censor.check(scene, plan.rating) for scene in scenes]
    allowed = all(verdict.allowed for verdict in verdicts)
    result: dict[str, object] = {
        "status": "draft" if allowed else "censored",
        "continuity_checked": False,
        "writer": WRITER,
        "censor": DEFAULT_MODEL,
        "chapter": chapter.number,
        "scenes": len(scenes),
        "words": len(text.split()),
        "input_tokens": tokens_in,
        "output_tokens": tokens_out,
        "draft_seconds": elapsed,
        "free_gib_before_writer": round(writer_free_gib, 2),
        "free_gib_before_censor": round(censor_free_gib, 2),
        "reserve_gib": reserve_gib,
        "model_headroom_gib": model_headroom_gib,
        "censor_reasons": [reason for verdict in verdicts for reason in verdict.reasons],
        "warnings": [warning.message for verdict in verdicts for warning in verdict.warnings],
        "labels": sorted({label for verdict in verdicts for label in verdict.labels}),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if allowed:
        (out_dir / "ch01.md").write_text(text, encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path, help="author-approved StoryPlan JSON")
    parser.add_argument("out_dir", type=Path, help="new or empty output directory")
    parser.add_argument(
        "--reserve-gib", type=float, default=24, help="free GiB reserved for others"
    )
    parser.add_argument(
        "--model-headroom-gib", type=float, default=24, help="estimated extra GiB per phase"
    )
    args = parser.parse_args()
    plan = StoryPlan.model_validate_json(args.plan.read_text(encoding="utf-8"))
    if not plan.approved:
        raise NotApproved("the author has not approved this plan")
    require_memory(args.reserve_gib, args.model_headroom_gib)
    from services.mlx_guard import memory_guard

    # Applies to this process only and stops it when shared-machine pressure is high.
    memory_guard(max_gb=36, min_free_percent=30)
    result = draft(plan, LazyCensor(), args.out_dir, args.reserve_gib, args.model_headroom_gib)
    print(json.dumps(result, indent=2))
    if result["status"] != "draft":
        raise SystemExit(1)


if __name__ == "__main__":
    if sys.argv[1:] == ["--writer-worker"]:
        writer_worker()
    else:
        main()
