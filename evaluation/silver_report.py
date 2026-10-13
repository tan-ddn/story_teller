"""Compare API labelling runs: quality, cost and latency.

Pilot runs are scored on every category against the hand-labelled pilot; LitBank runs on
entities only, against LitBank's human coreference annotations.

uv run python -m evaluation.silver_report
"""

import json
import statistics
from pathlib import Path

from evaluation.extraction import CATEGORIES, Counts, align_entities, total
from evaluation.extraction import score as score_extraction
from pipelines.labeling.models import LabelRun, SilverLabel
from services.extractor.schema import Entity, SceneExtraction

RUNS_DIR = Path("data/interim/silver")
PILOT = Path("data/processed/silver_pilot.jsonl")
LITBANK = Path("data/processed/litbank_entities.jsonl")


def load_runs(path: Path) -> list[LabelRun]:
    return [LabelRun.model_validate_json(line) for line in path.open(encoding="utf-8")]


def usage_summary(runs: list[LabelRun]) -> str:
    ok = [r for r in runs if r.extraction]
    seconds = sorted(r.seconds for r in runs)
    p95 = seconds[min(len(seconds) - 1, int(0.95 * len(seconds)))]
    spend = [r.cost_usd for r in runs]
    return (
        f"valid {len(ok)}/{len(runs)}  retried {sum(r.attempts > 1 for r in runs)}  "
        f"cost ${sum(spend):.2f} (${statistics.mean(spend):.3f}/scene)  "
        f"latency p50 {statistics.median(seconds):.0f}s p95 {p95:.0f}s"
    )


def f1_row(name: str, counts: Counts) -> str:
    return f"    {name:<11} P {counts.precision:.2f}  R {counts.recall:.2f}  F1 {counts.f1:.2f}"


def pilot_report(path: Path, gold: dict[str, SceneExtraction]) -> None:
    runs = load_runs(path)
    print(f"{path.stem}  [pilot]  {usage_summary(runs)}")
    scores = [score_extraction(r.extraction, gold[r.scene_id]) for r in runs if r.extraction]
    totals = total(scores)
    for category in CATEGORIES:
        print(f1_row(category, totals[category]))


def litbank_report(path: Path, gold: dict[str, list[Entity]]) -> None:
    runs = load_runs(path)
    print(f"{path.stem}  [LitBank entities]  {usage_summary(runs)}")
    counts = Counts()
    for r in runs:
        if r.extraction:
            reference = gold[r.scene_id]
            matched = align_entities(r.extraction.entities, reference)
            counts.add(
                Counts(tp=len(matched), pred=len(r.extraction.entities), gold=len(reference))
            )
    print(f1_row("entities", counts))


def main() -> None:
    pilot = {
        label.scene_id: label.extraction
        for label in map(SilverLabel.model_validate_json, PILOT.open(encoding="utf-8"))
    }
    litbank = {}
    for line in LITBANK.open(encoding="utf-8"):
        doc = json.loads(line)
        litbank[doc["doc_id"]] = [Entity.model_validate(e) for e in doc["entities"]]
    for path in sorted(RUNS_DIR.glob("*.jsonl")):
        if path.name.endswith(".litbank.jsonl"):
            litbank_report(path, litbank)
        else:
            pilot_report(path, pilot)


if __name__ == "__main__":
    main()
