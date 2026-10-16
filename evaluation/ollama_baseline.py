"""Baseline: un-finetuned local models served by Ollama, prompted like the silver labeller.

`run` labels scenes with the same guidelines, schema and evidence retry as
pipelines.labeling.silver, appending LabelRun lines to data/interim/baseline/<model>.jsonl
(resumable). `report` prints validity and latency over every scene run, and P/R/F1 on the
scenes that have a reference: the hand-labelled pilot by default, or any JSONL of
{"scene_id", "extraction"} records such as a full silver run. A second, lenient score also counts
answers that failed the evidence check, with the share of their quotes that are verbatim.

uv run python -m evaluation.ollama_baseline run --models qwen3:32b gemma3:27b
uv run python -m evaluation.ollama_baseline run --models qwen3:32b --host http://100.123.117.116:11434
uv run python -m evaluation.ollama_baseline run --models qwen3:8b --pilot --no-think --tag nothink
uv run python -m evaluation.ollama_baseline report
uv run python -m evaluation.ollama_baseline report --reference data/interim/silver/gpt-6.1-sol.jsonl
uv run python -m evaluation.ollama_baseline log data/interim/baseline/qwen3_8b.think.jsonl \
    --param think=true --param num_ctx=16384      # one MLflow run per file (MLFLOW_TRACKING_URI)
"""

import argparse
import json
import os
import re
import statistics
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from evaluation.extraction import CATEGORIES, Counts, total
from evaluation.extraction import score as score_extraction
from evaluation.silver_report import f1_row, load_runs, usage_summary
from pipelines.labeling.models import LabelRun
from pipelines.labeling.silver import PILOT, SCENES, Call, label_scene
from services.extractor.schema import SceneExtraction, check_evidence

OUT_DIR = Path("data/interim/baseline")
REQUEST_TIMEOUT = 1800  # seconds; large local models can be slow on long scenes


def _ollama(host: str, model: str, num_ctx: int, think: bool, num_predict: int | None) -> Call:
    # Ollama's grammar parser rejects the \d escape used in ID patterns.
    schema = json.loads(json.dumps(SceneExtraction.model_json_schema()).replace(r"\\d", "[0-9]"))

    def call(system: str, user: str) -> tuple[SceneExtraction, int, int]:
        body = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "format": schema,
            "stream": False,
            "think": think,
            "options": {"temperature": 0, "num_ctx": num_ctx},
        }
        if num_predict:
            body["options"]["num_predict"] = num_predict
        request = urllib.request.Request(  # noqa: S310 - host is an http(s) URL we chose
            f"{host.rstrip('/')}/api/chat",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:  # noqa: S310
            r = json.load(response)
        extraction = SceneExtraction.model_validate_json(r["message"]["content"])
        return extraction, r.get("prompt_eval_count", 0), r.get("eval_count", 0)

    return call


def out_path(model: str, suffix: str = "") -> Path:
    return OUT_DIR / f"{re.sub(r'[^\w.-]', '_', model)}{suffix}.jsonl"


def run_model(host: str, model: str, scenes: dict[str, str], args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = out_path(model, f".{args.tag}" if args.tag else "")
    done = set()
    if out.exists():
        done = {json.loads(line)["scene_id"] for line in out.open(encoding="utf-8")}
    todo = [sid for sid in scenes if sid not in done]
    with urllib.request.urlopen(f"{host.rstrip('/')}/api/tags", timeout=10) as r:  # noqa: S310
        pulled = {m["name"] for m in json.load(r)["models"]}
    if model not in pulled and f"{model}:latest" not in pulled:
        raise SystemExit(f"{model} is not pulled on {host}; run: ollama pull {model}")
    call = _ollama(host, model, args.num_ctx, not args.no_think, args.num_predict)
    print(f"{model}: {len(todo)} to label, {len(done)} already done")
    with ThreadPoolExecutor(args.workers) as pool, out.open("a", encoding="utf-8") as f:
        futures = [pool.submit(label_scene, call, model, sid, scenes[sid]) for sid in todo]
        for future in as_completed(futures):
            run = future.result()
            f.write(run.model_dump_json() + "\n")
            f.flush()
            status = "ok" if run.extraction else f"FAILED {run.error}"
            print(f"  {run.scene_id:<14} {run.seconds:>6}s {status}")


def load_reference(path: Path) -> dict[str, SceneExtraction]:
    """Read any JSONL of records with scene_id and extraction (SilverLabel or LabelRun)."""
    reference = {}
    for line in path.open(encoding="utf-8"):
        record = json.loads(line)
        if record.get("extraction"):
            reference[record["scene_id"]] = SceneExtraction.model_validate(record["extraction"])
    return reference


def quote_count(e: SceneExtraction) -> int:
    return len(e.entities) + len(e.attributes) + len(e.events) + len(e.relations) + len(e.time.cues)


@dataclass
class Scores:
    strict: dict[str, Counts]  # answers whose evidence checked out
    lenient: dict[str, Counts]  # any answer that parsed against the schema
    strict_scenes: int
    lenient_scenes: int
    parsed: int
    quotes: int
    bad_quotes: int


def score_runs(
    runs: list[LabelRun], reference: dict[str, SceneExtraction], scenes: dict[str, str]
) -> Scores:
    strict = [
        (r.extraction, reference[r.scene_id])
        for r in runs
        if r.extraction and r.scene_id in reference
    ]
    answers = [(r.scene_id, r.extraction or r.unverified) for r in runs]
    parsed = [(sid, e) for sid, e in answers if e]
    lenient = [(e, reference[sid]) for sid, e in parsed if sid in reference]
    return Scores(
        strict=total(score_extraction(pred, gold) for pred, gold in strict),
        lenient=total(score_extraction(pred, gold) for pred, gold in lenient),
        strict_scenes=len(strict),
        lenient_scenes=len(lenient),
        parsed=len(parsed),
        quotes=sum(quote_count(e) for _, e in parsed),
        bad_quotes=sum(len(check_evidence(e, scenes[sid])) for sid, e in parsed),
    )


def load_scenes() -> dict[str, str]:
    return {s["scene_id"]: s["text"] for s in map(json.loads, SCENES.open(encoding="utf-8"))}


def report(reference_path: Path) -> None:
    reference = load_reference(reference_path)
    scenes = load_scenes()
    print(f"reference: {reference_path} ({len(reference)} scenes)\n")
    for path in sorted(OUT_DIR.glob("*.jsonl")):
        runs = load_runs(path)
        print(f"{path.stem}  {usage_summary(runs)}")
        s = score_runs(runs, reference, scenes)
        print(f"  strict (evidence verified): scored on {s.strict_scenes} scenes")
        for category in CATEGORIES:
            print(f1_row(category, s.strict[category]))
        verbatim = s.quotes - s.bad_quotes
        print(f"    parsed {s.parsed}/{len(runs)}  verbatim quotes {verbatim}/{s.quotes}")
        print(f"  lenient (any parsed answer): scored on {s.lenient_scenes} scenes")
        for category in CATEGORIES:
            print(f1_row(category, s.lenient[category]))


def log_to_mlflow(paths: list[Path], reference_path: Path, params: dict[str, str]) -> None:
    """Log each result file as one MLflow run: settings, scores, latency and the file itself."""
    import mlflow

    reference = load_reference(reference_path)
    scenes = load_scenes()
    mlflow.set_experiment("archivist-baseline")
    for path in paths:
        runs = load_runs(path)
        s = score_runs(runs, reference, scenes)
        seconds = sorted(r.seconds for r in runs)
        metrics = {
            "scenes": len(runs),
            "valid": sum(bool(r.extraction) for r in runs),
            "parsed": s.parsed,
            "verbatim_quote_rate": 1 - s.bad_quotes / s.quotes if s.quotes else 0.0,
            "latency_p50_s": statistics.median(seconds),
            "latency_p95_s": seconds[min(len(seconds) - 1, int(0.95 * len(seconds)))],
        }
        for kind, totals in (("strict", s.strict), ("lenient", s.lenient)):
            for category, c in totals.items():
                metrics |= {
                    f"{kind}_{category}_precision": c.precision,
                    f"{kind}_{category}_recall": c.recall,
                    f"{kind}_{category}_f1": c.f1,
                }
        with mlflow.start_run(run_name=path.stem):
            mlflow.log_params({"model": runs[0].model, "reference": str(reference_path), **params})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
            mlflow.log_artifact(str(path))
        print(f"logged {path.stem}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline local Ollama models on scenes.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="label scenes with Ollama models")
    run.add_argument("--models", nargs="+", required=True)
    run.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    run.add_argument("--pilot", action="store_true", help="only the 50 hand-labelled scenes")
    run.add_argument("--workers", type=int, default=1, help="match OLLAMA_NUM_PARALLEL")
    run.add_argument("--num-ctx", type=int, default=16384, help="guidelines+scene need ~6k")
    run.add_argument("--tag", default="", help="suffix for a separate output file")
    run.add_argument("--num-predict", type=int, help="cap output tokens, stopping runaways")
    run.add_argument("--no-think", action="store_true", help="disable reasoning (qwen3 etc.)")
    rep = sub.add_parser("report", help="score baseline runs")
    rep.add_argument("--reference", type=Path, default=PILOT)
    log = sub.add_parser("log", help="log result files to MLflow, one run each")
    log.add_argument("paths", nargs="+", type=Path)
    log.add_argument("--reference", type=Path, default=PILOT)
    log.add_argument("--param", action="append", default=[], help="extra key=value run setting")
    args = parser.parse_args()

    if args.command == "report":
        report(args.reference)
        return
    if args.command == "log":
        log_to_mlflow(args.paths, args.reference, dict(p.split("=", 1) for p in args.param))
        return
    scenes = load_scenes()
    if args.pilot:
        pilot = [json.loads(line)["scene_id"] for line in PILOT.open(encoding="utf-8")]
        scenes = {sid: scenes[sid] for sid in pilot}
    for model in args.models:
        run_model(args.host, model, scenes, args)


if __name__ == "__main__":
    main()
