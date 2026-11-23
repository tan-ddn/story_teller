"""Measure Archivist extraction against held-out silver scenes, one local model per process."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from evaluation.extraction import CATEGORIES, score
from services.extractor.schema import SceneExtraction, check_evidence

GUIDELINES = Path("pipelines/labeling/guidelines.md")


def run_case(call: Any, system: str, messages: list[dict[str, str]]) -> dict[str, Any]:
    user = messages[1]["content"]
    scene = user.split("\n<scene>\n", 1)[1].rsplit("\n</scene>", 1)[0]
    gold = SceneExtraction.model_validate_json(messages[2]["content"])
    attempts: list[dict[str, Any]] = []
    prediction = None
    for _ in range(2):
        start = time.monotonic()
        try:
            candidate, tokens_in, tokens_out = call(system, user)
            bad = check_evidence(candidate, scene)
            status = "evidence_mismatch" if bad else "valid"
            attempts.append(
                {
                    "status": status,
                    "seconds": time.monotonic() - start,
                    "input_tokens": tokens_in,
                    "output_tokens": tokens_out,
                    "invalid_quotes": len(bad),
                }
            )
            if not bad:
                prediction = candidate
                break
            user += "\nSome evidence was not verbatim. Copy every quote exactly from the scene."
        except ValueError:
            attempts.append({"status": "invalid_schema", "seconds": time.monotonic() - start})
            user += "\nReturn a valid extraction matching the required schema."
    if prediction is None:
        prediction = SceneExtraction.model_validate(
            {
                "entities": [],
                "attributes": [],
                "events": [],
                "relations": [],
                "time": {"mode": "main"},
            }
        )
    scores = score(prediction, gold)
    return {
        "accepted": attempts[-1]["status"] == "valid",
        "attempts": attempts,
        "scores": {k: {"tp": c.tp, "pred": c.pred, "gold": c.gold} for k, c in scores.items()},
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"scenes": len(rows), "accepted": sum(r["accepted"] for r in rows)}
    times = sorted(sum(a["seconds"] for a in r["attempts"]) for r in rows)
    result["seconds"] = {
        "total": sum(times),
        "median": times[len(times) // 2] if times else None,
        "p95": times[(len(times) - 1) * 95 // 100] if times else None,
    }
    result["attempt_statuses"] = {
        status: sum(a["status"] == status for r in rows for a in r["attempts"])
        for status in ("valid", "invalid_schema", "evidence_mismatch")
    }
    result["scores"] = {}
    for category in CATEGORIES:
        counts = {k: sum(r["scores"][category][k] for r in rows) for k in ("tp", "pred", "gold")}
        precision = counts["tp"] / counts["pred"] if counts["pred"] else 0.0
        recall = counts["tp"] / counts["gold"] if counts["gold"] else 0.0
        result["scores"][category] = counts | {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="mlx:model or mlx:model+adapter")
    parser.add_argument("dataset", type=Path, help="Prepared valid.jsonl or test.jsonl")
    parser.add_argument("output", type=Path, help="Run directory; resumes existing results")
    parser.add_argument(
        "--memory-pressure-only",
        action="store_true",
        help="rely on the MLX allocation and memory-pressure guards",
    )
    args = parser.parse_args()
    if not args.model.startswith("mlx:"):
        raise ValueError("only local MLX models are allowed")
    from services.llm import parse_call
    from writers_room.smoke import watch_free_reserve

    args.output.mkdir(parents=True, exist_ok=True)
    meta = {
        "model": args.model,
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "guidelines_sha256": hashlib.sha256(GUIDELINES.read_bytes()).hexdigest(),
        "max_tokens": os.environ.get("MLX_MAX_TOKENS", "1024"),
    }
    metadata = args.output / "metadata.json"
    if metadata.exists() and json.loads(metadata.read_text()) != meta:
        raise ValueError("run metadata changed; use a new output directory")
    metadata.write_text(json.dumps(meta, indent=2) + "\n")
    path = args.output / "results.jsonl"
    done = (
        {json.loads(line)["scene_id"] for line in path.read_text().splitlines()}
        if path.exists()
        else set()
    )
    system = GUIDELINES.read_text(encoding="utf-8")
    started = time.monotonic()
    if not args.memory_pressure_only:
        watch_free_reserve(8)
    call = parse_call(args.model, SceneExtraction, name="archivist_eval")
    print(f"model loaded in {time.monotonic() - started:.1f}s", flush=True)
    with args.dataset.open(encoding="utf-8") as source, path.open("a", encoding="utf-8") as out:
        for line in source:
            row = json.loads(line)
            if row["scene_id"] in done:
                continue
            result = run_case(call, system, row["messages"])
            result |= {"scene_id": row["scene_id"], "book_id": row["book_id"]}
            out.write(json.dumps(result) + "\n")
            out.flush()
            print(
                f"{row['scene_id']}: accepted={result['accepted']} "
                f"seconds={sum(a['seconds'] for a in result['attempts']):.1f}",
                flush=True,
            )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    report = summarize(rows)
    report["books"] = {
        str(book): summarize([r for r in rows if r["book_id"] == book])
        for book in sorted({r["book_id"] for r in rows})
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
