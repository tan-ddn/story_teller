"""Teacher labels for the operational_harm ban (dataset v3).

The hand-labelled set has only 5 harmful-instruction positives, and every fine-tune learned to
never set the ban. This adds model-labelled Aegis 2.0 train responses from the categories where
real instructions occur, without anyone reading them:
- Qwen3.6-27B base (rubric profile; 96% recall and no false flags on the hard-ban suite) decides
  operational_harm;
- the best fine-tuned Censor (gemma-4-26b + censor-lora-v1) gives every other field.

Labels go to pipelines/censor_sft/labels/teacher.jsonl as IDs, text hashes and scores (quotes as
spans), tagged "teacher", never mixed up with hand labels. Texts are never committed.

uv run python -m pipelines.censor_sft.teacher candidates     # data/interim/censor_sft/teacher/
uv run python -m pipelines.censor_sft.teacher score --model mlx:Qwen3.6-27B-4bit     # on the Mac
uv run python -m pipelines.censor_sft.teacher score --model "$SCORER"   # gemma + censor-lora-v1
uv run python -m pipelines.censor_sft.teacher combine
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from evaluation.censor import classify_all
from guardrails.censor import Scores, make_censor
from pipelines.censor_sft.build import (
    CANDIDATES,
    LABELS,
    MAX_CHARS,
    MIN_CHARS,
    NearDuplicates,
    eval_exclusions,
    eval_texts,
    load_aegis_train,
    norm_hash,
)

DIR = Path("data/interim/censor_sft/teacher")
TEACHER_CANDIDATES = DIR / "candidates.jsonl"
OPS_JUDGE = "mlx:Qwen3.6-27B-4bit"
SCORER = "mlx:gemma-4-26b-a4b-it-4bit+censor-lora-v1-gemma4_26b_a4b"
OPS_CATEGORIES = {
    "Guns and Illegal Weapons",
    "Controlled/Regulated Substances",
    "Malware",
    "Criminal Planning/Confessions",
    "Fraud/Deception",
    "Illegal Activity",
}
SAFE_FLAG_LIMIT = 0.03  # Qwen3.6 may set the ban on at most this share of Aegis-"safe" responses


def candidates() -> None:
    """Aegis train responses in the harmful-instruction categories, minus anything an eval uses
    (by ID, hash or near-duplicate) and anything already in the hand-labelled set."""
    ids, hashes = eval_exclusions()
    near = NearDuplicates(eval_texts())
    taken = {json.loads(line)["text_sha"] for line in CANDIDATES.open(encoding="utf-8")}
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    dropped: Counter[str] = Counter()
    for r in load_aegis_train():
        text = r["response"]
        if not text or text.strip() == "REDACTED" or not MIN_CHARS <= len(text) <= MAX_CHARS:
            continue
        cats = {c.strip() for c in (r.get("violated_categories") or "").split(",") if c.strip()}
        if r["prompt_label"] != "unsafe" or not cats & OPS_CATEGORIES:
            continue
        cid, h = f"aegis-train-{r['id']}", norm_hash(text)
        if cid in ids or h in hashes or h in seen:
            dropped["eval or duplicate"] += 1
            continue
        if h in taken:
            dropped["already labelled"] += 1
            continue
        if near.match(text):
            dropped["near-duplicate of eval"] += 1
            continue
        seen.add(h)
        out.append({"id": cid, "text": text, "text_sha": h, "aegis_label": r["response_label"]})
    DIR.mkdir(parents=True, exist_ok=True)
    with TEACHER_CANDIDATES.open("w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"{len(out)} candidates {dict(Counter(r['aegis_label'] for r in out))}; "
        f"dropped {dict(dropped)}"
    )


def _texts() -> dict[str, str]:
    return {r["id"]: r["text"] for r in map(json.loads, TEACHER_CANDIDATES.open(encoding="utf-8"))}


def _cache(model: str) -> Path:
    return DIR / f"{model}.scores.jsonl"


def score(model: str, workers: int) -> None:
    censor = make_censor(model)
    scores, _ = classify_all(censor, model, _texts(), workers, _cache(model))
    print(
        f"{model}: {sum(v is not None for v in scores.values())} scored, "
        f"{sum(v is None for v in scores.values())} failed"
    )


REFUSAL = re.compile(
    r"\b(i can(no|')t|i won't|i'm sorry|i am sorry|i'm not able|i am not able|i'm unable|"
    r"cannot (help|assist|provide)|can't (help|assist|provide)|not able to (help|assist))\b",
    re.IGNORECASE,
)


def raw(model: str = OPS_JUDGE) -> None:
    """Why did the teacher fail? Re-ask it for every candidate without a cached answer, with the
    Censor's exact prompt, and classify the raw reply: "parsed" (valid Scores after all),
    "refusal" (declined, no JSON) or "other". Only the classification, the parsed decision and,
    for refusals, the first 160 characters are kept (data/interim, never committed)."""
    from services.llm import _raw_call, json_system, parse_json_answer

    done = set(_load(model))
    texts = {i: t for i, t in _texts().items() if i not in done}
    call = _raw_call(model, None)
    censor = make_censor(model)
    system = json_system(censor.rubric, Scores)
    out = DIR / "teacher_failures.jsonl"
    kinds: Counter[str] = Counter()
    with out.open("w", encoding="utf-8") as f:
        for i, text in texts.items():
            reply, _, _ = call(system, f"<text>\n{text}\n</text>")
            row: dict[str, Any] = {"id": i}
            try:
                parsed = parse_json_answer(reply, Scores)
                row |= {"kind": "parsed", "operational_harm": parsed.operational_harm}
            except ValueError:
                if REFUSAL.search(reply[:400]):
                    row |= {"kind": "refusal", "start": reply[:160]}
                else:
                    row |= {"kind": "other", "length": len(reply)}
            kinds[row["kind"]] += 1
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
    print(f"{len(texts)} failures re-asked: {dict(kinds)}")


def _load(model: str) -> dict[str, Scores]:
    out = {}
    for row in map(json.loads, _cache(model).open(encoding="utf-8")):
        out[row["id"]] = Scores.model_validate(row["scores"])
    return out


def combine() -> dict[str, Any]:
    rows = {r["id"]: r for r in map(json.loads, TEACHER_CANDIDATES.open(encoding="utf-8"))}
    judge, scorer = _load(OPS_JUDGE), _load(SCORER)
    labels: list[dict[str, Any]] = []
    report: Counter[str] = Counter()
    for cid, row in rows.items():
        if cid not in judge or cid not in scorer:
            report["missing a teacher answer"] += 1
            continue
        s = scorer[cid].model_copy(update={"operational_harm": judge[cid].operational_harm})
        text = " ".join(row["text"].split())
        spans = []
        for q in s.quotes:
            quote = " ".join(q.text.split())
            start = text.find(quote)
            if start >= 0:
                spans.append({"category": q.category, "start": start, "end": start + len(quote)})
            else:
                report["quote not in text (dropped)"] += 1
        stored = s.model_dump(exclude={"quotes"}) | {"quote_spans": spans}
        labels.append(
            {"id": cid, "text_sha": row["text_sha"], "scores": stored, "source": "teacher"}
        )
        report[f"aegis {row['aegis_label']}, ban {s.operational_harm}"] += 1
    safe = report["aegis safe, ban True"] + report["aegis safe, ban False"]
    unsafe = report["aegis unsafe, ban True"] + report["aegis unsafe, ban False"]
    checks = {
        "safe_flag_rate": round(report["aegis safe, ban True"] / safe, 4) if safe else None,
        "unsafe_flag_rate": round(report["aegis unsafe, ban True"] / unsafe, 4) if unsafe else None,
        "positives": sum(lbl["scores"]["operational_harm"] for lbl in labels),
    }
    checks["safe_check_passed"] = (checks["safe_flag_rate"] or 0) <= SAFE_FLAG_LIMIT
    with (LABELS / "teacher.jsonl").open("w", encoding="utf-8") as f:
        for lbl in labels:
            f.write(json.dumps(lbl, ensure_ascii=False) + "\n")
    result = {"counts": dict(report), "checks": checks, "labels": len(labels)}
    (DIR / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Teacher labels for the operational_harm ban.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("candidates")
    s = sub.add_parser("score")
    s.add_argument("--model", required=True)
    s.add_argument("--workers", type=int, default=1)
    sub.add_parser("combine")
    sub.add_parser("raw", help="re-ask the ops teacher on its failures and classify the replies")
    args = parser.parse_args()
    load_dotenv(".env")
    if args.cmd == "candidates":
        candidates()
    elif args.cmd == "score":
        score(args.model, args.workers)
    elif args.cmd == "raw":
        raw()
    else:
        combine()


if __name__ == "__main__":
    main()
