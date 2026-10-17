"""Score a Censor model on the fine-tuning validation split (in-domain, hand-labelled).

Each val text goes through the Censor exactly as in production; its Scores are compared with the
hand label:
- per category: exact accuracy, accuracy within one point, mean absolute error
- per rating: allow / warn / block agreement, and block recall and precision
- hard bans: recall and precision of real_person and operational_harm
- errors: answers that could not be parsed (Censor fails closed, so these count as blocks)

Run it on the base model before fine-tuning and on each adapter after, with the same split.

uv run python -m evaluation.censor_sft_val --model mlx:Qwen3-30B-A3B-4bit
uv run python -m evaluation.censor_sft_val --model mlx:Qwen3-30B-A3B-4bit+censor-lora-v1
"""

import argparse
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from evaluation.censor import REPORTS, action, classify_all
from evaluation.censor_mlflow import log_check
from guardrails.censor import CATEGORIES, DEFAULT_MODEL, Censor, Rating, Scores, judge
from services.llm import parse_call

VAL = Path("data/processed/censor_sft/val.jsonl")
RATINGS: tuple[Rating, ...] = ("children", "teen")
BANS = ("real_person", "operational_harm")
MAX_ERROR_RATE = 0.02


def load_val() -> tuple[dict[str, str], dict[str, Scores]]:
    texts, gold = {}, {}
    for row in map(json.loads, VAL.open(encoding="utf-8")):
        user = row["messages"][1]["content"]
        texts[row["id"]] = user.removeprefix("<text>\n").removesuffix("\n</text>")
        gold[row["id"]] = Scores.model_validate_json(row["messages"][2]["content"])
    return texts, gold


def _rate(hits: list[bool]) -> float | None:
    return round(sum(hits) / len(hits), 4) if hits else None


def report(gold: dict[str, Scores], pred: dict[str, Scores | None]) -> dict[str, Any]:
    ids = list(gold)
    ok = [i for i in ids if pred.get(i) is not None]
    out: dict[str, Any] = {"n": len(ids), "error_rate": round(1 - len(ok) / len(ids), 4)}
    for c in CATEGORIES:
        diffs = [abs(getattr(pred[i], c) - getattr(gold[i], c)) for i in ok]
        out[c] = {
            "exact": _rate([d == 0 for d in diffs]),
            "within_1": _rate([d <= 1 for d in diffs]),
            "mae": round(sum(diffs) / len(diffs), 4) if diffs else None,
        }
    for rating in RATINGS:
        pairs = []
        for i in ids:
            want = action(judge(gold[i], rating))
            p = pred.get(i)
            pairs.append((want, "block" if p is None else action(judge(p, rating))))
        out[rating] = {
            "agreement": _rate([w == g for w, g in pairs]),
            "block_recall": _rate([g == "block" for w, g in pairs if w == "block"]),
            "block_precision": _rate([w == "block" for w, g in pairs if g == "block"]),
        }
    for ban in BANS:
        out[ban] = {
            "recall": _rate([bool(getattr(pred[i], ban)) for i in ok if getattr(gold[i], ban)]),
            "precision": _rate([bool(getattr(gold[i], ban)) for i in ok if getattr(pred[i], ban)]),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Censor model on the fine-tuning val split.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="first N samples only; 0 = all")
    args = parser.parse_args()
    load_dotenv(".env")

    texts, gold = load_val()
    if args.limit:
        keep = list(texts)[: args.limit]
        texts, gold = {k: texts[k] for k in keep}, {k: gold[k] for k in keep}
    censor = Censor(parse_call(args.model, Scores, name="censor"))
    cache = REPORTS / f"{args.model}.scores.jsonl"
    pred, spent = classify_all(censor, args.model, texts, args.workers, cache)
    result = {"model": args.model, **report(gold, pred)}
    out = REPORTS / f"sft-val.{args.model}{f'.limit{args.limit}' if args.limit else ''}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    params = {"split": str(VAL), "n": len(texts), "limit": args.limit}
    log_check("sft-val", args.model, params, result, result["error_rate"] <= MAX_ERROR_RATE, out)


if __name__ == "__main__":
    main()
