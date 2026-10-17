"""Plan Censor fine-tuning runs: derive configs, and pick the best finished fine-tune.

`config` writes a training config from a base one for another dataset and epoch count, deriving
the batch count, per-epoch checkpoints and the learning-rate schedule from the training-set size
(mlx-lm counts batches; the schedule advances once per optimizer update).

`best` ranks finished fine-tunes by their full Censor check: fewest gate failures first, then the
mean of children and teen fiction agreement. It prints "<model> <adapter> <epochs>".

uv run python -m training.censor.plan config training/configs/censor_gemma4_26b_a4b_lora.yaml \
    --dataset data/processed/censor_sft_v2 --epochs 3 --adapter censor-lora-v2-gemma --out cfg.yaml
uv run python -m training.censor.plan best mlx:A+adapter1:1 mlx:B+adapter2:3
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml

from evaluation import censor, censor_bans, censor_fiction
from evaluation.censor import REPORTS


def config(base: Path, dataset: Path, epochs: int, adapter: str, out: Path) -> dict[str, Any]:
    cfg: dict[str, Any] = yaml.safe_load(base.read_text(encoding="utf-8"))
    n_train = sum(1 for _ in (dataset / "train.jsonl").open(encoding="utf-8"))
    per_epoch = n_train // cfg["batch_size"]
    iters = per_epoch * epochs
    updates = math.ceil(iters / cfg.get("grad_accumulation_steps", 1))
    lr = cfg["learning_rate"]
    cfg |= {
        "adapter_name": adapter,
        "dataset": str(dataset),
        "iters": iters,
        "save_every": per_epoch,
        "steps_per_eval": per_epoch // 3,
        "lr_schedule": {"name": "cosine_decay", "warmup": 30, "arguments": [lr, updates, 1.0e-6]},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"{out}: {epochs} epochs = {iters} batches, checkpoint every {per_epoch}")
    return cfg


def failures(model: str) -> int | None:
    """Gate failures in the full Censor check, or None if a report is missing."""
    slug = model.replace(":", "_")
    paths = [
        censor_fiction.REPORTS / f"report.{slug}.vs.reference-claude-opus-5.json",
        REPORTS / f"bans.{model}.json",
        REPORTS / f"{model}.json",
    ]
    if not all(p.exists() for p in paths):
        return None
    fiction, bans, bench = (json.loads(p.read_text(encoding="utf-8")) for p in paths)
    return len(censor_fiction.gate(fiction)) + len(censor_bans.gate(bans)) + len(censor.gate(bench))


def fiction_agreement(model: str) -> float:
    slug = model.replace(":", "_")
    path = censor_fiction.REPORTS / f"report.{slug}.vs.reference-claude-opus-5.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    return float((report["children"]["agreement"] + report["teen"]["agreement"]) / 2)


def best(candidates: list[str]) -> str:
    """Each candidate is "<mlx model id>:<epochs>"; returns "<model> <adapter> <epochs>"."""
    ranked = []
    for spec in candidates:
        model, _, n_epochs = spec.rpartition(":")
        fails = failures(model)
        if fails is None:
            print(f"skip {model}: no full Censor check")
            continue
        ranked.append((fails, -fiction_agreement(model), model, int(n_epochs)))
        print(f"{model}: {fails} gate failures, fiction agreement {-ranked[-1][1]:.3f}")
    if not ranked:
        raise SystemExit("no finished fine-tune to choose from")
    _, _, model, epochs = min(ranked)
    base, _, adapter = model.removeprefix("mlx:").partition("+")
    line = f"{base} {adapter} {epochs}"
    print(line)
    return line


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan Censor fine-tuning runs.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("config")
    c.add_argument("base", type=Path)
    c.add_argument("--dataset", type=Path, required=True)
    c.add_argument("--epochs", type=int, required=True)
    c.add_argument("--adapter", required=True)
    c.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("best")
    b.add_argument("candidates", nargs="+")
    args = parser.parse_args()
    if args.cmd == "config":
        config(args.base, args.dataset, args.epochs, args.adapter, args.out)
    else:
        best(args.candidates)


if __name__ == "__main__":
    main()
