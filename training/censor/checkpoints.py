"""Per-epoch checkpoints of a Censor LoRA run: split them into adapters and pick the best.

mlx-lm saves checkpoints as NNNNNNN_adapters.safetensors in the adapter folder. `split` turns the
ones at each epoch boundary into sibling adapter folders (<adapter>-epoch<k>, hard-linked, so no
extra disk), which services.llm can load as "mlx:<model>+<adapter>-epoch<k>". `best` reads their
val-split reports (evaluation.censor_sft_val) and prints the best epoch's adapter name.

uv run python -m training.censor.checkpoints split adapters/censor-lora-v1-3ep --every 2400
uv run python -m training.censor.checkpoints best mlx:Qwen3-30B-A3B-4bit censor-lora-v1-3ep 3
"""

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from evaluation.censor import REPORTS


def split(adapter_dir: Path, every: int) -> list[Path]:
    out = []
    for ckpt in sorted(adapter_dir.glob("*_adapters.safetensors")):
        step = int(re.match(r"(\d+)_", ckpt.name).group(1))  # type: ignore[union-attr]
        if step % every:
            continue
        target = adapter_dir.with_name(f"{adapter_dir.name}-epoch{step // every}")
        target.mkdir(exist_ok=True)
        shutil.copy(adapter_dir / "adapter_config.json", target / "adapter_config.json")
        weights = target / "adapters.safetensors"
        weights.unlink(missing_ok=True)
        os.link(ckpt, weights)
        out.append(target)
        print(f"{ckpt.name} -> {target}")
    return out


def score(report: dict[str, Any]) -> tuple[float, float]:
    """Mean allow/warn/block agreement over both ratings; teen block F1 breaks ties."""
    agreement = (report["children"]["agreement"] + report["teen"]["agreement"]) / 2
    r, p = report["teen"]["block_recall"] or 0.0, report["teen"]["block_precision"] or 0.0
    return agreement, (2 * r * p / (r + p) if r + p else 0.0)


def best(model: str, adapter: str, epochs: int) -> str:
    scored = {}
    for k in range(1, epochs + 1):
        name = f"{adapter}-epoch{k}"
        path = REPORTS / f"sft-val.{model}+{name}.json"
        if path.exists():
            scored[name] = score(json.loads(path.read_text(encoding="utf-8")))
            print(name, scored[name])
    if not scored:
        raise SystemExit("no epoch reports found")
    winner = max(scored, key=lambda n: scored[n])
    print(winner)
    return winner


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-epoch Censor LoRA checkpoints.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split")
    s.add_argument("adapter_dir", type=Path)
    s.add_argument("--every", type=int, required=True, help="batches per epoch")
    b = sub.add_parser("best")
    b.add_argument("model")
    b.add_argument("adapter")
    b.add_argument("epochs", type=int)
    args = parser.parse_args()
    if args.cmd == "split":
        split(args.adapter_dir, args.every)
    else:
        best(args.model, args.adapter, args.epochs)


if __name__ == "__main__":
    main()
