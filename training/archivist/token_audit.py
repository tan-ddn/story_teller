"""Count full rendered chat lengths without loading model weights or truncating.

Runs on the Mac with the project's MLX environment. Outputs only scene IDs and lengths.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def audit(model: Path, data: Path, max_seq_length: int) -> dict[str, Any]:
    from mlx_lm.utils import load_tokenizer

    tokenizer = load_tokenizer(model, tokenizer_config_extra={"trust_remote_code": False})
    if "deepseek-r1-distill" in str(model).lower():
        from training.archivist.deepseek_chat import align_deepseek_template

        align_deepseek_template(tokenizer)
    report: dict[str, Any] = {"model": str(model), "max_seq_length": max_seq_length, "splits": {}}
    for split in ("train", "valid", "test"):
        entries = []
        path = data / f"{split}.jsonl"
        with path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                tokens = tokenizer.apply_chat_template(
                    row["messages"],
                    tokenize=True,
                    add_generation_prompt=False,
                    enable_thinking=False,
                    tools=None,
                    return_dict=False,
                )
                entries.append({"scene_id": row["scene_id"], "tokens": len(tokens)})
        lengths = sorted(e["tokens"] for e in entries)
        report["splits"][split] = {
            "count": len(lengths),
            "min": lengths[0],
            "median": lengths[len(lengths) // 2],
            "p95": lengths[(len(lengths) - 1) * 95 // 100],
            "max": lengths[-1],
            "longest_scene_id": max(entries, key=lambda entry: entry["tokens"])["scene_id"],
            "examples": entries,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "over_limit": [e for e in entries if e["tokens"] > max_seq_length],
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path, help="Local MLX model directory")
    parser.add_argument("data", type=Path, help="Prepared Archivist dataset directory")
    parser.add_argument("--max-seq-length", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.model, args.data, args.max_seq_length)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
