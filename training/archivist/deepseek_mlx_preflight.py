"""Check that MLX training masks exactly the inference prompt for every scene."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("data", type=Path)
    args = parser.parse_args()
    from mlx_lm.utils import load_tokenizer

    from training.archivist.deepseek_chat import align_deepseek_template

    tokenizer = load_tokenizer(args.model, tokenizer_config_extra={"trust_remote_code": False})
    align_deepseek_template(tokenizer)
    for split in ("train", "valid"):
        with (args.data / f"{split}.jsonl").open() as source:
            for line in source:
                row = json.loads(line)
                messages = row["messages"]
                options = {"tokenize": True, "enable_thinking": False, "return_dict": False}
                prompt = tokenizer.apply_chat_template(
                    messages[:-1], add_generation_prompt=True, **options
                )
                full = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=False, **options
                )
                if full[: len(prompt)] != prompt or len(prompt) >= len(full):
                    raise ValueError(f"{split} {row['scene_id']}: prompt boundary mismatch")
    print("All train and validation prompt boundaries match")


if __name__ == "__main__":
    main()
