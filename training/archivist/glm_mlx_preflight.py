"""Check full-scene GLM prompt masking and sequence lengths before training."""

import json
from pathlib import Path

from mlx_lm.utils import load_tokenizer

model = Path("/Volumes/Untitled/Shared/models/glm-4-9b-chat-1m-4bit")
data = Path("data/interim/archivist_sft")
tokenizer = load_tokenizer(model, tokenizer_config_extra={"trust_remote_code": False})
for split in ("train", "valid"):
    longest = 0
    count = 0
    with (data / f"{split}.jsonl").open() as source:
        for line in source:
            row = json.loads(line)
            messages = row["messages"]
            options = {"tokenize": True, "enable_thinking": False, "return_dict": False}
            prompt = tokenizer.apply_chat_template(
                messages[:-1], add_generation_prompt=True, **options
            )
            full = tokenizer.apply_chat_template(messages, add_generation_prompt=False, **options)
            if full[: len(prompt)] != prompt or len(prompt) >= len(full):
                raise ValueError(f"{split} {row['scene_id']}: prompt boundary mismatch")
            if len(full) > 10240:
                raise ValueError(f"{split} {row['scene_id']}: exceeds 10240 tokens")
            longest = max(longest, len(full))
            count += 1
    print(f"{split}: {count} scenes, max {longest} tokens", flush=True)
