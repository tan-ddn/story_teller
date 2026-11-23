"""Convert a standard PEFT LoRA adapter to mlx-lm's dense LoRA representation."""

import argparse
import json
from pathlib import Path

import mlx.core as mx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    config = json.loads((args.source / "adapter_config.json").read_text())
    if config["peft_type"] != "LORA" or config.get("use_rslora") or config.get("use_dora"):
        raise ValueError("only ordinary LoRA adapters are supported")
    weights = mx.load(str(args.source / "adapter_model.safetensors"))
    converted = {}
    keys = set()
    layers = set()
    for name, value in weights.items():
        prefix = "base_model.model.model.layers."
        if not name.startswith(prefix):
            raise ValueError(f"unexpected adapter weight {name}")
        layer, tail = name[len(prefix) :].split(".", 1)
        parts = tail.rsplit(".", 2)
        if len(parts) != 3 or parts[1] not in ("lora_A", "lora_B") or parts[2] != "weight":
            raise ValueError(f"unexpected adapter weight {name}")
        keys.add(parts[0])
        layers.add(int(layer))
        converted[f"model.layers.{layer}.{parts[0]}.{parts[1].lower()}"] = value.T
    if layers != set(range(32, 48)) or len(converted) != len(layers) * len(keys) * 2:
        raise ValueError("unexpected layer coverage or missing LoRA weights")
    args.output.mkdir(parents=True, exist_ok=False)
    mx.save_safetensors(str(args.output / "adapters.safetensors"), converted)
    output = {
        "num_layers": 16,
        "fine_tune_type": "lora",
        "lora_parameters": {
            "rank": config["r"],
            "scale": config["lora_alpha"] / config["r"],
            "dropout": 0.0,
            "keys": sorted(keys),
        },
    }
    (args.output / "adapter_config.json").write_text(json.dumps(output, indent=2) + "\n")
    print(f"Converted {len(converted)} tensors across {len(layers)} layers")


if __name__ == "__main__":
    main()
