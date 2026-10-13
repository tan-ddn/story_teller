"""Label scenes with API models: the guidelines as system prompt, SceneExtraction as output.

Responses whose evidence quotes are not in the scene get one retry with the bad quotes listed.
Runs append to data/interim/silver/<model>.jsonl and skip scenes already there, so an
interrupted run resumes without paying twice.

uv run python -m pipelines.labeling.silver --pilot                  # the 50 pilot scenes
uv run python -m pipelines.labeling.silver --pilot --models gpt-6-astra
uv run python -m pipelines.labeling.silver --litbank                # LitBank excerpts (baseline)
"""

import argparse
import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from pipelines.labeling.models import LabelRun
from services.extractor.schema import SceneExtraction, check_evidence

GUIDELINES = Path(__file__).with_name("guidelines.md")
SCENES = Path("data/processed/scenes.jsonl")
PILOT = Path("data/processed/silver_pilot.jsonl")
LITBANK = Path("data/processed/litbank_entities.jsonl")
OUT_DIR = Path("data/interim/silver")
MAX_ATTEMPTS = 2
REQUEST_TIMEOUT = 600.0  # seconds; the Anthropic and OpenAI SDKs default to the same

# USD per 1M tokens (input, output). Thinking/reasoning tokens are billed as output.
# Cached input is counted at the full input price, so Claude's cost is a slight overestimate.
PRICES = {
    "claude-opus-5": (5.0, 25.0),
    "gpt-6-astra": (10.0, 50.0),
    "gpt-6-luna": (0.10, 0.50),
    "gpt-6.1-sol": (2.0, 10.0),
    "gemini-3.1-pro-preview": (2.0, 12.0),
}

Call = Callable[[str, str], tuple[SceneExtraction, int, int]]


def _anthropic(model: str) -> Call:
    import anthropic

    client = anthropic.Anthropic()

    def call(system: str, user: str) -> tuple[SceneExtraction, int, int]:
        r = client.messages.parse(
            model=model,
            max_tokens=16000,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_format=SceneExtraction,
        )
        if r.parsed_output is None:
            raise ValueError(f"no parsed output, stop_reason={r.stop_reason}")
        u = r.usage
        tokens_in = u.input_tokens + (u.cache_creation_input_tokens or 0)
        tokens_in += u.cache_read_input_tokens or 0
        return r.parsed_output, tokens_in, u.output_tokens

    return call


def _openai(model: str) -> Call:
    import openai

    client = openai.OpenAI()

    def call(system: str, user: str) -> tuple[SceneExtraction, int, int]:
        r = client.responses.parse(
            model=model,
            input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            text_format=SceneExtraction,
        )
        if r.output_parsed is None or r.usage is None:
            raise ValueError("no parsed output")
        return r.output_parsed, r.usage.input_tokens, r.usage.output_tokens

    return call


def _gemini(model: str) -> Call:
    from google import genai

    client = genai.Client()
    schema = SceneExtraction.model_json_schema()

    def call(system: str, user: str) -> tuple[SceneExtraction, int, int]:
        # Typed as Interaction | Stream; we never request a stream, and Interaction is not exported.
        r: Any = client.interactions.create(
            model=model,
            system_instruction=system,
            input=user,
            response_format={"type": "text", "mime_type": "application/json", "schema": schema},
            timeout=REQUEST_TIMEOUT,
        )
        if r.usage is None or r.output_text is None:
            raise ValueError("no complete interaction")
        u = r.usage
        tokens_out = (u.total_output_tokens or 0) + (u.total_thought_tokens or 0)
        extraction = SceneExtraction.model_validate_json(r.output_text)
        return extraction, u.total_input_tokens or 0, tokens_out

    return call


def client_for(model: str) -> Call:
    if model.startswith("claude-"):
        return _anthropic(model)
    if model.startswith("gpt-"):
        return _openai(model)
    if model.startswith("gemini-"):
        return _gemini(model)
    raise ValueError(f"unknown provider for {model}")


def cost(model: str, tokens_in: int, tokens_out: int) -> float:
    price_in, price_out = PRICES[model]
    return (tokens_in * price_in + tokens_out * price_out) / 1_000_000


def label_scene(call: Call, model: str, scene_id: str, text: str) -> LabelRun:
    """Label one scene; retry once with the unverifiable quotes listed."""
    system = GUIDELINES.read_text(encoding="utf-8")
    user = f"<scene>\n{text}\n</scene>"
    run = LabelRun(scene_id=scene_id, model=model)
    start = time.monotonic()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        run.attempts = attempt
        try:
            extraction, tokens_in, tokens_out = call(system, user)
        except Exception as e:  # noqa: BLE001 - record any provider or parse failure
            run.error = f"{type(e).__name__}: {e}"[:500]
            continue
        run.input_tokens += tokens_in
        run.output_tokens += tokens_out
        bad = check_evidence(extraction, text)
        if not bad:
            run.extraction, run.error = extraction, None
            break
        run.error = f"{len(bad)} evidence quotes not in scene"
        quotes = "\n".join(f"- {q}" for q in bad)
        user = (
            f"<scene>\n{text}\n</scene>\n\nA previous answer used these evidence quotes, which "
            f"are not verbatim in the scene. Copy evidence exactly from the scene.\n{quotes}"
        )
    run.seconds = round(time.monotonic() - start, 1)
    run.cost_usd = round(cost(model, run.input_tokens, run.output_tokens), 4)
    return run


def run_model(model: str, scenes: dict[str, str], workers: int, suffix: str = "") -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{model}{suffix}.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(line)["scene_id"] for line in out.open(encoding="utf-8")}
    todo = [sid for sid in scenes if sid not in done]
    call = client_for(model)
    print(f"{model}: {len(todo)} to label, {len(done)} already done")
    with ThreadPoolExecutor(workers) as pool, out.open("a", encoding="utf-8") as f:
        futures = [pool.submit(label_scene, call, model, sid, scenes[sid]) for sid in todo]
        for future in as_completed(futures):  # write each scene as it finishes
            run = future.result()
            f.write(run.model_dump_json() + "\n")
            f.flush()
            status = "ok" if run.extraction else f"FAILED {run.error}"
            print(f"  {run.scene_id:<14} {run.seconds:>6}s ${run.cost_usd:.3f} {status}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Silver-label scenes with API models.")
    parser.add_argument("--models", nargs="+", default=list(PRICES))
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--pilot", action="store_true", help="only the manual-pilot scenes")
    group.add_argument("--litbank", action="store_true", help="the LitBank excerpts")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--tag", default="", help="suffix for a separate output file")
    args = parser.parse_args()
    load_dotenv(".env")

    scenes: dict[str, str] = {}
    if args.litbank:
        for line in LITBANK.open(encoding="utf-8"):
            doc = json.loads(line)
            scenes[doc["doc_id"]] = doc["text"]
    else:
        for line in SCENES.open(encoding="utf-8"):
            s = json.loads(line)
            scenes[s["scene_id"]] = s["text"]
    if args.pilot:
        pilot = [json.loads(line)["scene_id"] for line in PILOT.open(encoding="utf-8")]
        scenes = {sid: scenes[sid] for sid in pilot}
    for model in args.models:
        suffix = (".litbank" if args.litbank else "") + (f".{args.tag}" if args.tag else "")
        run_model(model, scenes, args.workers, suffix)


if __name__ == "__main__":
    main()
