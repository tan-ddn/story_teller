"""One way to call LLMs: provider by model ID, one price table, Langfuse tracing.

Local models are the default for running, training and testing. A model ID with an Ollama tag
("qwen3.6:27b") goes to the Ollama server at OLLAMA_HOST (default http://127.0.0.1:11435, an SSH
tunnel to the Mac Studio). claude-*, gpt-*, o3/o4-* and gemini-* IDs still reach the API
providers, for reference runs only.

Cost is computed here from PRICES at call time, because budgets (Showrunner) need it before
Langfuse has ingested anything. The same numbers are sent to Langfuse as `cost_details`, so the
run ledger and the Langfuse dashboard agree. Tracing is on when LANGFUSE_PUBLIC_KEY is set.
Prompts and outputs are not sent to Langfuse (manuscript text stays out of traces, §12.3).
"""

import json
import os
import re
import urllib.request
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

MAX_TOKENS = 16000
REQUEST_TIMEOUT = 600.0  # seconds
OLLAMA_HOST = "http://127.0.0.1:11435"
OLLAMA_TIMEOUT = 1800.0  # local models on a shared machine can be slow
OLLAMA_NUM_CTX = 16384  # rubric + a 1,500-word scene + reasoning fit comfortably

# USD per 1M tokens (input, output). Thinking/reasoning tokens are billed as output.
# Cached input is counted at the full input price, so Claude's cost is a slight overestimate.
PRICES = {
    "claude-opus-5": (5.0, 25.0),
    "gpt-6-astra": (10.0, 50.0),
    "gpt-6-luna": (0.10, 0.50),
    "gpt-6.1-sol": (2.0, 10.0),
    "o4-mini": (1.10, 4.40),
    "o3": (2.0, 8.0),
    "gemini-3.1-pro-preview": (2.0, 12.0),
}

TextCall = Callable[[str, str], tuple[str, int, int]]


def cost(model: str, tokens_in: int, tokens_out: int) -> float:
    price_in, price_out = PRICES.get(model, (0.0, 0.0))  # local models are free
    return (tokens_in * price_in + tokens_out * price_out) / 1_000_000


def _raw_call(
    model: str, schema: type[BaseModel] | None
) -> Callable[[str, str], tuple[Any, int, int]]:
    """(system, user) -> (text or parsed schema, tokens in, tokens out)."""
    if ":" in model:
        return _ollama(model, schema)
    if model.startswith("claude-"):
        import anthropic

        claude = anthropic.Anthropic()

        def call_claude(system: str, user: str) -> tuple[Any, int, int]:
            kwargs: dict[str, Any] = {
                "model": model,
                "max_tokens": MAX_TOKENS,
                "system": [
                    {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
                ],
                "messages": [{"role": "user", "content": user}],
            }
            out: Any
            if schema is None:
                r: Any = claude.messages.create(**kwargs)
                out = "".join(b.text for b in r.content if b.type == "text")
            else:
                r = claude.messages.parse(**kwargs, output_format=schema)
                if r.parsed_output is None:
                    raise ValueError(f"no parsed output, stop_reason={r.stop_reason}")
                out = r.parsed_output
            u = r.usage
            tokens_in = u.input_tokens + (u.cache_creation_input_tokens or 0)
            tokens_in += u.cache_read_input_tokens or 0
            return out, tokens_in, u.output_tokens

        return call_claude
    if model.startswith("gpt-") or re.match(r"o\d", model):  # GPT and o-series reasoning
        import openai

        gpt = openai.OpenAI()

        def call_gpt(system: str, user: str) -> tuple[Any, int, int]:
            messages: Any = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            out: Any
            if schema is None:
                r: Any = gpt.responses.create(model=model, input=messages)
                out = r.output_text
            else:
                r = gpt.responses.parse(model=model, input=messages, text_format=schema)
                if r.output_parsed is None:
                    raise ValueError("no parsed output")
                out = r.output_parsed
            if r.usage is None:
                raise ValueError("no usage in response")
            return out, r.usage.input_tokens, r.usage.output_tokens

        return call_gpt
    if model.startswith("gemini-"):
        from google import genai

        gemini = genai.Client()

        def call_gemini(system: str, user: str) -> tuple[Any, int, int]:
            kwargs: dict[str, Any] = {}
            if schema is not None:
                kwargs["response_format"] = {
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": schema.model_json_schema(),
                }
            # Typed as Interaction | Stream; we never request a stream.
            r: Any = gemini.interactions.create(
                model=model,
                system_instruction=system,
                input=user,
                timeout=REQUEST_TIMEOUT,
                **kwargs,
            )
            if r.usage is None or r.output_text is None:
                raise ValueError("no complete interaction")
            u = r.usage
            tokens_out = (u.total_output_tokens or 0) + (u.total_thought_tokens or 0)
            out = r.output_text if schema is None else schema.model_validate_json(r.output_text)
            return out, u.total_input_tokens or 0, tokens_out

        return call_gemini
    raise ValueError(f"unknown provider for {model}")


def _ollama(
    model: str, schema: type[BaseModel] | None
) -> Callable[[str, str], tuple[Any, int, int]]:
    host = os.environ.get("OLLAMA_HOST", OLLAMA_HOST).rstrip("/")
    fmt: Any = None
    if schema is not None:
        # Ollama's grammar parser rejects the \d escape used in some ID patterns.
        fmt = json.loads(json.dumps(schema.model_json_schema()).replace(r"\\d", "[0-9]"))

    def call_ollama(system: str, user: str) -> tuple[Any, int, int]:
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "num_ctx": OLLAMA_NUM_CTX},
        }
        if fmt is not None:
            body["format"] = fmt
        request = urllib.request.Request(  # noqa: S310 - host is an http(s) URL we chose
            f"{host}/api/chat",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=OLLAMA_TIMEOUT) as response:  # noqa: S310  # nosec B310
            r = json.load(response)
        text = r["message"]["content"]
        out = text if schema is None else schema.model_validate_json(text)
        return out, r.get("prompt_eval_count", 0), r.get("eval_count", 0)

    return call_ollama


def traced[T](
    model: str, name: str, call: Callable[[str, str], tuple[T, int, int]]
) -> Callable[[str, str], tuple[T, int, int]]:
    """Record each call as a Langfuse generation (model, tokens, cost; no text)."""
    if not os.environ.get("LANGFUSE_PUBLIC_KEY"):
        return call
    from langfuse import get_client

    langfuse = get_client()

    def wrapped(system: str, user: str) -> tuple[T, int, int]:
        with langfuse.start_as_current_observation(
            name=name, as_type="generation", model=model
        ) as gen:
            out, tokens_in, tokens_out = call(system, user)
            price_in, price_out = PRICES.get(model, (0.0, 0.0))
            gen.update(
                usage_details={"input": tokens_in, "output": tokens_out},
                cost_details={
                    "input": tokens_in * price_in / 1_000_000,
                    "output": tokens_out * price_out / 1_000_000,
                },
                metadata={"input_chars": len(system) + len(user)},
            )
            return out, tokens_in, tokens_out

    return wrapped


def text_call(model: str, name: str = "llm") -> TextCall:
    call: TextCall = _raw_call(model, None)
    return traced(model, name, call)


def parse_call[M: BaseModel](
    model: str, schema: type[M], name: str = "llm"
) -> Callable[[str, str], tuple[M, int, int]]:
    """Structured output validated against `schema`, on every provider."""
    call: Callable[[str, str], tuple[M, int, int]] = _raw_call(model, schema)
    return traced(model, name, call)
