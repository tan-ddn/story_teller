"""Run every Censor check before merging a change to the rubric, policies, model or code.

1. evaluation.censor_fiction: decisions on 200 Gutenberg scenes vs the frozen judge reference.
2. evaluation.censor_bans: hard bans (harmful instructions, real people) and prompt injection.
3. evaluation.censor: sexual-content hard ban on the full OpenAI moderation set, and the
   boundary passages.

Everything runs on the local Censor model (Ollama; no API calls). Scores are cached per model,
rubric and text, so only changed inputs are re-scored.

uv run python -m evaluation.censor_check
uv run python -m evaluation.censor_check --model qwen3:32b
"""

import argparse
import subprocess  # nosec B404
import sys

from guardrails.censor import DEFAULT_MODEL


def main() -> None:
    parser = argparse.ArgumentParser(description="Run every Censor check.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="the model that applies the rubric")
    args = parser.parse_args()
    checks = [
        ["evaluation.censor_fiction", "--censor", args.model, "--gate"],
        ["evaluation.censor_bans", "--model", args.model],
        ["evaluation.censor", "--model", args.model, "--gate"],
    ]
    failed = []
    for check in checks:
        print(f"\n=== {check[0]} ===", flush=True)
        if subprocess.run([sys.executable, "-m", *check], check=False).returncode:  # noqa: S603  # nosec B603
            failed.append(check[0])
    print("\nCensor check:", "FAILED " + ", ".join(failed) if failed else "all passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
