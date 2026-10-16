"""Run every Censor check before merging a change to the rubric, policies, model or code.

1. evaluation.censor_fiction: decisions on 200 Gutenberg scenes vs the frozen judge reference.
2. evaluation.censor_bans: hard bans (harmful instructions, real people) and prompt injection.
3. evaluation.censor: sexual-content hard ban on the full OpenAI moderation set, and the
   boundary passages.

Everything runs on the local Censor model (Ollama; no API calls). Scores are cached per model,
rubric and text, so only changed inputs are re-scored.

Each invocation is one parent run in MLflow ("censor-eval"), with a child run per check.
--smoke runs a few cases per check, logs to "censor-eval-smoke" and does not apply the gates.

uv run python -m evaluation.censor_check --smoke
uv run python -m evaluation.censor_check
uv run python -m evaluation.censor_check --model qwen3:32b
"""

import argparse
import os
import subprocess  # nosec B404
import sys

from dotenv import load_dotenv

from evaluation import censor_mlflow
from guardrails.censor import DEFAULT_MODEL

SMOKE_LIMIT = 3  # cases per group in a smoke test
SMOKE_SAMPLE = 10  # benchmark samples in a smoke test


def main() -> None:
    parser = argparse.ArgumentParser(description="Run every Censor check.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="the model that applies the rubric")
    parser.add_argument("--smoke", action="store_true", help="a few cases each, no gates")
    args = parser.parse_args()
    load_dotenv(".env")
    if args.smoke:
        os.environ[censor_mlflow.SMOKE_ENV] = "1"
    gate = [] if args.smoke else ["--gate"]
    fiction = ["--limit", str(SMOKE_LIMIT)] if args.smoke else []
    bans = ["--limit", str(SMOKE_LIMIT)] if args.smoke else []
    sample = ["--sample", str(SMOKE_SAMPLE)] if args.smoke else []
    checks = [
        ["evaluation.censor_fiction", "--censor", args.model, *fiction, *gate],
        ["evaluation.censor_bans", "--model", args.model, *bans, *gate],
        ["evaluation.censor", "--model", args.model, *sample, *gate],
    ]
    parent = censor_mlflow.start_parent(args.model)
    env = os.environ | ({censor_mlflow.PARENT_ENV: parent} if parent else {})
    failed = []
    for check in checks:
        print(f"\n=== {check[0]} ===", flush=True)
        cmd = [sys.executable, "-m", *check]
        if subprocess.run(cmd, check=False, env=env).returncode:  # noqa: S603  # nosec B603
            failed.append(check[0])
    censor_mlflow.end_parent(parent, not failed)
    print("\nCensor check:", "FAILED " + ", ".join(failed) if failed else "all passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
