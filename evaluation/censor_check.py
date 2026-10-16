"""Run every Censor check before merging a change to the rubric, policies, model or code.

1. evaluation.censor_fiction: decisions on 200 Gutenberg scenes vs an independent judge.
2. evaluation.censor_bans: hard bans (harmful instructions, real people) and prompt injection.
3. evaluation.censor: sexual-content hard ban on the full OpenAI moderation set, and the
   boundary passages.

Scores are cached per model, rubric and text, so an unchanged setup costs nothing; after a
rubric change a full run costs about $11 with o3.

uv run python -m evaluation.censor_check
uv run python -m evaluation.censor_check --model claude-opus-5 --judge o3
"""

import argparse
import subprocess  # nosec B404
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Run every Censor check.")
    parser.add_argument("--model", default="o3", help="the model that applies the rubric")
    parser.add_argument("--judge", default="claude-opus-5", help="independent fiction judge")
    args = parser.parse_args()
    if args.model == args.judge:
        sys.exit("The fiction judge must be a different model from the Censor model.")
    checks = [
        ["evaluation.censor_fiction", "--censor", args.model, "--judge", args.judge, "--gate"],
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
