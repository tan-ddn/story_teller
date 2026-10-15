"""Run every Censor check before merging a change to the rubric, policies, model or code.

1. evaluation.censor_fiction: decisions on 200 Gutenberg scenes vs an independent judge.
2. evaluation.censor_bans: hard bans (harmful instructions, real people) and prompt injection.
3. evaluation.censor: sexual-content hard ban and boundary passages on the OpenAI benchmark.

Scores are cached per model, rubric and text, so an unchanged setup costs nothing; after a
rubric change a full run costs about $4.30 with o3.

uv run python -m evaluation.censor_check
"""

import subprocess  # nosec B404
import sys

MODEL = "o3"
JUDGE = "claude-opus-5"
CHECKS = [
    ["evaluation.censor_fiction", "--censor", MODEL, "--judge", JUDGE, "--gate"],
    ["evaluation.censor_bans", "--model", MODEL],
    ["evaluation.censor", "--model", MODEL, "--sample", "300", "--gate"],
]


def main() -> None:
    failed = []
    for check in CHECKS:
        print(f"\n=== {check[0]} ===", flush=True)
        if subprocess.run([sys.executable, "-m", *check], check=False).returncode:  # noqa: S603  # nosec B603
            failed.append(check[0])
    print("\nCensor check:", "FAILED " + ", ".join(failed) if failed else "all passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
