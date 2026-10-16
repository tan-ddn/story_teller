"""MLflow logging for the Censor evals, kept apart from other experiments.

Runs go to their own experiments: "censor-eval" for real runs and "censor-eval-smoke" for smoke
tests, never to another team's experiment. evaluation.censor_check opens one parent run per
invocation and passes its ID to the three checks, which each log one nested child run with:
- tags: component=censor, check, censor_model, rubric_sha, policy versions, git commit, smoke
- params: the check's settings
- metrics: the check's numbers, plus gate_passed
- artifact: the check's report file

Logging is on only when MLFLOW_TRACKING_URI is set. The tracking server needs
MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD in the environment (.env).
"""

import hashlib
import os
import re
import shutil
import subprocess  # nosec B404
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from guardrails.censor import HERE as GUARDRAILS
from guardrails.censor import policy

EXPERIMENT = "censor-eval"
SMOKE_EXPERIMENT = "censor-eval-smoke"
PARENT_ENV = "CENSOR_MLFLOW_PARENT"  # parent run ID, set by censor_check for the child checks
SMOKE_ENV = "CENSOR_SMOKE"  # "1" while a smoke test runs


def enabled() -> bool:
    return bool(os.environ.get("MLFLOW_TRACKING_URI"))


def smoke() -> bool:
    return os.environ.get(SMOKE_ENV) == "1"


def experiment() -> str:
    return SMOKE_EXPERIMENT if smoke() else EXPERIMENT


def _git_commit() -> str:
    """Short commit of the checkout, with "-dirty" if Censor-related files have changes.
    CENSOR_GIT_COMMIT overrides it for copies without .git (e.g. the queue on the Mac)."""
    if commit := os.environ.get("CENSOR_GIT_COMMIT"):
        return commit
    git = shutil.which("git")
    if git is None:
        return "unknown"
    try:
        head = subprocess.run(  # noqa: S603  # nosec B603
            [git, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(  # noqa: S603  # nosec B603
            [git, "status", "--porcelain", "--", "guardrails", "evaluation", "services"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return head + ("-dirty" if dirty else "")


def tags(model: str) -> dict[str, str]:
    rubric = (GUARDRAILS / "rubric.md").read_text(encoding="utf-8")
    return {
        "component": "censor",
        "censor_model": model,
        "rubric_sha": hashlib.sha256(rubric.encode("utf-8")).hexdigest()[:16],
        "policy_children": str(policy("children").version),
        "policy_teen": str(policy("teen").version),
        "git_commit": _git_commit(),
        "smoke": str(smoke()).lower(),
    }


def _flatten(metrics: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
    """Numeric leaves of a nested report, with keys joined by '.'."""
    out: dict[str, float] = {}
    for key, value in metrics.items():
        name = re.sub(r"[^\w\-. /]", "_", f"{prefix}{key}".replace("->", "_to_"))
        if isinstance(value, bool):
            out[name] = float(value)
        elif isinstance(value, int | float):
            out[name] = float(value)
        elif isinstance(value, Mapping):
            out |= _flatten(value, f"{name}.")
    return out


def start_parent(model: str) -> str | None:
    """Open censor_check's parent run and return its ID (None when logging is off)."""
    if not enabled():
        return None
    import mlflow

    mlflow.set_experiment(experiment())
    with mlflow.start_run(run_name=f"censor_check {model}", tags=tags(model)) as run:
        return str(run.info.run_id)


def end_parent(run_id: str | None, passed: bool) -> None:
    """Record the outcome: gate_passed for a real run; for a smoke run, whose gates are not
    applied, only checks_completed (every check ran without crashing)."""
    if run_id is None:
        return
    import mlflow

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("checks_completed" if smoke() else "gate_passed", float(passed))


def log_check(
    check: str,
    model: str,
    params: Mapping[str, Any],
    metrics: Mapping[str, Any],
    passed: bool,
    artifact: Path,
) -> None:
    """Log one check as a run (nested under censor_check's parent run when there is one)."""
    if not enabled():
        return
    import mlflow

    mlflow.set_experiment(experiment())
    parent = os.environ.get(PARENT_ENV)
    with mlflow.start_run(run_name=f"{check} {model}", nested=bool(parent), parent_run_id=parent):
        mlflow.set_tags(tags(model) | {"check": check})
        mlflow.log_params({k: str(v) for k, v in params.items()})
        mlflow.log_metrics(_flatten(metrics) | {"gate_passed": float(passed)})
        mlflow.log_artifact(str(artifact))
