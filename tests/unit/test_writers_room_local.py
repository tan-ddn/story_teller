import subprocess

import pytest

from guardrails.censor import Verdict
from writers_room.local import call_worker, screen_windows
from writers_room.ports import WorkerFailure


def test_worker_memory_failure_reports_safe_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("writers_room.local.require_memory", lambda reserve, headroom: 40.0)

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(3, "worker", stderr="private draft text")

    monkeypatch.setattr("writers_room.local.subprocess.run", fail)
    with pytest.raises(WorkerFailure) as exc:
        call_worker("writer", system="private", user="private")
    assert str(exc.value) == "writer: memory guard stopped worker (status 3)"
    assert "private" not in str(exc.value)


@pytest.mark.parametrize(
    "message",
    [
        "free-page guard: free memory 23.9 GiB is below 24.0 GiB required",
        "memory guard: MLX 36.1 GB > 36 GB, stopping",
        "memory guard: system free memory 29% < 30%, stopping",
    ],
)
def test_specific_guard_reason_is_recorded(monkeypatch: pytest.MonkeyPatch, message: str) -> None:
    monkeypatch.setattr("writers_room.local.require_memory", lambda reserve, headroom: 40.0)

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(3, "worker", stderr=f"private prose\n{message}\n")

    monkeypatch.setattr("writers_room.local.subprocess.run", fail)
    with pytest.raises(WorkerFailure) as exc:
        call_worker("writer", system="private", user="private")
    assert message in str(exc.value)
    assert "private" not in str(exc.value)


def test_window_screen_fails_closed_when_any_window_fails() -> None:
    class FakeCensor:
        def check(self, text: str, rating: str) -> Verdict:
            return Verdict(False, ("blocked",), hard_ban=True) if "BAD" in text else Verdict(True)

    verdict = screen_windows(FakeCensor(), "good " * 20 + "BAD " + "good " * 20, "teen", 40)
    assert not verdict.allowed and verdict.hard_ban and "blocked" in verdict.reasons
