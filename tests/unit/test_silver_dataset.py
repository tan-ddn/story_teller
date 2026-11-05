import json
import sys
from pathlib import Path

import pytest

from evaluation import ollama_baseline
from pipelines.labeling.dataset import DEFAULT_SILVER_DIR, load_silver_reference


def record(scene_id: str) -> dict[str, object]:
    return {
        "scene_id": scene_id,
        "extraction": {
            "entities": [],
            "attributes": [],
            "events": [],
            "relations": [],
            "time": {"mode": "main"},
        },
    }


def test_directory_loads_book_labels_and_rejects_duplicates(tmp_path: Path) -> None:
    first = tmp_path / "fable-11.jsonl"
    second = tmp_path / "fable-12.jsonl"
    first.write_text(json.dumps(record("11-c001-s01")) + "\n", encoding="utf-8")
    second.write_text(json.dumps(record("12-c001-s01")) + "\n", encoding="utf-8")
    (tmp_path / "registry-11.json").write_text("[]", encoding="utf-8")

    assert set(load_silver_reference(tmp_path)) == {"11-c001-s01", "12-c001-s01"}

    second.write_text(json.dumps(record("11-c001-s01")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate silver scene ID 11-c001-s01"):
        load_silver_reference(tmp_path)


def test_file_reference_skips_failed_runs(tmp_path: Path) -> None:
    path = tmp_path / "other.jsonl"
    path.write_text(
        json.dumps(record("s1")) + "\n" + json.dumps({"scene_id": "s2", "extraction": None}),
        encoding="utf-8",
    )
    assert list(load_silver_reference(path)) == ["s1"]


def test_empty_directory_is_not_a_valid_reference(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no Fable label files"):
        load_silver_reference(tmp_path)


@pytest.mark.parametrize("command", ["report", "log"])
@pytest.mark.parametrize("override", [False, True])
def test_baseline_defaults_to_selected_silver(
    monkeypatch: pytest.MonkeyPatch, command: str, override: bool
) -> None:
    seen: list[Path] = []
    if command == "report":
        monkeypatch.setattr(ollama_baseline, "report", lambda reference: seen.append(reference))
        args = ["baseline", "report"]
    else:
        monkeypatch.setattr(
            ollama_baseline,
            "log_to_mlflow",
            lambda paths, reference, params: seen.append(reference),
        )
        args = ["baseline", "log", "example.jsonl"]
    if override:
        args += ["--reference", "another.jsonl"]
    monkeypatch.setattr(sys, "argv", args)
    ollama_baseline.main()
    assert seen == [Path("another.jsonl") if override else DEFAULT_SILVER_DIR]
