import json
import sys

import pytest

from pipelines.archivist_sft import build
from training.archivist.train import main, varied_scene_ids


def test_build_preserves_scene_and_removes_book_specific_ids(tmp_path, monkeypatch):
    labels = tmp_path / "labels"
    labels.mkdir()
    extraction = {
        "entities": [
            {
                "id": "e1",
                "type": "character",
                "name": "Alice",
                "aliases": [],
                "known_id": "B11-1",
                "evidence": "Alice walked.",
            }
        ],
        "attributes": [],
        "events": [],
        "relations": [],
        "time": {"mode": "main", "cues": []},
    }
    (labels / "fable-11.jsonl").write_text(
        json.dumps(
            {
                "scene_id": "11-c001-s01",
                "extraction": extraction,
            }
        )
        + "\n"
    )
    scenes = tmp_path / "scenes.jsonl"
    scenes.write_text(
        json.dumps(
            {
                "scene_id": "11-c001-s01",
                "book_id": 11,
                "text": "Alice walked.",
            }
        )
        + "\n"
    )
    monkeypatch.setattr(build, "EXPECTED", {"train": 1, "valid": 0, "test": 0})
    monkeypatch.setattr(build, "EXPECTED_BOOKS", 1)
    out = tmp_path / "out"
    build.build(out, scenes, labels)
    row = json.loads((out / "train.jsonl").read_text())
    assert "Alice walked." in row["messages"][1]["content"]
    target = json.loads(row["messages"][2]["content"])
    assert target["entities"][0]["known_id"] is None
    assert extraction["entities"][0]["known_id"] == "B11-1"


def test_prompt_contains_full_scene_and_disables_known_ids():
    text = "First line.\n\nLast line."
    assert build.prompt(text).endswith(f"<scene>\n{text}\n</scene>")
    assert "leave them null" in build.prompt(text)


def test_varied_smoke_selects_complete_range_deterministically():
    entries = [{"scene_id": str(i), "tokens": i} for i in range(100)]
    selected = varied_scene_ids(entries, 5)
    assert selected == {"0", "24", "49", "74", "99"}


def test_diagnostic_windows_cannot_be_used_for_full_training(tmp_path, monkeypatch):
    (tmp_path / "manifest.json").write_text(json.dumps({"diagnostic_only": True}))
    config = tmp_path / "config.yaml"
    config.write_text(f"model: qwen\nadapter_name: x\ndataset: {tmp_path}\n")
    monkeypatch.setattr(sys, "argv", ["train", str(config)])
    with pytest.raises(ValueError, match="diagnostic-only"):
        main()


@pytest.mark.parametrize("reserve", ["-1", "nan", "inf"])
def test_invalid_free_page_reserve_is_rejected(monkeypatch, reserve):
    monkeypatch.setattr(sys, "argv", ["train", "unused.yaml", "--reserve-gib", reserve])
    with pytest.raises(SystemExit) as result:
        main()
    assert result.value.code == 2
