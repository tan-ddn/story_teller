"""The selected silver dataset and its JSONL reader."""

import json
from pathlib import Path

from services.extractor.schema import SceneExtraction

DEFAULT_SILVER_DIR = Path("data/interim/silver/fable-v3-default")


def load_silver_reference(path: Path = DEFAULT_SILVER_DIR) -> dict[str, SceneExtraction]:
    """Load labeled scenes from one JSONL file or a directory of Fable book files.

    Model-run files can contain failed records with no extraction; skip those. Reject
    duplicate scene IDs so a mixed or accidentally duplicated dataset cannot silently win
    based on file order.
    """
    paths = sorted(path.glob("fable-*.jsonl")) if path.is_dir() else [path]
    if not paths:
        raise ValueError(f"no Fable label files in {path}")
    reference: dict[str, SceneExtraction] = {}
    for file in paths:
        with file.open(encoding="utf-8") as lines:
            for line in lines:
                record = json.loads(line)
                if record.get("extraction"):
                    scene_id = record["scene_id"]
                    if scene_id in reference:
                        raise ValueError(f"duplicate silver scene ID {scene_id} in {file}")
                    reference[scene_id] = SceneExtraction.model_validate(record["extraction"])
    return reference
