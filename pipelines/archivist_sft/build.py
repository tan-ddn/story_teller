"""Build the frozen, book-split Fable v3 Archivist corpus.

Run: python -m pipelines.archivist_sft.build
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from pipelines.labeling.dataset import DEFAULT_SILVER_DIR, load_silver_reference
from services.extractor.schema import SceneExtraction, check_evidence

SCENES = Path("data/processed/scenes.jsonl")
GUIDELINES = Path("pipelines/labeling/guidelines.md")
OUT = Path("data/interim/archivist_sft")
VALID_BOOKS = {55, 120, 1450}
TEST_BOOKS = {36, 113, 215, 421}
EXPECTED = {"train": 1165, "valid": 157, "test": 214}
EXPECTED_BOOKS = 27


def prompt(text: str) -> str:
    # Matches Archivist.extract for an empty known-character context.
    return (
        "Known character names (not IDs): []\n"
        "Do not invent known_id values; leave them null.\n<scene>\n" + text + "\n</scene>"
    )


def system_prompt(guidelines: str) -> str:
    # Matches services.llm.json_system, which the MLX provider applies at inference.
    schema = json.dumps(SceneExtraction.model_json_schema(), separators=(",", ":"))
    return (
        f"{guidelines}\n\nAnswer with one JSON object and nothing else, "
        f"matching this JSON schema:\n{schema}"
    )


def build(
    out: Path = OUT, scenes_path: Path = SCENES, labels: Path = DEFAULT_SILVER_DIR
) -> dict[str, Any]:
    reference = load_silver_reference(labels)
    scenes: dict[str, dict[str, Any]] = {}
    for line in scenes_path.read_text(encoding="utf-8").splitlines():
        scene = json.loads(line)
        sid = scene["scene_id"]
        if sid in scenes:
            raise ValueError(f"duplicate scene: {sid}")
        scenes[sid] = scene
    if missing := reference.keys() - scenes.keys():
        raise ValueError(f"silver scenes missing from corpus: {sorted(missing)}")
    system = system_prompt(GUIDELINES.read_text(encoding="utf-8"))
    rows: dict[str, list[dict[str, Any]]] = {k: [] for k in EXPECTED}
    books: dict[str, set[int]] = {k: set() for k in EXPECTED}
    hashes: dict[str, str] = {}
    for sid, extraction in sorted(reference.items()):
        scene = scenes[sid]
        book = int(scene["book_id"])
        if not sid.startswith(f"{book}-"):
            raise ValueError(f"scene/book mismatch: {sid}")
        split = "valid" if book in VALID_BOOKS else "test" if book in TEST_BOOKS else "train"
        bad = check_evidence(extraction, scene["text"])
        if bad:
            raise ValueError(f"{sid}: {len(bad)} evidence quotes absent from scene")
        target = extraction.model_copy(deep=True)
        for entity in target.entities:
            entity.known_id = None
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt(scene["text"])},
            {"role": "assistant", "content": target.model_dump_json()},
        ]
        rows[split].append({"scene_id": sid, "book_id": book, "messages": messages})
        books[split].add(book)
        hashes[sid] = hashlib.sha256(scene["text"].encode()).hexdigest()
    counts = {k: len(v) for k, v in rows.items()}
    if counts != EXPECTED or len(set.union(*books.values())) != EXPECTED_BOOKS:
        raise ValueError(f"unexpected corpus composition: {counts}, books={books}")
    if len(set(hashes.values())) != len(hashes):
        raise ValueError("duplicate scene text; inspect for leakage before splitting")
    out.mkdir(parents=True, exist_ok=True)
    for split, records in rows.items():
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    input_files = [scenes_path, GUIDELINES, *sorted(labels.glob("fable-*.jsonl"))]
    manifest = {
        "counts": counts,
        "books": {k: sorted(v) for k, v in books.items()},
        "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in input_files},
        "known_context": "empty; all target known_id values cleared",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
