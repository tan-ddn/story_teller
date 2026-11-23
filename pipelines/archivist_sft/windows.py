"""Build explicitly audited, evidence-contained training windows from prepared chats."""

import argparse
import json
from pathlib import Path
from typing import Any

from pipelines.archivist_sft.build import prompt
from services.extractor.schema import SceneExtraction, _squash, check_evidence

KINDS = ("entities", "attributes", "events", "relations", "time_cues")


def slice_scene(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("require size > overlap >= 0")
    windows = []
    for start in range(0, len(text), size - overlap):
        windows.append(text[start : start + size])
        if start + size >= len(text):
            break
    return windows


def window_records(
    row: dict[str, Any], size: int, overlap: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    messages = row["messages"]
    text = messages[1]["content"].split("\n<scene>\n", 1)[1].rsplit("\n</scene>", 1)[0]
    source = SceneExtraction.model_validate_json(messages[2]["content"])
    facts: dict[str, list[Any]] = {
        "entities": source.entities,
        "attributes": source.attributes,
        "events": source.events,
        "relations": source.relations,
        "time_cues": source.time.cues,
    }
    covered: dict[str, set[int]] = {kind: set() for kind in KINDS}
    results = []
    for i, text_window in enumerate(slice_scene(text, size, overlap)):
        normalized = _squash(text_window)

        def present(item: Any, normalized: str = normalized) -> bool:
            return _squash(item.evidence) in normalized

        entity_indexes = [j for j, item in enumerate(source.entities) if present(item)]
        ids = {source.entities[j].id for j in entity_indexes}
        indexes = {
            "entities": entity_indexes,
            "attributes": [
                j for j, a in enumerate(source.attributes) if a.entity in ids and present(a)
            ],
            "events": [
                j
                for j, e in enumerate(source.events)
                if present(e)
                and set(e.participants) <= ids
                and (e.location is None or e.location in ids)
            ],
            "relations": [
                j
                for j, r in enumerate(source.relations)
                if present(r) and r.src in ids and r.dst in ids
            ],
            "time_cues": [j for j, c in enumerate(source.time.cues) if present(c)],
        }
        for kind, numbers in indexes.items():
            covered[kind].update(numbers)
        target = SceneExtraction.model_validate(
            {
                "entities": [source.entities[j] for j in indexes["entities"]],
                "attributes": [source.attributes[j] for j in indexes["attributes"]],
                "events": [source.events[j] for j in indexes["events"]],
                "relations": [source.relations[j] for j in indexes["relations"]],
                "time": {
                    "mode": source.time.mode,
                    "cues": [source.time.cues[j] for j in indexes["time_cues"]],
                },
            }
        )
        if check_evidence(target, text_window):
            raise ValueError(f"unmatched window evidence for {row['scene_id']}")
        results.append(
            {
                "scene_id": f"{row['scene_id']}@w{i:03}",
                "book_id": row["book_id"],
                "messages": [
                    messages[0],
                    {"role": "user", "content": prompt(text_window)},
                    {"role": "assistant", "content": target.model_dump_json()},
                ],
            }
        )
    coverage = {
        kind: {
            "total": len(facts[kind]),
            "covered": len(covered[kind]),
            "uncovered_indexes": sorted(set(range(len(facts[kind]))) - covered[kind]),
        }
        for kind in KINDS
    }
    return results, coverage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--size", type=int, default=3000)
    parser.add_argument("--overlap", type=int, default=1500)
    args = parser.parse_args()
    if args.output.resolve() == args.source.resolve():
        raise ValueError("output must be separate from the source dataset")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "diagnostic_only": True,
        "size": args.size,
        "overlap": args.overlap,
        "splits": {},
    }
    for split in ("train", "valid"):
        rows = []
        coverage = {}
        for line in (args.source / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            records, counts = window_records(row, args.size, args.overlap)
            rows.extend(records)
            coverage[row["scene_id"]] = counts
        with (args.output / f"{split}.jsonl").open("w", encoding="utf-8") as out:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest["splits"][split] = {"windows": len(rows), "scenes": coverage}
    # The untouched full-scene test set is never used for window training.
    (args.output / "test.jsonl").write_bytes((args.source / "test.jsonl").read_bytes())
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for split, info in manifest["splits"].items():
        print(
            split,
            "windows",
            info["windows"],
            "uncovered",
            {
                kind: sum(c[kind]["total"] - c[kind]["covered"] for c in info["scenes"].values())
                for kind in KINDS
            },
        )


if __name__ == "__main__":
    main()
