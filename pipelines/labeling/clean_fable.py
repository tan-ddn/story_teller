"""Mechanical clean-up of the Fable silver labels (data/interim/silver/fable).

- Drop a fact's `mode` when it equals the scene's mode (the guidelines say to leave it unset).
- When two entities in one scene share a known_id, keep it on the entity whose name or aliases
  match the registry entry and clear it on the other. This never merges entities, so a wrong
  merge cannot be introduced; some correct links (identity reveals) are dropped instead.

Registry entries with duplicate names are left alone: several are genuinely different things
(different narrators of separate stories, different stations), which only judgement can tell.

uv run python -m pipelines.labeling.clean_fable
"""

import json
from collections.abc import Mapping
from pathlib import Path

from services.extractor.schema import (
    AttributeClaim,
    Entity,
    Event,
    Relation,
    SceneExtraction,
    TimeCue,
)

FABLE_DIR = Path("data/interim/silver/fable")


def _matches(entity: Entity, registry_entry: Mapping[str, object]) -> int:
    names = {n.lower() for n in [entity.name, *entity.aliases]}
    reg_aliases = registry_entry.get("aliases")
    reg_names = {str(registry_entry["name"]).lower()}
    if isinstance(reg_aliases, list):
        reg_names |= {str(a).lower() for a in reg_aliases}
    return len(names & reg_names)


def clean(
    x: SceneExtraction, registry: Mapping[str, Mapping[str, object]]
) -> tuple[SceneExtraction, int, int]:
    """Return the cleaned extraction and the number of modes and known_ids removed."""
    modes = 0
    facts: list[AttributeClaim | Event | Relation | TimeCue] = [
        *x.attributes,
        *x.events,
        *x.relations,
        *x.time.cues,
    ]
    for fact in facts:
        if fact.mode is not None and fact.mode == x.time.mode:
            fact.mode = None
            modes += 1
    links = 0
    by_id: dict[str, list[Entity]] = {}
    for e in x.entities:
        if e.known_id:
            by_id.setdefault(e.known_id, []).append(e)
    for known_id, entities in by_id.items():
        if len(entities) > 1:
            entry = registry.get(known_id, {"name": ""})
            keep = max(entities, key=lambda e: _matches(e, entry))
            for e in entities:
                if e is not keep:
                    e.known_id = None
                    links += 1
    return x, modes, links


def main() -> None:
    modes = links = 0
    for path in sorted(FABLE_DIR.glob("fable-*.jsonl")):
        book = path.stem.removeprefix("fable-")
        registry = {
            r["known_id"]: r
            for r in json.loads((FABLE_DIR / f"registry-{book}.json").read_text(encoding="utf-8"))
        }
        rows = []
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            x, m, k = clean(SceneExtraction.model_validate(row["extraction"]), registry)
            modes, links = modes + m, links + k
            rows.append(json.dumps({"scene_id": row["scene_id"], "extraction": x.model_dump()}))
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"removed {modes} redundant modes and {links} doubled known_ids")


if __name__ == "__main__":
    main()
