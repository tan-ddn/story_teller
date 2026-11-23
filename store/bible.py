"""A small persistent Story Bible storing the accepted scene extractions for one project."""

import json
import sqlite3
from pathlib import Path

from services.extractor.schema import SceneExtraction


class SQLiteBible:
    """Keep full evidence and modes; expose only unhedged main-story facts to prompts.

    Chapter order is narrative order, not an inferred story timeline. Facts are not merged or
    silently resolved; contradictions and alias ambiguity require a separate continuity checker.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with sqlite3.connect(self.path) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS chapters "
                "(number INTEGER PRIMARY KEY, scenes TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS chapter_text "
                "(number INTEGER PRIMARY KEY, text TEXT NOT NULL)"
            )
            db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")

    def bind_plan(self, plan_json: str) -> None:
        """Prevent an existing output directory from being reused for another plan."""
        canonical = json.dumps(json.loads(plan_json), sort_keys=True)
        with sqlite3.connect(self.path) as db:
            row = db.execute("SELECT value FROM metadata WHERE key = 'plan'").fetchone()
            if row and row[0] != canonical:
                raise ValueError("output directory is already bound to a different plan")
            db.execute("INSERT OR IGNORE INTO metadata VALUES ('plan', ?)", (canonical,))

    def commit(self, chapter: int, scenes: list[SceneExtraction]) -> None:
        if chapter < 1:
            raise ValueError("chapter number must be positive")
        payload = json.dumps([scene.model_dump(mode="json") for scene in scenes], sort_keys=True)
        with sqlite3.connect(self.path) as db:
            row = db.execute("SELECT scenes FROM chapters WHERE number = ?", (chapter,)).fetchone()
            if row is not None:
                if row[0] != payload:
                    raise ValueError(f"chapter {chapter} already has different committed facts")
                return
            db.execute("INSERT INTO chapters (number, scenes) VALUES (?, ?)", (chapter, payload))

    def facts_about(self, names: list[str]) -> list[str]:
        wanted = {name.casefold() for name in names}
        facts: list[str] = []
        with sqlite3.connect(self.path) as db:
            rows = db.execute("SELECT number, scenes FROM chapters ORDER BY number").fetchall()
        aliases: dict[str, set[str]] = {}
        for _, payload in rows:
            for raw in json.loads(payload):
                for entity in SceneExtraction.model_validate(raw).entities:
                    for name in [entity.name, *entity.aliases]:
                        aliases.setdefault(name.casefold(), set()).add(entity.name.casefold())
        resolved = set(wanted)
        for name in wanted:
            if len(aliases.get(name, set())) == 1:
                resolved.update(aliases[name])
        for chapter, payload in rows:
            for raw in json.loads(payload):
                scene = SceneExtraction.model_validate(raw)
                entities = {entity.id: entity for entity in scene.entities}
                matched = {
                    entity.id for entity in scene.entities if entity.name.casefold() in resolved
                }
                for claim in scene.attributes:
                    if (
                        claim.entity in matched
                        and not claim.hedged
                        and (claim.mode or scene.time.mode) == "main"
                    ):
                        name = entities[claim.entity].name
                        facts.append(f"{name}: {claim.key} = {claim.value} (chapter {chapter})")
                for event in scene.events:
                    if (
                        matched.intersection(event.participants)
                        and not event.hedged
                        and (event.mode or scene.time.mode) == "main"
                    ):
                        facts.append(f"Event: {event.description} (chapter {chapter})")
                for rel in scene.relations:
                    if (
                        {rel.src, rel.dst}.intersection(matched)
                        and not rel.hedged
                        and (rel.mode or scene.time.mode) == "main"
                    ):
                        src, dst = entities[rel.src].name, entities[rel.dst].name
                        facts.append(f"{src} {rel.type} {dst} (chapter {chapter})")
        return facts

    def commit_chapter(self, chapter: int, scenes: list[SceneExtraction], text: str) -> None:
        """Atomically store accepted text and extraction for crash recovery."""
        if chapter < 1:
            raise ValueError("chapter number must be positive")
        payload = json.dumps([scene.model_dump(mode="json") for scene in scenes], sort_keys=True)
        with sqlite3.connect(self.path) as db:
            row = db.execute("SELECT scenes FROM chapters WHERE number = ?", (chapter,)).fetchone()
            old_text = db.execute(
                "SELECT text FROM chapter_text WHERE number = ?", (chapter,)
            ).fetchone()
            if row and row[0] != payload:
                raise ValueError(f"chapter {chapter} already has different committed facts")
            if old_text and old_text[0] != text:
                raise ValueError(f"chapter {chapter} already has different committed text")
            db.execute("INSERT OR IGNORE INTO chapters VALUES (?, ?)", (chapter, payload))
            db.execute("INSERT OR IGNORE INTO chapter_text VALUES (?, ?)", (chapter, text))

    def chapter_texts(self) -> dict[int, str]:
        with sqlite3.connect(self.path) as db:
            return dict(db.execute("SELECT number, text FROM chapter_text ORDER BY number"))

    def extractions_before(self, chapter: int) -> list[tuple[int, SceneExtraction]]:
        with sqlite3.connect(self.path) as db:
            rows = db.execute(
                "SELECT number, scenes FROM chapters WHERE number < ? ORDER BY number", (chapter,)
            ).fetchall()
        return [
            (number, SceneExtraction.model_validate(raw))
            for number, payload in rows
            for raw in json.loads(payload)
        ]
