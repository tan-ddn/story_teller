"""Conservative continuity checks; uncertain chronology requires author review."""

from datetime import date

from services.extractor.schema import SceneExtraction
from store.bible import SQLiteBible
from writers_room.ports import Flag

# These may change legitimately. Without reliable story-time and change explanations, flag only.
CHECKED_KEYS = {"life_status", "appearance", "species", "age", "occupation"}
EXEMPT_MODES = {"dream", "prophecy", "told"}


def story_date(scene: SceneExtraction) -> date | None:
    """Resolve a scene date only when its unambiguous absolute cues are ISO dates."""
    dates: set[date] = set()
    for cue in scene.time.cues:
        if cue.kind != "absolute" or (cue.mode and cue.mode != scene.time.mode):
            continue
        try:
            dates.add(date.fromisoformat(cue.value))
        except ValueError:
            return None
    return next(iter(dates)) if len(dates) == 1 else None


class Scripty:
    def __init__(self, bible: SQLiteBible) -> None:
        self.bible = bible

    def check(self, chapter: int, scenes: list[SceneExtraction]) -> list[Flag]:
        history = self.bible.extractions_before(chapter)
        flags: list[Flag] = []
        seen: set[str] = set()
        for current in scenes:
            current_entities = {e.id: e for e in current.entities}
            for claim in current.attributes:
                mode = claim.mode or current.time.mode
                if claim.hedged or mode in EXEMPT_MODES or claim.key not in CHECKED_KEYS:
                    continue
                entity = current_entities[claim.entity]
                name = entity.name
                names = {value.casefold() for value in [name, *entity.aliases]}
                for old_chapter, old in history:
                    old_entities = {e.id: e for e in old.entities}
                    for prev in old.attributes:
                        old_mode = prev.mode or old.time.mode
                        old_entity = old_entities[prev.entity]
                        old_names = {
                            value.casefold() for value in [old_entity.name, *old_entity.aliases]
                        }
                        if (
                            prev.hedged
                            or old_mode in EXEMPT_MODES
                            or prev.key != claim.key
                            or not names.intersection(old_names)
                            or prev.value.casefold().strip() == claim.value.casefold().strip()
                        ):
                            continue
                        message = (
                            f"Possible {name} {claim.key} change: chapter {old_chapter} says "
                            f"{prev.value!r}; chapter {chapter} says {claim.value!r}. "
                            "Check chronology and whether the change is explained."
                        )
                        if message not in seen:
                            flags.append(Flag(message, "low"))
                            seen.add(message)
            for event in current.events:
                if event.hedged or (event.mode or current.time.mode) in EXEMPT_MODES:
                    continue
                for participant in event.participants:
                    entity = current_entities[participant]
                    names = {name.casefold() for name in [entity.name, *entity.aliases]}
                    for old_chapter, old in history:
                        old_date, current_date = story_date(old), story_date(current)
                        old_entities = {item.id: item for item in old.entities}
                        for claim in old.attributes:
                            if (
                                old_date is not None
                                and current_date is not None
                                and current_date < old_date
                                and (claim.mode or old.time.mode) == old.time.mode
                                and (event.mode or current.time.mode) == current.time.mode
                            ):
                                continue
                            previous = old_entities[claim.entity]
                            previous_names = {
                                name.casefold() for name in [previous.name, *previous.aliases]
                            }
                            if (
                                claim.hedged
                                or claim.key != "life_status"
                                or claim.value != "dead"
                                or (claim.mode or old.time.mode) in EXEMPT_MODES
                                or not names.intersection(previous_names)
                            ):
                                continue
                            message = (
                                f"Possible event involving {entity.name} after a death claim "
                                f"in chapter {old_chapter}: {event.description} "
                                "Check chronology and whether the event implies the person acts."
                            )
                            if message not in seen:
                                flags.append(Flag(message, "low"))
                                seen.add(message)
            history.append((chapter, current))
        return flags
