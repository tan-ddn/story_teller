"""Score a predicted SceneExtraction against a reference one.

Entities are aligned one-to-one by their surface forms (name and aliases): an exact form match
beats a token-subset match ("Jim" vs "Jim Hawkins"), and a type match breaks ties. Relations
and attributes are then compared through that alignment, events by overlapping evidence, and
time cues by their value. Each category reports micro precision, recall and F1.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass

from services.extractor.schema import Entity, SceneExtraction

SYMMETRIC = {
    "sibling_of",
    "spouse_of",
    "engaged_to",
    "former_partner_of",
    "friend_of",
    "enemy_of",
    "rival_of",
    "near",
}
EVENT_MIN_OVERLAP = 0.5  # token Jaccard between evidence quotes for two events to match


@dataclass
class Counts:
    tp: int = 0
    pred: int = 0
    gold: int = 0

    def add(self, other: "Counts") -> None:
        self.tp += other.tp
        self.pred += other.pred
        self.gold += other.gold

    @property
    def precision(self) -> float:
        return self.tp / self.pred if self.pred else 0.0

    @property
    def recall(self) -> float:
        return self.tp / self.gold if self.gold else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


CATEGORIES = ("entities", "attributes", "events", "relations", "time_cues")


def _norm(text: str) -> str:
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return " ".join(w for w in text.split() if w not in {"the", "a", "an"})


def _forms(entity: Entity) -> set[str]:
    return {f for f in map(_norm, [entity.name, *entity.aliases]) if f}


def _match_score(pred: Entity, gold: Entity) -> float:
    p, g = _forms(pred), _forms(gold)
    if p & g:
        score = 2.0
    elif any(
        set(a.split()) <= set(b.split()) or set(b.split()) <= set(a.split()) for a in p for b in g
    ):
        score = 1.0
    else:
        return 0.0
    return score + (0.5 if pred.type == gold.type else 0.0)


def align_entities(pred: Iterable[Entity], gold: Iterable[Entity]) -> dict[str, str]:
    """Map predicted entity IDs to reference entity IDs, best matches first, one-to-one."""
    pairs = [(_match_score(p, g), p.id, g.id) for p in pred for g in gold]
    alignment: dict[str, str] = {}
    used: set[str] = set()
    for score, pid, gid in sorted(pairs, key=lambda x: -x[0]):
        if score > 0 and pid not in alignment and gid not in used:
            alignment[pid] = gid
            used.add(gid)
    return alignment


def _set_counts(pred: set[tuple[str, ...]], gold: set[tuple[str, ...]]) -> Counts:
    return Counts(tp=len(pred & gold), pred=len(pred), gold=len(gold))


def _relation_key(src: str, dst: str, type_: str) -> tuple[str, ...]:
    if type_ in SYMMETRIC:
        src, dst = sorted((src, dst))
    return (src, dst, type_)


def _tokens(text: str) -> set[str]:
    return set(_norm(text).split())


def _event_counts(pred: SceneExtraction, gold: SceneExtraction) -> Counts:
    gold_tokens = [_tokens(e.evidence) for e in gold.events]
    used: set[int] = set()
    tp = 0
    for event in pred.events:
        tokens = _tokens(event.evidence)
        best, best_overlap = None, EVENT_MIN_OVERLAP
        for i, g in enumerate(gold_tokens):
            overlap = len(tokens & g) / len(tokens | g) if tokens | g else 0.0
            if i not in used and overlap >= best_overlap:
                best, best_overlap = i, overlap
        if best is not None:
            used.add(best)
            tp += 1
    return Counts(tp=tp, pred=len(pred.events), gold=len(gold.events))


def score(pred: SceneExtraction, gold: SceneExtraction) -> dict[str, Counts]:
    alignment = align_entities(pred.entities, gold.entities)

    def mapped(entity_id: str) -> str:
        return alignment.get(entity_id, f"unmatched:{entity_id}")

    return {
        "entities": Counts(tp=len(alignment), pred=len(pred.entities), gold=len(gold.entities)),
        "attributes": _set_counts(
            {(mapped(a.entity), a.key) for a in pred.attributes},
            {(a.entity, a.key) for a in gold.attributes},
        ),
        "events": _event_counts(pred, gold),
        "relations": _set_counts(
            {_relation_key(mapped(r.src), mapped(r.dst), r.type) for r in pred.relations},
            {_relation_key(r.src, r.dst, r.type) for r in gold.relations},
        ),
        "time_cues": _set_counts(
            {(_norm(c.value),) for c in pred.time.cues},
            {(_norm(c.value),) for c in gold.time.cues},
        ),
    }


def total(scores: Iterable[dict[str, Counts]]) -> dict[str, Counts]:
    """Sum per-scene counts into micro-averaged totals."""
    out = {c: Counts() for c in CATEGORIES}
    for s in scores:
        for category, counts in s.items():
            out[category].add(counts)
    return out
