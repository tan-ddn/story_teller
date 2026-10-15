"""Context Builder: assembles each writer prompt within a token budget.

Sections in priority order: rating limits, style sheet, the chapter's beats, Story Bible facts for
the characters in the chapter, then the end of the previous chapter. Lower-priority sections are
trimmed first. Strategies are named and versioned so they can be compared experimentally;
"no_memory" leaves out the Bible facts (the baseline for the Story Bible memory experiment).
"""

from typing import Literal

from writers_room.models import ChapterPlan, StoryPlan
from writers_room.ports import Bible

Strategy = Literal["bible_v1", "no_memory"]

CHARS_PER_TOKEN = 4  # rough estimate; good enough for budgeting
RECENT_CHARS = 6000  # at most this much of the previous chapter

SYSTEM = """You are a novelist in a writers' room, writing one chapter of "{title}".
Write only the chapter text: no title, notes or commentary. Follow the beats in order and keep \
every fact consistent with the Story Bible facts you are given. Separate scenes with a line \
containing only "* * *".

The book is family-friendly and rated {rating}. Sexual content is never allowed; keep peril, \
conflict and language suitable for that audience.

Style: {tense} tense; {pov_rules}. Tone: {tone}."""


def tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN + 1


def system_prompt(plan: StoryPlan) -> str:
    s = plan.style
    return SYSTEM.format(
        title=plan.title,
        rating=plan.rating,
        tense=s.tense,
        pov_rules=s.pov_rules,
        tone=s.tone,
    )


def _names(plan: StoryPlan, chapter: ChapterPlan) -> list[str]:
    """The POV character plus every planned character named in a beat."""
    text = " ".join(chapter.beats)
    named = [c.name for c in plan.characters if c.name != chapter.pov]
    named = [n for n in named if n in text or any(a in text for a in plan.character(n).aliases)]
    return [chapter.pov, *named]


def build(
    plan: StoryPlan,
    chapter: ChapterPlan,
    bible: Bible,
    previous: str,
    budget: int,
    strategy: Strategy = "bible_v1",
) -> str:
    """The user prompt for one chapter; `budget` is in tokens."""
    names = _names(plan, chapter)
    beats = "\n".join(f"{i}. {b}" for i, b in enumerate(chapter.beats, 1))
    voices = "\n".join(f"- {n}: {plan.style.voices[n]}" for n in names if n in plan.style.voices)
    head = (
        f"Chapter {chapter.number}: {chapter.title}\n"
        f"POV: {chapter.pov}. Story time: {chapter.story_time}. Mode: {chapter.mode}.\n\n"
        f"Beats:\n{beats}\n"
    )
    if voices:
        head += f"\nDialogue voices:\n{voices}\n"
    left = budget - tokens(head)

    facts = ""
    if strategy == "bible_v1":
        lines = bible.facts_about(names)
        kept: list[str] = []
        for line in lines:  # keep whole lines while they fit, earliest first
            if tokens("\n".join([*kept, line])) > left // 2:
                break
            kept.append(line)
        if kept:
            facts = "\nStory Bible facts so far:\n" + "\n".join(f"- {x}" for x in kept) + "\n"
        left -= tokens(facts)

    recent = ""
    wrapper = "\nEnd of the previous chapter:\n<previous>\n{}\n</previous>\n"
    left -= tokens(wrapper) + 1
    if previous and left > 0:
        tail = previous[-min(RECENT_CHARS, left * CHARS_PER_TOKEN) :]
        recent = wrapper.format(tail)
    return head + facts + recent
