# Writing comparison: The Gift of the Magi

The first reference is O. Henry's short story [The Gift of the Magi](https://www.gutenberg.org/files/7256/7256-h/7256-h.htm),
published in 1905 and available in full from Project Gutenberg. It is a complete short story, not
a chapter from a longer novel. The [SparkNotes plot summary](https://www.sparknotes.com/short-stories/the-gift-of-the-magi/summary/)
and the original informed our independently written beats in
[gift_of_the_magi_plan.json](gift_of_the_magi_plan.json). The summary is not an original author
outline. The plan is unapproved, so review it and set `approved` to `true` in a copy before use.

The reference stays outside the writer prompt. The existing smoke command uses only the plan and
the context builder, and produces a single draft. Treat the plan as a reconstruction of the story,
not a recovered outline the author actually used. The smoke prompt includes selected beats and
character attributes, but the current context builder does not include the plan's premise, places,
character arcs, or world rules. Do not infer that all fields in the JSON reach the writer.

After the run, read the complete generated draft and the original, then compare:

1. **Story execution:** sequence, motivations, the gifts and their reveal, and whether the ending
   follows naturally from the setup.
2. **Voice and prose:** narrative warmth and humor, sentence rhythm, specificity, dialogue and
   whether the prose sounds natural. Note that the plan asks for Della-centered narration, while
   the original narrator is more expansive; exact voice matching is not the goal of this prompt.
3. **Emotional effect:** whether Della's choices and the couple's affection feel earned rather than
   summarized.
4. **Practical results:** output length, drafting time, tokens and Censor result. The run does not
   measure peak memory or validate continuity.
5. **Possible memorization:** this famous story may occur in model training. Record conspicuously
   similar passages or phrasing; a close match may reflect recall rather than writing ability.

This is one small, derived-prompt comparison, not a controlled literary benchmark or a test of
long-novel continuity. The original has already been edited and published, whereas the model output
is a first draft. For a more representative evaluation, add other genres and less familiar material,
and include an original brief that cannot be reproduced from a known reference.

Follow [SMOKE_TEST.md](SMOKE_TEST.md) for environment setup and invocation, substituting an
approved copy of this plan. No model execution is required to prepare this fixture.
