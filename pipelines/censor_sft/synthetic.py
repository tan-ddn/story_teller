# ruff: noqa: E501  (template sentences are kept whole for readability)
"""Synthetic real-person cases for the Censor fine-tuning set.

Positives are demeaning or defamatory posts about a private person, framed as real (a named
person with a street, workplace, school or town). Negatives use the same kind of names but are
fiction, praise, neutral reviews or criticism of a public office. Each template carries its
label. Names are sampled by frequency from the US Census lists (2010 surnames, 1990 first
names; public domain), so they look like real names without pointing at a particular person.
None of these texts or templates appear in the Censor evals; build.py also checks by hash.

uv run python -m pipelines.censor_sft.synthetic   # writes pipelines/censor_sft/synthetic.jsonl
"""

import csv
import json
import random
import re
from pathlib import Path

NAMES = Path("data/raw/names")
OUT = Path(__file__).with_name("synthetic.jsonl")
SEED = 7
FILLS = 2  # samples per template
TOP = 3000  # most common names only

TOWNS = [
    "Dayton",
    "Reno",
    "Spokane",
    "Durham",
    "Leicester",
    "Bendigo",
    "Kelowna",
    "Macon",
    "Fargo",
    "Norwich",
    "Toowoomba",
    "Moncton",
    "Billings",
    "Exeter",
    "Ballarat",
    "Erie",
]
STREETS = [
    "Oak Street",
    "Mill Road",
    "Elm Avenue",
    "Station Road",
    "Pine Lane",
    "Church Street",
    "Lakeview Drive",
    "Park Road",
    "Maple Court",
    "High Street",
]
WORKPLACES = [
    "the Kroger on {street}",
    "the dental office on {street}",
    "the bank branch on {street}",
    "the gym on {street}",
    "the bakery on {street}",
    "the hardware store on {street}",
    "the vet clinic on {street}",
    "the call centre on {street}",
]
SCHOOLS = [
    "Riverside Middle School",
    "St. Anne's Primary",
    "North High",
    "Hillcrest Academy",
    "Westview Elementary",
    "Lincoln High",
]

POSITIVE = [
    "Everyone in {town} should know that {name}, who works at {work}, is a liar and a thief. She took money from my purse when I wasn't looking.",
    "{name} from {street} in {town} is the most disgusting slob in the neighbourhood. Her yard is a dump and so is she.",
    "Share this: {name}, a teacher at {school}, is a lazy fraud who shouldn't be allowed near kids. Get him out of that school.",
    "Don't trust {name} at {work}. He's a crook who shortchanges old people and laughs about it.",
    "{name} lives at {num} {street}, {town}. Total loser, no friends, still lives with his mum. Spread the word.",
    "My landlord {name} in {town} is a greedy pig who never fixes anything and pockets the deposit every time.",
    "Watch out for {name}, the receptionist at {work}. She is a nasty, two-faced gossip who spreads lies about patients.",
    "This is {name}, year 9 at {school}. Ugliest kid in the year and everyone knows it. Repost.",
    "{name}, the coach at the {town} youth league, is a drunk who shows up to games smelling of beer. Parents, keep your kids away.",
    "Avoid {name}'s plumbing business in {town} — he's a con artist and a creep who goes through people's drawers.",
    "{name} at {work} is the stupidest manager I have ever had. Fat, useless and everyone laughs at him.",
    "Reminder to the whole of {street}: {name} at number {num} steals parcels off porches. Lock your stuff up.",
    "{name} from my church in {town} is a hypocrite and a cheat who sleeps through the sermon and lies to everyone.",
    "My ex {name} works at {work}. Pathetic, broke and a compulsive liar. Ladies of {town}, stay away.",
    "{name}, the nurse on the night shift at {town} General, is cruel to old patients and should be fired.",
    "The kid at {num} {street}, {name}, is a little thief and his parents are trash. Everyone on the street knows.",
    "{name} who runs the corner shop on {street} sells out-of-date food on purpose. Crook. Tell everyone in {town}.",
    "Our neighbour {name} in {town} is a lunatic who screams at children. She's a nasty old witch.",
    "PSA {town}: {name} drives the blue pickup and works at {work}. Lying, cheating scumbag. Don't do business with him.",
    "{name} from {school} cheated on every exam and is a spoiled brat. Her parents bought her grades.",
    "Tagging {name} from {town} so everyone sees what a fake she is. Steals other people's work and lies about it.",
    "{name}, the guy who owns the car wash on {street}, is a racist creep and a thief. Boycott him.",
    "The babysitter we hired, {name} from {street} in {town}, is lazy and filthy and a liar. Never hire her.",
    "{name} at {work} is a moron who can't count and rips off every customer. Absolute joke of a person.",
    "If you live in {town}, beware of {name}. He borrows money and never pays back. Deadbeat and a fraud.",
    "{name} from apartment {num} on {street} is a junkie who robs cars at night. Watch your windows.",
    "{name}, cashier at {work}, is rude, stupid and ugly, and somebody should tell her to her face.",
    "Everyone at {school}: {name} in class 7B is a weirdo and a liar, don't talk to him.",
    "The woman who runs the daycare on {street}, {name}, is a cruel cow who neglects the kids. Pull your children out.",
    "{name} from {town} posts fake reviews and scams his customers. He's a lying little rat.",
]
NEGATIVE = [
    '"You\'re a thief and a liar, {first}," the pirate snarled, and the crew laughed as the boy climbed the rigging.',
    "Big thanks to {name} at {work}, who stayed late to help my mum with her pension forms.",
    "{name} from {street} shovels everyone's driveway after a snowfall. {town} is lucky to have him.",
    "In the novel, {name} is the meanest girl at the boarding school, until the night she rescues the headmistress's cat.",
    "My review of the bakery on {street}: the bread was stale on Tuesday and the queue was long, but the staff were polite.",
    "The mayor's plan to cut the {town} bus routes is a mistake, and the council should vote it down.",
    "Congratulations to {name} of {school}, who won the regional spelling bee this weekend!",
    "{name} is the villain of my short story: a greedy landlord who never fixes the roof, until the storm teaches him a lesson.",
    "We're throwing a farewell party for {name}, who has delivered the mail on {street} for thirty years.",
    "Lost dog near {num} {street}, {town}. If found, please call {name}. He is friendly and answers to Biscuit.",
    "The plumber, {name}, arrived an hour late and the invoice was higher than the quote, so I'd ask for a written estimate next time.",
    '"Lazy, that\'s what you are," said the old farmer, but {first} only grinned and went on whittling.',
    "{name} has been teaching piano in {town} for twenty years and every one of her students adores her.",
    "Shout-out to {name} at {work} for finding my lost wallet and handing it in.",
    "The governor's new budget underfunds schools and should be rejected by the legislature.",
    "In chapter three, {name} lies to her sister about the broken vase and feels terrible about it for days.",
    "{name}, our neighbour at number {num}, lent us his ladder and helped fix the gutter.",
    "Our team captain {name} from {school} scored the winning goal in the final minute!",
    "I found the service at {work} slow today, but the manager apologised and fixed the order.",
    "The detective suspected {name}, the butler, but the real thief was the vicar.",
    "Happy 90th birthday to {name} from {street} — still gardening every morning!",
    "{name} volunteers at the {town} food bank every Saturday and never asks for thanks.",
    "The senator's speech on farm subsidies was full of errors, and voters deserve a clearer answer.",
    '"Scoundrel!" cried the duchess, and {first} bowed so low that his wig fell into the soup.',
    "The new coach at {school}, {name}, has the whole team running laps before dawn, and they love it.",
    "Thanks to {name} for organising the {street} street party — best one yet.",
    "In the play, {name} is a con man who sells fake maps to tourists, and the audience boos him every night.",
    "{name} at {work} recommended a great book for my son. Lovely to see such kind staff in {town}.",
    "The council's decision to close the {town} library on Sundays is short-sighted and should be reversed.",
    "Our babysitter {name} taught the kids to bake bread, and now they want to do it every weekend.",
]


# Scores other than the bans, by (kind, template index): what the rubric gives the text itself.
EXTRA = {
    ("pos", 8): ({"substance": 1}, "is a drunk who shows up"),
    ("pos", 25): ({"substance": 1}, "is a junkie"),
}


# Matched negatives for the operational_harm ban (dataset v2): weapons, chemicals, drugs, hacking
# and break-ins mentioned the way fiction, history, news and safety advice mention them, with no
# usable instructions. Label: no ban; the scores the rubric gives the text itself.
OPS_NEGATIVE: list[tuple[str, dict[str, int], dict[str, str]]] = [
    (
        "The old soldier cleaned his rifle by the fire and told the boys it had jammed at the worst possible moment, the winter the river froze.",
        {},
        {},
    ),
    (
        "In chemistry class Ms. Ortiz explained why bleach and ammonia must never be mixed, and made everyone repeat it back to her.",
        {},
        {},
    ),
    (
        'The detective found a small bottle of rat poison under the sink and frowned. "Somebody wanted us to find this," she said.',
        {"violence": 1},
        {"violence": "rat poison"},
    ),
    (
        "Gunpowder was invented in China around the ninth century, probably by alchemists searching for a potion of long life.",
        {},
        {},
    ),
    (
        'The hacker in the film typed for ten seconds, said "I\'m in," and the whole audience laughed at how unrealistic it was.',
        {},
        {},
    ),
    (
        "Grandpa kept his hunting knife in a locked drawer and only took it out when he was gutting the fish he had caught.",
        {},
        {},
    ),
    (
        "The police warned residents that a new scam was going round: callers pretending to be from the bank and asking for card numbers. Never give them out.",
        {},
        {},
    ),
    (
        "The pirates rolled out the cannon and the whole ship shuddered as it fired; the gulls scattered across the bay.",
        {"violence": 1},
        {"violence": "the whole ship shuddered as it fired"},
    ),
    (
        "Mara picked the lock of the old diary with a hairpin, the way the girl in her favourite mystery novel always did, and felt terribly guilty.",
        {},
        {},
    ),
    (
        "The documentary showed how the bomb squad uses robots to inspect suspicious packages so that nobody has to get close.",
        {},
        {},
    ),
    (
        "If you find a syringe in the park, don't touch it; call the council and they will send someone to collect it safely.",
        {},
        {},
    ),
    (
        'The wizard\'s potion bubbled green and smelled of burnt toast. "Three newt eyes," he muttered, "or was it four?"',
        {},
        {},
    ),
    (
        "During the war, the village blacksmith made horseshoes by day and, the story goes, swords for the resistance by night.",
        {"violence": 1},
        {"violence": "swords for the resistance"},
    ),
    (
        "Our IT team reminds everyone to use a password manager and to report any email that asks you to click a strange link.",
        {},
        {},
    ),
    (
        "The archer drew back the string, held her breath and loosed the arrow straight into the centre of the straw target.",
        {},
        {},
    ),
    (
        "The museum's display explains how medieval castles were defended with boiling water, arrows and very thick walls.",
        {"violence": 1},
        {"violence": "boiling water, arrows"},
    ),
    (
        '"Stay away from the fireworks shed," Dad said. "Only the people from the council are allowed to set them off."',
        {},
        {},
    ),
    (
        "The news reported that customs officers had seized a shipment of illegal drugs hidden inside crates of bananas.",
        {"substance": 1},
        {"substance": "a shipment of illegal drugs"},
    ),
    (
        "In the game, Leo's character builds a catapult out of logs and rope and launches pumpkins at the castle wall.",
        {},
        {},
    ),
    (
        "The pharmacist explained that you should never take more than the dose on the label, and never share prescription pills.",
        {},
        {},
    ),
    (
        "The bank robbers in the story were caught because one of them left his library card at the scene.",
        {},
        {},
    ),
    (
        "Fire investigators found that the blaze started in an old toaster; nobody was hurt, but the kitchen was ruined.",
        {},
        {},
    ),
    (
        "The spy hid the microfilm inside a hollow coin and slipped it into the guard's pocket without him noticing.",
        {},
        {},
    ),
    (
        "A virus deleted half the school's files last year, so now the computers update every night and the teachers back up their work.",
        {},
        {},
    ),
    (
        "The knight sharpened his sword on the whetstone while the squire polished the dented helmet until it shone.",
        {},
        {},
    ),
    (
        "Scientists warned that some mushrooms in the forest are deadly and that nobody should eat a wild mushroom they cannot identify.",
        {},
        {},
    ),
    (
        "The smugglers in the novel used a secret tunnel under the inn to move barrels of brandy past the excise men.",
        {"substance": 1},
        {"substance": "barrels of brandy"},
    ),
    (
        "The safety leaflet says to keep matches and lighters out of children's reach and to test the smoke alarm every month.",
        {},
        {},
    ),
    (
        "Ellie's uncle, a locksmith, showed her the tiny pins inside a lock and explained why cheap padlocks are easy to open.",
        {},
        {},
    ),
    (
        "The general studied the map, moved two wooden blocks across the river and declared that the attack would come at dawn.",
        {"violence": 1},
        {"violence": "the attack would come at dawn"},
    ),
    (
        "The magician's trick knife folded into the handle, and the audience screamed with delight when he 'stabbed' the apple.",
        {},
        {},
    ),
    (
        "Cybersecurity experts say the best defence against phishing is to slow down and check who really sent the message.",
        {},
        {},
    ),
    (
        "The farmer kept a shotgun to scare off foxes, though in twenty years he had never once fired it.",
        {},
        {},
    ),
    (
        "A lab accident in the comic gave Max the ability to talk to ants, which was less useful than he had hoped.",
        {},
        {},
    ),
    (
        "The court heard that the defendant had bought the stolen credit card numbers online; he was sentenced to two years.",
        {},
        {},
    ),
    (
        "The volcano in the science fair project erupted with baking soda and vinegar, and the judges gave it second prize.",
        {},
        {},
    ),
    (
        "Rangers asked hikers not to light campfires during the drought, because a single spark could set the hills alight.",
        {},
        {},
    ),
    (
        "The heist crew argued for an hour about the plan, then gave up and ordered pizza instead.",
        {},
        {},
    ),
    (
        "Historians believe the poison in the old king's cup was arsenic, though the case has never been proven.",
        {"violence": 1},
        {"violence": "the poison in the old king's cup"},
    ),
    (
        "The police dog found the hidden weapons under the floorboards, and the officers sealed off the house.",
        {"violence": 1},
        {"violence": "the hidden weapons"},
    ),
]


def _names(path: Path, col: int, sep: str | None) -> tuple[list[str], list[float]]:
    names: list[str] = []
    weights: list[float] = []
    with path.open(encoding="utf-8") as f:
        rows = csv.reader(f) if sep == "," else (line.split() for line in f)
        for i, row in enumerate(rows):
            if sep == "," and i == 0:
                continue
            if len(names) == TOP:
                break
            names.append(row[0].title())
            weights.append(float(row[col]) if row[col] not in ("(S)", "") else 0.01)
    return names, weights


def _gender(template: str) -> str | None:
    """'f' or 'm' when the template's pronouns fix the person's gender, else None."""
    if re.search(r"\b(she|her|hers)\b", template, re.IGNORECASE):
        return "f"
    if re.search(r"\b(he|him|his)\b", template, re.IGNORECASE):
        return "m"
    return None


def generate() -> list[dict[str, object]]:
    rng = random.Random(SEED)  # noqa: S311  # nosec B311
    last, lw = _names(NAMES / "Names_2010Census.csv", 2, ",")
    female, fw = _names(NAMES / "dist.female.first", 1, None)
    male, mw = _names(NAMES / "dist.male.first", 1, None)
    rows: list[dict[str, object]] = []
    for kind, templates in (("pos", POSITIVE), ("neg", NEGATIVE)):
        for t, template in enumerate(templates):
            for k in range(FILLS):
                gender = _gender(template) or ("f" if rng.random() < 0.5 else "m")
                first = rng.choices(female, fw)[0] if gender == "f" else rng.choices(male, mw)[0]
                street = rng.choice(STREETS)
                fill = {
                    "first": first,
                    "name": f"{first} {rng.choices(last, lw)[0]}",
                    "town": rng.choice(TOWNS),
                    "street": street,
                    "work": rng.choice(WORKPLACES).format(street=street),
                    "school": rng.choice(SCHOOLS),
                    "num": rng.randint(2, 240),
                }
                scores, quote = EXTRA.get((kind, t), ({}, ""))
                text = template.format(**fill)
                quotes = [{"category": c, "text": quote} for c in scores]
                if any(q["text"] not in text for q in quotes):
                    raise ValueError(f"quote not in text for {kind}{t}")
                rows.append(
                    {
                        "id": f"synth-rp-{kind}{t:02d}-{k}",
                        "text": text,
                        "label": {"real_person": kind == "pos", **scores, "quotes": quotes},
                    }
                )
    for i, (text, ops_scores, ops_quotes) in enumerate(OPS_NEGATIVE):
        q = [{"category": c, "text": t} for c, t in ops_quotes.items()]
        if any(x["text"] not in text for x in q):
            raise ValueError(f"quote not in ops negative {i}")
        rows.append(
            {
                "id": f"synth-ops-neg{i:02d}",
                "text": text,
                "label": {
                    "operational_harm": False,
                    "real_person": False,
                    **ops_scores,
                    "quotes": q,
                },
            }
        )
    return rows


def main() -> None:
    rows = generate()
    with OUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(rows)} synthetic cases written to {OUT}")


if __name__ == "__main__":
    main()
