"""Import LitBank entity and coreference annotations as gold Archivist entities.

LitBank (Bamman et al.) annotates ~2,000-word excerpts of 100 novels and is licensed CC BY 4.0.
Each coreference cluster becomes one schema Entity; mention texts become its name, aliases and
evidence. LitBank events are single-word triggers without participants, so they are not imported.

uv run python -m pipelines.labeling.litbank
"""

import io
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path

from pydantic import BaseModel

from pipelines.ingest.download import USER_AGENT
from services.extractor.schema import Entity

COMMIT = "3e50db0ffc033d7ccbb94f4d88f6b99210328ed8"
URL = f"https://codeload.github.com/dbamman/litbank/tar.gz/{COMMIT}"
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw" / "litbank"
OUT_PATH = DATA_DIR / "processed" / "litbank_entities.jsonl"

# Hand-picked family-friendly works; stands in for the Censor ingest filter until Censor v1.
# Left out: horror, graphic violence, sexual themes, racial slurs or caricature.
ALLOWLIST = {
    "1023_bleak_house",
    "105_persuasion",
    "113_the_secret_garden",
    "1155_the_secret_adversary",
    "11_alices_adventures_in_wonderland",
    "1206_the_flying_u_ranch",
    "120_treasure_island",
    "1245_night_and_day",
    "1260_jane_eyre_an_autobiography",
    "12677_personality_plus_some_experiences_of_emma_mcchesney_and_her_son_jock",
    "1327_elizabeth_and_her_german_garden",
    "1342_pride_and_prejudice",
    "1400_great_expectations",
    "145_middlemarch",
    "155_the_moonstone",
    "158_emma",
    "1661_the_adventures_of_sherlock_holmes",
    "1695_the_man_who_was_thursday_a_nightmare",
    "18581_adrift_in_new_york_tom_and_florence_braving_the_world",
    "2005_piccadilly_jim",
    "208_daisy_miller_a_study",
    "215_the_call_of_the_wild",
    "24_o_pioneers",
    "2641_a_room_with_a_view",
    "271_black_beauty",
    "27_far_from_the_madding_crowd",
    "2852_the_hound_of_the_baskervilles",
    "2891_howards_end",
    "3268_the_mysteries_of_udolpho",
    "32_herland",
    "3457_the_man_of_the_forest",
    "367_country_of_the_pointed_firs",
    "36_the_war_of_the_worlds",
    "41286_miss_marjoribanks",
    "41_the_legend_of_sleepy_hollow",
    "4276_north_and_south",
    "432_the_ambassadors",
    "434_the_circular_staircase",
    "44_the_song_of_the_lark",
    "45_anne_of_green_gables",
    "502_desert_gold",
    "514_little_women",
    "521_the_life_and_adventures_of_robinson_crusoe",
    "5348_ragged_dick_or_street_life_in_new_york_with_the_bootblacks",
    "543_main_street",
    "550_silas_marner",
    "599_vanity_fair",
    "6053_evelina_or_the_history_of_a_young_ladys_entrance_into_the_world",
    "60_the_scarlet_pimpernel",
    "766_david_copperfield",
    "77_the_house_of_the_seven_gables",
    "8867_the_magnificent_ambersons",
    "95_the_prisoner_of_zenda",
}

ENTITY_TYPES = {
    "PER": "character",
    "FAC": "location",
    "GPE": "location",
    "LOC": "location",
    "VEH": "object",
    "ORG": "org",
}


class LitBankDoc(BaseModel):
    doc_id: str
    gutenberg_id: int  # overlaps the Gutenberg catalog: keep these excerpts out of training data
    text: str  # LitBank's tokenised text, one sentence per line
    entities: list[Entity]


def download_litbank(raw_dir: Path = RAW_DIR) -> Path:
    """Extract the README and the allowlisted coref files at the pinned commit, unless cached."""
    if (raw_dir / "README.md").exists():
        return raw_dir
    request = urllib.request.Request(URL, headers={"User-Agent": USER_AGENT})  # noqa: S310
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310  # nosec B310
        data = response.read()
    raw_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar.getmembers():
            rel = member.name.split("/", 1)[-1]
            stem = Path(rel).name.removesuffix("_brat.ann").removesuffix("_brat.txt")
            if rel == "README.md" or (rel.startswith("coref/tsv/") and stem in ALLOWLIST):
                member.name = Path(rel).name
                tar.extract(member, raw_dir, filter="data")
    return raw_dir


def parse_doc(doc_id: str, ann: str, text: str) -> LitBankDoc:
    """Turn one coref TSV file into schema entities, one per cluster, in order of first mention."""
    mentions: dict[str, list[str]] = {}  # mention ID -> [start_sent, start_tok, text, type, kind]
    clusters: dict[str, list[str]] = {}  # cluster name -> mention IDs
    for line in ann.splitlines():
        row = line.split("\t")
        if row[0] == "MENTION":
            mentions[row[1]] = [row[2], row[3], row[6], row[7], row[8]]
        elif row[0] == "COREF":
            clusters.setdefault(row[2], []).append(row[1])
    clustered = {m for ids in clusters.values() for m in ids}
    groups = list(clusters.values()) + [[m] for m in mentions if m not in clustered]

    def position(mention_id: str) -> tuple[int, int]:
        return int(mentions[mention_id][0]), int(mentions[mention_id][1])

    groups = [sorted(g, key=position) for g in groups]
    groups.sort(key=lambda g: position(g[0]))

    entities = []
    for n, group in enumerate(groups, start=1):
        rows = [mentions[m] for m in group]
        named = [r[2] for r in rows if r[4] == "PROP"] or [r[2] for r in rows if r[4] == "NOM"]
        name = max(named, key=len) if named else rows[0][2]
        aliases = {r[2] for r in rows if r[4] != "PRON"} - {name}
        kind = Counter(r[3] for r in rows).most_common(1)[0][0]
        entities.append(
            Entity(
                id=f"e{n}",
                type=ENTITY_TYPES[kind],  # type: ignore[arg-type]
                name=name,
                aliases=sorted(aliases),
                evidence=rows[0][2],
            )
        )
    return LitBankDoc(
        doc_id=doc_id, gutenberg_id=int(doc_id.split("_")[0]), text=text, entities=entities
    )


def main() -> None:
    raw_dir = download_litbank()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as out:
        for doc_id in sorted(ALLOWLIST):
            ann = (raw_dir / f"{doc_id}_brat.ann").read_text(encoding="utf-8")
            text = (raw_dir / f"{doc_id}_brat.txt").read_text(encoding="utf-8")
            doc = parse_doc(doc_id, ann, text)
            out.write(doc.model_dump_json() + "\n")
            print(f"{doc_id[:50]:<50} {len(doc.entities):>4} entities")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
