"""Starter corpus: public-domain, family-friendly novels from Project Gutenberg.

The list is curated by hand. It stands in for the Censor ingest filter until Censor v1 exists.
Each entry's expected title is checked against the downloaded file, so a wrong ID fails loudly.
"""

CATALOG: dict[int, str] = {
    11: "Alice's Adventures in Wonderland",
    12: "Through the Looking-Glass",
    16: "Peter Pan",
    35: "The Time Machine",
    36: "The War of the Worlds",
    45: "Anne of Green Gables",
    55: "The Wonderful Wizard of Oz",
    103: "Around the World in Eighty Days",
    113: "The Secret Garden",
    120: "Treasure Island",
    146: "A Little Princess",
    164: "Twenty Thousand Leagues under the Sea",
    215: "The Call of the Wild",
    236: "The Jungle Book",
    244: "A Study in Scarlet",
    271: "Black Beauty",
    289: "The Wind in the Willows",
    421: "Kidnapped",
    514: "Little Women",
    521: "Robinson Crusoe",
    778: "Five Children and It",
    1448: "Heidi",
    1450: "Pollyanna",
    1661: "The Adventures of Sherlock Holmes",
    1874: "The Railway Children",
    2097: "The Sign of the Four",
    2852: "The Hound of the Baskervilles",
}
