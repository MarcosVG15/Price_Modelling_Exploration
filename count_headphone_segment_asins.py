"""Count how many ASINs in an ASIN->keyword_id mapping fall under a segmentation level.

The mapping file (output of assign_keywords.py) has lines:
    ASIN,keyword_id;keyword_id;...,english|english|...
An empty keyword_id field means NO_MATCH.

"Under the Headphones segment" = the ASIN has at least one keyword_id whose
key_words row has Segmentation_level_2 = 'Headphones'. The set of qualifying
keyword_ids is resolved live from the scrape DB so it stays in sync with the taxonomy.

Usage:
    python count_headphone_segment_asins.py [mapping_file] [--level 2] [--value Headphones]
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

ENV_PATH = Path(__file__).resolve().parent / ".env"
DEFAULT_MAP = "/tmp/claude-1000/-home-marcos-vargas-Documents-PROJECT-COMMAXX-OFFICIAL-ANALYSIS/c414f756-d20f-4206-a6df-b27013ba47cc/scratchpad/asin_keyword_map.txt"

SEG_COLUMNS = {1: "Segmentation_level_1", 2: "Segmentation_level_2", 3: "Segmentation_level_3"}


def segment_keyword_ids(level: int, value: str) -> dict[int, str]:
    """Return {keyword_id: english_keyword} for rows matching the segmentation value."""
    load_dotenv(ENV_PATH)
    engine = create_engine(os.getenv("SCRAPE_DATABASE_URL"))
    col = SEG_COLUMNS[level]
    query = text(f'SELECT id, key_word_english FROM key_words WHERE "{col}" ILIKE :v')
    with engine.connect() as conn:
        rows = conn.execute(query, {"v": value}).fetchall()
    return {int(r[0]): r[1] for r in rows}


def parse_mapping(path: Path):
    """Yield (asin, [keyword_id, ...]) for each non-comment line."""
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            asin = parts[0].strip()
            kw_field = parts[1].strip() if len(parts) > 1 else ""
            ids = [int(x) for x in kw_field.split(";") if x.strip().isdigit()]
            yield asin, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mapping", nargs="?", default=DEFAULT_MAP)
    ap.add_argument("--level", type=int, default=2, choices=[1, 2, 3])
    ap.add_argument("--value", default="Headphones")
    args = ap.parse_args()

    seg = segment_keyword_ids(args.level, args.value)
    seg_ids = set(seg)
    print(f"Segmentation_level_{args.level} = '{args.value}' -> {len(seg_ids)} keyword_ids:")
    for kid, name in sorted(seg.items()):
        print(f"    {kid:>4}  {name}")

    total = matched = no_match = 0
    hits = []  # (asin, matching english keywords)
    for asin, ids in parse_mapping(Path(args.mapping)):
        total += 1
        if not ids:
            no_match += 1
        overlap = [seg[i] for i in ids if i in seg_ids]
        if overlap:
            matched += 1
            hits.append((asin, overlap))

    print(f"\nTotal ASINs in mapping:                 {total}")
    print(f"ASINs with NO_MATCH (no keyword):        {no_match}")
    print(f"ASINs under '{args.value}' segment:      {matched}")
    if total:
        print(f"Share of all ASINs:                      {matched / total:.1%}")

    print(f"\nThe {matched} ASINs under '{args.value}':")
    for asin, kws in hits:
        print(f"    {asin}  <- {', '.join(sorted(set(kws)))}")


if __name__ == "__main__":
    main()
