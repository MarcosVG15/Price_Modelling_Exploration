"""Of the SP-API ASINs that fall under the Headphones segment, how many are in the CSV?

Pipeline:
  1. asin_sp_api.txt is a CSV (asin,title,keyword_ids,english_keywords,...). Each row
     carries the keyword_ids that were matched to that ASIN.
  2. "Under the Headphones segment" = the ASIN has >=1 keyword_id whose key_words row
     has Segmentation_level_2 = 'Headphones'. That id set is resolved live from the
     scrape DB (reused from count_headphone_segment_asins.segment_keyword_ids).
  3. The Headphones feature CSV is saved with a 2-row (MultiIndex) header; the real
     ASIN lives in the ("clean", "asin") column.

So: from the SP-API list we keep only the Headphones-segment ASINs, then report how
many of THOSE are present in the CSV (e.g. "50 under Headphones -> 37 found in CSV").

Usage:
    python count_sp_api_asins_in_headphones.py [--level 2] [--value Headphones]
"""

import argparse
import csv
from pathlib import Path

import pandas as pd

from count_headphone_segment_asins import segment_keyword_ids

ASIN_TXT = Path("asin_sp_api.txt")
CSV_PATH = Path("data_files/all_feature_data_Headphones.csv")


def load_sp_api_keyword_map(path: Path) -> dict[str, set[int]]:
    """Return {asin: {keyword_id, ...}} from the SP-API CSV. Empty set = NO_MATCH."""
    mapping: dict[str, set[int]] = {}
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            asin = (row.get("asin") or "").strip()
            if not asin:
                continue
            kw_field = (row.get("keyword_ids") or "").strip()
            ids = {int(x) for x in kw_field.split(";") if x.strip().isdigit()}
            mapping.setdefault(asin, set()).update(ids)
    return mapping


def load_csv_asins(path: Path) -> set[str]:
    # 2-row MultiIndex header: level 0 = entity, level 1 = feature/id.
    df = pd.read_csv(path, header=[0, 1], low_memory=False)

    asin_cols = [c for c in df.columns if str(c[1]).strip().lower() == "asin"]
    if not asin_cols:
        raise KeyError(f"No 'asin' column found. Columns level-1: "
                       f"{sorted({str(c[1]) for c in df.columns})}")

    col = df[asin_cols[0]].dropna().astype(str).str.strip()
    return set(col[col != ""])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=2, choices=[1, 2, 3])
    ap.add_argument("--value", default="Headphones")
    args = ap.parse_args()

    sp_api_map = load_sp_api_keyword_map(ASIN_TXT)
    seg = segment_keyword_ids(args.level, args.value)
    seg_ids = set(seg)

    # SP-API ASINs whose keyword_ids place them under the segment.
    seg_asins = {asin for asin, ids in sp_api_map.items() if ids & seg_ids}

    csv_asins = load_csv_asins(CSV_PATH)
    found = seg_asins & csv_asins
    missing = seg_asins - csv_asins

    print(f"Segmentation_level_{args.level} = '{args.value}' -> {len(seg_ids)} keyword_ids")
    print(f"SP-API ASINs (from {ASIN_TXT.name}):        {len(sp_api_map)}")
    print(f"  ...under '{args.value}' segment:          {len(seg_asins)}")
    print(f"Unique ASINs in {CSV_PATH.name}:            {len(csv_asins)}")
    print(f"  ...of the {len(seg_asins)} found in CSV:  {len(found)}")
    print(f"  ...missing from CSV:                       {len(missing)}")
    if seg_asins:
        print(f"Coverage of Headphones-segment ASINs:      "
              f"{len(found) / len(seg_asins):.1%}")

    if missing:
        print(f"\nHeadphones-segment ASINs missing from CSV ({len(missing)}):")
        for a in sorted(missing):
            kws = sorted({seg[i] for i in sp_api_map[a] if i in seg_ids})
            print(f"    {a}  <- {', '.join(kws)}")


if __name__ == "__main__":
    main()
