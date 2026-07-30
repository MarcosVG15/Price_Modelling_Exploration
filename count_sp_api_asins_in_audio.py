"""Of the ASINs in asin_sp_api.txt, how many are present in all_feature_data_Audio.csv?

Pipeline:
  1. asin_sp_api.txt is a CSV (asin,title,keyword_ids,english_keywords,...).
  2. The Audio feature CSV is saved with a 2-row (MultiIndex) header; the real
     ASIN lives in the ("clean", "asin") column.

Usage:
    python count_sp_api_asins_in_audio.py
"""

import csv
from pathlib import Path

import pandas as pd

ASIN_TXT = Path("asin_sp_api.txt")
CSV_PATH = Path("data_files/all_feature_data_Audio.csv")


def load_sp_api_asins(path: Path) -> set[str]:
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        return {(row.get("asin") or "").strip() for row in reader if (row.get("asin") or "").strip()}


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
    sp_api_asins = load_sp_api_asins(ASIN_TXT)
    csv_asins = load_csv_asins(CSV_PATH)

    found = sp_api_asins & csv_asins
    missing = sp_api_asins - csv_asins

    print(f"SP-API ASINs (from {ASIN_TXT.name}):        {len(sp_api_asins)}")
    print(f"Unique ASINs in {CSV_PATH.name}:            {len(csv_asins)}")
    print(f"  ...found in CSV:                           {len(found)}")
    print(f"  ...missing from CSV:                       {len(missing)}")
    if sp_api_asins:
        print(f"Coverage of SP-API ASINs:                   {len(found) / len(sp_api_asins):.1%}")

    if missing:
        print(f"\nSP-API ASINs missing from CSV ({len(missing)}):")
        for a in sorted(missing):
            print(f"    {a}")


if __name__ == "__main__":
    main()
