FOLDER_PATH = "data_files"

# Product identifier column in the two-row-header feature frame (section, field).
ASIN_COL = ("clean", "asin_ean_id")

# Features excluded from grouping and voting (not real product attributes):
# scrape/marketplace metadata like the capture date. Identifiers (asin/id/upc)
# are handled separately by the type detector. Add more here as needed.
BANNED_COLUMNS = ["day"]

# Soft per-feature importance for voting (multiplies the rarity/IDF weight).
# Writing-style stats from the title/paragraph text are kept in but down-weighted
# (they reflect listing presentation, not product identity). Price stays at 1.0
# because it is meaningful. Add explicit overrides in FEATURE_WEIGHTS.
TEXT_STAT_WEIGHT = 0.75
FEATURE_WEIGHTS = {}   # e.g. {"price": 1.5} to boost a specific feature
