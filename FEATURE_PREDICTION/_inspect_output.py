"""
Inspect the *structure* of what matrix_factorization.feature_clusterize() yields.
Run from the project root:  python inspect_output.py
"""
import numpy as np
import pandas as pd

# make the repo root importable when run directly from inside the subfolder
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from helper_methods.general import find_csv
from Matrix_Factorization_Clustering.Matrix_Factorization import matrix_factorization

search_term = "Headphones"
path = find_csv(search_term)

mf = matrix_factorization(path, search_term)

print("=" * 70)
print("INPUT  self.feature_data")
print("=" * 70)
print("shape (rows=products, cols=feature columns):", mf.feature_data.shape)
print("column MultiIndex levels: level0=section, level1=datapoint")
print("first 6 columns:")
for c in mf.feature_data.columns[:6]:
    print("   ", c)

# feature_clusterize returns a GENERATOR yielding (feat, codes, present) per feature
gen = mf.feature_clusterize(mf.feature_data)

print()
print("=" * 70)
print("OUTPUT  feature_clusterize(...) -> generator of (feat, codes, present)")
print("=" * 70)
print("type of return value:", type(gen).__name__)

rows = []
first_examples = []
for i, (feat, codes, present) in enumerate(gen):
    rows.append({
        "feature": feat,
        "codes.shape": codes.shape,
        "n_present": int(present.sum()),
        "n_missing(-1)": int((codes == -1).sum()),
        "distinct_codes": int(len(np.unique(codes[codes != -1]))),
        "code_min": int(codes.min()),
        "code_max": int(codes.max()),
    })
    if i < 3:
        first_examples.append((feat, codes, present))

summary = pd.DataFrame(rows)
print(f"\nnumber of features yielded (one tuple each): {len(summary)}")
print(f"each 'codes' is 1 integer per product -> length = {rows[0]['codes.shape'][0]} products")
print("\nper-feature structure (first 15 features):")
with pd.option_context("display.max_columns", None, "display.width", 140):
    print(summary.head(15).to_string(index=False))

print("\n--- concrete peek at the first 3 features (first 12 products) ---")
for feat, codes, present in first_examples:
    print(f"\nfeature: {feat}")
    print("  codes  :", codes[:12], "...")
    print("  present:", present[:12].astype(int), "...")
