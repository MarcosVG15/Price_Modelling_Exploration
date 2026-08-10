import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd

from config import ASIN_COL
from Matrix_Factorization_Clustering.Matrix_Factorization import matrix_factorization
from game import pricing_game


def _estimate_nrows_for_mb(path, max_mb, sample_bytes=1_000_000):
    """pd.read_csv has no direct byte-size limit, only nrows -- approximate nrows for a
    target file size by sampling the first ~1MB to get avg bytes/row (fast: avoids
    reading the whole multi-GB file just to measure it)."""
    with open(path, "rb") as f:
        sample = f.read(sample_bytes)
    n_lines = sample.count(b"\n")
    if n_lines == 0:
        return None
    avg_bytes_per_row = len(sample) / n_lines
    return max(1, int((max_mb * 1024 * 1024) / avg_bytes_per_row))


def build_vn(search_term="Audio", k=1, resolution=1.0, max_rows=20_000, max_mb=None):
    path = os.path.join(ROOT, f"data_files/all_feature_data_{search_term}.csv")

    nrows = max_rows
    if max_mb is not None:
        est_rows = _estimate_nrows_for_mb(path, max_mb)
        nrows = est_rows if nrows is None else min(nrows, est_rows)

    data = pd.read_csv(path, header=[0, 1], low_memory=False, nrows=nrows)
    data = data.drop(columns=data.columns[0])

    vn = matrix_factorization(data)

    vn.matrix_factorization_tf_idf()

    vn.svd_product_communities(k=k, resolution=resolution)

    return vn, data



def pick_specific_target(data, asin, asin_col=("clean", "asin")):
    match = data.loc[data[asin_col] == asin]
    if match.empty:
        raise ValueError(f"asin {asin!r} not found in data")
    return match.iloc[[0]]

def pick_target(data, random_state=2):
    unique = data.drop_duplicates(subset=[ASIN_COL])
    rng = np.random.default_rng(random_state)
    i = int(rng.integers(len(unique)))
    return unique.iloc[[i]]


def main():
    print("building vn ")
    # max_rows=28_000 -> large enough to include B09DGN7MRD (row 26,445 in Audio.csv), still
    # safely under the ~30-40k row ceiling where svd_product_communities' dense similarity
    # matrix starts risking a segfault (see Matrix_Factorization.py:221).
    vn, data = build_vn(max_rows=28_000)



    print("getting target ")
    # B09DGN7MRD has 111 weeks of real sales_traffic_daily history (nb_dispersion and the
    # seasonal index calibrate for real, instead of falling back to flat defaults) -- swap
    # in pick_target(data) for a random draw instead, which may land on a scrape-only,
    # never-sold product with no real calibration data (see the cluster-966 comparison).
    target = pick_specific_target(data, "B09DGN7MRD")
    # target = pick_target(data)
    asin_col = ("clean", "asin")
    target_asin = target[asin_col].iloc[0] if asin_col in target.columns else None
    print("target ASIN:", target_asin if pd.notna(target_asin) else "unresolved")

    print("target : " , target )
    game = pricing_game(target, vn, competitor_strategy="promo_cycler")

    print("target cluster:", game.cluster_id)

    game.test()

    checkpoint_path = os.path.join(HERE, "checkpoints", f"ppo_phase1_cluster{game.cluster_id}.pt")
    # Phase 1: PPO vs the stationary promo_cycler (clean single-agent training).
    # Phase 2: warm-started RL-vs-RL self-play, alternating which side learns each
    # round, checked every round against the fixed promo_cycler benchmark.
    result = game.train_curriculum(promo_episodes=3000, rl_rounds=6, episodes_per_round=500,
                                    checkpoint_path=checkpoint_path)
    curriculum_path = game.plot_curriculum(result)
    print(f"  saved curriculum plot -> {curriculum_path}")

    single_run_path = game.plot_single_run()
    print(f"  saved single run -> {single_run_path}")

    # Run the assessment after training
    assessment = game.assess()


    print("\n--- MONTE CARLO (n=500) ---")
    mc = game.monte_carlo(n=500, randomize_week=False)
    pr, un = mc["profit"], mc["units"]
    print(f"Season profit: mean=${pr['mean']:,.0f}  "
          f"90% band [${pr['p05']:,.0f}, ${pr['p95']:,.0f}]  (std ${pr['std']:,.0f})")
    print(f"Season units:  mean={un['mean']:,.0f}  "
          f"90% band [{un['p05']:,.0f}, {un['p95']:,.0f}]")

    tag = str(target_asin) if pd.notna(target_asin) else f"cluster{game.cluster_id}"
    for metric in ("demand", "profit", "price"):
        out = os.path.join(HERE, f"mc_band_{metric}_{tag}.png")
        game.plot_bands(mc, metric=metric, path=out)
        print(f"  saved {metric} band -> {out}")

if __name__ == "__main__":
    main()
