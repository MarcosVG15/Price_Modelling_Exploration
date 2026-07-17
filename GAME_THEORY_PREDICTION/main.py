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


def build_vn(search_term="Headphones", k=1, resolution=1.0):
    path = os.path.join(ROOT, f"data_files/all_feature_data_{search_term}.csv")
    data = pd.read_csv(path, header=[0, 1], low_memory=False)
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
    vn, data = build_vn()

    
    
    print("getting target ")
    target = pick_target(data)
    # target = pick_specific_target(data  ,'B0DCVKJGLX')
    asin_col = ("clean", "asin")
    target_asin = target[asin_col].iloc[0] if asin_col in target.columns else None
    print("target ASIN:", target_asin if pd.notna(target_asin) else "unresolved")

    print("target : " , target )
    game = pricing_game(target, vn, competitor_strategy="promo_cycler")

    print("target cluster:", game.cluster_id)

    game.test()

    game.train(episodes=2000)
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
