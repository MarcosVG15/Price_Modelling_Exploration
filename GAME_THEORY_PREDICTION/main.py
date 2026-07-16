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

    asin_col = ("clean", "asin")
    target_asin = target[asin_col].iloc[0] if asin_col in target.columns else None
    print("target ASIN:", target_asin if pd.notna(target_asin) else "unresolved")

    print("target : " , target )
    game = pricing_game(target, vn)

    print("target cluster:", game.cluster_id)

    game.test()

    game.train(episodes=2000)
    # Run the assessment after training
    assessment = game.assess()


    print("--- REVEALING THE BLACK BOX ---")
    print(f"Starting Price: ${assessment['start_price']:.2f}")
    print(f"Final Price Recommended: ${assessment['final_price']:.2f}")
    print(f"Total Predicted Units (52 wks): {assessment['total_units']:.0f}")

    # Print the week-by-week price + predicted demand evolution
    print("\nWeek-by-Week Schedule (ISO week | price | predicted units):")
    for iso_week, price, units in zip(
        assessment['weeks'], assessment['schedule'], assessment['demand']
    ):
        print(f"  W{iso_week:02d}: ${price:6.2f}   {units:7.1f} units")

if __name__ == "__main__":
    main()
