
import os
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from dotenv import load_dotenv
# make the repo root importable when run directly from inside the subfolder
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from config import FOLDER_PATH

from FEATURE_PREDICTION.Techniques.cluster.voting_network import voting_network
from sqlalchemy import create_engine, text
from helper_methods.general import find_csv 
load_dotenv()


def visualize_voting_networks(cluster_voting_map, limit_size=100, thresholds=None,
                              output_dir="voting_network_plots"):
    """
    Visualizes the constructed voting networks (now co-presence-normalized scores in [0, 1]).
    - limit_size: Limits the heatmap size to the first N products so it doesn't get cluttered.
    - thresholds: optional {cluster_id: threshold} from filter_vote() to overlay the cut.
    - output_dir: Directory where the figures are saved (used because the environment
      runs matplotlib's non-interactive Agg backend, so plots cannot be shown in a window).
    """
    os.makedirs(output_dir, exist_ok=True)
    thresholds = thresholds or {}

    for cluster_id, score_matrix in cluster_voting_map.items():
        if score_matrix.shape[0] < 2:
            continue  # Skip clusters with too few products to compare

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Cluster {cluster_id} Voting Network Analysis", fontsize=16)

        # --- PLOT 1: Heatmap of pairwise agreement ---
        n = min(score_matrix.shape[0], limit_size)
        sns.heatmap(
            score_matrix[:n, :n],
            cmap="viridis", vmin=0, vmax=1,
            ax=axes[0],
            cbar_kws={'label': 'Agreement (matches / co-present features)'}
        )
        axes[0].set_title(f"Pairwise Agreement (First {n} products)")
        axes[0].set_xlabel("Product Index")
        axes[0].set_ylabel("Product Index")

        # --- PLOT 2: Similarity distribution (informative pairs only) ---
        upper_tri = score_matrix[np.triu_indices_from(score_matrix, k=1)]
        informative = upper_tri[upper_tri > 0]

        axes[1].hist(informative, bins=np.linspace(0, 1, 51),
                     color="skyblue", edgecolor="black", alpha=0.7)
        thr = thresholds.get(cluster_id)
        if thr is not None:
            axes[1].axvline(thr, color="crimson", linestyle="--",
                            label=f"threshold = {thr:.3f}")
            axes[1].legend()
        axes[1].set_title("Distribution of Pairwise Agreement")
        axes[1].set_xlabel("Agreement (matches / co-present features)")
        axes[1].set_ylabel("Number of Product Pairs")

        plt.tight_layout()

        # The environment uses matplotlib's non-interactive Agg backend, so a
        # window cannot be shown. Save each figure to disk instead.
        out_path = os.path.join(output_dir, f"cluster_{cluster_id}_voting_network.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {out_path}")





if __name__ == "__main__":
    search_term = "Headphones"
    path = find_csv(search_term)
    vn  = voting_network(path , search_term)

   
    for method in ["copresent", "cosine"]:
        print(f"\n=== scoring method: {method} ===")
        voting_map = vn.vote(method=method)
        filtered = vn.filter_vote(voting_map)
        thresholds = {cid: r["threshold"] for cid, r in filtered.items()}
        visualize_voting_networks(voting_map, 1000, thresholds=thresholds,
                                  output_dir=f"voting_network_plots/{method}")

 