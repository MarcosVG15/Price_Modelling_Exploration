import os
import numpy as np
import pandas as pd
import networkx as nx

import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
# make the repo root importable when run directly from inside the subfolder
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from helper_methods.general import find_csv
from helper_methods.calculations import calculate_unified_score
from Matrix_Factorization_Clustering.Matrix_Factorization import matrix_factorization
from Matrix_Factorization_Clustering.effective_tests import presence_auc , global_fidelity , modularity_k,conductance , LCC_computer

def visualize_mf_votes(agreement, limit_size=1000,
                       output_dir="voting_network_plots/voting_network_plots-VN-MF"):
    """Heatmap + distribution of the full MF voting matrix (no pre-clustering).

    Same layout/scale as the voting-network plots so they compare directly.
    Saved to disk because the environment runs matplotlib's Agg backend.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("MF Voting Network — All Products", fontsize=16)

    # --- PLOT 1: Heatmap of pairwise agreement ---
    n = min(agreement.shape[0], limit_size)
    sns.heatmap(
        agreement[:n, :n],
        cmap="viridis", vmin=0, vmax=1,
        ax=axes[0],
        cbar_kws={'label': 'Agreement (matches / co-present features)'}
    )
    axes[0].set_title(f"Pairwise Agreement (First {n} products)")
    axes[0].set_xlabel("Product Index")
    axes[0].set_ylabel("Product Index")

    # --- PLOT 2: Similarity distribution (informative pairs only) ---
    upper_tri = agreement[np.triu_indices_from(agreement, k=1)]
    informative = upper_tri[upper_tri > 0]

    axes[1].hist(informative, bins=np.linspace(0, 1, 51),
                 color="skyblue", edgecolor="black", alpha=0.7)
    axes[1].set_title("Distribution of Pairwise Agreement")
    axes[1].set_xlabel("Agreement (matches / co-present features)")
    axes[1].set_ylabel("Number of Product Pairs")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "mf_voting_network.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def visualize_product_communities(vn, limit_size=10000, graph_sample=500,
                                  output_dir="voting_network_plots/voting_network_plots-VN-MF"):
    """Show the Louvain product communities three ways so the clusters are
    visible: (1) the community graph itself, (2) the cosine-similarity matrix
    reordered by community (block structure), (3) the SVD embedding scattered
    in 2D and coloured by community. Call after vn.svd_product_communities()."""
    labels = vn.product_labels
    sim = vn.product_similarity
    G = vn.product_graph
    n = len(labels)
    n_comm = len(set(labels))

    os.makedirs(output_dir, exist_ok=True)
    fig = plt.figure(figsize=(17, 5.5))

    # --- PANEL 1: node-link community graph (subsample for a readable layout) ---
    ax1 = fig.add_subplot(1, 3, 1)
    sample = list(range(min(graph_sample, n)))
    H = G.subgraph(sample)
    pos = nx.spring_layout(H, seed=42)
    nx.draw_networkx_edges(H, pos, ax=ax1, alpha=0.12, width=0.5)
    nx.draw_networkx_nodes(H, pos, ax=ax1, node_size=18,
                           node_color=[labels[i] for i in H.nodes()], cmap="tab20")
    ax1.set_title(f"Community graph (first {len(sample)} products)")
    ax1.axis("off")

    # --- PANEL 2: cosine similarity reordered by community label ---
    ax2 = fig.add_subplot(1, 3, 2)
    order = np.argsort(labels, kind="stable")[:min(limit_size, n)]
    im = ax2.imshow(sim[np.ix_(order, order)], cmap="viridis", vmin=-0.2, vmax=1)
    ax2.set_title(f"Cosine sim reordered by community (first {len(order)})")
    fig.colorbar(im, ax=ax2, fraction=0.046)

    # --- PANEL 3: 2D SVD embedding coloured by community ---
    ax3 = fig.add_subplot(1, 3, 3)
    xy = vn.X_transformed[:, 1:3]   # dims 2-3 (dim 1 mostly tracks listing size)
    ax3.scatter(xy[:, 0], xy[:, 1], c=labels, cmap="tab20", s=6, alpha=0.5)
    ax3.set_title("SVD embedding (dims 2-3) by community")
    ax3.set_xlabel("latent dim 2")
    ax3.set_ylabel("latent dim 3")

    fig.suptitle(f"SVD Product Communities (Louvain): "
                 f"{n_comm} clusters over {n} products", fontsize=15)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "mf_product_communities.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def plot_modularity_grid(results, resolutions, save_dir="vote_effective_test_charts", filename="modularity_grid_search.png"):
    """
    Plots and saves the modularity scores to a local folder.
    """
    # Create the directory if it does not exist yet
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    for k_val, scores in results.items():
        plt.plot(
            resolutions,
            scores,
            marker='o',
            linewidth=2,
            label=f'k = {k_val}'
        )
    
    plt.title('Modularity Optimization Grid Search', fontsize=14, fontweight='bold')
    plt.xlabel('Resolution Factor', fontsize=12)
    plt.ylabel('Modularity Score', fontsize=12)
    plt.xticks(resolutions)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Neighbors (k)', fontsize=10, title_fontsize=11)
    
    plt.tight_layout()
    
    # Save the file instead of calling plt.show()
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Clean up memory
    
    print(f"\n[INFO] Chart successfully saved to: {save_path}")



def plot_tradeoff_frontier(results, resolutions, save_dir="vote_effective_test_charts", filename="modularity_conductance_tradeoff.png"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 7))
    
    for k_val, metrics in results.items():
        modularities = metrics[:, 0]  # Column 0
        conductances = metrics[:, 1]  # Column 1
        
        # Plot the trajectory line for this k across resolutions
        plt.plot(conductances, modularities, '-', linewidth=2, label=f'k = {k_val}')
        plt.scatter(conductances, modularities, s=40)
        
        # Annotate the start (0.5) and end (1.4) resolutions to show direction
        plt.text(conductances[0], modularities[0], f" {resolutions[0]}", fontsize=8, alpha=0.7, va='bottom')
        plt.text(conductances[-1], modularities[-1], f" {resolutions[-1]}", fontsize=8, alpha=0.7, va='top')

    plt.title('Modularity vs. Conductance Trade-off', fontsize=14, fontweight='bold')
    plt.xlabel('Average Conductance (Lower is Better)', fontsize=12)
    plt.ylabel('Modularity Score (Higher is Better)', fontsize=12)
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title='Neighbors (k)', fontsize=10, title_fontsize=11, loc='lower left')
    
    # Save the plot
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n[INFO] Trade-off chart saved to: {save_path}")



if __name__ == "__main__":
    search_term = "Headphones"
    path = find_csv(search_term)
    vn  = matrix_factorization(path , search_term)
    populated_data = vn.matrix_factorization_tf_idf()

    test_stats = presence_auc(vn.presence_mask, populated_data)
    test_stats2 = global_fidelity(vn.X_tfidf.toarray(), populated_data)

    print("presence AUC   :", test_stats)
    print("global fidelity:", test_stats2)

    # populated_data.to_csv("test.csv")
    vn.svd_product_communities(k=5)
    visualize_product_communities(vn)
    print(vn.product_labels)
    mod_k = modularity_k(vn.product_graph , vn.product_labels)
    print("MODE K " , mod_k)


    results = {}
    resolutions = [round(r, 1) for r in np.arange(0.5, 1.5, 0.1)] 

    for i in tqdm(range(5,30, 5)) :
        row_data = []
        for j in resolutions:
            
            graph = vn.product_graph
            labels = vn.product_labels


            vn.svd_product_communities(k=i , resolution= j)
            dp = modularity_k( graph , labels )
            cd = conductance(graph , labels)

            row_data.append([dp , cd])

        
        results[i] = np.array(row_data)
    
    plot_tradeoff_frontier(results, resolutions)


    
    scores = calculate_unified_score(results, vn, resolutions)
    # Find the best k automatically
    best_k = max(scores, key=scores.get)
    print(f"\n[RECOMMENDATION] The optimal parameter is k = {best_k} with a score of {scores[best_k]:.2f}")