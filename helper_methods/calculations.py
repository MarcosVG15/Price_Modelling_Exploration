import os
import numpy as np
import pandas as pd
import networkx as nx


def calculate_unified_score(results, vn, resolutions, epsilon=1e-5):
    """
    Calculates a single score for each k value based on performance,
    stability (conductance and modularity standard deviations), and graph connectivity (LCC).
    """
    unified_scores = {}
    
    
    for k_val, metrics in results.items():
        # metrics is a 2D array of shape (num_resolutions, 2)
        modularities = metrics[:, 0]
        conductances = metrics[:, 1]
        
        # 1. Calculate Mean Modularity and Mean Conductance
        mean_mod = np.mean(modularities)
        mean_cond = np.mean(conductances)
        
        # 2. Calculate Standard Deviations (Instabilities)
        std_cond = np.std(conductances)
        std_mod = np.std(modularities)  # Added Modularity Standard Deviation
        
        # 3. Calculate Graph Connectivity (LCC Fraction)
        vn.svd_product_communities(k=int(k_val), resolution=1.0)
        G = vn.product_graph
        lcc = max(nx.connected_components(G), key=len)
        lcc_fraction = len(lcc) / G.number_of_nodes()
        
        # 4. Compute the Updated Unified Score
        # We add both standard deviations to the denominator to penalize overall instability
        denominator = mean_cond + std_cond + std_mod + epsilon
        score = (mean_mod * lcc_fraction) / denominator
        
        unified_scores[k_val] = score
        
        # print(f"k = {k_val:2} | "
        #       f"Mean Mod: {mean_mod:.3f} | "
        #       f"LCC: {lcc_fraction:.2%} | "
        #       f"Mod StdDev: {std_mod:.4f} | "
        #       f"Cond StdDev: {std_cond:.4f} | "
        #       f"SCORE: {score:.2f}")
              
    return unified_scores