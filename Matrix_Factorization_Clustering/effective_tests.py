
import numpy as np

import networkx as nx
import networkx.algorithms.community as nx_comm

from sklearn.metrics import roc_auc_score, average_precision_score


def presence_auc(presence_mask, reconstructed):

    y_true = np.asarray(presence_mask).ravel().astype(int)
    y_score = np.asarray(reconstructed, dtype=float).ravel()

    if y_true.shape != y_score.shape:
        raise ValueError(
            f"shape mismatch: mask {presence_mask.shape} vs "
            f"reconstructed {reconstructed.shape}"
        )

    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "baseline_pr": float(y_true.mean()),
    }


def global_fidelity(original_data, reconstructed):

    original_data = np.asarray(original_data, dtype=float)
    reconstructed = np.asarray(reconstructed, dtype=float)

    if original_data.shape != reconstructed.shape:
        raise ValueError(
            f"shape mismatch: original {original_data.shape} vs "
            f"reconstructed {reconstructed.shape}"
        )

    diff = original_data - reconstructed
    sse = np.sum(diff ** 2)                        # sum of squared errors

    original_norm = np.linalg.norm(original_data)
    rel_frobenius = np.sqrt(sse) / original_norm 
    rmse = np.sqrt(sse / original_data.size)
    mae = np.mean(np.abs(diff))

    return {
        "rel_frobenius": float(rel_frobenius),
        "rmse": float(rmse),
        "mae": float(mae),
    }

def format_labels_comm(labels):
    communities = {}
    for node_id, label in enumerate(labels):
        if label in communities:
            # Using a set (.add) is preferred by NetworkX
            communities[label].add(node_id)
        else:
            communities[label] = {node_id}
    
    return communities
#  takes in the graph and the labels which should be an array where the position is the node id an the content is the cluster it belongs to 
def modularity_k(graph, labels):
    
    communities = format_labels_comm(labels)
    mod = nx_comm.modularity(G=graph, communities=list(communities.values())) # only passing the values no keys 
    return mod

# find out how much percentage is the lagrest community in the graph
def LCC_computer(graph , labels):
    all_nodes = graph.nodes  
    nodes_list = list(graph.nodes)

    communities  = format_labels_comm(labels)
    largest_comm  = -1
    for comm in list(communities.values()) :
        length = len(comm)

        if length > largest_comm :
            largest_comm = length

    return largest_comm / len(nodes_list) 


def conductance(graph, labels):
    communities = format_labels_comm(labels)

    conductances = []
    # Loop over the .values() (the actual sets of node IDs)
    for c in communities.values():
        if len(c) > 0:
            try:
                cond = nx.conductance(graph, c)
                conductances.append(cond)
            except ZeroDivisionError:
                # Handle isolated communities safely
                conductances.append(0.0)

    # Return the average, ensuring we don't divide by zero if list is empty
    return np.mean(conductances) if conductances else 1.0