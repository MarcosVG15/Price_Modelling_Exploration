'''

Stage 1 - create a matrix pair of distances for each feature
Stage 2 - depending on the type you use different clustering techniques.
Stage 2.1 - make the voting matrix
Stage 3 (Pruner): Use the Jaccard Index on your cluster assignments to filter out any pairs with a score of candidate pairs down to a fraction of the size.
Stage 3.5 (Filter): Apply PMI to the remaining pairs to remove weak connections caused by bloated, generic clusters.
Stage 4 (Deep Analysis): On the remaining high-probability pairs, calculate the SHAP-weighted similarity or run them through a GLM to output the final definitive similarity score.
'''


''' WHEN YOU CHECK YOU CAN SEARCH FOR PRODUCTS OF THE SAME ASIN IN DIFFERENT THROUGH THE CLUSTERS TO CHECK WHETHER THEY ARE ALL GROUPED TOGETHER OR NO'''


'''
CHECK :

(1) stop penalizing one-sided absence, and (2) add an explicit confidence/support factor so thin overlap doesn't masquerade as strong similarity.


'''

from tqdm import tqdm
import numpy as np
import pandas as pd

from .pre_clustering import pre_cluster
from helper_methods.feature_cluster import feature_cluster


def load_feature_matrix(path):
    # provenance (title/paragraph_n) on level 0, the actual feature on level 1
    return pd.read_csv(path, header=[0, 1], low_memory=False)


def otsu_threshold(values, bins=256):
    # parameter-free split of a bimodal distribution: the cut that maximizes
    # between-class variance (the valley between the two humps)
    values = np.asarray(values, dtype=float)
    hist, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    total = hist.sum()
    if total == 0:
        return float(np.median(values))
    p = hist / total
    omega = np.cumsum(p)
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    denom = omega * (1 - omega)
    sigma_b = np.where(denom > 0, (mu_t * omega - mu) ** 2 / np.where(denom > 0, denom, 1), 0)
    return float(centers[int(np.argmax(sigma_b))])


class voting_network :
    def __init__(self, path,search_term ):

        self.parsed_data = load_feature_matrix(path)
        self.pre_c = pre_cluster(path)
        self.feature_cluster = feature_cluster(self.parsed_data, search_term= search_term)

        cat_columns = self.pre_c.extract_categorical_columns()
        self.labels = self.pre_c.soft_quantization(cat_cols=cat_columns)

        # cluster_sizes = pd.Series(self.labels).value_counts().sort_index()

        self.id_cluster_map = pd.DataFrame({
            "product_id": self.pre_c.parsed_data.iloc[:, 0].values,
            "cluster": self.labels,
        })

        cluster_sizes = self.id_cluster_map["cluster"].value_counts().sort_index()
        print("Number of elements per cluster:")
        for cluster_id, size in cluster_sizes.items():
            print(f"  Cluster {cluster_id}: {size}")
        print(f"Total clusters: {cluster_sizes.shape[0]} | Total elements: {cluster_sizes.sum()}")


    def vote(self, method="cosine"):
        # Stage 2.1: per soft cluster, score every pair of products.
        #   "copresent" - baseline: fraction of co-present features they agree on
        #                 (unweighted, co-present denominator). Simple and stable.
        #   "cosine"    - IDF-weighted cosine over (feature, value) tokens; shared
        #                 rare values dominate, missing/disagreeing forgiven.
        # Both run on identical clusters so they can be compared head to head.
        scorer = {"cosine": self._score_cosine, "copresent": self._score_copresent}[method]

        cluster_score_map = {}
        self.cluster_index_map = {}

        fc = self.feature_cluster
        if fc.feature_types is None:
            fc.resolve_types()
        self._total_features = sum(1 for f, k in fc.feature_types.items() if not fc._skip(f, k))

        for cluster_id, group in self.id_cluster_map.groupby("cluster"):
            idx = group.index.to_numpy()
            self.cluster_index_map[cluster_id] = idx
            cluster_score_map[cluster_id] = scorer(idx, cluster_id)

        return cluster_score_map

    def _iter(self, idx, cluster_id):
        return tqdm(self.feature_cluster.iter_feature_codes(idx=idx),
                    desc=f"  Cluster {cluster_id}", total=self._total_features, leave=False)

    def _score_cosine(self, idx, cluster_id):
        fc = self.feature_cluster
        n = len(idx)
        dot = np.zeros((n, n))      # Σ over shared tokens of a_i * a_j
        sq_norm = np.zeros(n)       # Σ over present tokens of a_i^2

        for feat, codes, present in self._iter(idx, cluster_id):
            n_present = int(present.sum())
            if n_present == 0:
                continue
            vals, counts = np.unique(codes[present], return_counts=True)
            idf = dict(zip(vals.tolist(), (-np.log(counts / n_present)).tolist()))
            wf = fc._feature_weight(feat)
            a = np.array([wf * idf.get(c, 0.0) if c != -1 else 0.0 for c in codes])

            equal = (codes[:, None] == codes[None, :]) & (codes[:, None] != -1)
            dot += np.where(equal, a[:, None] * a[None, :], 0.0)
            sq_norm += a * a

        norm = np.sqrt(sq_norm)
        with np.errstate(invalid="ignore", divide="ignore"):
            denom = norm[:, None] * norm[None, :]
            score = np.where(denom > 0, dot / denom, 0.0)
        np.fill_diagonal(score, 1.0)
        return score

    def _score_copresent(self, idx, cluster_id):
        n = len(idx)
        match = np.zeros((n, n), dtype=np.int32)
        copresent = np.zeros((n, n), dtype=np.int32)

        for feat, codes, present in self._iter(idx, cluster_id):
            both = (codes[:, None] != -1) & (codes[None, :] != -1)
            equal = both & (codes[:, None] == codes[None, :])
            match += equal
            copresent += both

        with np.errstate(invalid="ignore", divide="ignore"):
            score = np.where(copresent > 0, match / copresent, 0.0)
        np.fill_diagonal(score, 1.0)
        return score

    def filter_vote(self, voting_scores=None, max_pairs=None):
        # Auto-threshold each cluster with Otsu. Pass max_pairs to cap the output
        # to a top-N budget by score (useful when the distribution is unimodal and
        # Otsu just bisects it); by default the Otsu cut stands uncapped.
        if voting_scores is None:
            voting_scores = self.vote()

        product_ids = self.id_cluster_map["product_id"].to_numpy()
        results = {}
        for cluster_id, score in voting_scores.items():
            iu = np.triu_indices(score.shape[0], k=1)
            vals = score[iu]
            informative = vals[vals > 0]
            if informative.size < 2:
                results[cluster_id] = {"threshold": None, "pairs": np.empty((0, 2)), "n_pairs": 0}
                continue

            thr = otsu_threshold(informative)
            keep = vals >= thr
            capped = False
            if max_pairs is not None and keep.sum() > max_pairs:   # optional top-N budget
                cutoff = np.partition(vals, -max_pairs)[-max_pairs]
                keep = vals >= cutoff
                thr = max(thr, float(cutoff))
                capped = True

            local = np.column_stack((iu[0][keep], iu[1][keep]))
            id_pairs = product_ids[self.cluster_index_map[cluster_id][local]]

            results[cluster_id] = {
                "threshold": float(thr),
                "scores": vals[keep],
                "pairs": id_pairs,
                "n_pairs": int(keep.sum()),
                "kept_fraction": float(keep.mean()),
                "capped": capped,
            }
            print(f"Cluster {cluster_id}: threshold={thr:.3f} "
                  f"kept {keep.sum()}/{keep.size} pairs ({keep.mean():.1%})"
                  f"{' [capped]' if capped else ''}")
        return results
