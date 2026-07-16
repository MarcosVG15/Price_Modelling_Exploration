import os
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import AgglomerativeClustering
import networkx as nx

from helper_methods.feature_cluster import feature_cluster


class matrix_factorization :
    def __init__( self ,feature_data ):
        self.feature_data = feature_data



        
    #  treates 0 as missing data point you should then try to use 
    def matrix_factorization_tf_idf(self):
        numeric = self.feature_data.apply(pd.to_numeric, errors="coerce")
        present = self.feature_data.notna() & ~numeric.eq(0)          
        mock_data = present.to_numpy(dtype=np.uint8)
        X_raw = csr_matrix(mock_data)

        self.presence_mask = mock_data

        density = X_raw.nnz / (X_raw.shape[0] * X_raw.shape[1])
        print(f"Original Matrix Shape: {X_raw.shape}")
        print(f"Presence density: {density:.2%}")

        tfidf = TfidfTransformer()
        X_tfidf = tfidf.fit_transform(X_raw)

        self.X_tfidf = X_tfidf


        n_components = 50
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        transform = svd.fit_transform(X_tfidf)
        populated_data = svd.inverse_transform(transform)

        self.tfidf = tfidf
        self.svd = svd

        self.X_transformed = transform

        
        populated_data = pd.DataFrame(
            populated_data,
            index=self.feature_data.index,
            columns=self.feature_data.columns,
        )
        return populated_data

    def _cluster_features(self, S, corr_threshold=0.8):

        if S.shape[0] < 2:
            return np.zeros(S.shape[0], dtype=int)

        corr = np.nan_to_num(np.corrcoef(S), nan=0.0)
        distance = 1.0 - np.abs(corr)
        np.fill_diagonal(distance, 0.0)

        model = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="average",
            distance_threshold=1.0 - corr_threshold,
        )
        return model.fit_predict(distance)

    def _svd_impute(self, S, P, n_components=50, n_iters=10):
       
        S = S.astype(float).copy()
        observed = P.astype(bool)

        # seed each feature's missing cells with its observed mean (0.5 if unseen)
        for f in range(S.shape[0]):
            obs = observed[f]
            S[f, ~obs] = S[f, obs].mean() if obs.any() else 0.5

        k = min(n_components, min(S.shape) - 1)
        if k < 1:
            return S

        for _ in range(n_iters):
            svd = TruncatedSVD(n_components=k, random_state=42)
            S_hat = svd.inverse_transform(svd.fit_transform(S))
            S[~observed] = S_hat[~observed]   # keep observed fixed, fill only holes

        return np.clip(S, 0.0, 1.0)

    def feature_clusterize(self, data, corr_threshold=0.8, populate=True,
                           cluster_method="agglomerative"):
        self.feature_cluster = feature_cluster(data, self.search_term)
        fc = self.feature_cluster
        if fc.feature_types is None:
            fc.resolve_types()
        total_features = sum(1 for f, k in fc.feature_types.items() if not fc._skip(f, k))

        names, scores, presents = [], [], []
        for feat, s, p in tqdm(fc.iter_feature_codes(encoding="soft"),
                               total=total_features):
            names.append(feat)
            scores.append(np.asarray(s, dtype=float))
            presents.append(np.asarray(p, dtype=bool))

        S = np.vstack(scores)     # (n_features, n_products)
        P = np.vstack(presents)   # (n_features, n_products)

        
        if populate:
            density = P.mean()
            S = self._svd_impute(S, P)
            P = np.ones_like(P, dtype=bool)
            print(f"SVD-populated {(1 - density):.1%} missing cells "
                  f"(observed density was {density:.1%})")

        # 1. SCALE SOFT SCORES: Stretch your compressed SVD values to [0.0, 1.0]
        for f in range(S.shape[0]):
            row_min = S[f].min()
            row_max = S[f].max()
            row_range = row_max - row_min
            if row_range > 0:
                S[f] = (S[f] - row_min) / row_range

        # Cluster the FEATURES (for the redundancy down-weighting below).
        # "agglomerative" = correlation-distance hierarchical; "louvain" =
        # community detection on the |corr|-thresholded feature graph.
        if cluster_method == "louvain":
            feat_labels = self._cluster_features_louvain(S, corr_threshold)
        else:
            feat_labels = self._cluster_features(S, corr_threshold)

        _, inverse, counts = np.unique(feat_labels, return_inverse=True,
                                       return_counts=True)
        weights = 1.0 / counts[inverse]

        n = S.shape[1]
        agree = np.zeros((n, n))
        denom = np.zeros((n, n))
        for f in range(S.shape[0]):
            both = P[f][:, None] & P[f][None, :]
            
            # 2. GAUSSIAN KERNEL: Penalize small SVD-imputed differences
            diff  = np.abs(S[f][:, None] - S[f][None, :])
            gamma = 15.0  # Adjust this: higher values increase the strictness of matches
            sim   = np.exp(-gamma * (diff ** 2))
            
            agree += weights[f] * np.where(both, sim, 0.0)
            denom += weights[f] * both

        with np.errstate(divide="ignore", invalid="ignore"):
            agreement = np.where(denom > 0, agree / denom, 0.0)

        self.feature_labels  = dict(zip(names, feat_labels.tolist()))
        self.feature_weights = dict(zip(names, weights.tolist()))
        self.agreement       = agreement
        print(f"{S.shape[0]} features -> {counts.size} feature clusters")

  
        agreement_min = agreement.min()
        agreement_max = agreement.max()
        stretched = (agreement - agreement_min) / (agreement_max - agreement_min)

        #  TRY THIS AFTER YOU RUN THE CURRENT ONE 
        agreement_final = stretched ** 3 # Try 2 (squared) or 3 (cubed) 

        return agreement_final



    def _cluster_features_louvain(self, S, corr_threshold=0.8):
        """
        Groups features using the Louvain community detection algorithm.
        Returns a 1D numpy array of cluster labels for each feature.
        """
        if S.shape[0] < 2:
            return np.zeros(S.shape[0], dtype=int)

        # 1. Calculate Pearson correlation coefficient
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.nan_to_num(np.corrcoef(S), nan=0.0)
            
        # 2. Convert to absolute correlation (similarities)
        abs_corr = np.abs(corr)
        
        # 3. Apply threshold: keep edge weights only if they are >= corr_threshold
        #    Louvain works best when weak/noisy edges are removed (set to 0)
        adj_matrix = np.where(abs_corr >= corr_threshold, abs_corr, 0.0)
        np.fill_diagonal(adj_matrix, 0.0) # Remove self-loops
        
        # 4. Create a NetworkX Graph from the weighted adjacency matrix
        G = nx.from_numpy_array(adj_matrix)
        
        # 5. Run Louvain Community Detection
        #    It automatically determines the optimal number of clusters
        communities = nx.community.louvain_communities(G, weight='weight', seed=42)
        
        # 6. Convert NetworkX's output format (list of node sets) 
        #    back to a 1D NumPy array of labels matching S.shape[0]
        feat_labels = np.zeros(S.shape[0], dtype=int)
        for cluster_id, community in tqdm(enumerate(communities)):
            for feature_index in community:
                feat_labels[feature_index] = cluster_id

        return feat_labels

    def svd_product_communities(self, k=10, resolution=1.0, drop_first=False):

        if not hasattr(self, "X_transformed"):
            raise RuntimeError("run matrix_factorization_tf_idf() first")

        U = self.X_transformed[:, 1:] if drop_first else self.X_transformed
        U = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)  # L2 normalize
        sim = U @ U.T                                               # cosine sims
        np.fill_diagonal(sim, -1.0)                                 # ignore self

        # keep only each product's top-k neighbours -> a sparse, tractable graph
        n = sim.shape[0]
        kk = min(k, n - 1)
        nbrs = np.argpartition(-sim, kk - 1, axis=1)[:, :kk]

        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in nbrs[i]:
                w = sim[i, int(j)]
                if w > 0:
                    G.add_edge(i, int(j), weight=float(w))

        communities = nx.community.louvain_communities(
            G, weight="weight", resolution=resolution, seed=42)

        labels = np.zeros(n, dtype=int)
        for cid, comm in tqdm(enumerate(communities)):
            for node in comm:
                labels[node] = cid

        np.fill_diagonal(sim, 1.0)  # restore for downstream use / heatmaps

        self.product_labels = labels
        self.product_similarity = sim
        self.product_graph = G
        # print(f"{n} products -> {len(communities)} product communities "
        #       f"(k={kk} neighbours, {G.number_of_edges()} edges)")
        return labels, sim

    def fit_new_products(self, new_data, k=15, drop_first=False, weighted_vote=True):
        """Fold brand-new products into the trained vn WITHOUT re-fitting.

        new_data : DataFrame with the SAME feature columns as self.feature_data
                   (one row per new product). Missing columns are treated as
                   absent; extra/unseen columns are dropped (fixed SVD vocab).

        The new products are projected onto the EXISTING latent axes (reusing the
        fitted tfidf + svd) and each is assigned a community by a cosine k-NN vote
        over the ORIGINAL trained products (new products don't vote on each other,
        so assignment is deterministic and consistent with the trained model).

        Returns a lightweight COPY of the vn with the new products appended to
        feature_data / X_transformed / product_labels. The trained vn is NOT
        mutated. `new_product_indices` on the copy marks the appended rows.
        """
        if not hasattr(self, "svd") or not hasattr(self, "tfidf"):
            raise RuntimeError("run matrix_factorization_tf_idf() first "
                               "(need the fitted tfidf + svd to fold in)")
        if not hasattr(self, "product_labels"):
            raise RuntimeError("run svd_product_communities() first "
                               "(need existing communities to vote on)")

        # 1. align new rows to the exact trained feature schema (order + set)
        new_data = new_data.reindex(columns=self.feature_data.columns)

        # 2. same presence rule as training: value present AND not zero
        numeric = new_data.apply(pd.to_numeric, errors="coerce")
        present = new_data.notna() & ~numeric.eq(0)
        X_raw_new = csr_matrix(present.to_numpy(dtype=np.uint8))

        # 3. FOLD IN: reuse the fitted transformers (.transform, never .fit)
        emb_new = self.svd.transform(self.tfidf.transform(X_raw_new))   # (m, 50)

        # 4. cosine k-NN vote over the original trained products
        U_old = self.X_transformed[:, 1:] if drop_first else self.X_transformed
        U_new = emb_new[:, 1:] if drop_first else emb_new
        U_old = U_old / (np.linalg.norm(U_old, axis=1, keepdims=True) + 1e-12)
        U_new = U_new / (np.linalg.norm(U_new, axis=1, keepdims=True) + 1e-12)

        sims = U_new @ U_old.T                       # (m, n) cosine to every product
        kk = min(k, sims.shape[1])
        new_labels = np.empty(emb_new.shape[0], dtype=int)
        for i in range(sims.shape[0]):
            nbr = np.argpartition(-sims[i], kk - 1)[:kk]
            if weighted_vote:                        # heaviest similarity-weighted community
                votes = {}
                for j in nbr:
                    lbl = int(self.product_labels[j])
                    votes[lbl] = votes.get(lbl, 0.0) + max(float(sims[i, j]), 0.0)
                new_labels[i] = max(votes, key=votes.get)
            else:                                    # plain majority
                vals, counts = np.unique(self.product_labels[nbr], return_counts=True)
                new_labels[i] = int(vals[counts.argmax()])

        # 5. lightweight COPY: shallow copy shares the big arrays; we reassign only
        #    the extended ones as NEW objects, so the trained vn is never mutated.
        aug = copy.copy(self)
        aug.feature_data = pd.concat([self.feature_data, new_data], ignore_index=True)
        aug.X_transformed = np.vstack([self.X_transformed, emb_new])
        aug.product_labels = np.concatenate([self.product_labels, new_labels])
        aug.new_product_indices = np.arange(self.X_transformed.shape[0],
                                            aug.X_transformed.shape[0])
        aug.product_similarity = None   # (n+m)x(n+m) not rebuilt; use embeddings for distance
        print(f"folded in {emb_new.shape[0]} new products -> "
              f"community labels {new_labels.tolist()}")
        return aug