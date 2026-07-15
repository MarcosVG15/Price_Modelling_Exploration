import torch
import numpy as np
import pandas as pd

from kmodes.kmodes import KModes
from config import BANNED_COLUMNS



'''
Davies-Bouldin Index: Can be adapted for non-Euclidean distances.
DBCV (Density-Based Clustering Validation): This is often much better than Silhouette for non-spherical/non-Euclidean clusters.
Connectivity-based measures: If your data has a manifold structure, look at the local neighborhood consistency.
'''

class pre_cluster :

    def __init__(self , path) :
        self.path = path 
        self.parsed_data = pd.read_csv(self.path, skiprows=[0],  low_memory=False)

        self.banned_columns   = list(BANNED_COLUMNS)

    def extract_categorical_columns(self):

        data  = self.get_useful_columns()

        unique_ratio_threshold = 0.01
        max_unique = 75
        cat_columns = set() 
        total_rows = len(data)

        for col in data.columns:
            
            base_name = self.extract_base_name(col)

            if base_name in self.banned_columns :
                continue

            if data[col].dtype == 'object':
                sample = data[col].dropna().astype(str)
                if sample.str.len().mean() > 50:
                    continue 
            
            if base_name in cat_columns:
                continue

            unique_count = data[col].nunique()
            unique_ratio = unique_count / total_rows

            if  unique_count <= max_unique:
                cat_columns.add(base_name)



        # sorted, not list(set): set iteration order is randomized per process,
        # which would reshuffle the KModes input columns and change the clusters
        # every run even with a fixed random_state
        return sorted(cat_columns)
    

    def get_useful_columns(self, min_fill_rate=0.25):
        is_useful = self.parsed_data.notna().copy()
        for col in self.parsed_data.columns:
            # Change: Correct type-safe numeric evaluation on columns using Series wrapper
            numeric_col = pd.to_numeric(self.parsed_data[col], errors='coerce')
            is_zero = numeric_col.eq(0)
            is_useful[col] = is_useful[col] & (~is_zero)

        fill_rate = is_useful.mean()
        useful_cols = []

        for name, rate in fill_rate.items():
            if rate >= min_fill_rate:
                useful_cols.append(name)

        return self.parsed_data[useful_cols].copy()

    def extract_base_name(self, col):
        base_name = col.split('.')[0]
        if base_name is None:
            return col
        else :
            return base_name



    ''' 
    I will user the  K prototype for fast initial soft quantization such 
    that I can then make more elaborate similarity clusters 
    '''
    def _categorical_frame(self, cat_cols):
        cat_indexes = []

        for col in cat_cols:
            for title in self.parsed_data.columns:
                if self.extract_base_name(title) == col:
                    cat_indexes.append(title)

        return self.parsed_data[cat_indexes].fillna("Unknown").astype(str)

    def soft_quantization(self, cat_cols , target_cluster_size=3000):

        data_array = self._categorical_frame(cat_cols).to_numpy()

        total_df = len(data_array)
        n_clusters = max(1, total_df // target_cluster_size)

        # n_jobs=1 for reproducibility (the main determinism fix is sorting the
        # categorical columns in extract_categorical_columns; single-thread avoids
        # any residual parallel-seeding variation on top of that)
        kmodes = KModes(n_jobs=1, n_clusters=n_clusters, init='Huang', random_state=42)
        labels = kmodes.fit_predict(data_array)

        return labels

    ''' Section dedicated to checking the validity of the systen that I have developed'''
    def cost_elbow(self, cat_cols, k_values):
        data_array = self._categorical_frame(cat_cols).to_numpy()

        costs = {}
        for k in k_values:
            kmodes = KModes(n_jobs=1, n_clusters=k, init='Huang', random_state=42)
            kmodes.fit(data_array)
            costs[k] = kmodes.cost_

        return pd.Series(costs).sort_index()

    def mode_purity(self, cat_cols, labels):
        data = self._categorical_frame(cat_cols)
        value_cols = list(data.columns)
        data['cluster'] = labels

        rows = []
        for cluster_id, group in data.groupby('cluster'):
            for col in value_cols:
                purity = group[col].value_counts(normalize=True).iloc[0]
                rows.append({'cluster': cluster_id, 'column': col, 'purity': purity})

        return pd.DataFrame(rows)

    def validation(self, cat_cols, labels, k_values=None):
        if k_values is None:
            base_k = max(1, len(self.parsed_data) // 3000)
            k_values = range(max(1, base_k - 2), base_k + 3)

        costs = self.cost_elbow(cat_cols, k_values)
        purity = self.mode_purity(cat_cols, labels)

        return costs, purity