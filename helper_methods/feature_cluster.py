import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from config import BANNED_COLUMNS, TEXT_STAT_WEIGHT, FEATURE_WEIGHTS
from data_extraction.normalize import parse_numeric
from helper_methods.feature_type import FeatureType


class feature_cluster:
    def __init__(self, data, search_term , typer=None, banned=None ):
        self.parsed_data = data
        self.search_term  = search_term
        self.typer = typer or FeatureType(cache_path= f"data_files/feature_types_{self.search_term}.json")
        self.feature_types = None
        self.banned = set(BANNED_COLUMNS if banned is None else banned)

        # writing-style stats = any feature coming from title/paragraph text
        prov = self.parsed_data.columns
        text_prov = [p for p in prov.get_level_values(0).unique()
                     if str(p) == "title" or str(p).startswith("paragraph")]
        self.linguistic = set(prov[prov.get_level_values(0).isin(text_prov)]
                              .get_level_values(1))

    def _skip(self, feat, kind):
        # identifiers carry no similarity signal; banned columns are metadata
        # (e.g. the capture date) that must not influence grouping/voting
        return kind == "identifier" or feat in self.banned

    def _feature_weight(self, feat):
        # soft importance multiplier on top of the IDF value weight
        if feat in FEATURE_WEIGHTS:
            return FEATURE_WEIGHTS[feat]
        if feat in self.linguistic:
            return TEXT_STAT_WEIGHT
        return 1.0

    def resolve_types(self):
        self.feature_types = self.typer.resolve(self.parsed_data)
        return self.feature_types

    def _feature_cols(self, feat):
        return self.parsed_data.loc[:, self.parsed_data.columns.get_level_values(1) == feat]

    def _feature_vector(self, feat, kind, idx):
        # collapse a feature's provenance columns into one value per product,
        # and track which products actually have it (vs missing) so co-absence
        # does not count as agreement downstream
        cols = self._feature_cols(feat).iloc[idx]
        present = cols.notna().any(axis=1).to_numpy()
        if kind == "numeric":
            vals = cols.apply(parse_numeric).mean(axis=1)  # unit-aware ("40mm" -> 40)
            present = present & vals.notna().to_numpy()
            return vals.fillna(vals.mean()).fillna(0.0), present
        vals = cols.astype(str).where(cols.notna())
        mode = vals.mode(axis=1, dropna=True)
        vec = mode.iloc[:, 0] if mode.shape[1] else pd.Series(np.nan, index=cols.index)
        return vec.fillna("Unknown"), present

    def _distance(self, vec, kind, present=None):
        if kind == "numeric":
            x = vec.to_numpy(dtype=float)
            ref = x[present] if present is not None and present.any() else x
            std = ref.std()
            z = ((x - ref.mean()) / std if std else np.zeros_like(x)).reshape(-1, 1)
            return pdist(z, metric="cityblock")
        codes = pd.factorize(vec)[0].reshape(-1, 1)
        return pdist(codes, metric="hamming")

    def iter_feature_distances(self, idx=None):
        if self.feature_types is None:
            self.resolve_types()
        if idx is None:
            idx = np.arange(len(self.parsed_data))

        for feat, kind in self.feature_types.items():
            if self._skip(feat, kind):
                continue
            vec, present = self._feature_vector(feat, kind, idx)
            yield feat, kind, self._distance(vec, kind, present), present

    def _encode(self, feat, kind, idx, n_bins=10):
        # discrete code per product (-1 = missing). numerics are quantile-binned
        # so "same bucket" is an achievable match, not a knife-edge distance.
        vec, present = self._feature_vector(feat, kind, idx)
        if kind == "numeric":
            x = pd.to_numeric(vec, errors="coerce").to_numpy()
            codes = np.full(len(x), -1, dtype=np.int64)
            m = present & ~np.isnan(x)
            if m.any():
                q = min(n_bins, len(np.unique(x[m])))
                codes[m] = pd.qcut(x[m], q=q, labels=False, duplicates="drop") if q >= 2 else 0
            return feat, codes, present
        codes = pd.factorize(vec.where(present))[0]   # -1 where absent
        return feat, codes, present
    
    def _soft_encode(self, feat, kind, idx):
        vec, present = self._feature_vector(feat, kind, idx)
        
        if kind == "numeric":
            x = pd.to_numeric(vec, errors="coerce").to_numpy()
            scores = np.zeros(len(x), dtype=np.float64)
            m = present & ~np.isnan(x)
            
            if m.any():
                # Min-Max Scaling: Normalizes the number to a continuous [0.0, 1.0] scale
                min_val = x[m].min()
                max_val = x[m].max()
                rng = max_val - min_val
                if rng > 0:
                    scores[m] = (x[m] - min_val) / rng
                else:
                    scores[m] = 0.5
            return feat, scores, present

        # For Categorical, factorize remains hard, but you can normalize 
        # based on frequency or represent them as soft scores if you use string distance.
        # Alternatively, returning a normalized scale of string matches is possible.
        codes = pd.factorize(vec.where(present))[0]
        # Normalize categorical codes to [0.0, 1.0] so they can be processed numerically
        max_code = codes.max()
        scores = np.where(codes >= 0, codes / max_code if max_code > 0 else 1.0, 0.0)
        
        return feat, scores, present




    def iter_feature_codes(self, idx=None, encoding="soft"):
        # encoding selects how a feature is represented per product:
        #   "hard" - discrete bin/factorize codes (_encode)
        #   "soft" - continuous [0, 1] scores (_soft_encode)
        encoder = {"hard": self._encode, "soft": self._soft_encode}[encoding]

        if self.feature_types is None:
            self.resolve_types()
        if idx is None:
            idx = np.arange(len(self.parsed_data))

        for feat, kind in self.feature_types.items():
            if self._skip(feat, kind):
                continue
            yield encoder(feat, kind, idx)

