import os
import json
import numpy as np
import pandas as pd
import networkx as nx

from pathlib import Path
from collections import defaultdict

import shap
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error

from config import ASIN_COL
from Matrix_Factorization_Clustering.Matrix_Factorization import matrix_factorization


def load_scale_params(params_path, feature="price"):
    """Return (center, scale) for a feature from a normalizer .params.json
    sidecar. A normalized value maps back to real units via value*scale + center
    (robust scaling: center = median, scale = IQR)."""
    blob = json.loads(Path(params_path).read_text())
    params = blob.get("params", blob)
    spec = params[feature]
    return spec["center"], spec["scale"]


'''
I need to make a predictor that will use the clusteres and a weighted average in order to compute the price of a product.
to do this I need to get a product Id find the cluster , use the dp in those clusters and then tune the parameters to extracted the data.

'''

PRICE_COL = ("clean", "price")

# Marketplace / scrape metadata + price-derived leakage — NOT intrinsic product
# attributes. Pass as `drop_fields` to train_price_model for a clean hedonic
# (product-features-only) model.
MARKETPLACE_FIELDS = {
    "discount",                             # price-derived -> leakage
    "keyword_id",     # scrape artifacts
     "btf_amazon best sellers ranking",
     "average_rating",
    "buy_box",  "image_url",
}


class predictor:

    def __init__(self ,k , feature_data, target_data , vn ) :
        self.feature_data = feature_data
        self.target_data = target_data
        self.vn = vn
        self.k  = k

        shape = self.feature_data.shape
        print("SHAPE : " , shape)
        self.params = np.ones([shape[1] , 1 ])
        print("PARAM SHAPE : " , self.params.shape)


        
    def find_cluster(self):

        fit_vn = self.vn.fit_new_products(new_data=self.target_data, k=self.k)
        self.fit_vn        = fit_vn
        self.feature_data  = fit_vn.feature_data          
        self.X_transformed = fit_vn.X_transformed         
        labels             = fit_vn.product_labels
        query_idxs         = fit_vn.new_product_indices
        query_set          = set(int(q) for q in query_idxs)

        U = self.X_transformed / (np.linalg.norm(self.X_transformed, axis=1, keepdims=True) + 1e-12)
  
        members_by_comm = defaultdict(list)
        for idx, lab in enumerate(labels):
            if idx in query_set:
                continue
            members_by_comm[int(lab)].append(idx)

        prices = self.feature_data[PRICE_COL].to_numpy(dtype=float)

        query_clusters = {}
        for q in query_idxs:
            q = int(q)
            comm    = int(labels[q])
            members = np.asarray(members_by_comm[comm], dtype=int)

            if members.size:
                sims = U[q] @ U[members].T           # cosine query -> each member
                member_prices = prices[members]
            else:                                    
                sims = np.empty(0)
                member_prices = np.empty(0)

            query_clusters[q] = {
                "community":  comm,
                "members":    members,
                "similarity": sims,
                "prices":     member_prices,
            }

        self.query_clusters = query_clusters
        return query_clusters


    #  This is goign to be a weighted model based on a training process. 


    #  AI suggestion simply based on distance only 
    def predict_basic(self, gamma=25.0, price_center=None, price_scale=None):

        if not hasattr(self, "query_clusters"):
            self.find_cluster()

        predictions = {}
        for q, info in self.query_clusters.items():
            sims, prices = info["similarity"], info["prices"]

            if prices.size == 0:                     
                predictions[q] = np.nan
                continue

            dist = 1.0 - sims                        
            w = np.exp(-gamma * dist ** 2)          
            total = w.sum()

            w = w / total if total > 0 else np.full_like(w, 1.0 / w.size)

            predictions[q] = float(w @ prices)

        # optional: map normalized predictions back to real price units
        if price_center is not None and price_scale is not None:
            predictions = {q: v * price_scale + price_center
                           for q, v in predictions.items()}

        self.predictions = predictions
        return predictions

    # ------------------------------------------------------------------ #
    #  SHAP price model: train a GBM on the real feature VALUES (numeric +
    #  categorical), then reconstruct each product's price additively from
    #  its own per-feature SHAP contributions.
    # ------------------------------------------------------------------ #
    def _prepare_columns(self, feature_types, drop_fields=None):
        """Select the 'clean'-section numeric & categorical feature columns to
        model on, excluding identifiers, price-derived columns (leakage), and any
        field in drop_fields (e.g. MARKETPLACE_FIELDS for an intrinsic model)."""
        if isinstance(feature_types, (str, Path)):
            feature_types = json.loads(Path(feature_types).read_text())
        drop = set(drop_fields or [])

        def typ(field):
            v = feature_types.get(field)
            return v.get("type") if isinstance(v, dict) else v

        def price_like(c):
            return "price" in str(c[1]).lower()

        def usable(c, kind):
            return typ(c[1]) == kind and not price_like(c) and c[1] not in drop

        clean = [c for c in self.feature_data.columns if c[0] == "clean"]
        self.num_cols = [c for c in clean if usable(c, "numeric")]
        self.cat_cols = [c for c in clean if usable(c, "categorical")]

    def _numeric_block(self, df):
        return df[self.num_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    def _categorical_block(self, df):
        # strings, missing -> "MISSING"; the fitted encoder handles unseen levels
        arr = df[self.cat_cols].astype("string").fillna("MISSING").to_numpy()
        return self.ohe.transform(arr)

    def _build_matrix(self, df):
        return np.hstack([self._numeric_block(df), self._categorical_block(df)])

    def train_price_model(self, feature_types , price_params_path=None,
                          test_size=0.3, min_frequency=20, random_state=0,
                          drop_fields=None, log_price=False):
        """log_price=True trains on log(price in dollars): tames the heavy tail,
        stabilizes R2, and makes negative predictions impossible (exp > 0).
        Requires price_params_path to de-normalize to real dollars first."""
        self._prepare_columns(feature_types, drop_fields=drop_fields)
        df = self.feature_data

        # price de-normalization params up-front (needed for the log transform
        # and for reporting/inverting to dollars later)
        self.price_center, self.price_scale = (None, None)
        if price_params_path is not None:
            self.price_center, self.price_scale = load_scale_params(price_params_path)
        self.log_price = log_price

        y_all = pd.to_numeric(df[PRICE_COL], errors="coerce")
        keep  = (~df[ASIN_COL].duplicated()).to_numpy() & y_all.notna().to_numpy()
        y_norm = y_all.to_numpy(dtype=float)[keep]

        if log_price:
            if self.price_center is None:
                raise ValueError("log_price=True needs price_params_path to recover dollars")
            y_dollars = y_norm * self.price_scale + self.price_center
            y = np.log(np.clip(y_dollars, 1e-6, None))    # model target = log(dollars)
        else:
            y = y_norm                                     # normalized price (as before)


        Xnum_df = df[self.num_cols].apply(pd.to_numeric, errors="coerce").loc[keep]
        good = (Xnum_df.notna().sum() >= 50) & (Xnum_df.nunique() >= 5)
        self.num_cols = [c for c, g in zip(self.num_cols, good) if g]
        num_names = [str(c[1]) for c in self.num_cols]

        # categorical: drop constant columns
        Xcat_df = df[self.cat_cols].astype("string").fillna("MISSING").loc[keep]
        goodc = (Xcat_df.nunique() >= 2).to_numpy()
        self.cat_cols = [c for c, g in zip(self.cat_cols, goodc) if g]
        cat_names = [str(c[1]) for c in self.cat_cols]

        # split first, fit the one-hot encoder on TRAIN only
        idx = np.arange(len(y))
        tr, te = train_test_split(idx, test_size=test_size, random_state=random_state)
        cat_raw = df[self.cat_cols].astype("string").fillna("MISSING").loc[keep].to_numpy()

        # unique, prefix-free names (c0, c1, ...) so dummies map cleanly to parents
        tokens = [f"c{i}" for i in range(len(cat_names))]
        self.ohe = OneHotEncoder(min_frequency=min_frequency,
                                 handle_unknown="infrequent_if_exist",
                                 sparse_output=False)
        self.ohe.fit(cat_raw[tr])
        oh_names = list(self.ohe.get_feature_names_out(tokens))

        # parent feature + human-readable display name for every model column
        self.feat_parent, self.feat_names = [], []
        for n in num_names:
            self.feat_parent.append(n); self.feat_names.append(n)
        for n in oh_names:
            tok, level = n.split("_", 1)
            parent = cat_names[int(tok[1:])]
            self.feat_parent.append(parent)
            self.feat_names.append(f"{parent}={level}")

        Xnum = Xnum_df.loc[:, good].to_numpy(dtype=float)
        Xcat_oh = self.ohe.transform(cat_raw)
        X = np.hstack([Xnum, Xcat_oh])

        self.model = HistGradientBoostingRegressor(
            random_state=random_state, max_iter=400, learning_rate=0.05)
        self.model.fit(X[tr], y[tr])

        # evaluate + SHAP
        yp = self.model.predict(X[te])
        self.explainer = shap.TreeExplainer(self.model)
        sv = self.explainer.shap_values(X[te])

        r2 = r2_score(y[te], yp)                          # in the model's target space
        if log_price:
            mae = mean_absolute_error(np.exp(y[te]), np.exp(yp))   # dollars
            print(f"\nSHAP price model (LOG) | test R2(log) = {r2:.3f} | "
                  f"MAE = ${mae:,.2f}  ({len(num_names)} numeric + "
                  f"{len(oh_names)} one-hot cat features)")
        else:
            c, s = (self.price_center or 0.0), (self.price_scale or 1.0)
            mae = mean_absolute_error(y[te] * s + c, yp * s + c)
            unit = "$" if self.price_scale is not None else "(norm) "
            print(f"\nSHAP price model | test R2 = {r2:.3f} | "
                  f"MAE = {unit}{mae:,.2f}  ({len(num_names)} numeric + "
                  f"{len(oh_names)} one-hot cat features)")


        return self.model





    def predict_price(self, df=None, top_n=10):

        if not hasattr(self, "model"):
            raise RuntimeError("call train_price_model(...) first")
        if df is None:
            df = self.target_data

        X = self._build_matrix(df)
        preds = self.model.predict(X)
        sv = self.explainer.shap_values(X)
        base = float(np.ravel(self.explainer.expected_value)[0])

        log_price = getattr(self, "log_price", False)
        c = self.price_center if self.price_center is not None else 0.0
        s = self.price_scale if self.price_scale is not None else 1.0

        out = {}
        for i, row_idx in enumerate(df.index):
            # aggregate one-hot contributions back to the original feature
            agg = defaultdict(float)
            for parent, val in zip(self.feat_parent, sv[i]):
                agg[parent] += val                  # model-space contribution
            top = sorted(agg.items(), key=lambda kv: -abs(kv[1]))[:top_n]

            if log_price:
                # model space is log(dollars): invert with exp; each feature's
                # effect is MULTIPLICATIVE -> report as a % change in price.
                out[int(row_idx)] = {
                    "predicted":    float(np.exp(preds[i])),
                    "base":         float(np.exp(base)),
                    "contrib_unit": "%",
                    "top_features": [(str(f), round((np.exp(v) - 1) * 100, 1)) for f, v in top],
                }
            else:
                out[int(row_idx)] = {
                    "predicted":    float(preds[i]) * s + c,
                    "base":         base * s + c,
                    "contrib_unit": "$",
                    "top_features": [(str(f), round(v * s, 2)) for f, v in top],
                }
        self.price_predictions = out
        return out

    def predict_price_parsed_columns(self, feature_types , price_params_path=None,
                                     df=None, drop_fields=MARKETPLACE_FIELDS, top_n=0,
                                     **train_kwargs):
        self.train_price_model(feature_types, price_params_path=price_params_path,
                               drop_fields=drop_fields, **train_kwargs)
        return self.predict_price(df=df, top_n=top_n)


