
import os 

import shap
import joblib
import numpy as np
import pandas as pd 

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

'''  
for each of the csv generate a vn cluster network and add it to a dict such that you can reference based on the segmenation level 


for this you will need to extract all of the search terms for a specific segmentation level that were used to load the data
if the csv is not found then it should be genereted. 

then we will find in each of the csv the commaxx products that we hae extracted in the bbox feature panel
    - to each of these we will append the keyword and the cluster id that it contains
    WHY ? = such that I can extract the competitor data - this means that we can accurately trace the buybox


We will do a shap analysis on the features to understand which are important to the buybox prediction
    - this will include all of the ones that are in the data panel as well as the data that is extracted from the competitor in the cluster id
        - so we have in fact so many more features to understand the buybox
    

Storing 
The model will be stored in a joblib file for easy access - later this model will be updated every 3 weeks or so - I dont fuking know

TESTING 
for the commax product we will for each period stored in the csv we will predict the buybox percetange vs the actual one and see how much they differ in size
'''



# Economically-required direction for price/cost-like features: a HistGradientBoosting
# tree has no built-in notion that "pricier than the market -> never a HIGHER chance of
# winning the buy-box" -- on sparse real data it can and does learn local reversals of
# that (see e.g. a $20 price increase flipping predicted win probability from 58% to
# 99.7%). monotonic_cst forces the direction; keyed by name so it survives fit_buybox()'s
# SHAP top-k selection regardless of which subset of these actually gets picked, and any
# column not listed here (including all categoricals) stays unconstrained (0).
MONOTONIC_DIRECTION = {
    "price": -1, "own_landed": -1, "own_shipping": -1,          # my own price/cost: higher -> never helps
    "price_vs_lowest": -1, "price_vs_bb": -1, "landed_vs_bb": -1,  # my gap vs the market: wider -> never helps
    "lowest_price": +1, "min_competitor_landed": +1,             # the floor I have to beat: higher -> never hurts
}


class create_bbox_predictor :

    #  this method needs to know the location of the path that contains all of the csv that will be used to train the cvr predictor
    #  lookback_days: None = use the full panel history; otherwise only weeks within that many days
    #  of the panel's latest week are used (e.g. 21 to match the ~3-week competitor/stock snapshot window).
    def __init__(self , cvs_folder_path , seg_level, seg_terms, lookback_days=None):
        self.folder_path = cvs_folder_path
        self.seg_level = seg_level
        self.lookback_days = lookback_days

        self.aggregate_data = None


        for term in seg_terms : 
            temp_data = pd.read_csv(f'{self.folder_path}/all_feature_data_{term}.csv')
            self.aggregate_data = self.union(self.aggregate_data , temp_data)


        HERE = os.path.dirname(os.path.abspath(__file__))
        self.BBOX_path = os.path.join(HERE, "bbox_feature_panel.csv")

        self.BUYBOX_SHAP_MODEL = None
        self.BUYBOX_SHAP_MODEL_PATH = os.path.join(HERE, "bbox_shap_model.joblib")

        self._BUYBOX_MODEL = None
        self._BUYBOX_MODEL_PATH = os.path.join(HERE, "bbox_model.joblib")
        self._BUYBOX_FEATURES = None
        self._BUYBOX_CAT_COLS = None
        self._BUYBOX_CAT_LEVELS = None
        self._BUYBOX_X_test = None
        self._BUYBOX_Y_test = None

        self._BUYBOX_PANEL_DF = None


    def _read_panel(self):
        """Load bbox_feature_panel.csv, optionally clipped to the last `self.lookback_days`
        days (relative to the panel's own latest week) -- use this to train/evaluate on just
        the recent window where the competitor/stock/rank snapshot columns actually have data,
        instead of the full 2-year history where they're near-empty."""
        panel = pd.read_csv(self.BBOX_path)
        if self.lookback_days is None:
            return panel
        week = pd.to_datetime(panel["week"])
        cutoff = week.max() - pd.Timedelta(days=self.lookback_days)
        panel = panel[week >= cutoff].reset_index(drop=True)
        print(f"[bbox] lookback_days={self.lookback_days} -> {len(panel):,} rows "
              f"(weeks {week[week >= cutoff].min().date()} -> {week.max().date()})")
        return panel


    def get_model_shap(self):
        return self._load_shap_model()

    def get_model(self):
        return self._load_buybox_model()

    def reset_models(self):
  
        for path in (self.BUYBOX_SHAP_MODEL_PATH, self._BUYBOX_MODEL_PATH):
            if os.path.exists(path):
                os.remove(path)
                print(f"[bbox] removed {path}")

        self.BUYBOX_SHAP_MODEL = None
        self._BUYBOX_MODEL = None
        self._BUYBOX_FEATURES = None
        self._BUYBOX_CAT_COLS = None
        self._BUYBOX_CAT_LEVELS = None
        self._BUYBOX_X_test = None
        self._BUYBOX_Y_test = None


    #  this method should check what columns coiincide and merge the data and if not add new columns
    def union(self , aggregate_data , temp_data ) :
        return pd.concat([aggregate_data, temp_data], axis=0, ignore_index=True)


    def _load_shap_model(self):

        if self.BUYBOX_SHAP_MODEL is not None:
            return self.BUYBOX_SHAP_MODEL

        if os.path.exists(self.BUYBOX_SHAP_MODEL_PATH):
            try:
                self.BUYBOX_SHAP_MODEL = joblib.load(self.BUYBOX_SHAP_MODEL_PATH)
                return self.BUYBOX_SHAP_MODEL
            except Exception as e:
                print(f"[bbox] cached SHAP model at {self.BUYBOX_SHAP_MODEL_PATH} unreadable ({e}); retraining")

        self.extract_feature_importance_BBox()
        return self.BUYBOX_SHAP_MODEL

    def _load_buybox_model(self):

        if self._BUYBOX_MODEL is not None and self._BUYBOX_X_test is not None:
            return self._BUYBOX_MODEL

        if os.path.exists(self._BUYBOX_MODEL_PATH):
            bundle = None
            try:
                bundle = joblib.load(self._BUYBOX_MODEL_PATH)
            except Exception as e:
                print(f"[bbox] cached model at {self._BUYBOX_MODEL_PATH} unreadable ({e}); retraining")

            if bundle is not None:
                self._BUYBOX_MODEL = bundle["model"]
                self._BUYBOX_FEATURES = bundle["features"]
                self._BUYBOX_CAT_COLS = bundle["cat_cols"]
                self._BUYBOX_CAT_LEVELS = bundle["cat_levels"]

                bbox_data = self._read_panel()
                Y = (bbox_data["buybox_pct"] / 100.0).clip(0.0, 1.0)
                X = bbox_data[[c for c in self._BUYBOX_FEATURES if c in bbox_data.columns]].copy()
                for c in self._BUYBOX_CAT_COLS:
                    X[c] = X[c].astype("category")
                _, X_test, _, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
                self._BUYBOX_X_test = X_test
                self._BUYBOX_Y_test = Y_test
                return self._BUYBOX_MODEL

        self.fit_buybox()
        return self._BUYBOX_MODEL


    def extract_feature_importance_BBox(self):
        if not os.path.exists(self.BBOX_path):
            raise FileNotFoundError(
                f"buy-box feature panel not found: {self.BBOX_path}\n"
                f"Run `python GAME_THEORY_PREDICTION/BBOX/build_feature_panel_BBox.py` first.")
        shap_data = self._read_panel()

        drop_columns = [
            "asin", "marketplace_id", "week", "buybox_pct",
            "units", "sessions", "revenue",
            "implied_price", "oi_price", "listing_price", "raw_price",
            "rank_value", "main_browser_node_id",
        ]

        Y = (shap_data["buybox_pct"] / 100.0).clip(0.0, 1.0)  
        X = shap_data.drop(columns=[c for c in drop_columns if c in shap_data.columns])

        sparse = [c for c in X.columns if X[c].notna().mean() < 0.02]
        X = X.drop(columns=sparse)
        cat_cols = [c for c in ["brand", "product_type", "manufacturer", "has_aplus"]
                    if c in X.columns]

        X = pd.get_dummies(X, columns=cat_cols, drop_first=True).astype(float)
        print(f"[shap-bbox] {X.shape[1]} features; dropped {len(sparse)} near-empty cols: {sparse}")

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

        model = HistGradientBoostingRegressor(max_iter=200, random_state=0)
        model.fit(X_train, Y_train)
        self.BUYBOX_SHAP_MODEL = {'model': model , 'features': X}
        joblib.dump(self.BUYBOX_SHAP_MODEL, self.BUYBOX_SHAP_MODEL_PATH)
        print(f"[bbox] saved SHAP-ranking model -> {self.BUYBOX_SHAP_MODEL_PATH}")


    def fit_buybox(self, top_k=15):

        shap_bundle = self._load_shap_model()
        model = shap_bundle["model"]
        feats = shap_bundle["features"]

        shap_vals = shap.TreeExplainer(model).shap_values(feats.iloc[:2000])
        importance = (pd.Series(abs(shap_vals).mean(axis=0), index=feats.columns)
                        .sort_values(ascending=False))
        top_features = importance.index.tolist()[:top_k]

        cats = ["brand", "product_type", "manufacturer", "has_aplus"]
        raw_keep = []
        for f in top_features:
            parent = next((c for c in cats if f == c or f.startswith(c + "_")), f)
            raw_keep.append(parent)
        raw_keep = list(dict.fromkeys(raw_keep))        
        print(f"[bbox] top {top_k} one-hot features -> {len(raw_keep)} raw cols: {raw_keep}")

        bbox_data = self._read_panel()
        Y = (bbox_data["buybox_pct"] / 100.0).clip(0.0, 1.0)
        X = bbox_data[[c for c in raw_keep if c in bbox_data.columns]].copy()
        cat_in_X = [c for c in cats if c in X.columns]
        for c in cat_in_X:
            X[c] = X[c].astype("category")
        cat_mask = [c in cat_in_X for c in X.columns]

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=0)
        monotonic_cst = [MONOTONIC_DIRECTION.get(c, 0) for c in X.columns]
        best_tuned = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.08, max_leaf_nodes=31,
            min_samples_leaf=10, l2_regularization=0.5,
            categorical_features=cat_mask, monotonic_cst=monotonic_cst, random_state=0,
        )
        
        best_tuned.fit(X_train, Y_train)
        self._BUYBOX_MODEL = best_tuned
        self._BUYBOX_FEATURES = list(X.columns)
        self._BUYBOX_CAT_COLS = cat_in_X
        self._BUYBOX_CAT_LEVELS = {c: list(X[c].cat.categories) for c in cat_in_X}
        self._BUYBOX_X_test = X_test
        self._BUYBOX_Y_test = Y_test

        joblib.dump({
            "model": self._BUYBOX_MODEL,
            "features": self._BUYBOX_FEATURES,
            "cat_cols": self._BUYBOX_CAT_COLS,
            "cat_levels": self._BUYBOX_CAT_LEVELS,
        }, self._BUYBOX_MODEL_PATH)

        preds = best_tuned.predict(X_test)
        r2 = r2_score(Y_test, preds)
        mse = mean_squared_error(Y_test, preds)
        mae = mean_absolute_error(Y_test, preds)
        try:
            # ROC-AUC needs a binary label -- "won majority of the buy-box that week" (>=0.5)
            auc = roc_auc_score((Y_test >= 0.5).astype(int), preds)
        except ValueError:
            auc = float("nan")   # holdout had only one class

        print(f"[bbox] refit on {X.shape[1]} raw cols -> holdout R^2={r2:.3f}  MSE={mse:.3f}  "
              f"MAE={mae:.3f}  ROC-AUC={auc:.3f}")
        print(f"[bbox] saved tuned model -> {self._BUYBOX_MODEL_PATH}")
        return X_test, Y_test, self._BUYBOX_MODEL

    def _eval_out_path(self, filename):
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bbox_eval")
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, filename)

    def plot_predicted_vs_actual(self, path=None):
        """Held-out predicted vs actual buy-box share, from fit_buybox()'s test split."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        model = self._load_buybox_model()

        Y_test = self._BUYBOX_Y_test
        Y_pred = model.predict(self._BUYBOX_X_test)
        r2 = r2_score(Y_test, Y_pred)
        mae = mean_absolute_error(Y_test, Y_pred)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(Y_test, Y_pred, alpha=0.4, s=12)
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="perfect")
        ax.set_xlabel("Actual buy-box share")
        ax.set_ylabel("Predicted buy-box share")
        ax.set_title(f"Buy-box prediction -- R2={r2:.3f}, MAE={mae:.3f}")
        ax.legend(loc="upper left")
        fig.tight_layout()

        path = path or self._eval_out_path("predicted_vs_actual.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"[bbox-eval] predicted_vs_actual -> {path}")
        return path

    def plot_residuals(self, path=None):
        model = self._load_buybox_model()

        Y_test = self._BUYBOX_Y_test
        Y_pred = model.predict(self._BUYBOX_X_test)
        residual = Y_pred - Y_test

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].scatter(Y_test, residual, alpha=0.4, s=12)
        axes[0].axhline(0, color="gray", ls="--", lw=1)
        axes[0].set_xlabel("Actual buy-box share")
        axes[0].set_ylabel("Residual (pred - actual)")
        axes[0].set_title("Residual vs actual")

        axes[1].hist(residual, bins=30, color="steelblue", edgecolor="black", alpha=0.8)
        axes[1].axvline(0, color="gray", ls="--", lw=1)
        axes[1].set_xlabel("Residual (pred - actual)")
        axes[1].set_title(f"Residual distribution (mean={residual.mean():+.3f})")
        fig.tight_layout()

        path = path or self._eval_out_path("residuals.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"[bbox-eval] residuals -> {path}")
        return path

    def plot_shap_summary(self, top_n=20, sample=2000, path=None):

        shap_bundle = self._load_shap_model()
        model = shap_bundle["model"]
        feats = shap_bundle["features"].iloc[:sample]

        shap_vals = shap.TreeExplainer(model).shap_values(feats)
        mean_abs = pd.Series(np.abs(shap_vals).mean(axis=0), index=feats.columns)
        
        importance_pct = (mean_abs / mean_abs.sum() * 100).sort_values(ascending=False)
        top = importance_pct.head(top_n)

        fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(top))))
        top.iloc[::-1].plot.barh(ax=ax, color="steelblue")
        for i, v in enumerate(top.iloc[::-1]):
            ax.text(v, i, f" {v:.1f}%", va="center", fontsize=8)
        ax.set_xlim(0, top.max() * 1.15)
        ax.set_xlabel("% of total SHAP importance")
        ax.set_title(f"Top {len(top)} buy-box features ({top.sum():.0f}% of total importance)")
        fig.tight_layout()

        path = path or self._eval_out_path("shap_importance.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"[bbox-eval] shap_importance -> {path}")
        return path

    BBOX_DIST_BINS = 50

    def plot_bbox_distribution(self, data=None, label="", path=None):
        """Distribution of the buy-box target across data points. Unlike CVR (zero-
        inflated, mass piled at one end), buy-box share is BIMODAL: heavy mass at both
        0 (lost the buy-box every session that week) and 1 (won it every session) --
        see the vertical stripes at 0/1 in predicted_vs_actual.png. Doesn't need a
        fitted model, just the raw panel (or a filtered one, via `data`)."""
        panel = data if data is not None else self._buybox_panel()
        y = (panel["buybox_pct"] / 100.0).clip(0.0, 1.0).dropna()
        frac_zero = (y == 0).mean()
        frac_one = (y == 1).mean()
        edges = np.linspace(0.0, 1.0, self.BBOX_DIST_BINS + 1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        counts = [(y == 0).sum(), ((y > 0) & (y < 1)).sum(), (y == 1).sum()]
        axes[0].bar(["== 0 (always lost)", "0 < x < 1 (mixed)", "== 1 (always won)"], counts,
                    color=["indianred", "steelblue", "seagreen"], edgecolor="black")
        for i, v in enumerate(counts):
            axes[0].text(i, v, f" {v:,}\n({v/len(y)*100:.1f}%)", ha="center", va="bottom")
        axes[0].set_ylim(0, max(counts) * 1.15)
        axes[0].set_ylabel("data point count")
        axes[0].set_title("Always-lost vs mixed vs always-won weeks")

        axes[1].hist(y, bins=edges, color="steelblue", edgecolor="black", alpha=0.85)
        axes[1].set_yscale("log")
        axes[1].set_xlabel("Buy-box share")
        axes[1].set_ylabel("data point count (log scale)")
        axes[1].set_title(f"Full buy-box distribution ({self.BBOX_DIST_BINS} bins)")
        prefix = f"{label} -- " if label else ""
        fig.suptitle(f"{prefix}Buy-box distribution across {len(y):,} data points -- "
                     f"{frac_zero*100:.1f}% always lost, {frac_one*100:.1f}% always won")
        fig.tight_layout(rect=[0, 0, 1, 0.94])

        default_name = f"bbox_distribution_{label.lower()}.png" if label else "bbox_distribution.png"
        path = path or self._eval_out_path(default_name)
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"[bbox-eval] bbox_distribution{'_' + label.lower() if label else ''} -> {path}")
        return path

    def evaluate_bbox(self):
        """Run the full evaluation suite and return {name: saved_path}."""
        return {
            "predicted_vs_actual": self.plot_predicted_vs_actual(),
            "residuals": self.plot_residuals(),
            "shap_importance": self.plot_shap_summary(),
            "bbox_distribution": self.plot_bbox_distribution(),
        }


    def _buybox_panel(self):
        if self._BUYBOX_PANEL_DF is None:
            self._BUYBOX_PANEL_DF = pd.read_csv(self.BBOX_path)
        return self._BUYBOX_PANEL_DF

    def _snapshot_buybox_features(self, asin):
  
        if self._BUYBOX_FEATURES is None:
            return None

        panel = self._buybox_panel()
        rows = panel[panel["asin"].astype(str) == str(asin)] if "asin" in panel.columns else panel.iloc[0:0]

        cat_cols = self._BUYBOX_CAT_COLS or []
        snap = {}
        for col in self._BUYBOX_FEATURES:
            if rows.empty or col not in panel.columns:
                snap[col] = np.nan
            elif col in cat_cols:
                m = rows[col].mode(dropna=True)
                snap[col] = m.iloc[0] if not m.empty else np.nan
            else:
                snap[col] = float(pd.to_numeric(rows[col], errors="coerce").median())
        return snap


if __name__ == "__main__":
    import sys

    # optional CLI arg: how many days back (from the panel's latest week) to train/evaluate on.
    # omit for the full 2-year history; pass e.g. 21 to match the competitor/stock snapshot window.
    lookback_days = 370

    HERE = os.path.dirname(os.path.abspath(__file__))
    predictor = create_bbox_predictor(cvs_folder_path=HERE, seg_level=None, seg_terms=[],
                                       lookback_days=lookback_days)
    predictor.reset_models()   # cached joblib models are trained on the old panel schema -- force a refit
    predictor.fit_buybox()
    for name, path in predictor.evaluate_bbox().items():
        print(f"{name}: {path}")
