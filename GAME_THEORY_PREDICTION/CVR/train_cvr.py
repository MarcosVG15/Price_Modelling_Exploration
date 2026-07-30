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


class create_cvr_predictor:

    def __init__(self, cvs_folder_path, seg_level, seg_terms, bbox_predictor_path):
        self.folder_path = cvs_folder_path
        self.seg_level = seg_level

        self.aggregate_data = None

        for term in seg_terms:
            temp_data = pd.read_csv(f'{self.folder_path}/all_feature_data_{term}.csv')
            self.aggregate_data = self.union(self.aggregate_data, temp_data)

        HERE = os.path.dirname(os.path.abspath(__file__))
        self.CVR_path = os.path.join(HERE, "cvr_feature_panel.csv")

        self.bbox_predictor_path = bbox_predictor_path
        self._BBOX_BUNDLE = None

        self.CVR_SHAP_MODEL = None
        self.CVR_SHAP_MODEL_PATH = os.path.join(HERE, "cvr_shap_model.joblib")

        self._CVR_MODEL = None
        self._CVR_MODEL_PATH = os.path.join(HERE, "cvr_model.joblib")
        self._CVR_FEATURES = None
        self._CVR_CAT_COLS = None
        self._CVR_CAT_LEVELS = None
        self._CVR_X_test = None
        self._CVR_Y_test = None

        self._CVR_PANEL_DF = None


    def union(self, aggregate_data, temp_data):
        return pd.concat([aggregate_data, temp_data], axis=0, ignore_index=True)

    def get_model_shap(self):
        return self._load_shap_model()

    def get_model(self):
        return self._load_cvr_model()

    def reset_models(self):
        for path in (self.CVR_SHAP_MODEL_PATH, self._CVR_MODEL_PATH):
            if os.path.exists(path):
                os.remove(path)
                print(f"[cvr] removed {path}")

        self.CVR_SHAP_MODEL = None
        self._CVR_MODEL = None
        self._CVR_FEATURES = None
        self._CVR_CAT_COLS = None
        self._CVR_CAT_LEVELS = None
        self._CVR_X_test = None
        self._CVR_Y_test = None


    def _load_bbox_bundle(self):
        if self._BBOX_BUNDLE is None:
            self._BBOX_BUNDLE = joblib.load(self.bbox_predictor_path)
        return self._BBOX_BUNDLE

    def _predict_buybox(self, df):
        bundle = self._load_bbox_bundle()
        model = bundle["model"]
        features = bundle["features"]
        cat_cols = bundle["cat_cols"]
        cat_levels = bundle["cat_levels"]

        X = pd.DataFrame(index=df.index)
        for col in features:
            if col in cat_cols:
                values = df[col] if col in df.columns else pd.Series(np.nan, index=df.index)
                X[col] = pd.Categorical(values, categories=cat_levels[col])
            else:
                values = df[col] if col in df.columns else pd.Series(np.nan, index=df.index)
                X[col] = pd.to_numeric(values, errors="coerce")

        return model.predict(X)


    def _load_shap_model(self):
        if self.CVR_SHAP_MODEL is not None:
            return self.CVR_SHAP_MODEL

        if os.path.exists(self.CVR_SHAP_MODEL_PATH):
            try:
                self.CVR_SHAP_MODEL = joblib.load(self.CVR_SHAP_MODEL_PATH)
                return self.CVR_SHAP_MODEL
            except Exception as e:
                print(f"[cvr] cached SHAP model at {self.CVR_SHAP_MODEL_PATH} unreadable ({e}); retraining")

        self.extract_feature_importance_CVR()
        return self.CVR_SHAP_MODEL

    def _load_cvr_model(self):
        if self._CVR_MODEL is not None and self._CVR_X_test is not None:
            return self._CVR_MODEL

        if os.path.exists(self._CVR_MODEL_PATH):
            bundle = None
            try:
                bundle = joblib.load(self._CVR_MODEL_PATH)
            except Exception as e:
                print(f"[cvr] cached model at {self._CVR_MODEL_PATH} unreadable ({e}); retraining")

            if bundle is not None:
                self._CVR_MODEL = bundle["model"]
                self._CVR_FEATURES = bundle["features"]
                self._CVR_CAT_COLS = bundle["cat_cols"]
                self._CVR_CAT_LEVELS = bundle["cat_levels"]

                cvr_data = pd.read_csv(self.CVR_path, low_memory=False)
                cvr_data["buybox_pred"] = self._predict_buybox(cvr_data)
                Y = cvr_data["cvr"].clip(0.0, 1.0)
                X = cvr_data[[c for c in self._CVR_FEATURES if c in cvr_data.columns]].copy()
                for c in self._CVR_CAT_COLS:
                    X[c] = X[c].astype("category")
                keep_idx = self._cap_indices_by_zero_cvr(Y)   # match fit_cvr's rebalanced split
                X, Y = X.loc[keep_idx], Y.loc[keep_idx]
                _, X_test, _, Y_test = train_test_split(X, Y, test_size=0.4, random_state=0)
                self._CVR_X_test = X_test
                self._CVR_Y_test = Y_test
                return self._CVR_MODEL

        self.fit_cvr()
        return self._CVR_MODEL


    def extract_feature_importance_CVR(self):
        if not os.path.exists(self.CVR_path):
            raise FileNotFoundError(
                f"CVR feature panel not found: {self.CVR_path}\n"
                f"Run `python GAME_THEORY_PREDICTION/CVR/build_feature_panel_CVR.py` first.")
        shap_data = pd.read_csv(self.CVR_path, low_memory=False)
        shap_data["buybox_pred"] = self._predict_buybox(shap_data)

        drop_columns = [
            "asin", "marketplace_id", "week", "cvr", "buybox_pct",
            "units", "sessions", "revenue",
            "implied_price", "oi_price", "listing_price", "raw_price",
            "rank_value", "main_browser_node_id",
            "scrape_title",   # raw scraped title text -- its signal already lives in title_* stylistic scores
        ]

        Y = shap_data["cvr"].clip(0.0, 1.0)
        X = shap_data.drop(columns=[c for c in drop_columns if c in shap_data.columns])

        sparse = [c for c in X.columns if X[c].notna().mean() < 0.02]
        X = X.drop(columns=sparse)

        cat_cols = [c for c in ["brand", "product_type", "has_aplus"] if c in X.columns]
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True).astype(float)
        print(f"[shap-cvr] {X.shape[1]} features; dropped {len(sparse)} near-empty cols: {sparse}")

        # undersample cvr==0 rows BEFORE splitting (cvr>0 rows always kept, untouched),
        # so both train and test come from the rebalanced distribution.
        n_before = len(X)
        keep_idx = self._cap_indices_by_zero_cvr(Y)
        X, Y = X.loc[keep_idx], Y.loc[keep_idx]
        print(f"[cvr] rebalanced panel: {n_before:,} -> {len(X):,} rows "
              f"({(Y==0).sum():,} zero / {(Y!=0).sum():,} non-zero)")

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

        model = HistGradientBoostingRegressor(max_iter=200, random_state=0)
        model.fit(X_train, Y_train)
        self.CVR_SHAP_MODEL = {'model': model, 'features': X}
        joblib.dump(self.CVR_SHAP_MODEL, self.CVR_SHAP_MODEL_PATH)
        print(f"[cvr] saved SHAP-ranking model -> {self.CVR_SHAP_MODEL_PATH}")


    def fit_cvr(self, top_k=15):
        shap_bundle = self._load_shap_model()
        model = shap_bundle["model"]
        feats = shap_bundle["features"]

        shap_vals = shap.TreeExplainer(model).shap_values(feats.iloc[:2000])
        importance = (pd.Series(abs(shap_vals).mean(axis=0), index=feats.columns)
                        .sort_values(ascending=False))
        top_features = importance.index.tolist()[:top_k]

        cats = ["brand", "product_type", "has_aplus"]
        raw_keep = []
        for f in top_features:
            parent = next((c for c in cats if f == c or f.startswith(c + "_")), f)
            raw_keep.append(parent)
        raw_keep = list(dict.fromkeys(raw_keep))
        print(f"[cvr] top {top_k} one-hot features -> {len(raw_keep)} raw cols: {raw_keep}")

        cvr_data = pd.read_csv(self.CVR_path, low_memory=False)
        cvr_data["buybox_pred"] = self._predict_buybox(cvr_data)
        Y = cvr_data["cvr"].clip(0.0, 1.0)
        X = cvr_data[[c for c in raw_keep if c in cvr_data.columns]].copy()
        cat_in_X = [c for c in cats if c in X.columns]
        for c in cat_in_X:
            X[c] = X[c].astype("category")
        cat_mask = [c in cat_in_X for c in X.columns]

        # undersample cvr==0 rows BEFORE splitting (cvr>0 rows always kept, untouched),
        # so both train and test come from the rebalanced distribution.
        n_before = len(X)
        keep_idx = self._cap_indices_by_zero_cvr(Y)
        X, Y = X.loc[keep_idx], Y.loc[keep_idx]
        print(f"[cvr] rebalanced panel: {n_before:,} -> {len(X):,} rows "
              f"({(Y==0).sum():,} zero / {(Y!=0).sum():,} non-zero)")

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=0)
        best_tuned = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.03,
            max_leaf_nodes=31,
            min_samples_leaf=5,
            l2_regularization=0.05,
            categorical_features=cat_mask,
            random_state=0,
        )
        best_tuned.fit(X_train, Y_train)
        self._CVR_MODEL = best_tuned
        self._CVR_FEATURES = list(X.columns)
        self._CVR_CAT_COLS = cat_in_X
        self._CVR_CAT_LEVELS = {c: list(X[c].cat.categories) for c in cat_in_X}
        self._CVR_X_test = X_test
        self._CVR_Y_test = Y_test
        joblib.dump({
            "model": self._CVR_MODEL,
            "features": self._CVR_FEATURES,
            "cat_cols": self._CVR_CAT_COLS,
            "cat_levels": self._CVR_CAT_LEVELS,
        }, self._CVR_MODEL_PATH)
        preds = best_tuned.predict(X_test)
        r2 = r2_score(Y_test, preds)
        mse = mean_squared_error(Y_test, preds)
        mae = mean_absolute_error(Y_test, preds)
        try:
            # ROC-AUC needs a binary label. Unlike buy-box share, CVR is tiny and right-skewed
            # (median is 0, only ~11.5% of rows have cvr > 0) so a 0.5 cutoff would be degenerate --
            # binarize as "converted at all that week" instead.
            auc = roc_auc_score((Y_test > 0).astype(int), preds)
        except ValueError:
            auc = float("nan")   # holdout had only one class

        print(f"[cvr] refit on {X.shape[1]} raw cols -> holdout R^2={r2:.3f}  MSE={mse:.4f}  "
              f"MAE={mae:.4f}  ROC-AUC={auc:.3f}")
        print(f"[cvr] saved tuned model -> {self._CVR_MODEL_PATH}")
        return X_test, Y_test, self._CVR_MODEL


    def _eval_out_path(self, filename):
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cvr_eval")
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, filename)

    def plot_predicted_vs_actual(self, path=None):
        model = self._load_cvr_model()

        Y_test = self._CVR_Y_test
        Y_pred = model.predict(self._CVR_X_test)
        r2 = r2_score(Y_test, Y_pred)
        mae = mean_absolute_error(Y_test, Y_pred)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(Y_test, Y_pred, alpha=0.4, s=12)
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="perfect")
        ax.set_xlabel("Actual CVR")
        ax.set_ylabel("Predicted CVR")
        ax.set_title(f"CVR prediction -- R2={r2:.3f}, MAE={mae:.4f}")
        ax.legend(loc="upper right")
        fig.tight_layout()

        path = path or self._eval_out_path("predicted_vs_actual.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"[cvr-eval] predicted_vs_actual -> {path}")
        return path

    def plot_residuals(self, path=None):
        model = self._load_cvr_model()

        Y_test = self._CVR_Y_test
        Y_pred = model.predict(self._CVR_X_test)
        residual = Y_pred - Y_test

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].scatter(Y_test, residual, alpha=0.4, s=12)
        axes[0].axhline(0, color="gray", ls="--", lw=1)
        axes[0].set_xlabel("Actual CVR")
        axes[0].set_ylabel("Residual (pred - actual)")
        axes[0].set_title("Residual vs actual")

        axes[1].hist(residual, bins=30, color="steelblue", edgecolor="black", alpha=0.8)
        axes[1].axvline(0, color="gray", ls="--", lw=1)
        axes[1].set_xlabel("Residual (pred - actual)")
        axes[1].set_title(f"Residual distribution (mean={residual.mean():+.4f})")
        fig.tight_layout()

        path = path or self._eval_out_path("residuals.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"[cvr-eval] residuals -> {path}")
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
        ax.set_title(f"Top {len(top)} CVR features ({top.sum():.0f}% of total importance)")
        fig.tight_layout()

        path = path or self._eval_out_path("shap_importance.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"[cvr-eval] shap_importance -> {path}")
        return path

    CVR_DIST_BINS = 50        # only affects the distribution PLOT's bin width, not resampling

    def plot_cvr_distribution(self, data=None, label="", path=None):
        """Distribution of the CVR target across data points -- how zero-inflated it is.
        Doesn't need a fitted model, just the raw panel (or a filtered one, via `data`),
        since this is meant to be looked at BEFORE/AFTER rebalancing the zero-CVR rows."""
        panel = data if data is not None else self._cvr_panel()
        y = panel["cvr"].clip(0.0, 1.0).dropna()
        zero_frac = (y == 0).mean()
        edges = np.linspace(0.0, 1.0, self.CVR_DIST_BINS + 1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        counts = [(y == 0).sum(), (y > 0).sum()]
        axes[0].bar(["cvr == 0", "cvr > 0"], counts,
                    color=["indianred", "steelblue"], edgecolor="black")
        for i, v in enumerate(counts):
            axes[0].text(i, v, f" {v:,}\n({v/len(y)*100:.1f}%)", ha="center", va="bottom")
        axes[0].set_ylim(0, max(counts) * 1.15)
        axes[0].set_ylabel("data point count")
        axes[0].set_title("Zero vs non-zero CVR rows")

        axes[1].hist(y, bins=edges, color="steelblue", edgecolor="black", alpha=0.85)
        axes[1].set_yscale("log")
        axes[1].set_xlabel("CVR")
        axes[1].set_ylabel("data point count (log scale)")
        axes[1].set_title(f"Full CVR distribution ({self.CVR_DIST_BINS} bins)")
        prefix = f"{label} -- " if label else ""
        fig.suptitle(f"{prefix}CVR distribution across {len(y):,} data points -- "
                     f"{zero_frac*100:.1f}% are exactly 0")
        fig.tight_layout(rect=[0, 0, 1, 0.94])

        default_name = f"cvr_distribution_{label.lower()}.png" if label else "cvr_distribution.png"
        path = path or self._eval_out_path(default_name)
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"[cvr-eval] cvr_distribution{'_' + label.lower() if label else ''} -> {path}")
        return path

    def _cap_indices_by_zero_cvr(self, cvr, cap=None, random_state=0):
        """Core of the rebalancing: which of `cvr`'s index labels to KEEP. Undersamples
        ONLY the cvr==0 rows down to `cap`; every cvr>0 row is kept untouched, at its
        original value -- no binning, so exact zeros are never lumped in with barely-
        positive rows (that lumping is what caused an earlier ROC-AUC regression).

        cap=None (default) balances 1:1 with the non-zero row count. That's not arbitrary:
        swept caps from 316 up to the full 50,119 zero rows and measured mean predicted CVR
        for true-zero vs true-nonzero test rows -- below the 1:1 point the model
        systematically predicts HIGHER cvr for true zeros (ROC-AUC < 0.5, a real inversion,
        not noise: at cap=316, pred(zero)=0.227 vs pred(nonzero)=0.081). That crosses over
        and self-corrects almost exactly at 1:1 (pred(zero)=0.045 vs pred(nonzero)=0.047,
        ROC-AUC=0.559). R2 keeps falling as cap grows (0.415 at cap=316 -> 0.123 at 1:1 ->
        0.016 at the full natural distribution), so 1:1 is the smallest cap that avoids the
        inversion -- not the smallest cap, period. Pass an explicit cap to override.

        Works on any Series indexed the same way as the rows you'll filter (a raw panel's
        cvr column, or a training split's Y_train)."""
        y = cvr.clip(0.0, 1.0)
        zero_idx = y.index[y == 0]
        nonzero_idx = y.index[y != 0]
        cap = len(nonzero_idx) if cap is None else cap
        if len(zero_idx) > cap:
            rng = np.random.RandomState(random_state)
            zero_idx = rng.choice(zero_idx, size=int(cap), replace=False)
        return pd.Index(sorted(list(zero_idx) + list(nonzero_idx)))

    def _reduce_zero_cvr(self, panel, cap=None, random_state=0):
        """Panel-level wrapper around _cap_indices_by_zero_cvr, for the before/after plot."""
        keep_idx = self._cap_indices_by_zero_cvr(panel["cvr"], cap=cap, random_state=random_state)
        return panel.loc[keep_idx]

    def plot_cvr_rebalance_before_after(self, cap=None, random_state=0):
        """Undersample cvr==0 rows down to `cap` (default: 1:1 with the non-zero count --
        see _cap_indices_by_zero_cvr for why), keeping every cvr>0 row untouched, save a
        before/after distribution plot pair to cvr_eval/, and return the filtered panel +
        both plot paths. Doesn't retrain anything -- just filters + visualizes."""
        panel = self._cvr_panel()

        before_path = self.plot_cvr_distribution(data=panel, label="before")
        filtered = self._reduce_zero_cvr(panel, cap=cap, random_state=random_state)
        after_path = self.plot_cvr_distribution(data=filtered, label="after")

        print(f"[cvr-rebalance] zero cap={cap if cap is not None else 'auto (1:1)'} -> "
              f"{len(panel):,} rows -> {len(filtered):,} rows ({len(filtered)/len(panel)*100:.1f}% kept)")
        return {"panel": filtered, "before": before_path, "after": after_path}

    def evaluate_cvr(self):
        return {
            "predicted_vs_actual": self.plot_predicted_vs_actual(),
            "residuals": self.plot_residuals(),
            "shap_importance": self.plot_shap_summary(),
            "cvr_distribution": self.plot_cvr_distribution(),
        }


    def _cvr_panel(self):
        if self._CVR_PANEL_DF is None:
            self._CVR_PANEL_DF = pd.read_csv(self.CVR_path, low_memory=False)
        return self._CVR_PANEL_DF

    def _snapshot_cvr_features(self, asin):
        if self._CVR_FEATURES is None:
            return None

        panel = self._cvr_panel()
        rows = panel[panel["asin"].astype(str) == str(asin)] if "asin" in panel.columns else panel.iloc[0:0]

        cat_cols = self._CVR_CAT_COLS or []
        snap = {}
        for col in self._CVR_FEATURES:
            if rows.empty or col not in panel.columns:
                snap[col] = np.nan
            elif col in cat_cols:
                m = rows[col].mode(dropna=True)
                snap[col] = m.iloc[0] if not m.empty else np.nan
            else:
                snap[col] = float(pd.to_numeric(rows[col], errors="coerce").median())
        return snap


if __name__ == "__main__":
    HERE = os.path.dirname(os.path.abspath(__file__))
    BBOX_MODEL_PATH = os.path.join(HERE, "..", "BBOX", "bbox_model.joblib")

    # CVR is chained off the buy-box model (fit_cvr/extract_feature_importance_CVR both call
    # _predict_buybox(), which loads this file) -- fail fast with a clear message instead of
    # letting joblib.load() raise deep inside training.
    if not os.path.exists(BBOX_MODEL_PATH):
        raise FileNotFoundError(
            f"buy-box predictor not found: {BBOX_MODEL_PATH}\n"
            f"Run `python GAME_THEORY_PREDICTION/BBOX/train_bbox.py` first -- CVR features "
            f"include the buy-box model's own prediction (buybox_pred).")
    print(f"[cvr] using buy-box predictor -> {BBOX_MODEL_PATH}")

    predictor = create_cvr_predictor(cvs_folder_path=HERE, seg_level=None, seg_terms=[],
                                      bbox_predictor_path=BBOX_MODEL_PATH)
    predictor.reset_models()   # cached joblib models may be stale -- force a refit
    predictor.fit_cvr()
    for name, path in predictor.evaluate_cvr().items():
        print(f"{name}: {path}")
