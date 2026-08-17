"""
Two-stage (hurdle) CVR predictor.

Stage 1 classifier: P(cvr > 0)              -- did this asin/week convert at all.
Stage 2 regressor:  E[cvr | cvr > 0]         -- conversion RATE magnitude, trained ONLY on
                                                 rows that actually converted (zero rows are
                                                 fully excluded here, not resampled -- no
                                                 rebalancing needed since the regressor never
                                                 sees a zero).
Combined:           P(cvr > 0) * E[cvr | cvr > 0]   -- the standard hurdle-model expectation,
                                                        continuous across the whole population.

Why this exists: train_cvr.py's plain regression, once zero rows are undersampled so it
stops just predicting ~0 everywhere, trades R2 for a systematic ranking inversion -- at a
1:1 zero:nonzero balance, R2 drops from 0.415 (4.6% zero) to 0.117 while ROC-AUC recovers
from 0.263 (worse than random) to 0.563. That's one regression being asked two different
questions ("will it convert" and "how much, given it converts") at once. Splitting them
lets each be modeled and evaluated on its own terms: the classifier is trained on the
natural (untouched) class distribution -- no resampling, no class_weight, see below for
why -- and the regressor's training signal is completely clean (only genuine conversions,
never contaminated by zeros).

Both stages share ONE train/test split, done once on the natural (untouched) distribution --
the classifier trains on the full train split, the regressor trains on just the cvr>0 rows
within it, and the held-out test split (both classes, natural mix) is what
evaluate_two_stage() scores the combined prediction against.

Deliberately NOT using class_weight="balanced" on the classifier: it inflates predict_proba
well past the true base rate (checked: mean predicted P(nonzero)=0.278 vs true rate=0.118
with class_weight="balanced", vs 0.113 without) for the exact same AUC/PR-AUC -- fine if you
only read off rankings, but this classifier's probability feeds a MULTIPLICATIVE combination
with the regressor, so a systematic 2x+ inflation there silently wrecks the combined R2 even
though each stage looks great in isolation. Calibration matters here in a way it wouldn't for
a classifier used alone.

Run:  python GAME_THEORY_PREDICTION/CVR/train_cvr_two_stage.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)

import shap
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                              roc_auc_score, average_precision_score)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DROP_COLUMNS = [
    "asin", "marketplace_id", "week", "cvr", "buybox_pct",
    "units", "sessions", "revenue",
    "implied_price", "oi_price", "listing_price", "raw_price",
    "rank_value", "main_browser_node_id",
    "scrape_title",   # raw scraped title text -- signal already lives in title_* stylistic scores
]
CAT_COLS = ["brand", "product_type", "has_aplus"]

# Same reasoning as BBOX/train_bbox.py's MONOTONIC_DIRECTION: unconstrained trees can (and
# did -- see the isolated CVR spike at one price point flanked by much lower neighbors)
# learn a locally non-monotonic price response on sparse real data. buybox_pred is the
# chained buy-box prediction (top feature for both stages, per market_env.py) -- winning
# the buy-box more often should never translate to LOWER conversion, so it's constrained
# positive rather than left alone like price is negative.
MONOTONIC_DIRECTION = {
    "price": -1, "price_vs_lowest": -1, "lowest_price": +1,
    "buybox_pred": +1,
}


def _binary_shap_importance(model, feats):
    """shap.TreeExplainer's return shape for binary classifiers varies by shap/sklearn
    version (a 2-item list, or a single (n, features) array, or a (n, features, 2) array).
    Normalize to a single (n, features) array of "importance toward class 1" before
    averaging |.| per feature."""
    shap_vals = shap.TreeExplainer(model).shap_values(feats)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1] if len(shap_vals) == 2 else shap_vals[0]
    shap_vals = np.asarray(shap_vals)
    if shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, 1] if shap_vals.shape[2] == 2 else shap_vals[:, :, 0]
    return pd.Series(np.abs(shap_vals).mean(axis=0), index=feats.columns).sort_values(ascending=False)


class create_cvr_two_stage_predictor:

    def __init__(self, cvs_folder_path, seg_level, seg_terms, bbox_predictor_path):
        self.folder_path = cvs_folder_path
        self.seg_level = seg_level

        self.aggregate_data = None
        for term in seg_terms:
            temp_data = pd.read_csv(f'{self.folder_path}/all_feature_data_{term}.csv')
            self.aggregate_data = (pd.concat([self.aggregate_data, temp_data], axis=0, ignore_index=True)
                                    if self.aggregate_data is not None else temp_data)

        HERE = os.path.dirname(os.path.abspath(__file__))
        self.CVR_path = os.path.join(HERE, "cvr_feature_panel.csv")
        self.bbox_predictor_path = bbox_predictor_path
        self._BBOX_BUNDLE = None
        self._PANEL_DF = None

        # stage 1: classifier P(cvr > 0)
        self.CLF_SHAP_MODEL = None
        self.CLF_SHAP_MODEL_PATH = os.path.join(HERE, "cvr_clf_shap_model.joblib")
        self._CLF_MODEL = None
        self._CLF_MODEL_PATH = os.path.join(HERE, "cvr_clf_model.joblib")
        self._CLF_FEATURES = None
        self._CLF_CAT_COLS = None
        self._CLF_CAT_LEVELS = None

        # stage 2: regressor E[cvr | cvr > 0]
        self.REG_SHAP_MODEL = None
        self.REG_SHAP_MODEL_PATH = os.path.join(HERE, "cvr_reg_shap_model.joblib")
        self._REG_MODEL = None
        self._REG_MODEL_PATH = os.path.join(HERE, "cvr_reg_model.joblib")
        self._REG_FEATURES = None
        self._REG_CAT_COLS = None
        self._REG_CAT_LEVELS = None

        # shared holdout (natural distribution) for combined evaluation
        self._TEST_IDX = None
        self._STATE_PATH = os.path.join(HERE, "cvr_two_stage_state.joblib")

    def reset_models(self):
        for path in (self.CLF_SHAP_MODEL_PATH, self._CLF_MODEL_PATH,
                     self.REG_SHAP_MODEL_PATH, self._REG_MODEL_PATH, self._STATE_PATH):
            if os.path.exists(path):
                os.remove(path)
                print(f"[cvr-2stage] removed {path}")
        self.CLF_SHAP_MODEL = self._CLF_MODEL = None
        self.REG_SHAP_MODEL = self._REG_MODEL = None
        self._CLF_FEATURES = self._CLF_CAT_COLS = self._CLF_CAT_LEVELS = None
        self._REG_FEATURES = self._REG_CAT_COLS = self._REG_CAT_LEVELS = None
        self._TEST_IDX = None


    def _load_bbox_bundle(self):
        if self._BBOX_BUNDLE is None:
            self._BBOX_BUNDLE = joblib.load(self.bbox_predictor_path)
        return self._BBOX_BUNDLE

    def _predict_buybox(self, df):
        bundle = self._load_bbox_bundle()
        if bundle.get("type") == "manual_rule":
            return self._predict_buybox_manual_rule(df, bundle["params"])
        model, features = bundle["model"], bundle["features"]
        cat_cols, cat_levels = bundle["cat_cols"], bundle["cat_levels"]
        X = pd.DataFrame(index=df.index)
        for col in features:
            values = df[col] if col in df.columns else pd.Series(np.nan, index=df.index)
            X[col] = pd.Categorical(values, categories=cat_levels[col]) if col in cat_cols \
                else pd.to_numeric(values, errors="coerce")
        return model.predict(X)

    @staticmethod
    def _predict_buybox_manual_rule(df, params):
        """Mirrors market_env.py's _buybox_prob() manual-rule branch exactly (same formula,
        same gap/abs_gap definitions), so the buybox_pred feature CVR trains on matches what
        the live simulation actually feeds it -- instead of the old fitted tree's prediction,
        which was fit on a different feature set (own_landed/est_margin/fba_fee_per_unit/...)
        that mostly doesn't even exist in this panel and came back as near-all-NaN.

        own_price = price (the CVR panel's own per-row listing price, ~93% populated).
        comp_ref  = bb_price (avg buy-box-landed price that week) -- ~0.2% populated, so gap
                    falls back to 0.0 wherever it's missing, the SAME convention market_env.py
                    already uses live (`gap = ... if comp_ref else 0.0`).
        ref_price = lowest_price (avg lowest-landed price that week), same fallback -- the
                    closest real per-row analogue to market_env.py's fixed reference_price
                    anchor, which doesn't exist in this flat (non-clustered) panel.
        own_fba/own_prime/own_feedback = 1.0/1.0/4.5, the exact simulation-time constants
        market_env.py hardcodes for every product (see MarketEnv.__init__) -- not re-derived
        here since the live simulation never varies them either.
        """
        own_price = pd.to_numeric(df.get("price"), errors="coerce")
        comp_ref = pd.to_numeric(df.get("bb_price"), errors="coerce")
        ref_price = pd.to_numeric(df.get("lowest_price"), errors="coerce")

        gap = ((own_price - comp_ref) / comp_ref).where(comp_ref.notna() & (comp_ref != 0), 0.0)
        abs_gap = ((own_price - ref_price) / ref_price).where(ref_price.notna() & (ref_price != 0), 0.0)

        own_fba, own_prime, own_feedback = 1.0, 1.0, 4.5
        z = (params["buybox_intercept"] + params["buybox_gap_coef"] * gap
             + params["buybox_abs_gap_coef"] * abs_gap
             + params["buybox_fba_coef"] * own_fba
             + params["buybox_prime_coef"] * own_prime
             + params["buybox_feedback_coef"] * own_feedback)
        z = z.clip(-30.0, 30.0)
        return (1.0 / (1.0 + np.exp(-z))).to_numpy()

    def _load_panel(self):
        if self._PANEL_DF is None:
            if not os.path.exists(self.CVR_path):
                raise FileNotFoundError(
                    f"CVR feature panel not found: {self.CVR_path}\n"
                    f"Run `python GAME_THEORY_PREDICTION/CVR/build_feature_panel_CVR.py` first.")
            panel = pd.read_csv(self.CVR_path, low_memory=False)
            panel["buybox_pred"] = self._predict_buybox(panel)
            self._PANEL_DF = panel
        return self._PANEL_DF

    def _prepare_X(self, panel, index, sparse_cutoff=0.02):
     
        rows = panel.loc[index]
        X = rows.drop(columns=[c for c in DROP_COLUMNS if c in rows.columns])
        sparse = [c for c in X.columns if X[c].notna().mean() < sparse_cutoff]
        X = X.drop(columns=sparse)
        cat_cols = [c for c in CAT_COLS if c in X.columns]
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True).astype(float)
        return X, sparse

    def _build_X_for(self, panel, index, features, cat_cols, cat_levels):
      
        X = pd.DataFrame(index=index)
        for col in features:
            values = panel.loc[index, col] if col in panel.columns else pd.Series(np.nan, index=index)
            X[col] = pd.Categorical(values, categories=cat_levels[col]) if col in cat_cols \
                else pd.to_numeric(values, errors="coerce")
        return X


    def _load_clf_shap_model(self, train_idx):
        if self.CLF_SHAP_MODEL is not None:
            return self.CLF_SHAP_MODEL
        if os.path.exists(self.CLF_SHAP_MODEL_PATH):
            try:
                self.CLF_SHAP_MODEL = joblib.load(self.CLF_SHAP_MODEL_PATH)
                return self.CLF_SHAP_MODEL
            except Exception as e:
                print(f"[cvr-2stage] cached classifier SHAP model unreadable ({e}); retraining")
        self._extract_feature_importance_clf(train_idx)
        return self.CLF_SHAP_MODEL

    def _extract_feature_importance_clf(self, train_idx):
        panel = self._load_panel()
        X, sparse = self._prepare_X(panel, train_idx)
        Y = (panel.loc[train_idx, "cvr"].clip(0.0, 1.0) > 0).astype(int)
        print(f"[shap-cvr-clf] {X.shape[1]} features; dropped {len(sparse)} near-empty cols: {sparse}")

        model = HistGradientBoostingClassifier(max_iter=200, random_state=0)
        model.fit(X, Y)
        self.CLF_SHAP_MODEL = {"model": model, "features": X}
        joblib.dump(self.CLF_SHAP_MODEL, self.CLF_SHAP_MODEL_PATH)
        print(f"[cvr-2stage] saved classifier SHAP-ranking model -> {self.CLF_SHAP_MODEL_PATH}")

    def fit_classifier(self, train_idx, top_k=15):
        shap_bundle = self._load_clf_shap_model(train_idx)
        model, feats = shap_bundle["model"], shap_bundle["features"]
        importance = _binary_shap_importance(model, feats.iloc[:2000])
        top_features = importance.index.tolist()[:top_k]

        raw_keep = []
        for f in top_features:
            parent = next((c for c in CAT_COLS if f == c or f.startswith(c + "_")), f)
            raw_keep.append(parent)
        raw_keep = list(dict.fromkeys(raw_keep))
        print(f"[cvr-2stage] classifier top {top_k} one-hot features -> {len(raw_keep)} raw cols: {raw_keep}")

        panel = self._load_panel()
        X = panel.loc[train_idx, [c for c in raw_keep if c in panel.columns]].copy()
        cat_in_X = [c for c in CAT_COLS if c in X.columns]
        for c in cat_in_X:
            X[c] = X[c].astype("category")
        cat_mask = [c in cat_in_X for c in X.columns]
        Y = (panel.loc[train_idx, "cvr"].clip(0.0, 1.0) > 0).astype(int)

        monotonic_cst = [MONOTONIC_DIRECTION.get(c, 0) for c in X.columns]
        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=10, l2_regularization=0.1,
            categorical_features=cat_mask, monotonic_cst=monotonic_cst, random_state=0,
        )
        clf.fit(X, Y)
        self._CLF_MODEL = clf
        self._CLF_FEATURES = list(X.columns)
        self._CLF_CAT_COLS = cat_in_X
        self._CLF_CAT_LEVELS = {c: list(X[c].cat.categories) for c in cat_in_X}

        joblib.dump({
            "model": self._CLF_MODEL, "features": self._CLF_FEATURES,
            "cat_cols": self._CLF_CAT_COLS, "cat_levels": self._CLF_CAT_LEVELS,
        }, self._CLF_MODEL_PATH)
        print(f"[cvr-2stage] classifier refit on {X.shape[1]} raw cols -> saved -> {self._CLF_MODEL_PATH}")
        return self._CLF_MODEL

    # ---------------- Stage 2: regressor (nonzero rows only) ----------------

    def _load_reg_shap_model(self, nz_train_idx):
        if self.REG_SHAP_MODEL is not None:
            return self.REG_SHAP_MODEL
        if os.path.exists(self.REG_SHAP_MODEL_PATH):
            try:
                self.REG_SHAP_MODEL = joblib.load(self.REG_SHAP_MODEL_PATH)
                return self.REG_SHAP_MODEL
            except Exception as e:
                print(f"[cvr-2stage] cached regressor SHAP model unreadable ({e}); retraining")
        self._extract_feature_importance_reg(nz_train_idx)
        return self.REG_SHAP_MODEL

    def _extract_feature_importance_reg(self, nz_train_idx):
        panel = self._load_panel()
        X, sparse = self._prepare_X(panel, nz_train_idx)
        Y = panel.loc[nz_train_idx, "cvr"].clip(0.0, 1.0)
        print(f"[shap-cvr-reg] {X.shape[1]} features; dropped {len(sparse)} near-empty cols: {sparse}")

        model = HistGradientBoostingRegressor(max_iter=200, random_state=0)
        model.fit(X, Y)
        self.REG_SHAP_MODEL = {"model": model, "features": X}
        joblib.dump(self.REG_SHAP_MODEL, self.REG_SHAP_MODEL_PATH)
        print(f"[cvr-2stage] saved regressor SHAP-ranking model -> {self.REG_SHAP_MODEL_PATH}")

    def fit_regressor(self, nz_train_idx, top_k=15):
        shap_bundle = self._load_reg_shap_model(nz_train_idx)
        model, feats = shap_bundle["model"], shap_bundle["features"]

        shap_vals = shap.TreeExplainer(model).shap_values(feats.iloc[:2000])
        importance = pd.Series(np.abs(shap_vals).mean(axis=0), index=feats.columns).sort_values(ascending=False)
        top_features = importance.index.tolist()[:top_k]

        raw_keep = []
        for f in top_features:
            parent = next((c for c in CAT_COLS if f == c or f.startswith(c + "_")), f)
            raw_keep.append(parent)
        raw_keep = list(dict.fromkeys(raw_keep))
        print(f"[cvr-2stage] regressor top {top_k} one-hot features -> {len(raw_keep)} raw cols: {raw_keep}")

        panel = self._load_panel()
        X = panel.loc[nz_train_idx, [c for c in raw_keep if c in panel.columns]].copy()
        cat_in_X = [c for c in CAT_COLS if c in X.columns]
        for c in cat_in_X:
            X[c] = X[c].astype("category")
        cat_mask = [c in cat_in_X for c in X.columns]
        Y = panel.loc[nz_train_idx, "cvr"].clip(0.0, 1.0)

        monotonic_cst = [MONOTONIC_DIRECTION.get(c, 0) for c in X.columns]
        reg = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.03, max_leaf_nodes=31,
            min_samples_leaf=5, l2_regularization=0.05,
            categorical_features=cat_mask, monotonic_cst=monotonic_cst, random_state=0,
        )
        reg.fit(X, Y)
        self._REG_MODEL = reg
        self._REG_FEATURES = list(X.columns)
        self._REG_CAT_COLS = cat_in_X
        self._REG_CAT_LEVELS = {c: list(X[c].cat.categories) for c in cat_in_X}

        joblib.dump({
            "model": self._REG_MODEL, "features": self._REG_FEATURES,
            "cat_cols": self._REG_CAT_COLS, "cat_levels": self._REG_CAT_LEVELS,
        }, self._REG_MODEL_PATH)
        print(f"[cvr-2stage] regressor refit on {X.shape[1]} raw cols -> saved -> {self._REG_MODEL_PATH}")
        return self._REG_MODEL

    # ---------------- combined fit / predict / evaluate ----------------

    def fit_two_stage(self, top_k=15, test_size=0.3, random_state=0):
        """Fit both stages on ONE shared train/test split of the natural distribution.
        Classifier trains on the whole train split; regressor trains on just its cvr>0
        rows. Test split (untouched, both classes) is reserved for evaluate_two_stage()."""
        panel = self._load_panel()
        y = panel["cvr"].clip(0.0, 1.0)

        train_idx, test_idx = train_test_split(panel.index, test_size=test_size, random_state=random_state)
        nz_train_idx = train_idx[y.loc[train_idx] > 0]
        print(f"[cvr-2stage] train={len(train_idx):,} ({(y.loc[train_idx] > 0).sum():,} non-zero) "
              f"-> regressor trains on {len(nz_train_idx):,} rows; "
              f"test={len(test_idx):,} (natural distribution, {(y.loc[test_idx] > 0).sum():,} non-zero)")

        self.fit_classifier(train_idx, top_k=top_k)
        self.fit_regressor(nz_train_idx, top_k=top_k)

        self._TEST_IDX = test_idx
        joblib.dump({"test_idx": test_idx}, self._STATE_PATH)
        return test_idx

    def _load_test_idx(self):
        if self._TEST_IDX is not None:
            return self._TEST_IDX
        if os.path.exists(self._STATE_PATH) and self._CLF_MODEL is not None and self._REG_MODEL is not None:
            self._TEST_IDX = joblib.load(self._STATE_PATH)["test_idx"]
            return self._TEST_IDX
        return self.fit_two_stage()

    def combined_predict(self, index=None):
        """P(cvr>0) * E[cvr | cvr>0] for `index` rows (defaults to the held-out test
        split). Returns (p_nonzero, magnitude, combined) arrays."""
        if self._CLF_MODEL is None or self._REG_MODEL is None:
            self.fit_two_stage()
        index = self._load_test_idx() if index is None else index
        panel = self._load_panel()

        X_clf = self._build_X_for(panel, index, self._CLF_FEATURES, self._CLF_CAT_COLS, self._CLF_CAT_LEVELS)
        X_reg = self._build_X_for(panel, index, self._REG_FEATURES, self._REG_CAT_COLS, self._REG_CAT_LEVELS)

        p_nonzero = self._CLF_MODEL.predict_proba(X_clf)[:, 1]
        magnitude = self._REG_MODEL.predict(X_reg)
        return p_nonzero, magnitude, p_nonzero * magnitude

    def evaluate_two_stage(self):
        """Print classifier / regressor / combined metrics on the shared, natural-
        distribution test split, and save the combined predicted-vs-actual plot."""
        test_idx = self._load_test_idx()
        panel = self._load_panel()
        y_test = panel.loc[test_idx, "cvr"].clip(0.0, 1.0)
        is_nonzero_test = (y_test > 0).astype(int).values

        p_nonzero, magnitude, combined = self.combined_predict(test_idx)

        clf_auc = roc_auc_score(is_nonzero_test, p_nonzero)
        clf_ap = average_precision_score(is_nonzero_test, p_nonzero)

        nz_mask = is_nonzero_test.astype(bool)
        reg_r2 = r2_score(y_test[nz_mask], magnitude[nz_mask]) if nz_mask.sum() else float("nan")
        reg_mae = mean_absolute_error(y_test[nz_mask], magnitude[nz_mask]) if nz_mask.sum() else float("nan")

        combined_r2 = r2_score(y_test, combined)
        combined_mse = mean_squared_error(y_test, combined)
        combined_mae = mean_absolute_error(y_test, combined)
        combined_auc = roc_auc_score(is_nonzero_test, combined)

        print(f"[cvr-2stage-eval] n_test={len(test_idx):,} ({int(nz_mask.sum()):,} non-zero)")
        print(f"[cvr-2stage-eval] stage1 classifier : ROC-AUC={clf_auc:.3f}  PR-AUC={clf_ap:.3f}")
        print(f"[cvr-2stage-eval] stage2 regressor  : R2={reg_r2:.3f}  MAE={reg_mae:.4f}  "
              f"(nonzero rows only, n={int(nz_mask.sum()):,})")
        print(f"[cvr-2stage-eval] combined P*E[.]   : R2={combined_r2:.3f}  MSE={combined_mse:.4f}  "
              f"MAE={combined_mae:.4f}  ROC-AUC={combined_auc:.3f}")

        self.plot_combined_predicted_vs_actual(y_test, combined)
        return {
            "clf_auc": clf_auc, "clf_ap": clf_ap,
            "reg_r2": reg_r2, "reg_mae": reg_mae,
            "combined_r2": combined_r2, "combined_mse": combined_mse,
            "combined_mae": combined_mae, "combined_auc": combined_auc,
        }

    def _eval_out_path(self, filename):
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cvr_eval")
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, filename)

    def plot_combined_predicted_vs_actual(self, y_test, combined, path=None):
        r2 = r2_score(y_test, combined)
        mae = mean_absolute_error(y_test, combined)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_test, combined, alpha=0.4, s=12)
        lim = max(float(np.max(y_test)), float(np.max(combined)), 1.0)
        ax.plot([0, lim], [0, lim], "--", color="gray", lw=1, label="perfect")
        ax.set_xlabel("Actual CVR")
        ax.set_ylabel("Predicted CVR = P(convert) x E[cvr | convert]")
        ax.set_title(f"Two-stage CVR prediction -- R2={r2:.3f}, MAE={mae:.4f}")
        ax.legend(loc="upper right")
        fig.tight_layout()

        path = path or self._eval_out_path("two_stage_predicted_vs_actual.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"[cvr-2stage-eval] combined_predicted_vs_actual -> {path}")
        return path


if __name__ == "__main__":
    HERE = os.path.dirname(os.path.abspath(__file__))
    BBOX_MODEL_PATH = os.path.join(HERE, "..", "BBOX", "bbox_model.joblib")
    # Same precedence as market_env.py's _buybox_prob(): the hand-authored manual rule
    # (BBOX/train_bbox_manual_rule.py) takes priority over the old fitted tree whenever it
    # exists, so CVR is always trained against whichever buy-box mechanism the live
    # simulation actually uses -- not silently left behind it.
    BBOX_MANUAL_RULE_PATH = os.path.join(HERE, "..", "BBOX", "bbox_manual_rule.joblib")
    if os.path.exists(BBOX_MANUAL_RULE_PATH):
        BBOX_PREDICTOR_PATH = BBOX_MANUAL_RULE_PATH
    elif os.path.exists(BBOX_MODEL_PATH):
        BBOX_PREDICTOR_PATH = BBOX_MODEL_PATH
    else:
        raise FileNotFoundError(
            f"no buy-box predictor found at {BBOX_MANUAL_RULE_PATH} or {BBOX_MODEL_PATH}\n"
            f"Run `python GAME_THEORY_PREDICTION/BBOX/train_bbox_manual_rule.py` (or "
            f"train_bbox.py) first -- CVR features include the buy-box model's own "
            f"prediction (buybox_pred).")
    print(f"[cvr-2stage] using buy-box predictor -> {BBOX_PREDICTOR_PATH}")

    predictor = create_cvr_two_stage_predictor(cvs_folder_path=HERE, seg_level=None, seg_terms=[],
                                                bbox_predictor_path=BBOX_PREDICTOR_PATH)
    predictor.reset_models()   # cached joblib models may be stale -- force a refit
    predictor.fit_two_stage()
    predictor.evaluate_two_stage()
