"""
Tests for create_bbox_predictor (train_bbox.py).

Two groups:
  - Unit tests run against a small synthetic panel in a temp dir. They monkeypatch the
    predictor's joblib/output paths so they NEVER touch the real bbox_model.joblib /
    bbox_shap_model.joblib / bbox_eval outputs.
  - Assessment tests (TestRealPanelAssessment) run the actual pipeline against the real
    bbox_feature_panel.csv and print/assert sanity bounds on the real model. Skipped if
    the panel hasn't been built yet. Model outputs still go to a scratch dir, so re-running
    this file never clobbers whatever you last trained via `python train_bbox.py`.

Run:  python GAME_THEORY_PREDICTION/BBOX/test_train_bbox.py
  or: python -m unittest GAME_THEORY_PREDICTION.BBOX.test_train_bbox -v
"""
import os
import sys
import shutil
import tempfile
import unittest
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train_bbox import create_bbox_predictor  # noqa: E402

REAL_PANEL_PATH = os.path.join(HERE, "bbox_feature_panel.csv")


def _make_synthetic_panel(n_per_asin=40, n_asins=6, start=date(2025, 1, 6)):
    """Small deterministic panel shaped like bbox_feature_panel.csv: a handful of fully
    populated features, one sparse (~0.4%) numeric feature, and one fully-empty numeric
    feature -- the exact pattern that crashes HistGradientBoostingRegressor's binning if
    the <2%-coverage drop in extract_feature_importance_BBox is ever weakened."""
    rows = []
    for a in range(n_asins):
        for w in range(n_per_asin):
            week = start + timedelta(weeks=w)
            rows.append({
                "asin": f"ASIN{a:03d}", "marketplace_id": "MKT1", "week": week.isoformat(),
                "buybox_pct": float((a * 13 + w * 7) % 101),
                "units": 10, "sessions": 100, "revenue": 500.0,
                "price": 10.0 + a + 0.1 * w,
                "brand": f"brand{a % 3}", "product_type": f"type{a % 2}",
                "manufacturer": f"maker{a % 2}", "has_aplus": bool(a % 2),
                "image_count": 5.0, "own_landed": 10.0 + a, "return_rate": 0.05,
                "sparse_feature": 1.0 if (a == 0 and w == 0) else np.nan,   # ~0.4% coverage
                "empty_feature": np.nan,                                    # 0% coverage
            })
    return pd.DataFrame(rows)


def _isolate(predictor, tmp_dir, panel_path=None):
    """Redirect a predictor's joblib output paths (and optionally its panel source) into
    tmp_dir, so tests never read/write the real cached BBOX artifacts."""
    if panel_path is not None:
        predictor.BBOX_path = panel_path
    predictor.BUYBOX_SHAP_MODEL_PATH = os.path.join(tmp_dir, "shap_model.joblib")
    predictor._BUYBOX_MODEL_PATH = os.path.join(tmp_dir, "model.joblib")
    return predictor


class SyntheticPanelTestCase(unittest.TestCase):
    """Common setup: a fresh temp dir + synthetic CSV + isolated predictor per test."""

    lookback_days = None

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.panel = _make_synthetic_panel()
        self.panel_path = os.path.join(self.tmp, "panel.csv")
        self.panel.to_csv(self.panel_path, index=False)
        p = create_bbox_predictor(cvs_folder_path=self.tmp, seg_level=None, seg_terms=[],
                                   lookback_days=self.lookback_days)
        self.predictor = _isolate(p, self.tmp, panel_path=self.panel_path)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestReadPanel(SyntheticPanelTestCase):

    def test_full_history_returns_every_row(self):
        panel = self.predictor._read_panel()
        self.assertEqual(len(panel), len(self.panel))

    def test_lookback_keeps_only_recent_weeks(self):
        self.predictor.lookback_days = 14
        panel = self.predictor._read_panel()
        week = pd.to_datetime(panel["week"])
        max_week = pd.to_datetime(self.panel["week"]).max()
        self.assertTrue((week >= max_week - pd.Timedelta(days=14)).all())
        self.assertLess(len(panel), len(self.panel))
        self.assertGreater(len(panel), 0)


class TestResetModels(SyntheticPanelTestCase):

    def test_reset_removes_cached_files_and_clears_state(self):
        for path in (self.predictor.BUYBOX_SHAP_MODEL_PATH, self.predictor._BUYBOX_MODEL_PATH):
            with open(path, "wb") as f:
                f.write(b"stale")
        self.predictor.BUYBOX_SHAP_MODEL = {"stale": True}
        self.predictor._BUYBOX_MODEL = object()

        self.predictor.reset_models()

        self.assertFalse(os.path.exists(self.predictor.BUYBOX_SHAP_MODEL_PATH))
        self.assertFalse(os.path.exists(self.predictor._BUYBOX_MODEL_PATH))
        self.assertIsNone(self.predictor.BUYBOX_SHAP_MODEL)
        self.assertIsNone(self.predictor._BUYBOX_MODEL)

    def test_reset_is_a_noop_when_nothing_cached(self):
        self.predictor.reset_models()  # must not raise


class TestFeatureImportanceSurvivesEmptyColumns(SyntheticPanelTestCase):
    """Regression test for the crash hit while iterating on the coverage cutoff:
    HistGradientBoostingRegressor's binning raises ValueError on any column with fewer
    than 2 distinct values (e.g. avg_delivery_days, which is 100% NaN in the real panel).
    The <2%-coverage drop in extract_feature_importance_BBox (train_bbox.py) is what
    keeps such columns out of the model -- this test fails loudly (with that same
    ValueError) if that filter is ever removed or weakened."""

    def test_extract_feature_importance_drops_empty_and_sparse_columns(self):
        self.predictor.extract_feature_importance_BBox()
        bundle = self.predictor.BUYBOX_SHAP_MODEL
        self.assertIsNotNone(bundle)
        feats = bundle["features"]
        self.assertNotIn("empty_feature", feats.columns)
        self.assertNotIn("sparse_feature", feats.columns)
        preds = bundle["model"].predict(feats.iloc[:5])
        self.assertEqual(len(preds), 5)


class TestFitBuybox(SyntheticPanelTestCase):

    def test_fit_buybox_produces_a_usable_model(self):
        X_test, Y_test, model = self.predictor.fit_buybox()
        self.assertGreater(len(X_test), 0)
        preds = model.predict(X_test)
        self.assertEqual(len(preds), len(Y_test))
        self.assertTrue(np.all(np.isfinite(preds)))
        self.assertTrue(os.path.exists(self.predictor._BUYBOX_MODEL_PATH))


class TestPlots(SyntheticPanelTestCase):

    def test_evaluate_writes_all_three_plots(self):
        self.predictor.fit_buybox()
        out = {
            "predicted_vs_actual": os.path.join(self.tmp, "pva.png"),
            "residuals": os.path.join(self.tmp, "resid.png"),
            "shap_importance": os.path.join(self.tmp, "shap.png"),
        }
        self.predictor.plot_predicted_vs_actual(path=out["predicted_vs_actual"])
        self.predictor.plot_residuals(path=out["residuals"])
        self.predictor.plot_shap_summary(path=out["shap_importance"])
        for path in out.values():
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)


@unittest.skipUnless(os.path.exists(REAL_PANEL_PATH),
                      "bbox_feature_panel.csv not found -- run build_feature_panel_BBox.py first")
class TestRealPanelAssessment(unittest.TestCase):
    """Runs against the real panel and reports actual holdout metrics. Joblib output is
    redirected to a scratch dir so this never overwrites your real cached model."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_full_history_model_is_sane(self):
        predictor = _isolate(
            create_bbox_predictor(cvs_folder_path=HERE, seg_level=None, seg_terms=[]),
            self.tmp)
        X_test, Y_test, model = predictor.fit_buybox()
        preds = model.predict(X_test)
        r2, mae = r2_score(Y_test, preds), mean_absolute_error(Y_test, preds)
        print(f"\n[assess] full-history holdout: R2={r2:.3f} MAE={mae:.3f} n_test={len(Y_test):,}")

        self.assertTrue(np.all(np.isfinite(preds)))
        # regression guard, not a target: current holdout is R2=0.376 MAE=0.260 (2026-07-30).
        # Bounds have real slack either side so normal data drift doesn't make this flaky --
        # they exist to catch a change that silently collapses accuracy, not to enforce today's number.
        self.assertGreater(r2, 0.25, "R2 dropped well below the last known-good 0.376")
        self.assertLess(mae, 0.35, "MAE rose well above the last known-good 0.260")

    def test_recent_window_surfaces_competitive_features(self):
        predictor = _isolate(
            create_bbox_predictor(cvs_folder_path=HERE, seg_level=None, seg_terms=[],
                                   lookback_days=21),
            self.tmp)
        predictor.extract_feature_importance_BBox()
        cols = set(predictor.BUYBOX_SHAP_MODEL["features"].columns)
        competitive = cols & {"offer_count", "bb_price", "lowest_price", "units_in_stock",
                              "frac_oos", "best_category_rank", "comp_threshold"}
        print(f"\n[assess] 21-day window surfaces: {sorted(competitive)}")
        self.assertTrue(competitive, "a 21-day lookback should surface at least one real "
                                     "offer/stock/rank column above the 2% coverage cutoff")


@unittest.skipUnless(os.path.exists(REAL_PANEL_PATH),
                      "bbox_feature_panel.csv not found -- run build_feature_panel_BBox.py first")
class TestAccuracy(unittest.TestCase):
    """How good are the predictions, really -- not just "does it run". Compares the
    model against two honest baselines and breaks down error by actual-value range,
    which is where the predicted-vs-actual scatter plot showed it struggling most."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.predictor = _isolate(
            create_bbox_predictor(cvs_folder_path=HERE, seg_level=None, seg_terms=[]),
            self.tmp)
        self.X_test, self.Y_test, self.model = self.predictor.fit_buybox()
        self.preds = self.model.predict(self.X_test)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_beats_constant_mean_baseline(self):
        """Weakest possible baseline: always predict the training-set mean."""
        bbox_data = self.predictor._read_panel()
        y_all = (bbox_data["buybox_pct"] / 100.0).clip(0.0, 1.0)
        train_mean = y_all.loc[y_all.index.difference(self.X_test.index)].mean()

        model_mae = mean_absolute_error(self.Y_test, self.preds)
        baseline_mae = mean_absolute_error(self.Y_test, np.full(len(self.Y_test), train_mean))
        print(f"\n[accuracy] model MAE={model_mae:.3f} vs constant-mean baseline MAE={baseline_mae:.3f}")
        self.assertLess(model_mae, baseline_mae, "model should beat predicting the global mean every time")

    def test_beats_per_asin_history_mean_baseline(self):
        """Strongest honest baseline given the SHAP diagnosis (the surviving features
        are mostly static per-ASIN attributes): does the model add anything beyond
        'use this ASIN's own historical average', computed leak-free from train rows only?"""
        bbox_data = self.predictor._read_panel()
        y_all = (bbox_data["buybox_pct"] / 100.0).clip(0.0, 1.0)
        test_idx = self.X_test.index
        train_idx = bbox_data.index.difference(test_idx)

        asin_train_mean = y_all.loc[train_idx].groupby(bbox_data.loc[train_idx, "asin"]).mean()
        global_train_mean = y_all.loc[train_idx].mean()
        baseline_pred = bbox_data.loc[test_idx, "asin"].map(asin_train_mean).fillna(global_train_mean)

        model_mae = mean_absolute_error(self.Y_test, self.preds)
        baseline_mae = mean_absolute_error(self.Y_test, baseline_pred)
        print(f"\n[accuracy] model MAE={model_mae:.3f} vs per-ASIN-history-mean baseline MAE={baseline_mae:.3f}")
        # not asserting strictly-better: a per-ASIN historical mean is a genuinely strong
        # baseline here. Allow some slack, but flag if the model is meaningfully worse.
        self.assertLess(model_mae, baseline_mae * 1.05,
                         "model is meaningfully worse than just remembering each ASIN's "
                         "own historical average buy-box share")

    def test_error_by_actual_value_bucket(self):
        """Mirrors what the predicted-vs-actual scatter plot showed visually: error and
        bias should be reported per actual-value range, since low/high buckets are where
        the model (lacking any competitive/time signal) tends to regress to the middle."""
        df = pd.DataFrame({"actual": np.asarray(self.Y_test), "pred": self.preds})
        df["bucket"] = pd.cut(df["actual"], bins=[-0.01, 0.2, 0.8, 1.01],
                               labels=["low (<=0.2)", "mid (0.2-0.8)", "high (>0.8)"])
        report = df.groupby("bucket", observed=True).apply(
            lambda g: pd.Series({
                "n": len(g),
                "mae": mean_absolute_error(g["actual"], g["pred"]),
                "mean_bias": (g["pred"] - g["actual"]).mean(),   # >0 = over-predicts, <0 = under-predicts
            }), include_groups=False)
        print(f"\n[accuracy] error by actual-value bucket:\n{report.round(3).to_string()}")

        self.assertEqual(int(report["n"].sum()), len(df))
        self.assertTrue((report["n"] > 0).all(), "holdout set should cover low/mid/high buckets")


if __name__ == "__main__":
    unittest.main(verbosity=2)
