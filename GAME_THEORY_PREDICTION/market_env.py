"""
MarketEnv — the simulated marketplace for ONE cluster.

Imported by game.py (not run directly):
    from GAME_THEORY_PREDICTION.market_env import MarketEnv

It's the "world" an RL agent acts in. Core loop: step(price) -> (state, reward, done).
The three brains live here: consumer (logit), buy-box (gate), competitor (pluggable).
"""




import os
import sys
import json
import shap
import joblib
from pathlib import Path

import numpy as np
import pandas as pd

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "CVR"))
from train_cvr_two_stage import create_cvr_two_stage_predictor  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "BBOX"))
from train_bbox import create_bbox_predictor  # noqa: E402


PRICE_COL = ("clean", "price")
HORIZON = 52
ACTION_GRID = [0.90, 0.925, 0.95, 0.975,1.00, 1.025, 1.05,1.075, 1.10]
PARAMS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data_files", "all_feature_data_Audio.params.json")

BBOX_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "BBOX", "bbox_feature_panel.csv")

CVR_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "CVR", "cvr_feature_panel.csv")

CVR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CVR")
BBOX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "BBOX")
BBOX_MODEL_PATH = os.path.join(BBOX_DIR, "bbox_model.joblib")
# two-stage CVR predictor (see train_cvr_two_stage.py): P(cvr>0) classifier x E[cvr|cvr>0] regressor
CVR_CLF_MODEL_PATH = os.path.join(CVR_DIR, "cvr_clf_model.joblib")
CVR_REG_MODEL_PATH = os.path.join(CVR_DIR, "cvr_reg_model.joblib")

def _price_scale(path=PARAMS_PATH):
    blob = json.loads(Path(path).read_text())
    spec = blob.get("params", blob)["price"]
    return float(spec["center"]), float(spec["scale"])


def _api_engine():
    load_dotenv()
    url = os.getenv("API_DATABASE_URL")
    if not url:
        for line in Path(".env").read_text().splitlines():
            if line.strip().startswith("API_DATABASE_URL"):
                url = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
    return create_engine(url, connect_args={"connect_timeout": 20})


class MarketEnv:

    _CACHE = {}

    # Two-stage CVR model state, shared across all instances/clusters -- see
    # _load_or_fit_cvr_two_stage_model(). predict_cvr() rebuilds its input row against each
    # stage's own trained feature/cat schema, never hardcoded column names.
    # _CVR_MODEL is kept as a "is a CVR model ready" flag for game.py/backtest.py's status
    # checks (MarketEnv._CVR_MODEL is not None) -- it mirrors _CVR_REG_MODEL, not a model
    # used for prediction directly; predict_cvr() always goes through clf x reg below.
    _CVR_MODEL = None
    _CVR_CLF_MODEL = None
    _CVR_CLF_FEATURES = None
    _CVR_CLF_CAT_COLS = None
    _CVR_CLF_CAT_LEVELS = None
    _CVR_REG_MODEL = None
    _CVR_REG_FEATURES = None
    _CVR_REG_CAT_COLS = None
    _CVR_REG_CAT_LEVELS = None

    # buy-box model state, shared across all instances/clusters -- see _load_or_fit_buybox_model().
    _BUYBOX_MODEL = None
    _BUYBOX_FEATURES = None
    _BUYBOX_CAT_COLS = None
    _BUYBOX_CAT_LEVELS = None

    # cached panel CSVs (read once, shared across all instances) backing the snapshot
    # methods below -- these give bind_target_features() a real per-ASIN starting row
    # instead of an all-NaN one.
    _CVR_PANEL_DF = None
    _BUYBOX_PANEL_DF = None

    #  most of these methods are used to provide an estimate of a product that was never here before so we can have a basis on where to start.
    @classmethod
    def for_cluster(cls, vn, cluster_id, strategy="static", target_asin=None):
        # Cache key includes target_asin: two different target ASINs sharing the same
        # cluster must each get their OWN comp_prices/reference_price with themselves
        # excluded, not a shared env fit for whichever target asked first.
        key = (int(cluster_id), str(target_asin) if target_asin is not None else None)
        if key not in cls._CACHE:
            cls._CACHE[key] = cls.fit(vn, cluster_id, strategy=strategy, target_asin=target_asin)
        env = cls._CACHE[key]
        # The env is cached per (cluster, target_asin) and may be shared across
        # pricing_game instances that each want a different competitor. Set the
        # requested strategy on the (possibly cached) env so the caller always gets theirs.
        env.competitor_strategy = strategy
        return env


    def __init__(self, cluster_id, comp_prices, params, strategy="static", horizon=HORIZON, action_grid=ACTION_GRID):
        
        self.cluster_id = int(cluster_id)
        self.comp_prices = np.asarray(comp_prices, dtype=float)
        # Baseline competitor prices as fitted. step() overwrites self.comp_prices
        # for reactive strategies, so reset() restores this to keep episodes independent.
        self._comp_prices_base = self.comp_prices.copy()
        self.params = params
        self.horizon = horizon
        self.action_grid = np.asarray(action_grid, dtype=float)
        self.action_dim = len(self.action_grid)
        self.state_dim = 4
        self.target = None
        self._cvr_feat_snapshot = None    # set by bind_target_features()/reset()
        self._buybox_feat_snapshot = None # set by bind_target_features()/reset()
        self.own_price = None
        self.own_quality = 0.0
        self.own_fba = 1.0
        self.own_prime = 1.0
        self.own_feedback = params.get("buybox_feedback_median", 4.5)
        self.t = 0
        self.competitor_strategy = strategy
       


        self.rng = np.random.default_rng()
        self.stochastic = False
      

        # Negative-Binomial dispersion for stochastic demand: var = mean + mean^2 / r.
        # Smaller r = burstier; r -> inf recovers Poisson. Calibrated per cluster from
        # real weekly demand in fit() (params["nb_dispersion"]); defaults to 1.0.
        self.nb_dispersion = params.get("nb_dispersion", 1.0)

        self.competitor_agent = None
        self.competitor_learning = True   # gate observe()/on_episode_end() -- learn or stay frozen
        self.competitor_explore = True    # gate sample-vs-greedy -- independent of learning
        self.comp_price = None            # the RL competitor's single scalar price
        self.comp_cost = None
        self._comp_pending = None         # (state, action) the competitor just took, for observe()
        self.last_comp_profit = 0.0       # competitor's profit last step (for match/eval readouts)





    @classmethod
    def fit(cls, vn, cluster_id, strategy="static", target_asin=None):

        members = cls._extract_cluster(vn, cluster_id, target_asin=target_asin)
        if target_asin is not None and members.empty:
            print(f"[fit] cluster {cluster_id}: excluding target ASIN {target_asin} left "
                  f"ZERO other members -- reference_price/comp_prices will fall back to defaults")

        center, scale = _price_scale()
        comp_norm = pd.to_numeric(members[PRICE_COL], errors="coerce").dropna().to_numpy()
        comp_prices = comp_norm * scale + center

        params = {}
        params["price_center"] = center
        params["price_scale"] = scale
        params["reference_price"] = float(np.median(comp_prices)) if comp_prices.size else 1.0
        # buy-box/CVR calibration now runs lazily via the real ML models instead (see
        # _load_or_fit_buybox_model()/_load_or_fit_cvr_two_stage_model()), not here.

        nb_dispersion = 1.0
        seasonal_index = np.ones(52)
        try:
            asins = members[("clean", "asin")].dropna().astype(str).unique().tolist()
            if asins:
                in_list = ",".join("'" + a.replace("'", "") + "'" for a in asins)
                wq = text(f"""
                    SELECT date_trunc('week', data_date)::date AS week, SUM(units_ordered) AS units
                    FROM sales_traffic_daily
                    WHERE asin IN ({in_list})
                    GROUP BY asin, marketplace_id, date_trunc('week', data_date)
                """)
                with _api_engine().connect() as c:
                    wk = pd.read_sql(wq, c)
                units = pd.to_numeric(wk["units"], errors="coerce").fillna(0.0).to_numpy()
                if units.size >= 8:
                    hist_mean = float(units.mean())
                    hist_var = float(units.var())
                    if hist_var > hist_mean and hist_mean > 0:
                        nb_dispersion = (hist_mean ** 2) / (hist_var - hist_mean)
                    else:
                        nb_dispersion = 10.0   # variance ~ mean -> near-Poisson
                    print(f"[fit] cluster {cluster_id}: NB dispersion r={nb_dispersion:.2f} "
                          f"from {units.size} product-weeks "
                          f"(weekly units mean={hist_mean:.2f}, var={hist_var:.2f})")

                # seasonal index: average relative demand by week-of-year (1-52), normalized
                # to mean=1.0 so _seasonal_multiplier() is a pure multiplier on market_size.
                week_of_year = pd.to_datetime(wk["week"]).dt.isocalendar().week.clip(upper=52)
                by_week = (pd.to_numeric(wk["units"], errors="coerce").fillna(0.0)
                             .groupby(week_of_year).mean())
                if len(by_week) >= 26 and by_week.mean() > 0:   # need at least half the year covered
                    idx = by_week.reindex(range(1, 53)).fillna(by_week.mean())
                    seasonal_index = (idx / idx.mean()).to_numpy()
                    print(f"[fit] cluster {cluster_id}: seasonal index from {len(by_week)}/52 weeks "
                          f"(min={seasonal_index.min():.2f}, max={seasonal_index.max():.2f})")
        except Exception as e:
            print(f"[fit] NB dispersion/seasonality calibration failed ({e}); "
                  f"using r={nb_dispersion}, flat seasonal index")
        params["nb_dispersion"] = float(nb_dispersion)
        params["seasonal_index"] = seasonal_index

        return cls(int(cluster_id), comp_prices, params, strategy=strategy)

    @staticmethod
    def _extract_cluster(vn, cluster_id, target_asin=None):

        cluster = []
        for idx, label in enumerate(vn.product_labels) :
            if label == cluster_id :
                cluster.append(idx)

        members = vn.feature_data.iloc[cluster]
        # The target itself is frequently a member of its own cluster (it's a real row
        # in the scraped panel, not a synthetic "new" product) -- drop it so comp_prices/
        # reference_price reflect actual rivals, not the target benchmarked against itself.
        if target_asin is not None and ("clean", "asin") in members.columns:
            members = members[members[("clean", "asin")].astype(str) != str(target_asin)]
        return members


    @staticmethod
    def _quality(row):
        r = row.get(("clean", "average_rating"))
        try:
            return float(r)
        except (TypeError, ValueError):
            return 0.0
   
    

  
   
    @classmethod
    def _cvr_panel(cls):
        if cls._CVR_PANEL_DF is None:
            cls._CVR_PANEL_DF = pd.read_csv(CVR_path, low_memory=False)
        return cls._CVR_PANEL_DF

    @classmethod
    def _buybox_panel(cls):
        if cls._BUYBOX_PANEL_DF is None:
            cls._BUYBOX_PANEL_DF = pd.read_csv(BBOX_path, low_memory=False)
        return cls._BUYBOX_PANEL_DF

    @staticmethod
    def _snapshot_row(panel, asin, features, cat_cols):
        """Per-ASIN starting row for predict_cvr()/_buybox_prob(): median of each numeric
        feature's history, mode of each categorical -- NaN wherever the ASIN/column has no
        history. build_X()/the dyn-overrides dict then overwrite the live-simulation fields
        (price, week, buybox_pred, ...) on top of this."""
        if not features:
            return None
        rows = panel[panel["asin"].astype(str) == str(asin)] if "asin" in panel.columns else panel.iloc[0:0]
        cat_cols = set(cat_cols or [])
        snap = {}
        for col in features:
            if rows.empty or col not in panel.columns:
                snap[col] = np.nan
            elif col in cat_cols:
                m = rows[col].mode(dropna=True)
                snap[col] = m.iloc[0] if not m.empty else np.nan
            else:
                snap[col] = float(pd.to_numeric(rows[col], errors="coerce").median())
        return snap

    def _snapshot_cvr_features(self, asin):
        if self._CVR_CLF_MODEL is None or self._CVR_REG_MODEL is None:
            self._load_or_fit_cvr_two_stage_model()
        # union of both stages' feature lists -- predict_cvr() builds one shared row dict
        # and slices out each stage's own columns from it, so it needs every column either
        # stage might ask for.
        features = list(dict.fromkeys((self._CVR_CLF_FEATURES or []) + (self._CVR_REG_FEATURES or [])))
        cat_cols = set(self._CVR_CLF_CAT_COLS or []) | set(self._CVR_REG_CAT_COLS or [])
        return self._snapshot_row(self._cvr_panel(), asin, features, cat_cols)

    def _snapshot_buybox_features(self, asin):
        if self._BUYBOX_MODEL is None:
            self._load_or_fit_buybox_model()
        return self._snapshot_row(self._buybox_panel(), asin, self._BUYBOX_FEATURES, self._BUYBOX_CAT_COLS)

    def bind_target_features(self, asin):

        self._cvr_feat_snapshot = self._snapshot_cvr_features(asin)
        self._buybox_feat_snapshot = self._snapshot_buybox_features(asin)
        return self._cvr_feat_snapshot

    @classmethod
    def _load_or_fit_cvr_two_stage_model(cls):
        """Two-stage CVR predictor (train_cvr_two_stage.py): P(cvr>0) classifier x
        E[cvr | cvr>0] regressor. Checks both joblib files first, loads if present;
        fits both (create_cvr_two_stage_predictor.fit_two_stage()) if either is missing."""
        if cls._CVR_CLF_MODEL is not None and cls._CVR_REG_MODEL is not None:
            return cls._CVR_CLF_MODEL, cls._CVR_REG_MODEL

        if os.path.exists(CVR_CLF_MODEL_PATH) and os.path.exists(CVR_REG_MODEL_PATH):
            print(f"[market_env] loaded cached two-stage CVR models -> "
                  f"{CVR_CLF_MODEL_PATH}, {CVR_REG_MODEL_PATH}")
        else:
            print("[market_env] no cached two-stage CVR models -- fitting now")
            predictor = create_cvr_two_stage_predictor(cvs_folder_path=CVR_DIR, seg_level=None,
                                                         seg_terms=[], bbox_predictor_path=BBOX_MODEL_PATH)
            predictor.fit_two_stage()

        clf_bundle = joblib.load(CVR_CLF_MODEL_PATH)
        reg_bundle = joblib.load(CVR_REG_MODEL_PATH)
        cls._CVR_CLF_MODEL = clf_bundle["model"]
        cls._CVR_CLF_FEATURES = clf_bundle["features"]
        cls._CVR_CLF_CAT_COLS = clf_bundle["cat_cols"]
        cls._CVR_CLF_CAT_LEVELS = clf_bundle["cat_levels"]
        cls._CVR_REG_MODEL = reg_bundle["model"]
        cls._CVR_REG_FEATURES = reg_bundle["features"]
        cls._CVR_REG_CAT_COLS = reg_bundle["cat_cols"]
        cls._CVR_REG_CAT_LEVELS = reg_bundle["cat_levels"]
        cls._CVR_MODEL = cls._CVR_REG_MODEL   # "is a CVR model ready" flag for game.py/backtest.py
        return cls._CVR_CLF_MODEL, cls._CVR_REG_MODEL

    def predict_cvr(self, own_price, buybox_pred, comp_ref=None, week=None):

        if self._CVR_CLF_MODEL is None or self._CVR_REG_MODEL is None:
            self._load_or_fit_cvr_two_stage_model()

        if comp_ref is None:
            comp_ref = self._competitor_reference()
        if week is None:
            week = ((getattr(self, "start_week", 1) - 1 + getattr(self, "t", 0)) % 52) + 1

        month = int(min(12, max(1, int((int(week) - 1) // 4.345) + 1)))
        dyn = {
            "price": float(own_price),
            "listing_price": float(own_price),
            "bb_price": float(comp_ref) if comp_ref else np.nan,
            "week_of_year": float(week),
            "month": float(month),
            "buybox_pct": float(buybox_pred) * 100.0,
            "buybox_pred": float(buybox_pred),   # the chained BBOX prediction -- top feature for both stages
        }

        snap = getattr(self, "_cvr_feat_snapshot", None)

        def build_X(features, cat_cols, cat_levels):
            row = dict(snap) if snap else {f: np.nan for f in features}
            for k, v in dyn.items():
                if k in row:
                    row[k] = v
            X_df = pd.DataFrame([row])[features]        # enforce trained column order
            levels = cat_levels or {}
            for c in (cat_cols or []):                   # rebuild with the fitted levels
                X_df[c] = pd.Categorical(X_df[c], categories=levels.get(c))
            return X_df

        X_clf = build_X(self._CVR_CLF_FEATURES, self._CVR_CLF_CAT_COLS, self._CVR_CLF_CAT_LEVELS)
        X_reg = build_X(self._CVR_REG_FEATURES, self._CVR_REG_CAT_COLS, self._CVR_REG_CAT_LEVELS)

        p_nonzero = float(self._CVR_CLF_MODEL.predict_proba(X_clf)[0, 1])
        magnitude = float(self._CVR_REG_MODEL.predict(X_reg)[0])
        cvr = p_nonzero * magnitude

        # Tree models don't extrapolate: priced far above comp_ref (outside anything seen
        # in training) they plateau at a boundary-leaf value instead of decaying toward 0.
        # Mirror _buybox_prob()'s gap correction so CVR also collapses on an over-price
        # gap -- otherwise reward keeps rising with price indefinitely. Only kick in past
        # a free zone, though: real (price vs bb_price) gaps in cvr_feature_panel.csv are
        # tight (median 0%, 90th pct ~6%, 95th pct ~31%) -- applying this from gap=0 was
        # double-discounting completely ordinary competitive pricing on top of the buy-box
        # model's own already-steep price sensitivity, not just guarding true extrapolation.
        gap = (own_price - comp_ref) / comp_ref if comp_ref else 0.0
        free_zone = float(self.params.get("cvr_gap_free_zone", 0.4))
        decay_coef = float(self.params.get("cvr_gap_decay", 4.0))
        cvr *= float(np.exp(-decay_coef * max(0.0, gap - free_zone)))

        return float(np.clip(cvr, 0.0, 1.0))
    
    @classmethod
    def _load_or_fit_buybox_model(cls):

        if cls._BUYBOX_MODEL is not None:
            return cls._BUYBOX_MODEL

        if os.path.exists(BBOX_MODEL_PATH):
            print(f"[market_env] loaded cached buy-box model -> {BBOX_MODEL_PATH}")
        else:
            print(f"[market_env] no cached buy-box model at {BBOX_MODEL_PATH} -- fitting one now")
            predictor = create_bbox_predictor(cvs_folder_path=BBOX_DIR, seg_level=None, seg_terms=[])
            predictor.fit_buybox()

        bundle = joblib.load(BBOX_MODEL_PATH)
        cls._BUYBOX_MODEL = bundle["model"]
        cls._BUYBOX_FEATURES = bundle["features"]
        cls._BUYBOX_CAT_COLS = bundle["cat_cols"]
        cls._BUYBOX_CAT_LEVELS = bundle["cat_levels"]
        return cls._BUYBOX_MODEL

    def _buybox_prob(self, own_price, comp_ref=None, week=None):

        if comp_ref is None:
            comp_ref = self._competitor_reference()


        if self._BUYBOX_MODEL is None:
            self._load_or_fit_buybox_model()


        gap = (own_price - comp_ref) / comp_ref if comp_ref else 0.0
        if week is None:
            week = ((getattr(self, "start_week", 1) - 1 + getattr(self, "t", 0)) % 52) + 1

        if self._BUYBOX_MODEL is not None and self._BUYBOX_FEATURES is not None:
            snap = getattr(self, "_buybox_feat_snapshot", None)
            row = dict(snap) if snap else {f: np.nan for f in self._BUYBOX_FEATURES}
            ship = float(row.get("own_shipping") or 0.0)
            dyn = {
                "price": float(own_price),
                "own_shipping": ship,
                "own_landed": float(own_price) + ship,      # item + own shipping
            }
            for k, v in dyn.items():
                if k in row:
                    row[k] = v
            X_df = pd.DataFrame([row])[self._BUYBOX_FEATURES]       # trained column order
            levels = self._BUYBOX_CAT_LEVELS or {}
            for c in (self._BUYBOX_CAT_COLS or []):                # align to fitted levels
                X_df[c] = pd.Categorical(X_df[c], categories=levels.get(c))
            base = float(np.clip(self._BUYBOX_MODEL.predict(X_df)[0], 1e-4, 1.0 - 1e-4))

        
            coef = float(self.params.get("buybox_gap_coef", -12.0))
            logit = np.log(base / (1.0 - base)) + coef * gap
            logit = float(np.clip(logit, -30.0, 30.0))
            return float(1.0 / (1.0 + np.exp(-logit)))


        z = (self.params.get("buybox_intercept", 1.0)
             + self.params.get("buybox_gap_coef", -12.0) * gap
             + self.params.get("buybox_fba_coef", 0.5) * self.own_fba
             + self.params.get("buybox_prime_coef", 0.5) * self.own_prime
             + self.params.get("buybox_feedback_coef", 0.1) * self.own_feedback)
        z = float(np.clip(z, -30.0, 30.0))
        return 1.0 / (1.0 + np.exp(-z))


    def _draw_demand(self, mean):

        mean = max(float(mean), 0.0)
        if mean <= 0.0:
            return 0.0
        r = float(self.nb_dispersion)
        p = r / (r + mean)
        return float(self.rng.negative_binomial(r, p))

    def expected_demand(self, own_price):
        comp_ref = self._competitor_reference()
        bb_prob = self._buybox_prob(own_price, comp_ref)
        cvr = self.predict_cvr(own_price, bb_prob, comp_ref)
        # cached for diagnostics (game.py's per-run trace plot) -- mirrors last_units below,
        # since this is otherwise the only place either prediction is computed per step.
        self.last_buybox_prob = bb_prob
        self.last_cvr = cvr
        self.last_comp_ref = comp_ref


        return self._sessions() * cvr


    # ---- RL interface --------------------------------------------------
    def reset(self, target_product, start_week=None, competitor_strategy=None, seed=None):

        row = target_product.iloc[0]
        self.target = target_product

        # Snapshot this product's static CVR features into the trained schema so
        # predict_cvr has real brand/product_type/quality values to reason over.
        self.bind_target_features(row.get(("clean", "asin")))

        raw = float(pd.to_numeric(pd.Series([row[PRICE_COL]]), errors="coerce").iloc[0])
        self.own_price = raw * self.params["price_scale"] + self.params["price_center"]
        self.own_quality = self._quality(row)
        self.t = 0

        if seed is not None:
            self.rng = np.random.default_rng(seed)
      
        self.comp_prices = self._comp_prices_base.copy()

        if competitor_strategy is not None:
            self.competitor_strategy = competitor_strategy
        if start_week is None:
            start_week = pd.Timestamp.today().isocalendar().week
        self.start_week = int(start_week)
        if self.params.get("cost") is None:
            self.params["cost"] = 0.6 * self.own_price

        self.comp_price = float(self.params["reference_price"])
        self.comp_cost = 0.6 * self.comp_price
        self._comp_pending = None
        return self._state()

    def step(self, action):
        self.own_price = self.own_price * self.action_grid[int(action)]
       
        self.comp_prices = self._competitor_prices(self.competitor_strategy)
        units = self.expected_demand(self.own_price)
        if self.stochastic:
            units = self._draw_demand(units)   # Negative Binomial (mean-preserving, overdispersed)
        reward = self._profit(self.own_price, units)
        self.last_units = float(units)
     
        self.last_comp_profit = (self._competitor_profit(self._competitor_reference())
                                 if self.competitor_strategy == "RL" else 0.0)
        self.iso_week = ((self.start_week - 1 + self.t) % 52) + 1
        self.t += 1
        done = self.t >= self.horizon
        next_state = self._state()

       
        if self.competitor_learning and self._comp_pending is not None:
            s_c, a_c = self._comp_pending
            self.competitor_agent.observe(s_c, a_c, self.last_comp_profit,
                                          self._competitor_state(), done)
            if done:
                self.competitor_agent.on_episode_end()
            self._comp_pending = None
        return next_state, reward, done


    def _state(self):
        ref = self.params["reference_price"]
        comp_ref = self._competitor_reference()
        season = min(self._seasonal_multiplier(self.t) / 3.0, 1.0)
        return np.array([
            self.own_price / ref if ref else 0.0,
            comp_ref / ref if ref else 0.0,
            self._buybox_prob(self.own_price, comp_ref),
            season,
        ], dtype=float)



    def _seasonal_multiplier(self, t):
        week = ((self.start_week - 1 + t) % 52) + 1
        return self.params["seasonal_index"][week - 1]

    def plot_cvr_seasonality(self, path=None):
        """Plot predict_cvr()'s week_of_year/month response at a fixed reference
        price -- price (and therefore buy-box) held constant, so what moves is
        purely the model's learned seasonal component."""
        import matplotlib
        if path is not None:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ref_price = float(self.params.get("reference_price", 1.0))
        comp_ref = self._competitor_reference()
        weeks = np.arange(1, 53)
        bb = [self._buybox_prob(ref_price, comp_ref, week=int(w)) for w in weeks]
        cvr = [self.predict_cvr(ref_price, b, comp_ref=comp_ref, week=int(w))
               for b, w in zip(bb, weeks)]

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(weeks, cvr, marker="o", ms=3, lw=1.6, color="#E45756")
        ax.set_xlabel("ISO week of year")
        ax.set_ylabel("predicted CVR")
        ax.set_title(f"Learned CVR seasonality at reference price={ref_price:.2f} "
                     f"(cluster {self.cluster_id})")
        fig.tight_layout()

        if path is None:
            out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CVR")
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"cvr_seasonality_cluster{self.cluster_id}.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print(f"[market_env] CVR seasonality plot -> {path}")
        return path



    def _competitor_prices(self, strategy_type):


        base_prices = self.comp_prices
        reference = self.params["reference_price"]
        floor = self.params.get("comp_floor_price", reference * 0.7)

        match strategy_type:
            case "static":
                return base_prices

            case "undercutter":
                floor = self.params.get("comp_floor_price", self.params["reference_price"] * 0.7)
                return np.maximum(self.own_price - 0.01, floor)

            case "matcher":
                floor = self.params.get("comp_floor_price", self.params["reference_price"] * 0.7)
                return np.maximum(self.own_price, floor)

            case "RL":
                # The competitor is an RL Agent maximising its OWN profit -> a 2-player
                # game. It observes a mirrored state, picks a multiplier off the same
                # action grid, and (in step()) learns from its own reward.
                if self.competitor_agent is None:
                    print("NO COMPETITOR CHOSEN")
                    return base_prices           # no agent attached yet -> behave like static
                s_c = self._competitor_state()
                a = int(self.competitor_agent.choose(s_c, explore=self.competitor_explore))
                self._comp_pending = (s_c, a)    # remember for observe() in step()
                self.comp_price = float(max(self.comp_price * self.action_grid[a], floor))
                return np.array([self.comp_price])

            case "promo_cycler":
                # Use the env RNG (seeded per rollout) instead of a t-seeded local RNG,
                # so promo draws actually vary across Monte Carlo rollouts.
                rng = self.rng

                is_promo_week = (self.t % 10) in [0, 1]

                if is_promo_week:
                    # Drop prices stochastically by 15% to 25% during the promotion
                    promo_discount = rng.uniform(0.75, 0.85)
                    return np.maximum(base_prices * promo_discount, floor)
                else:
                    # Standard pricing with slight daily market noise (+/- 2%)
                    noise = rng.uniform(0.98, 1.02, size=base_prices.shape)
                    return np.maximum(base_prices * noise, floor)

            case _:
                print(f"Strategy {strategy_type} not recognized. Falling back to static.")
                return base_prices


    def _sessions(self):
        # Traffic available this week: market size scaled by seasonality. Shared by
        # our demand and the competitor's so the two stay on the same scale.
        return self.params.get("market_size",100) * self._seasonal_multiplier(self.t)


    def _profit(self, own_price, units):
        return (own_price - self.params["cost"]) * units

    def _competitor_reference(self):
        return float(self.comp_prices.min()) if self.comp_prices.size else self.params["reference_price"]

    def _competitor_state(self):
       

        ref = self.params["reference_price"]
        comp_p = self._competitor_reference()
        season = min(self._seasonal_multiplier(self.t) / 3.0, 1.0)
        return np.array([
            comp_p / ref if ref else 0.0,
            self.own_price / ref if ref else 0.0,
            1.0 - self._buybox_prob(self.own_price, comp_p),   # competitor wins what we don't
            season,
        ], dtype=float)

    def _competitor_profit(self, comp_price):
        """Competitor's per-step profit, mirroring expected_demand(): (1 - our_buybox)
        is the competitor's own win probability, fed into the CVR model as a feature
        the same way bb_prob is on the main side -- NOT also applied as an extra
        multiplier. Doing both double-counted the buy-box effect and made undercutting
        to the price floor look far more rewarding than it should."""
        bb_us = self._buybox_prob(self.own_price, comp_price)
        comp_cvr = self.predict_cvr(comp_price, 1.0 - bb_us, comp_ref=self.own_price)
        comp_units = self._sessions() * comp_cvr
        return (comp_price - self.comp_cost) * comp_units

    # ---- persistence ---------------------------------------------------
    def save(self, path):
        blob = {
            "cluster_id": self.cluster_id,
            "params": self.params,
            "comp_prices": self._comp_prices_base.tolist(),
            "horizon": self.horizon,
            "action_grid": self.action_grid.tolist(),
            "strategy": self.competitor_strategy,
        }
        Path(path).write_text(json.dumps(blob, indent=2))
        return path

    @classmethod
    def load(cls, path):
        blob = json.loads(Path(path).read_text())
        return cls(blob["cluster_id"], np.asarray(blob["comp_prices"], dtype=float),
                   blob["params"], strategy=blob.get("strategy", "static"),
                   horizon=blob.get("horizon", HORIZON),
                   action_grid=blob.get("action_grid", ACTION_GRID))
