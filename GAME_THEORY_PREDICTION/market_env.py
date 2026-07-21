"""
MarketEnv — the simulated marketplace for ONE cluster.

Imported by game.py (not run directly):
    from GAME_THEORY_PREDICTION.market_env import MarketEnv

It's the "world" an RL agent acts in. Core loop: step(price) -> (state, reward, done).
The three brains live here: consumer (logit), buy-box (gate), competitor (pluggable).
"""




import os
import json
import shap
from pathlib import Path

import numpy as np
import pandas as pd

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split


PRICE_COL = ("clean", "price")
HORIZON = 52
ACTION_GRID = [0.90, 0.95, 1.00, 1.05, 1.10]
PARAMS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data_files", "all_feature_data_Headphones.params.json")

BBOX_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "BBOX", "bbox_feature_panel.csv")

CVR_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "CVR", "cvr_feature_panel.csv")

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
    _BUYBOX_SHAP_MODEL = None
    _CVR_SHAP_MODEL  = None 
    _BUYBOX_MODEL = None
    _CVR_MODEL = None



    #  most of these methods are used to provide an estimate of a product that was never here before so we can have a basis on where to start.
    @classmethod
    def for_cluster(cls, vn, cluster_id, strategy="static"):
        key = int(cluster_id)
        if key not in cls._CACHE:
            cls._CACHE[key] = cls.fit(vn, key, strategy=strategy)
        env = cls._CACHE[key]
        # The env is cached per cluster and may be shared across pricing_game
        # instances that each want a different competitor. Set the requested
        # strategy on the (possibly cached) env so the caller always gets theirs.
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
        self.competitor_learning = True   # gate learning/exploration off during greedy eval
        self.comp_price = None            # the RL competitor's single scalar price
        self.comp_cost = None
        self._comp_pending = None         # (state, action) the competitor just took, for observe()
        self.last_comp_profit = 0.0       # competitor's profit last step (for match/eval readouts)





    @classmethod
    def fit(cls, vn, cluster_id, strategy="static"):

        members = cls._extract_cluster(vn, cluster_id)

        center, scale = _price_scale()
        comp_norm = pd.to_numeric(members[PRICE_COL], errors="coerce").dropna().to_numpy()
        comp_prices = comp_norm * scale + center

        params = {}
        params["price_center"] = center
        params["price_scale"] = scale
        params["reference_price"] = float(np.median(comp_prices)) if comp_prices.size else 1.0
        # params.update(cls._estimate_buybox(members, comp_prices))
        # params.update(cls._estimate_seasonality())
        # cls._estimate_conversion_rate()



        # --- calibrate Negative-Binomial dispersion from real weekly demand --------
        # Extract this cluster's own units sold, aggregate to (asin, marketplace, week)
        # totals, and estimate r by method of moments: r = mean^2 / (var - mean).
        # Falls back to a near-Poisson r if the query fails / data is too thin / the
        # series isn't overdispersed.
        nb_dispersion = 1.0
        try:
            asins = members[("clean", "asin")].dropna().astype(str).unique().tolist()
            if asins:
                in_list = ",".join("'" + a.replace("'", "") + "'" for a in asins)
                wq = text(f"""
                    SELECT SUM(units_ordered) AS units
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
        except Exception as e:
            print(f"[fit] NB dispersion calibration failed ({e}); using r={nb_dispersion}")
        params["nb_dispersion"] = float(nb_dispersion)

        return cls(int(cluster_id), comp_prices, params, strategy=strategy)

    @staticmethod
    def _extract_cluster(vn, cluster_id):

        cluster = []
        for idx, label in enumerate(vn.product_labels) :
            if label == cluster_id :
                cluster.append(idx)

        return vn.feature_data.iloc[cluster]


    @staticmethod
    def _quality(row):
        r = row.get(("clean", "average_rating"))
        try:
            return float(r)
        except (TypeError, ValueError):
            return 0.0
   
    

    ''' This previously was a CVR and buybox prediction that would then be factored in the model the issue was that the system was innacurate 
    the demand didn't fluctuate based on the price and thus the model though that rising the price would be  the best way to get profit '''



    # temp method to try and understand what the fuck matters for the Conversion rate - realized that I need to predict buybox and hoping that it isnt 
    # a circular dependency 
    @classmethod
    def extract_feature_importance_CVR(cls):
        
        csv_path = CVR_path

        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"feature panel not found: {csv_path}\n"
                f"Run `python GAME_THEORY_PREDICTION/build_feature_panel.py` first.")
        shap_data = pd.read_csv(csv_path)

    
        drop_columns = [
            "asin", "marketplace_id", "week", "cvr",
            "units", "sessions", "revenue",
            "implied_price", "oi_price", "listing_price", "raw_price",
            "rank_value", "main_browser_node_id",
        ]

        Y = shap_data["cvr"].clip(0.0, 1.0)                     # 
        X = shap_data.drop(columns=[c for c in drop_columns if c in shap_data.columns])

     
        sparse = [c for c in X.columns if X[c].notna().mean() < 0.02]
        X = X.drop(columns=sparse)

        cat_cols = [c for c in ["brand", "product_type", "has_aplus"] if c in X.columns]
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True).astype(float)
        print(f"[shap] {X.shape[1]} features; dropped {len(sparse)} near-empty cols: {sparse}")

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

        model = HistGradientBoostingRegressor(max_iter=200, random_state=0)
        model.fit(X_train, Y_train)
        cls._CVR_SHAP_MODEL = {'model': model , 'features': X}            


    @classmethod
    def extract_feature_importance_BBox(cls):
               
        if not os.path.exists(BBOX_path):
            raise FileNotFoundError(
                f"buy-box feature panel not found: {BBOX_path}\n"
                f"Run `python GAME_THEORY_PREDICTION/BBOX/build_feature_panel_BBox.py` first.")
        shap_data = pd.read_csv(BBOX_path)

        # drop identifiers, the target, demand OUTCOMES, and the duplicate raw
        # price sources (keep the coalesced+ffilled `price`).
        drop_columns = [
            "asin", "marketplace_id", "week", "buybox_pct",
            "units", "sessions", "revenue",
            "implied_price", "oi_price", "listing_price", "raw_price",
            "rank_value", "main_browser_node_id",
        ]

        Y = (shap_data["buybox_pct"] / 100.0).clip(0.0, 1.0)   # buy-box share in [0, 1]
        X = shap_data.drop(columns=[c for c in drop_columns if c in shap_data.columns])

        # near-empty numeric cols (the ~2-week competition/stock snapshots over a
        # 2-yr panel) break HistGBR's binning -> drop anything under 2% coverage.
        sparse = [c for c in X.columns if X[c].notna().mean() < 0.02]
        X = X.drop(columns=sparse)

        # one-hot the low-card string/bool categoricals so SHAP never sees a str.
        cat_cols = [c for c in ["brand", "product_type", "manufacturer", "has_aplus"]
                    if c in X.columns]
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True).astype(float)
        print(f"[shap-bbox] {X.shape[1]} features; dropped {len(sparse)} near-empty cols: {sparse}")

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

        model = HistGradientBoostingRegressor(max_iter=200, random_state=0)
        model.fit(X_train, Y_train)
        cls._BUYBOX_SHAP_MODEL = {'model': model , 'features': X}            
    

    ''' Using a similar approach to make a prediction model for the buybox'''
    @classmethod
    def predict_buybox(cls, top_k=15):
 
        if cls._BUYBOX_SHAP_MODEL is None:
            cls.extract_feature_importance_BBox()

        if cls._BUYBOX_MODEL is not None:
            return cls._BUYBOX_MODEL

        model = cls._BUYBOX_SHAP_MODEL["model"]
        feats = cls._BUYBOX_SHAP_MODEL["features"]        

        shap_vals = shap.TreeExplainer(model).shap_values(feats.iloc[:2000])
        importance = (pd.Series(abs(shap_vals).mean(axis=0), index=feats.columns)
                        .sort_values(ascending=False))
        
        feature_names = importance.index.tolist()
        top_features = feature_names[:top_k]

        bbox_data = pd.read_csv(BBOX_path)

        drop_columns = [] 
        for column in bbox_data.columns:
            if column not in feature_names:
                drop_columns.append(column)

        Y = (bbox_data["buybox_pct"] / 100.0).clip(0.0, 1.0)
        X = bbox_data.drop(columns=[c for c in drop_columns if c in bbox_data.columns])


        raw_columns_to_keep = []
        for feature in top_features:
            matched = False
            for parent_col in ["brand", "product_type", "manufacturer", "has_aplus"]:
                if feature.startswith(parent_col):
                    raw_columns_to_keep.append(parent_col)
                    matched = True
                    break
            if not matched:
                raw_columns_to_keep.append(feature)
        
        raw_columns_to_keep = list(set(raw_columns_to_keep))

        X = X[[c for c in raw_columns_to_keep if c in X.columns]]

        for col in ["brand", "product_type", "manufacturer"]:
            if col in X.columns:
                X[col] = X[col].astype('category')

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=0)
        categorical_mask = [
            True if col in ["brand", "product_type", "manufacturer"] else False
            for col in X_train.columns
        ]


        # tuned by ai 
        best_tuned = HistGradientBoostingRegressor(
            max_iter=200,             # Build more trees
            learning_rate=0.08,       # Increase learning rate slightly (default is 0.1)
            max_leaf_nodes=31,        # Allow larger, more complex trees (default is 31)
            min_samples_leaf=10,      # LOWER THIS: Allows splits on smaller brands (e.g., 10 rows)
            l2_regularization=0.5,    # Lower regularization to let the model fit the categories better
            categorical_features=categorical_mask,
            random_state=0,
        ) 
        best_tuned.fit(X_train, Y_train)
        cls._BUYBOX_MODEL = best_tuned

        return X_test, Y_test  , cls._BUYBOX_MODEL


    @classmethod
    def fit_cvr(cls, top_k=15):
        """Rank CVR features by SHAP, then refit a tuned HistGBR on the top-k
        (parent) features using NATIVE categoricals. Returns (X_test, Y_test, model).

        NB: renamed from predict_cvr to avoid colliding with the instance-level
        predict_cvr(self, own_price, ...) inference method further down the class.
        """
        # 1. SHAP model (lazy-train once)
        if cls._CVR_SHAP_MODEL is None:
            cls.extract_feature_importance_CVR()
        model = cls._CVR_SHAP_MODEL["model"]
        feats = cls._CVR_SHAP_MODEL["features"]          # one-hot training matrix (DataFrame)

        # 2. SHAP -> mean|impact| per one-hot column -> ranked names -> top-k
        shap_vals = shap.TreeExplainer(model).shap_values(feats.iloc[:2000])
        importance = (pd.Series(abs(shap_vals).mean(axis=0), index=feats.columns)
                        .sort_values(ascending=False))
        top_features = importance.index.tolist()[:top_k]

        # 3. map one-hot names back to RAW panel columns. A dummy like
        #    "product_type_Speaker" -> keep the whole raw "product_type" column;
        #    a plain numeric column keeps its own name.
        cats = ["brand", "product_type", "manufacturer", "has_aplus"]
        raw_keep = []
        for f in top_features:
            parent = next((c for c in cats if f == c or f.startswith(c + "_")), f)
            raw_keep.append(parent)
        raw_keep = list(dict.fromkeys(raw_keep))         # dedupe, preserve order
        print(f"[cvr] top {top_k} one-hot features -> {len(raw_keep)} raw cols: {raw_keep}")

        # 4. build X/Y from the panel using ONLY those raw columns; keep categoricals
        #    NATIVE (no one-hot) so the tuned model can split on them directly.
        cvr_data = pd.read_csv(CVR_path)
        Y = cvr_data["cvr"].clip(0.0, 1.0)
        X = cvr_data[[c for c in raw_keep if c in cvr_data.columns]].copy()
        cat_in_X = [c for c in cats if c in X.columns]
        for c in cat_in_X:
            X[c] = X[c].astype("category")
        cat_mask = [c in cat_in_X for c in X.columns]

        # 5. refit the tuned model on the selected features
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=0)
        best_tuned = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.08,
            max_leaf_nodes=31,
            min_samples_leaf=10,
            l2_regularization=0.5,
            categorical_features=cat_mask,
            random_state=0,
        )
        best_tuned.fit(X_train, Y_train)
        cls._CVR_MODEL = best_tuned
        print(f"[cvr] refit on {X.shape[1]} raw cols -> holdout R^2 = {best_tuned.score(X_test, Y_test):.3f}")
        return X_test, Y_test, cls._CVR_MODEL


    @classmethod
    def predict_demand(cls):
        pass 



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
        cvr = self.predict_cvr(own_price, comp_ref)

        # Expected demand is the combination of visibility, intent, and traffic
        return self._sessions() * bb_prob * cvr


    # ---- RL interface --------------------------------------------------
    def reset(self, target_product, start_week=None, competitor_strategy=None, seed=None):

        row = target_product.iloc[0]
        self.target = target_product

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

    def _buybox_prob(self, own_price, comp_ref, week=None):
        # gap = (own_price - comp_ref) / comp_ref if comp_ref else 0.0
        # # Same clock as predict_cvr / _seasonal_multiplier: default to the week currently
        # # being simulated (start_week + t); the backtest passes the real historical week.
        # if week is None:
        #     week = ((getattr(self, "start_week", 1) - 1 + getattr(self, "t", 0)) % 52) + 1

        # # _BUYBOX_MODEL is now a REGRESSOR predicting the buy-box SHARE (0-1) from
        # # [gap, fba, prime, feed, week] -- feature order must match _estimate_buybox exactly.
        # # Clip since a regressor can overshoot [0,1].
        # if self._BUYBOX_MODEL is not None:
        #     X = np.array([[gap, self.own_fba, self.own_prime, self.own_feedback, float(week)]], dtype=float)
        #     return float(np.clip(self._BUYBOX_MODEL.predict(X)[0], 0.0, 1.0))

        # # No model fit (DB unavailable / too little data): logistic fallback on the
        # # stored coefficients still yields a [0,1] share.
        # z = (self.params["buybox_intercept"]
        #      + self.params["buybox_gap_coef"] * gap
        #      + self.params["buybox_fba_coef"] * self.own_fba
        #      + self.params["buybox_prime_coef"] * self.own_prime
        #      + self.params["buybox_feedback_coef"] * self.own_feedback)
        # z = float(np.clip(z, -30.0, 30.0))
        z = 1
        return 1.0 / (1.0 + np.exp(-z))

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
                a = int(self.competitor_agent.choose(s_c, explore=self.competitor_learning))
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

    # ---- helpers -------------------------------------------------------
    def predict_cvr(self, own_price, comp_ref=None, week=None):

        if self._CVR_MODEL is None:
            return 0.03

        if comp_ref is None:
            comp_ref = self._competitor_reference()
        if week is None:
            week = ((getattr(self, "start_week", 1) - 1 + getattr(self, "t", 0)) % 52) + 1

        feedback_count_default = self.params.get("buybox_feedback_median", 100.0)
        feedback_count_log = np.log(feedback_count_default + 1)
        seller_score = feedback_count_log * self.own_feedback

        # Normalise by this cluster's reference price so the features are in the
        # same "fraction of typical price" space the model was trained on (train
        # divides by the per-product_type median; here we divide by the cluster
        # reference, the predict-time analogue). Feature names/order must match
        # _estimate_conversion_rate's X_df exactly.
        ref = self.params.get("reference_price") or 1.0
        X_df = pd.DataFrame([{
            'price_ratio': float(own_price / ref),
            'min_price_ratio': float(comp_ref / ref),
            'listing_ratio': float(own_price / ref),
            'shipping_ratio': 0.0,                  # Assuming FBA free shipping
            'daily_average_offer_position': 1.0,    # Assuming Buy Box placement to convert
            'prime_availability': float(self.own_prime),
            'seller_score': float(seller_score),
            'week_of_year': float(week),            # <-- time feature (order must match training)
        }])

        return float(np.clip(self._CVR_MODEL.predict(X_df)[0], 0.0, 1.0))

    def plot_cvr_seasonality(self, prices=None, path=None):
        """Diagnostic: the model's predicted CVR across all 52 weeks at fixed price
        level(s) -- i.e. the learned seasonal *conversion* curve, isolated from the
        seasonal traffic in _sessions(). Defaults to three price points around the
        reference. With no fitted model the curve is flat at the 0.03 fallback (which
        the title flags). Saves a PNG under cvr_seasonality/ unless ``path`` is given."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ref = self.params.get("reference_price") or 1.0
        if prices is None:
            prices = [round(0.9 * ref, 2), round(ref, 2), round(1.1 * ref, 2)]

        weeks = np.arange(1, 53)
        fig, ax = plt.subplots(figsize=(11, 5))
        for p in prices:
            cvr = [self.predict_cvr(float(p), week=int(w)) for w in weeks]
            ax.plot(weeks, cvr, lw=2, marker="o", ms=3, label=f"price €{p:.2f}")
        fitted = "fitted" if self._CVR_MODEL is not None else "CONSTANT 0.03 fallback"
        ax.set_xlabel("ISO week")
        ax.set_ylabel("predicted CVR")
        ax.set_title(f"Predicted CVR across the year — cluster {self.cluster_id} (model: {fitted})")
        ax.legend(loc="best", fontsize=8, title="fixed price")
        fig.tight_layout()

        if path is None:
            out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cvr_seasonality")
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"cvr_seasonality_cluster{self.cluster_id}.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def _sessions(self):
        # Traffic available this week: market size scaled by seasonality. Shared by
        # our demand and the competitor's so the two stay on the same scale.
        return self.params.get("market_size",100) * self._seasonal_multiplier(self.t)


    def _profit(self, own_price, units):
        return (own_price - self.params["cost"]) * units

    def _competitor_reference(self):
        return float(self.comp_prices.min()) if self.comp_prices.size else self.params["reference_price"]

    # ---- 2-player: the RL competitor's state & payoff ------------------
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
        """Competitor's per-step profit. The buy-box splits the market: it wins the
        (1 - our_buybox) share of sessions. CVR reuses our model with the competitor's
        price (seller-side features are approximated as symmetric)."""
        bb_us = self._buybox_prob(self.own_price, comp_price)
        comp_units = self._sessions() * (1.0 - bb_us) * self.predict_cvr(comp_price, comp_ref=self.own_price)
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
