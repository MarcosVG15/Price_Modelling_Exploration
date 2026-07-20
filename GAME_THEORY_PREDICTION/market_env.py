"""
MarketEnv — the simulated marketplace for ONE cluster.

Imported by game.py (not run directly):
    from GAME_THEORY_PREDICTION.market_env import MarketEnv

It's the "world" an RL agent acts in. Core loop: step(price) -> (state, reward, done).
The three brains live here: consumer (logit), buy-box (gate), competitor (pluggable).
"""




import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingRegressor


PRICE_COL = ("clean", "price")
HORIZON = 52
ACTION_GRID = [0.90, 0.95, 1.00, 1.05, 1.10]
PARAMS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data_files", "all_feature_data_Headphones.params.json")


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
    _BUYBOX = None
    _BUYBOX_MODEL = None
    _SEASONALITY = None
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
        # Monte Carlo support: when stochastic is True, step() draws realised demand
        # from a Poisson around the expected value instead of returning the mean. A
        # single RNG drives both demand noise and stochastic competitor moves, so a
        # rollout is fully reproducible from its seed (set via reset(seed=...)).
        self.rng = np.random.default_rng()
        self.stochastic = False
        # --- 2-player self-play ---------------------------------------------
        # An optional RL competitor (any Agent) that sets its OWN price via the
        # "RL" strategy and learns inside step(). Attached by pricing_game.
        self.competitor_agent = None
        self.competitor_learning = True   # gate learning/exploration off during greedy eval
        self.comp_price = None            # the RL competitor's single scalar price
        self.comp_cost = None
        self._comp_pending = None         # (state, action) the competitor just took, for observe()

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
        params.update(cls._estimate_buybox(members, comp_prices))
        params.update(cls._estimate_seasonality())
        cls._estimate_conversion_rate()

        return cls(int(cluster_id), comp_prices, params, strategy=strategy)

    @staticmethod
    def _extract_cluster(vn, cluster_id):

        cluster = []
        for idx, label in enumerate(vn.product_labels) :
            if label == cluster_id :
                cluster.append(idx)

        return vn.feature_data.iloc[cluster]



   
    _MIN_CVR_SAMPLES = 30
    _MIN_PRICE_LEVELS = 5

    @classmethod
    def _estimate_conversion_rate(cls, members=None, comp_prices=None):

        if cls._CVR_MODEL is not None :
            return cls._CVR_MODEL

        try:
            eng = _api_engine()

            query = text("""
                    SELECT
                    t.asin,
                    t.marketplace_id,
                    t.data_date,

                    t.units_ordered,
                    t.sessions,
                    t.ordered_product_sales,

                    o.landed_price,
                    o.listing_price_amount,
                    o.shipping_amount,
                    o.is_prime,
                    o.seller_feedback_rating,
                    o.seller_feedback_count,
                    o.offer_position,

                    p.product_type
                    FROM sales_traffic_daily t
                    LEFT JOIN (
                        SELECT asin, marketplace_id, captured_at::date AS d,
                               AVG(landed_price)          AS landed_price,
                               AVG(listing_price_amount)  AS listing_price_amount,
                               AVG(shipping_amount)       AS shipping_amount,
                               BOOL_OR(is_prime)          AS is_prime,
                               MIN(offer_position)        AS offer_position,
                               MAX(seller_feedback_count) AS seller_feedback_count,
                               AVG(seller_feedback_rating) AS seller_feedback_rating
                        FROM offers_daily
                        GROUP BY asin, marketplace_id, captured_at::date
                    ) o
                        ON o.asin = t.asin
                    AND o.marketplace_id = t.marketplace_id
                    AND o.d = t.data_date
                    LEFT JOIN (
                        SELECT DISTINCT ON (asin, marketplace_id)
                               asin, marketplace_id, product_type
                        FROM product_content
                    ) p
                        ON p.asin = t.asin
                    AND p.marketplace_id = t.marketplace_id
                    WHERE t.sessions IS NOT NULL AND t.sessions > 0
                """)

            with eng.connect() as c:
                data = pd.read_sql(query, c)
        except Exception as e:
            print(e)
            cls._CVR_MODEL = None
            return cls._CVR_MODEL

        if data.empty:
            print("[_estimate_conversion_rate] no own-catalog traffic rows; using constant CVR")
            cls._CVR_MODEL = None
            return cls._CVR_MODEL

        data = data.sort_values(["asin", "marketplace_id", "data_date"]).reset_index(drop=True)

        units    = pd.to_numeric(data["units_ordered"], errors="coerce").fillna(0.0)
        sessions = pd.to_numeric(data["sessions"], errors="coerce").fillna(0.0)
        revenue  = pd.to_numeric(data["ordered_product_sales"], errors="coerce").fillna(0.0)


        data["price"] = revenue / units.where(units > 0)
        data["price"] = (data.groupby(["asin", "marketplace_id"])["price"]
                             .transform(lambda s: s.ffill().bfill()))
        data["price"] = data["price"].fillna(pd.to_numeric(data["landed_price"], errors="coerce"))


        global_ref = float(data["price"].median(skipna=True))
        ref = (data.groupby("product_type")["price"].transform("median")
                   .replace(0, np.nan).fillna(global_ref))
        data["price"] = data["price"].fillna(ref).fillna(global_ref)

        # Supplementary offer features: real where offers_daily matched, else the same
        # defaults expected_demand/predict_cvr assume (FBA free shipping, own buy box,
        # prime). This keeps the train-time feature distribution aligned with predict.
        listing  = pd.to_numeric(data["listing_price_amount"], errors="coerce").fillna(data["price"])
        shipping = pd.to_numeric(data["shipping_amount"], errors="coerce").fillna(0.0)
        offerpos = pd.to_numeric(data["offer_position"], errors="coerce").fillna(1.0)
        prime    = pd.to_numeric(data["is_prime"], errors="coerce").fillna(1.0)

        fb_count = pd.to_numeric(data["seller_feedback_count"], errors="coerce")
        fb_rating = pd.to_numeric(data["seller_feedback_rating"], errors="coerce")
        seller_score = np.log(fb_count + 1) * fb_rating
        default_ss = float(np.log(100 + 1) * 4.5)   # matches predict_cvr's fallback
        seller_score = seller_score.fillna(seller_score.median()).fillna(default_ss)


        X_df = pd.DataFrame({
            'price_ratio': data["price"] / ref,
            'min_price_ratio': data["price"] / ref,
            'listing_ratio': listing / ref,
            'shipping_ratio': shipping / ref,
            'daily_average_offer_position': offerpos,
            'prime_availability': prime,
            'seller_score': seller_score,
        })
        
        y = np.clip((units / sessions.where(sessions > 0)).to_numpy(dtype=float), 0.0, 1.0)
        weights = sessions.to_numpy(dtype=float)

        good = np.isfinite(y) & np.isfinite(weights) & (weights > 0)
        X_df, y, weights = X_df[good].reset_index(drop=True), y[good], weights[good]

      
        n_prices = X_df['price_ratio'].round(2).nunique() if not X_df.empty else 0
        if len(X_df) < cls._MIN_CVR_SAMPLES:
            print(f"[_estimate_conversion_rate] only {len(X_df)} usable rows "
                  f"(need >= {cls._MIN_CVR_SAMPLES}); not enough own-catalog data -- using constant CVR")
            cls._CVR_MODEL = None
            return cls._CVR_MODEL
        if n_prices < cls._MIN_PRICE_LEVELS:
            print(f"[_estimate_conversion_rate] price_ratio takes only {n_prices} distinct level(s) across "
                  f"{len(X_df)} rows; cannot learn a price->CVR curve -- using constant CVR")
            cls._CVR_MODEL = None
            return cls._CVR_MODEL

        print(f"[_estimate_conversion_rate] fitting on {len(X_df)} (asin, marketplace, day) rows across "
              f"{data['product_type'].nunique()} product types, {n_prices} distinct price_ratio levels "
              f"(weighted mean CVR = {np.average(y, weights=weights):.4f})")

        cls._CVR_MODEL = HistGradientBoostingRegressor(random_state=1, max_iter=200)
        cls._CVR_MODEL.fit(X_df, y, sample_weight=weights)

        return cls._CVR_MODEL




    @classmethod
    def _estimate_buybox(cls, members, comp_prices):
        if cls._BUYBOX is not None:
            return cls._BUYBOX

        default = {
            "buybox_intercept": 0.0,
            "buybox_gap_coef": -8.0,
            "buybox_fba_coef": 3.0,
            "buybox_prime_coef": 1.0,
            "buybox_feedback_coef": 0.0,
            "buybox_feedback_median": 4.5,
        }

        try:
            eng = _api_engine()
            with eng.connect() as c:
                df = pd.read_sql(text("""
                    SELECT asin, captured_at, landed_price, is_buy_box_winner,
                           is_fulfilled_by_amazon, is_prime, seller_feedback_rating
                    FROM offers_daily
                    WHERE landed_price IS NOT NULL
                """), c)
        except Exception:
            cls._BUYBOX = default
            return cls._BUYBOX

        df["listing_min"] = df.groupby(["asin", "captured_at"])["landed_price"].transform("min")
        counts = df.groupby(["asin", "captured_at"])["landed_price"].transform("count")
        df = df[counts >= 2].copy()
        df["gap"] = (df["landed_price"] - df["listing_min"]) / df["listing_min"]
        df["fba"] = df["is_fulfilled_by_amazon"].fillna(False).astype(float)
        df["prime"] = df["is_prime"].fillna(False).astype(float)
        feed_med = float(df["seller_feedback_rating"].median())
        df["feed"] = df["seller_feedback_rating"].fillna(feed_med)

        X = df[["gap", "fba", "prime", "feed"]].to_numpy(dtype=float)
        y = df["is_buy_box_winner"].astype(int).to_numpy()

        if len(y) < 10 or len(np.unique(y)) < 2:
            default["buybox_feedback_median"] = feed_med
            cls._BUYBOX = default
            return cls._BUYBOX

        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        # Keep the fitted estimator so _buybox_prob can call predict_proba directly
        # instead of re-implementing the logistic from extracted coefficients.
        cls._BUYBOX_MODEL = model
        b = model.coef_[0]
        cls._BUYBOX = {
            "buybox_intercept": float(model.intercept_[0]),
            "buybox_gap_coef": float(b[0]),
            "buybox_fba_coef": float(b[1]),
            "buybox_prime_coef": float(b[2]),
            "buybox_feedback_coef": float(b[3]),
            "buybox_feedback_median": feed_med,
        }
        return cls._BUYBOX




    #  weekly demand for the whole account swings hard around Black Friday / Christmas
    #  (real sell-out data shows week 48-51 running 1.5x-2.3x a normal week). This gives
    #  the logit a real seasonal baseline to sit on top of instead of a flat market_size.
    @classmethod
    def _estimate_seasonality(cls):
        if cls._SEASONALITY is not None:
            return cls._SEASONALITY

        default = {"seasonal_index": [1.0] * 52}

        try:
            eng = _api_engine()
            with eng.connect() as c:
                df = pd.read_sql(text("SELECT data_date, units_ordered FROM sales_traffic_daily"), c)
        except Exception:
            cls._SEASONALITY = default
            return cls._SEASONALITY

        if df.empty:
            cls._SEASONALITY = default
            return cls._SEASONALITY

        df["data_date"] = pd.to_datetime(df["data_date"])
        iso = df["data_date"].dt.isocalendar()
        df["iso_year"] = iso["year"]
        df["iso_week"] = iso["week"].clip(upper=52)

        weekly = df.groupby(["iso_year", "iso_week"])["units_ordered"].sum().reset_index()
        year_means = weekly.groupby("iso_year")["units_ordered"].transform("mean")
        weekly["ratio"] = weekly["units_ordered"] / year_means.replace(0, np.nan)

        index_by_week = weekly.groupby("iso_week")["ratio"].mean()
        seasonal_index = [float(index_by_week.get(w, 1.0)) for w in range(1, 53)]

        cls._SEASONALITY = {"seasonal_index": seasonal_index}
        return cls._SEASONALITY



    @staticmethod
    def _quality(row):
        r = row.get(("clean", "average_rating"))
        try:
            return float(r)
        except (TypeError, ValueError):
            return 0.0




    # ---- RL interface --------------------------------------------------
    def reset(self, target_product, start_week=None, competitor_strategy="promo_cycler", seed=None):
        row = target_product.iloc[0]
        self.target = target_product
        raw = float(pd.to_numeric(pd.Series([row[PRICE_COL]]), errors="coerce").iloc[0])
        self.own_price = raw * self.params["price_scale"] + self.params["price_center"]
        self.own_quality = self._quality(row)
        self.t = 0
        # Reseed for a reproducible Monte Carlo rollout when a seed is given.
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # Restore the fitted competitor baseline so reactive strategies start each
        # episode from the same prices instead of inheriting the previous episode's.
        self.comp_prices = self._comp_prices_base.copy()
        self.competitor_strategy = competitor_strategy
        if start_week is None:
            start_week = pd.Timestamp.today().isocalendar().week
        self.start_week = int(start_week)
        if self.params.get("cost") is None:
            self.params["cost"] = 0.6 * self.own_price
        # RL competitor starts each episode at the cluster's typical price.
        self.comp_price = float(self.params["reference_price"])
        self.comp_cost = 0.6 * self.comp_price
        self._comp_pending = None
        return self._state()

    def step(self, action):
        self.own_price = self.own_price * self.action_grid[int(action)]
        # Competitor moves. For the "RL" strategy this queries self.competitor_agent
        # (see _competitor_prices) and records (state, action) in self._comp_pending.
        self.comp_prices = self._competitor_prices(self.competitor_strategy)
        units = self.expected_demand(self.own_price)
        if self.stochastic:
            # Sample realised demand around the expected value so repeated rollouts
            # form a distribution (Monte Carlo). Poisson is the natural count model.
            units = float(self.rng.poisson(max(units, 0.0)))
        reward = self._profit(self.own_price, units)
        self.last_units = float(units)
        self.iso_week = ((self.start_week - 1 + self.t) % 52) + 1
        self.t += 1
        done = self.t >= self.horizon
        next_state = self._state()

        # The RL competitor learns from its OWN transition, encapsulated here so the
        # agent-facing step() signature stays single-player. Skipped when learning is
        # gated off (e.g. greedy eval), which freezes the opponent.
        if self.competitor_learning and self._comp_pending is not None:
            s_c, a_c = self._comp_pending
            comp_reward = self._competitor_profit(self._competitor_reference())
            self.competitor_agent.observe(s_c, a_c, comp_reward, self._competitor_state(), done)
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

    def _buybox_prob(self, own_price, comp_ref):
        gap = (own_price - comp_ref) / comp_ref if comp_ref else 0.0

        # Use the fitted LogisticRegression directly when one was trained. Feature
        # order must match _estimate_buybox's training matrix exactly:
        # [gap, fba, prime, feed]. This also means swapping in a non-linear buy-box
        # classifier later requires no change here -- predict_proba handles it.
        if self._BUYBOX_MODEL is not None:
            X = np.array([[gap, self.own_fba, self.own_prime, self.own_feedback]], dtype=float)
            return float(self._BUYBOX_MODEL.predict_proba(X)[0, 1])

        # No model fit (DB unavailable / too little data): fall back to the stored
        # coefficients -- either those extracted from a past fit or the defaults.
        z = (self.params["buybox_intercept"]
             + self.params["buybox_gap_coef"] * gap
             + self.params["buybox_fba_coef"] * self.own_fba
             + self.params["buybox_prime_coef"] * self.own_prime
             + self.params["buybox_feedback_coef"] * self.own_feedback)
        z = float(np.clip(z, -30.0, 30.0))
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
    def predict_cvr(self, own_price, comp_ref=None):
        """Model's predicted conversion rate at a given own price. Single source of
        truth for the CVR feature vector, shared by expected_demand and the backtest
        so the two can never drift out of sync on feature names/order."""
        if self._CVR_MODEL is None:
            return 0.03

        if comp_ref is None:
            comp_ref = self._competitor_reference()

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
            'seller_score': float(seller_score)
        }])

        return float(np.clip(self._CVR_MODEL.predict(X_df)[0], 0.0, 1.0))

    def _sessions(self):
        # Traffic available this week: market size scaled by seasonality. Shared by
        # our demand and the competitor's so the two stay on the same scale.
        return self.params.get("market_size", 1000) * self._seasonal_multiplier(self.t)

    def expected_demand(self, own_price):
        comp_ref = self._competitor_reference()
        bb_prob = self._buybox_prob(own_price, comp_ref)
        cvr = self.predict_cvr(own_price, comp_ref)

        # Expected demand is the combination of visibility, intent, and traffic
        return self._sessions() * bb_prob * cvr

    def _profit(self, own_price, units):
        return (own_price - self.params["cost"]) * units

    def _competitor_reference(self):
        return float(self.comp_prices.min()) if self.comp_prices.size else self.params["reference_price"]

    # ---- 2-player: the RL competitor's state & payoff ------------------
    def _competitor_state(self):
        """The competitor's view of the market -- the mirror image of _state():
        its own price, OUR price, its buy-box share, and season."""
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
