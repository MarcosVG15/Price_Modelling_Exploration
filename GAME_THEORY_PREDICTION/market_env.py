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
    _SEASONALITY = None
    _CVR_MODEL = None



    #  most of these methods are used to provide an estimate of a product that was never here before so we can have a basis on where to start.
    @classmethod
    def for_cluster(cls, vn, cluster_id):
        key = int(cluster_id)
        if key not in cls._CACHE:
            cls._CACHE[key] = cls.fit(vn, key)
        return cls._CACHE[key]


    def __init__(self, group_id , cluster_id, comp_prices, params, horizon=HORIZON, action_grid=ACTION_GRID):
        
        self.group_id
        self.cluster_id = int(cluster_id)
        self.comp_prices = np.asarray(comp_prices, dtype=float)
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

    @classmethod
    def fit(cls, vn, cluster_id):

        members = cls._extract_cluster(vn, cluster_id)

        center, scale = _price_scale()
        comp_norm = pd.to_numeric(members[PRICE_COL], errors="coerce").dropna().to_numpy()
        comp_prices = comp_norm * scale + center

        params = {}
        params["price_center"] = center
        params["price_scale"] = scale
        params["reference_price"] = float(np.median(comp_prices)) if comp_prices.size else 1.0
        params.update(cls._estimate_demand(members, comp_prices))
        params.update(cls._estimate_buybox(members, comp_prices))
        params.update(cls._estimate_seasonality())

        return cls(int(cluster_id), comp_prices, params)

    @staticmethod
    def _extract_cluster(vn, cluster_id):

        cluster = []
        for idx, label in enumerate(vn.product_labels) :
            if label == cluster_id :
                cluster.append(idx)

        return vn.feature_data.iloc[cluster]



    #  in order to do this you can add the commaxx products to the vn and then find the cluster such that then you can train a logistic regressor to estimate the demand
    #  I will estimate using the conversion rate instead of raw orders
    @classmethod
    def _estimate_conversion_rate(cls , members, comp_prices, group_id):

        if cls._CVR_MODEL is not None :
            return cls._CVR_MODEL
        

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
    def reset(self, target_product, start_week=None):
        row = target_product.iloc[0]
        self.target = target_product
        raw = float(pd.to_numeric(pd.Series([row[PRICE_COL]]), errors="coerce").iloc[0])
        self.own_price = raw * self.params["price_scale"] + self.params["price_center"]
        self.own_quality = self._quality(row)
        self.t = 0
        if start_week is None:
            start_week = pd.Timestamp.today().isocalendar().week
        self.start_week = int(start_week)
        if self.params.get("cost") is None:
            self.params["cost"] = 0.6 * self.own_price
        return self._state()

    def step(self, action):
        self.own_price = self.own_price * self.action_grid[int(action)]
        self.comp_prices = self._competitor_prices(self.t)
        units = self.expected_demand(self.own_price)
        reward = self._profit(self.own_price, units)
        self.last_units = float(units)
        self.iso_week = ((self.start_week - 1 + self.t) % 52) + 1
        self.t += 1
        done = self.t >= self.horizon
        return self._state(), reward, done

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

    # ---- the three brains ---------------------------------------------
    def _logit_demand(self, own_price):
        a = self.params["alpha"]
        u0 = self.params["outside_utility"]
        comp_ref = self._competitor_reference()
        u_own = self.own_quality - a * own_price
        u_comp = -a * comp_ref
        n_comp = max(len(self.comp_prices), 1)
        num = np.exp(u_own)
        den = np.exp(u0) + num + n_comp * np.exp(u_comp)
        market_share = num / den
        return market_share * self.params["market_size"] * self._seasonal_multiplier(self.t)

    def _seasonal_multiplier(self, t):
        week = ((self.start_week - 1 + t) % 52) + 1
        return self.params["seasonal_index"][week - 1]

    def _buybox_prob(self, own_price, comp_ref):
        gap = (own_price - comp_ref) / comp_ref if comp_ref else 0.0
        z = (self.params["buybox_intercept"]
             + self.params["buybox_gap_coef"] * gap
             + self.params["buybox_fba_coef"] * self.own_fba
             + self.params["buybox_prime_coef"] * self.own_prime
             + self.params["buybox_feedback_coef"] * self.own_feedback)
        z = float(np.clip(z, -30.0, 30.0))
        return 1.0 / (1.0 + np.exp(-z))

    def _competitor_prices(self, strategy_type):
        match strategy_type:
            case "static":
                return self.comp_prices
                
            case "undercutter":
                floor = self.params.get("comp_floor_price", self.params["reference_price"] * 0.7)
                return np.maximum(self.own_price - 0.01, floor)
                
            case "matcher":
                floor = self.params.get("comp_floor_price", self.params["reference_price"] * 0.7)
                return np.maximum(self.own_price, floor)
                
            case "tit_for_tat":
                # Game-theory punishment logic
                ...
                
            case "promo_cycler":
                # Stochastic promotion logic
                ...
                
            case "SL":
                pass

            case "RL" :
                pass 
                
            case _:
                print(f"Strategy {strategy_type} not recognized. Falling back to static.")
                return self.comp_prices

    # ---- helpers -------------------------------------------------------
    def expected_demand(self, own_price):
        comp_ref = self._competitor_reference()
        return self._buybox_prob(own_price, comp_ref) * self._logit_demand(own_price)

    def _profit(self, own_price, units):
        return (own_price - self.params["cost"]) * units

    def _competitor_reference(self):
        return float(self.comp_prices.min()) if self.comp_prices.size else self.params["reference_price"]

    # ---- persistence ---------------------------------------------------
    def save(self, path):
        blob = {
            "cluster_id": self.cluster_id,
            "params": self.params,
            "comp_prices": self.comp_prices.tolist(),
            "horizon": self.horizon,
            "action_grid": self.action_grid.tolist(),
        }
        Path(path).write_text(json.dumps(blob, indent=2))
        return path

    @classmethod
    def load(cls, path):
        blob = json.loads(Path(path).read_text())
        return cls(blob["cluster_id"], np.asarray(blob["comp_prices"], dtype=float),
                   blob["params"], horizon=blob.get("horizon", HORIZON),
                   action_grid=blob.get("action_grid", ACTION_GRID))
