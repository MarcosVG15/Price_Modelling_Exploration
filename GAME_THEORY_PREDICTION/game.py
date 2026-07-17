import numpy as np
from tqdm import tqdm

from market_env import MarketEnv



import torch
import torch.nn as nn
import random
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, x):
        return self.fc(x)

class pricing_game:
    def __init__(self, target_product, vn, k=1, bins=12, seed=0, competitor_strategy="static"):
        fit = vn.fit_new_products(new_data=target_product, k=k)
        q = int(fit.new_product_indices[0])
        self.cluster_id = int(fit.product_labels[q])
        self.competitor_strategy = competitor_strategy
        self.env = MarketEnv.for_cluster(vn, self.cluster_id, strategy=competitor_strategy)
        self.vn = vn


        self.target = target_product

        self.bins = bins
        self.rng = np.random.default_rng(seed)
        self.q_table = {}
        self.low = np.array([0.0, 0.0, 0.0, 0.0])
        self.high = np.array([2.0, 2.0, 1.0, 1.0])
        self.history = []


    def test(self):
        # _estimate_conversion_rate now runs automatically inside MarketEnv.fit()
        # (triggered by for_cluster() in __init__ above), so there's nothing to
        # trigger here anymore -- this just reports whether a real model got fit
        # for this process, or whether expected_demand will use the constant fallback.
        status = "fitted" if MarketEnv._CVR_MODEL is not None else "constant fallback (see log above)"
        print(f"CVR model status: {status}")


    def _key(self, state):
        s = np.clip((np.asarray(state) - self.low) / (self.high - self.low), 0.0, 1.0)
        idx = np.minimum((s * self.bins).astype(int), self.bins - 1)
        return tuple(int(i) for i in idx)

    def _q(self, key):
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.env.action_dim)
        return self.q_table[key]

    def choose(self, state, epsilon):
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.env.action_dim))
        return int(np.argmax(self._q(self._key(state))))
    

    def agentic_model(self , type, state, total , epsilon, gamma , alpha):

        match type :
            case 'TQL' : 
                a = self.choose(state, epsilon)
                nxt, r, done = self.env.step(a)
                q = self._q(self._key(state))
                target = r if done else r + gamma * float(np.max(self._q(self._key(nxt))))
                q[a] += alpha * (target - q[a])
                state = nxt
                total += r

                return done, total , state

            case 'DQN' : 
                pass


        return None






    def train(self, episodes=3000, alpha=0.1, gamma=0.95,  epsilon=1.0, epsilon_min=0.05, decay=0.999 , type = 'TQL'):
        self.history = []
        pbar = tqdm(range(episodes), desc=f"training {type}", unit="ep")
       
       
        for _ in pbar:
            start_week = int(self.rng.integers(1, 53))
            state = self.env.reset(self.target, start_week=start_week, competitor_strategy=self.competitor_strategy)
            done = False
            total = 0.0
            while not done:
                a = self.choose(state, epsilon)
                nxt, r, done = self.env.step(a)
                q = self._q(self._key(state))
                target = r if done else r + gamma * float(np.max(self._q(self._key(nxt))))
                q[a] += alpha * (target - q[a])
                state = nxt
                total += r

            epsilon = max(epsilon_min, epsilon * decay)
            self.history.append(total)
            pbar.set_postfix(avg_reward=f"{np.mean(self.history[-50:]):.1f}", eps=f"{epsilon:.3f}")
        return self.history

    def best_policy(self, start_week=None, seed=None):
        state = self.env.reset(self.target, start_week=start_week, seed=seed,
                               competitor_strategy=self.competitor_strategy)
        done = False
        prices, multipliers, rewards, buybox, demand, weeks = [], [], [], [], [], []
        while not done:
            a = int(np.argmax(self._q(self._key(state))))
            state, r, done = self.env.step(a)
            prices.append(self.env.own_price)
            multipliers.append(float(self.env.action_grid[a]))
            rewards.append(r)
            buybox.append(float(state[2]))
            demand.append(self.env.last_units)
            weeks.append(self.env.iso_week)
        return {
            "prices": prices,
            "multipliers": multipliers,
            "rewards": rewards,
            "buybox": buybox,
            "demand": demand,
            "weeks": weeks,
            "total_profit": float(np.sum(rewards)),
            "total_units": float(np.sum(demand)),
        }

    def _run_hold(self):
        state = self.env.reset(self.target, competitor_strategy=self.competitor_strategy)
        hold = int(np.argmin(np.abs(self.env.action_grid - 1.0)))
        done = False
        rewards = []
        while not done:
            state, r, done = self.env.step(hold)
            rewards.append(r)
        return float(np.sum(rewards))

    def assess(self):
        self.env.reset(self.target, competitor_strategy=self.competitor_strategy)
        start_price = float(self.env.own_price)
        opt = self.best_policy()
        hold = self._run_hold()
        return {
            "cluster_id": self.cluster_id,
            "start_price": start_price,
            "hold_profit": hold,
            "optimal_profit": opt["total_profit"],
            "uplift": opt["total_profit"] - hold,
            "final_price": opt["prices"][-1] if opt["prices"] else start_price,
            "schedule": opt["prices"],
            "demand": opt["demand"],
            "weeks": opt["weeks"],
            "total_units": opt["total_units"],
        }

    def monte_carlo(self, n=500, seed0=0, stochastic=True, randomize_week=True):
        """Monte Carlo the current greedy policy against the current competitor.

        Runs ``n`` independent rollouts, each reseeded (so demand noise and stochastic
        competitor moves vary) and optionally started on a random ISO week.

        Returns season-total distributions (profit / units / final price) AND per-week
        confidence bands over the horizon: for every prediction week it reports the mean
        and the 5/25/50/75/95th percentiles across rollouts, so the trajectory can be
        drawn as a fan chart (see ``plot_bands``).

        On ``randomize_week``: leave it True for a scenario band (robust across seasons).
        Set it False for a *clean seasonal forecast* band -- then every rollout forecasts
        the same calendar weeks, so the band reflects only demand/competitor noise and the
        ISO week of each prediction step is returned too.

        Requires a trained policy (call ``train()`` first) and, for real variance,
        ``stochastic=True`` so ``step()`` samples demand instead of using the mean.
        """
        wk_rng = np.random.default_rng(seed0)   # dedicated stream -> reproducible from seed0
        self.env.stochastic = stochastic
        totals = {"profit": [], "units": [], "final_price": []}
        traj = {"demand": [], "profit": [], "price": [], "buybox": []}
        iso_weeks = None
        try:
            for i in range(n):
                week = int(wk_rng.integers(1, 53)) if randomize_week else None
                roll = self.best_policy(start_week=week, seed=seed0 + i)
                totals["profit"].append(roll["total_profit"])
                totals["units"].append(roll["total_units"])
                totals["final_price"].append(roll["prices"][-1] if roll["prices"] else np.nan)
                traj["demand"].append(roll["demand"])     # per-week trajectories,
                traj["profit"].append(roll["rewards"])    # each of length == horizon,
                traj["price"].append(roll["prices"])      # so they stack into (n, H)
                traj["buybox"].append(roll["buybox"])
                if not randomize_week:
                    iso_weeks = roll["weeks"]             # identical across fixed-start rollouts
        finally:
            self.env.stochastic = False   # never leave the shared/cached env in stochastic mode

        def _summ(a):
            a = np.asarray(a, dtype=float)
            return {
                "mean": float(np.mean(a)), "std": float(np.std(a)),
                "p05": float(np.percentile(a, 5)),
                "p50": float(np.percentile(a, 50)),
                "p95": float(np.percentile(a, 95)),
            }

        def _band(mat):
            M = np.asarray(mat, dtype=float)   # (n, horizon) -> percentile per week (column)
            qs = np.percentile(M, [5, 25, 50, 75, 95], axis=0)
            return {"mean": M.mean(axis=0),
                    "p05": qs[0], "p25": qs[1], "p50": qs[2], "p75": qs[3], "p95": qs[4]}

        horizon = len(traj["demand"][0]) if traj["demand"] else 0
        bands = {
            "week_index": np.arange(1, horizon + 1),      # prediction week 1..H (x-axis)
            "iso_weeks": np.asarray(iso_weeks) if iso_weeks is not None else None,
            "demand": _band(traj["demand"]),
            "profit": _band(traj["profit"]),
            "price": _band(traj["price"]),
            "buybox": _band(traj["buybox"]),
        }

        return {
            "n": n,
            "strategy": self.competitor_strategy,
            "profit": _summ(totals["profit"]),
            "units": _summ(totals["units"]),
            "final_price": _summ(totals["final_price"]),
            "profits": np.asarray(totals["profit"]),   # raw draws, for histograms / box plots
            "units_raw": np.asarray(totals["units"]),
            "bands": bands,
        }

    @staticmethod
    def plot_bands(mc, metric="demand", path=None, ax=None):
        """Draw a Monte Carlo fan chart for one per-week metric from a monte_carlo()
        result: median + mean lines with shaded 25-75% and 5-95% bands over the weeks.
        ``metric`` is one of 'demand', 'profit', 'price', 'buybox'. Pass ``path`` to save
        a PNG (headless-safe); otherwise it shows interactively.
        """
        import matplotlib
        if path is not None and ax is None:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        b = mc["bands"][metric]
        x = mc["bands"]["week_index"]

        created = ax is None
        if created:
            _, ax = plt.subplots(figsize=(11, 5))
        ax.fill_between(x, b["p05"], b["p95"], alpha=0.20, label="5-95%")
        ax.fill_between(x, b["p25"], b["p75"], alpha=0.35, label="25-75%")
        ax.plot(x, b["p50"], lw=2, label="median")
        ax.plot(x, b["mean"], lw=1, ls="--", label="mean")
        ax.set_xlabel("prediction week")
        ax.set_ylabel(metric)
        ax.set_title(f"Monte Carlo {metric} band vs '{mc['strategy']}' (n={mc['n']})")
        ax.legend(loc="best", fontsize=8)

        if not created:
            return ax
        ax.figure.tight_layout()
        if path:
            ax.figure.savefig(path, dpi=120)
            plt.close(ax.figure)
            return path
        plt.show()
        return ax
