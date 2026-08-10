import numpy as np
from tqdm import tqdm

from market_env import MarketEnv



import torch
import torch.nn as nn
import random
from collections import deque
from agents import TQL , DQN , PPO 


class pricing_game:
    def __init__(self, target_product, vn, k=1, bins=12, seed=0, competitor_strategy="static"):
        fit = vn.fit_new_products(new_data=target_product, k=k)
        q = int(fit.new_product_indices[0])
        self.cluster_id = int(fit.product_labels[q])
        self.competitor_strategy = competitor_strategy
        target_asin = (target_product[("clean", "asin")].iloc[0]
                       if ("clean", "asin") in target_product.columns else None)
        self.env = MarketEnv.for_cluster(vn, self.cluster_id, strategy=competitor_strategy,
                                         target_asin=target_asin)
        self.vn = vn


        self.target = target_product

        self.bins = bins
        self.rng = np.random.default_rng(seed)
        self.q_table = {}
        self.low = np.array([0.0, 0.0, 0.0, 0.0])
        self.high = np.array([2.0, 2.0, 1.0, 1.0])
        self.history = []
        # 2-player game: when the competitor strategy is "RL", attach a learning
        # opponent (its own Agent) to the env. It maximises its own profit in step().
        self.env.competitor_agent = self.choose_agent("PPO") if competitor_strategy == "RL" else None


    def test(self):
        # _estimate_conversion_rate now runs automatically inside MarketEnv.fit()
        # (triggered by for_cluster() in __init__ above), so there's nothing to
        # trigger here anymore -- this just reports whether a real model got fit
        # for this process, or whether expected_demand will use the constant fallback.
        status = "fitted" if MarketEnv._CVR_MODEL is not None else "constant fallback (see log above)"
        print(f"CVR model status: {status}")



    def choose_agent(self , type):
        
        match type :
            case 'TQL' :
                tql_agent = TQL(action_dim=self.env.action_dim, low=self.low, high=self.high)
                return tql_agent
            
            case 'DQN' :
                dqn_agent = DQN(state_dim=self.env.state_dim, action_dim=self.env.action_dim)
                return dqn_agent

            case 'PPO' :
                ppo_agent = PPO(state_dim=self.env.state_dim, action_dim=self.env.action_dim)
                return ppo_agent
            
            # case 'A2C': 

            # case 'SAC':

            # case 'MVN' marcos special. 

            case _ :
                print("AGENT NOT FOUND HABIBI")
                return None


    def train(self, episodes=3000, alpha=0.1, gamma=0.95,  epsilon=1.0, epsilon_min=0.05, decay=0.999,
              type='PPO', eval_every=50, eval_week=1, eval_seed=0):
        self.history = []
        self.eval_history = []       # (episode, greedy_return) -- clean policy-quality checkpoints
        # Remember the run's key params so plot_cumulative_reward() can name the file.
        self.train_config = {"agent": type, "episodes": episodes,
                             "gamma": gamma, "epsilon": epsilon, "decay": decay}
        pbar = tqdm(range(episodes), desc=f"training {type}", unit="ep")
        self.agent = self.choose_agent(type)

        for i in pbar:
            start_week = int(self.rng.integers(1, 53))
            state = self.env.reset(self.target, start_week=start_week, competitor_strategy=self.competitor_strategy)
            done = False
            total = 0.0
            while not done:
                a = self.agent.choose(state)
                nxt, r, done = self.env.step(a)
                self.agent.observe(state , a , r , nxt , done)
                state = nxt
                total += r

            self.agent.on_episode_end()                  # anneal exploration (TQL/DQN ε; no-op for PPO)
            self.history.append(total)

            # Periodic GREEDY eval: roll out the CURRENT policy with NO exploration on a
            # FIXED week/seed, so the only thing changing between checkpoints is the policy
            # itself. best_policy() never calls observe(), so it can't perturb learning.
            if (i + 1) % eval_every == 0:
                rng_state = self.env.rng                 # keep the training RNG stream intact ...
                greedy = self.best_policy(start_week=eval_week, seed=eval_seed)
                self.env.rng = rng_state                 # ... (the eval ran on its own seeded stream)
                self.eval_history.append((i + 1, greedy["total_profit"]))

            postfix = {"avg_reward": f"{np.mean(self.history[-50:]):.1f}"}
            if self.eval_history:
                postfix["greedy"] = f"{self.eval_history[-1][1]:.0f}"
            eps = getattr(self.agent, "epsilon", None)   # PPO has no ε -> omit it from the bar
            if eps is not None:
                postfix["eps"] = f"{eps:.3f}"
            pbar.set_postfix(**postfix)
        return self.history

    def train_curriculum(self, promo_episodes=3000, rl_rounds=6, episodes_per_round=500,
                          rl_lr=1e-4, rl_entropy_coef=0.01, eval_every=50, eval_week=1,
                          eval_seed=0, checkpoint_path=None):
        """Two-phase curriculum for a more stable RL-vs-RL pricing policy.

        Phase 1: train PPO to convergence against the stationary 'promo_cycler'
        competitor (plain train()) -- a clean single-agent problem.

        Phase 2: switch to competitor_strategy='RL' and warm-start BOTH sides from
        the phase-1 checkpoint (they play the identical game, so starting from the
        same competent baseline avoids one side crushing a naive opponent). Train
        via alternating best-response rounds -- one side frozen (greedy, no
        learning) while the other trains -- instead of naive simultaneous
        self-play, which is what actually destabilizes multi-agent PPO: each round
        is single-agent PPO against a stationary target, so neither side ever
        faces a genuinely moving one. Progress is checked every round against the
        fixed promo_cycler benchmark, not just the live opponent, since self-play's
        own reward can look healthy while both sides are cycling into a jointly
        worse equilibrium.
        """
        print(f"[curriculum] Phase 1: {promo_episodes} episodes vs promo_cycler")
        self.competitor_strategy = "promo_cycler"
        self.env.competitor_agent = None
        self.train(episodes=promo_episodes, type='PPO', eval_every=eval_every,
                   eval_week=eval_week, eval_seed=eval_seed)
        phase1_history = list(self.history)
        phase1_eval = list(self.eval_history)

        checkpoint = self.agent.state_dict()
        if checkpoint_path:
            import os
            os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
            torch.save(checkpoint, checkpoint_path)
            print(f"[curriculum] saved phase-1 checkpoint -> {checkpoint_path}")

        print(f"[curriculum] Phase 2: {rl_rounds} rounds x {episodes_per_round} episodes "
              f"vs RL, warm-started from phase-1 policy")
        self.competitor_strategy = "RL"

        main_agent = PPO(state_dim=self.env.state_dim, action_dim=self.env.action_dim,
                          lr=rl_lr, entropy_coef=rl_entropy_coef)
        comp_agent = PPO(state_dim=self.env.state_dim, action_dim=self.env.action_dim,
                          lr=rl_lr, entropy_coef=rl_entropy_coef)
        main_agent.load_state_dict(checkpoint)
        comp_agent.load_state_dict(checkpoint)
        self.agent = main_agent
        self.env.competitor_agent = comp_agent

        self.selfplay_history = {"main": [], "comp": [], "benchmark_eval": []}

        for round_idx in range(rl_rounds):
            # Round A: main agent learns, competitor frozen (greedy, stationary target)
            self.env.competitor_learning = False
            pbar = tqdm(range(episodes_per_round),
                        desc=f"round {round_idx + 1}/{rl_rounds} [main learns]", unit="ep")
            for _ in pbar:
                start_week = int(self.rng.integers(1, 53))
                state = self.env.reset(self.target, start_week=start_week, competitor_strategy="RL")
                done, total = False, 0.0
                while not done:
                    a = main_agent.choose(state)
                    nxt, r, done = self.env.step(a)
                    main_agent.observe(state, a, r, nxt, done)
                    state = nxt
                    total += r
                main_agent.on_episode_end()
                self.selfplay_history["main"].append(total)
                pbar.set_postfix(avg=f"{np.mean(self.selfplay_history['main'][-50:]):.1f}")

            # Round B: competitor learns (via env.step()'s internal observe()), main frozen greedy
            self.env.competitor_learning = True
            pbar = tqdm(range(episodes_per_round),
                        desc=f"round {round_idx + 1}/{rl_rounds} [comp learns]", unit="ep")
            for _ in pbar:
                start_week = int(self.rng.integers(1, 53))
                state = self.env.reset(self.target, start_week=start_week, competitor_strategy="RL")
                done, comp_total = False, 0.0
                while not done:
                    a = main_agent.choose(state, explore=False)
                    _, _, done = self.env.step(a)
                    comp_total += self.env.last_comp_profit
                self.selfplay_history["comp"].append(comp_total)
                pbar.set_postfix(avg=f"{np.mean(self.selfplay_history['comp'][-50:]):.1f}")

            # Benchmark eval: greedy main agent vs the FIXED promo_cycler, not the live opponent
            self.agent = main_agent
            prior_strategy = self.competitor_strategy
            self.competitor_strategy = "promo_cycler"
            rng_state = self.env.rng
            bench = self.best_policy(start_week=eval_week, seed=eval_seed)
            self.env.rng = rng_state
            self.competitor_strategy = prior_strategy
            self.selfplay_history["benchmark_eval"].append((round_idx + 1, bench["total_profit"]))
            print(f"[curriculum] round {round_idx + 1}/{rl_rounds}: "
                  f"main avg reward={np.mean(self.selfplay_history['main'][-episodes_per_round:]):.1f}  "
                  f"benchmark vs promo_cycler={bench['total_profit']:.1f}")

        self.agent = main_agent
        self.env.competitor_agent = comp_agent
        self.env.competitor_learning = True
        self.competitor_strategy = "RL"
        return {"phase1_history": phase1_history, "phase1_eval": phase1_eval,
                "selfplay_history": self.selfplay_history}

    def _curriculum_path(self):
        import os
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cumulative_reward")
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, f"curriculum_cluster{self.cluster_id}.png")

    def plot_curriculum(self, result, path=None):
        """Plot the two-phase curriculum: phase-1 promo_cycler learning curve, phase-2
        self-play main-agent learning curve, and the benchmark-vs-promo_cycler eval
        trend across self-play rounds -- the metric that actually tells you whether
        self-play is converging or just cycling.
        """
        import matplotlib
        if path is not None:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(11, 8))

        p1 = np.asarray(result["phase1_history"], dtype=float)
        axes[0].plot(np.arange(1, len(p1) + 1), p1, lw=0.8, alpha=0.3,
                     label="phase 1 episode return (vs promo_cycler)")
        if result["phase1_eval"]:
            ex, ey = zip(*result["phase1_eval"])
            axes[0].plot(ex, ey, lw=2, color="green", marker="o", ms=3, label="phase 1 greedy eval")
        sp_main = np.asarray(result["selfplay_history"]["main"], dtype=float)
        if sp_main.size:
            offset = len(p1)
            axes[0].plot(offset + np.arange(1, len(sp_main) + 1), sp_main, lw=0.8, alpha=0.3,
                         color="steelblue", label="phase 2 main-agent return (self-play rounds)")
        axes[0].axvline(len(p1), color="black", ls="--", lw=1, label="phase 1 -> phase 2")
        axes[0].set_ylabel("episode return")
        axes[0].set_title(f"Curriculum training vs '{self.competitor_strategy}' (cluster {self.cluster_id})")
        axes[0].legend(loc="best", fontsize=8)

        bench = result["selfplay_history"]["benchmark_eval"]
        if bench:
            bx, by = zip(*bench)
            axes[1].plot(bx, by, lw=2, marker="o", color="darkorange")
        axes[1].set_xlabel("self-play round")
        axes[1].set_ylabel("greedy profit vs promo_cycler\n(fixed benchmark)")

        fig.tight_layout()
        path = path or self._curriculum_path()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"[game] curriculum plot -> {path}")
        return path

    def _cumreward_path(self):
        """Descriptive PNG path under cumulative_reward/, built from the last
        train() config + competitor/cluster. Same params -> same file (overwrite)."""
        import os
        cfg = getattr(self, "train_config", {})
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cumulative_reward")
        os.makedirs(out_dir, exist_ok=True)
        name = "_".join([
            "cumreward",
            str(cfg.get("agent", "NA")),                 # the agent used
            f"ep{cfg.get('episodes', 'NA')}",            # key params ...
            str(self.competitor_strategy),
            f"cluster{self.cluster_id}",
        ]) + ".png"
        return os.path.join(out_dir, name)

    def plot_cumulative_reward(self, path=None, window=50, ax=None):
        """Plot the learning curve from ``self.history``: the raw per-episode
        cumulative reward (season profit) plus a moving-average trend. Saves a PNG
        under ``cumulative_reward/`` named by the agent + key params, unless ``path``
        is given. Run ``train()`` first.
        """
        import matplotlib
        if ax is None:
            matplotlib.use("Agg")          # headless-safe when saving to file
        import matplotlib.pyplot as plt

        if not self.history:
            raise RuntimeError("no training history -- call train() first")

        hist = np.asarray(self.history, dtype=float)
        x = np.arange(1, len(hist) + 1)

        created = ax is None
        if created:
            _, ax = plt.subplots(figsize=(11, 5))
        ax.plot(x, hist, lw=0.8, alpha=0.30, label="episode return (training, exploring)")
        if len(hist) >= window:            # smoothed trend (raw curve is noisy)
            ma = np.convolve(hist, np.ones(window) / window, mode="valid")
            ax.plot(np.arange(window, len(hist) + 1), ma, lw=2.0,
                    label=f"{window}-ep moving average")

        ev = getattr(self, "eval_history", [])
        if ev:                             # clean policy-quality signal: greedy, fixed week
            ex, ey = zip(*ev)
            ax.plot(ex, ey, lw=2.0, color="green", marker="o", ms=3,
                    label="greedy eval (fixed week, no exploration)")

        cfg = getattr(self, "train_config", {})
        ax.set_xlabel("episode")
        ax.set_ylabel("cumulative reward (season profit)")
        ax.set_title(f"Cumulative reward — {cfg.get('agent', '?')} "
                     f"vs '{self.competitor_strategy}' (cluster {self.cluster_id})")
        ax.legend(loc="best", fontsize=8)

        if not created:
            return ax
        ax.figure.tight_layout()
        if path is None:
            path = self._cumreward_path()
        ax.figure.savefig(path, dpi=120)
        plt.close(ax.figure)
        return path

    def best_policy(self, start_week=None, seed=None):
        if getattr(self, "agent", None) is None:
            raise RuntimeError("no trained agent -- call train() before best_policy()")
        state = self.env.reset(self.target, start_week=start_week, seed=seed,
                               competitor_strategy=self.competitor_strategy)
        done = False
        prices, comp_prices, multipliers, rewards = [], [], [], []
        buybox, cvr, demand, weeks = [], [], [], []
        while not done:
            a = self.agent.choose(state, explore=False)   # greedy: use the trained policy
            state, r, done = self.env.step(a)
            prices.append(self.env.own_price)
            comp_prices.append(self.env.last_comp_ref)
            multipliers.append(float(self.env.action_grid[a]))
            rewards.append(r)
            buybox.append(float(state[2]))
            cvr.append(self.env.last_cvr)
            demand.append(self.env.last_units)
            weeks.append(self.env.iso_week)
        return {
            "prices": prices,
            "comp_prices": comp_prices,
            "multipliers": multipliers,
            "rewards": rewards,
            "buybox": buybox,
            "cvr": cvr,
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

    def _single_run_path(self):
        import os
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "single_run")
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, f"single_run_{self.competitor_strategy}_cluster{self.cluster_id}.png")

    def plot_single_run(self, roll=None, path=None):
        """One concrete rollout under the trained (greedy) policy: agent vs competitor
        price, predicted buy-box probability, and predicted CVR, week by week. Lets you
        sanity-check the whole price -> buy-box -> CVR -> demand chain on a single
        trace, rather than only the aggregate Monte Carlo bands."""
        import matplotlib
        if path is not None:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        roll = roll if roll is not None else self.best_policy()
        x = np.arange(1, len(roll["prices"]) + 1)

        fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

        axes[0].plot(x, roll["prices"], color="steelblue", lw=2, marker="o", ms=3, label="agent price")
        axes[0].plot(x, roll["comp_prices"], color="darkorange", lw=2, marker="o", ms=3, label="competitor price")
        axes[0].set_ylabel("price")
        axes[0].set_title(f"Single run vs '{self.competitor_strategy}' (cluster {self.cluster_id})")
        axes[0].legend(loc="best", fontsize=8)

        axes[1].plot(x, roll["buybox"], color="seagreen", lw=2, marker="o", ms=3)
        axes[1].set_ylabel("P(buy-box)")
        axes[1].set_ylim(-0.05, 1.05)

        axes[2].plot(x, roll["cvr"], color="indianred", lw=2, marker="o", ms=3)
        axes[2].set_ylabel("predicted CVR")
        axes[2].set_xlabel("prediction week")

        fig.tight_layout()
        path = path or self._single_run_path()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"[game] single_run -> {path}")
        return path

    def monte_carlo(self, n=500, seed0=0, stochastic=True, randomize_week=True):
        
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
