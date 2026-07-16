import numpy as np
from tqdm import tqdm

from market_env import MarketEnv


class pricing_game:
    def __init__(self, target_product, vn, k=1, bins=12, seed=0):
        fit = vn.fit_new_products(new_data=target_product, k=k)
        q = int(fit.new_product_indices[0])
        self.cluster_id = int(fit.product_labels[q])
        self.env = MarketEnv.for_cluster(vn, self.cluster_id)
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

    def train(self, episodes=3000, alpha=0.1, gamma=0.95,
              epsilon=1.0, epsilon_min=0.05, decay=0.999):
        self.history = []
        pbar = tqdm(range(episodes), desc="training", unit="ep")
        for _ in pbar:
            start_week = int(self.rng.integers(1, 53))
            state = self.env.reset(self.target, start_week=start_week)
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

    def best_policy(self):
        state = self.env.reset(self.target)
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
        state = self.env.reset(self.target)
        hold = int(np.argmin(np.abs(self.env.action_grid - 1.0)))
        done = False
        rewards = []
        while not done:
            state, r, done = self.env.step(hold)
            rewards.append(r)
        return float(np.sum(rewards))

    def assess(self):
        self.env.reset(self.target)
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
