import torch
import random
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from collections import deque

from market_env import MarketEnv
from abstract_agent import Agent


class TQL(Agent):
    def __init__(self, action_dim, low, high, bins=12, alpha=0.1, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.05, decay=0.999, seed=0):

        super().__init__()
        self.action_dim = action_dim
        self.low  = np.asarray(low,  dtype=float)
        self.high = np.asarray(high, dtype=float)
        self.bins = bins
        self.alpha, self.gamma = alpha, gamma
        self.epsilon, self.epsilon_min, self.decay = epsilon, epsilon_min, decay
        self.rng = np.random.default_rng(seed)
        self.q_table = {}

    def _key(self, state):
        s = np.clip((np.asarray(state) - self.low) / (self.high - self.low), 0.0, 1.0)
        idx = np.minimum((s * self.bins).astype(int), self.bins - 1)
        return tuple(int(i) for i in idx)

    def _q(self, key):
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.action_dim)
        return self.q_table[key]

    def choose(self, state, explore=True):
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.action_dim))
        return int(np.argmax(self._q(self._key(state))))

    def observe(self, s, a, r, s2, done):
        q = self._q(self._key(s))                                   # s, not self.state
        target = r if done else r + self.gamma * float(np.max(self._q(self._key(s2))))
        q[a] += self.alpha * (target - q[a])

    def on_episode_end(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay)




class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64),        nn.ReLU(),
            nn.Linear(64, action_dim),
        )
    def forward(self, x):            # x: (batch, state_dim) -> (batch, action_dim)
        return self.fc(x)


class DQN(Agent):
    def __init__(self, state_dim, action_dim, gamma=0.95, lr=1e-3,
             epsilon=1.0, epsilon_min=0.05, decay=0.999,
             buffer=50_000, batch=64, sync=500, seed=0):

        super().__init__()
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon, self.epsilon_min, self.decay = epsilon, epsilon_min, decay
        self.rng = np.random.default_rng(seed)

        self.net    = QNetwork(state_dim, action_dim)          # the value function
        self.target = QNetwork(state_dim, action_dim)          # a frozen copy, for stable targets
        self.target.load_state_dict(self.net.state_dict())
        self.opt    = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.buffer = deque(maxlen=buffer)
        self.batch, self.sync, self.steps = batch, sync, 0


    def observe(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))     # 1. just store the transition
        if len(self.buffer) < self.batch:
            return                                   # 2. not enough data yet — do nothing
        self._train_step()                           # 3. sample a minibatch, one gradient step
        self.steps += 1
        if self.steps % self.sync == 0:
            self.target.load_state_dict(self.net.state_dict())   # 4. periodically refresh target


    def _train_step(self):
        batch = random.sample(self.buffer, self.batch)          # 64 random past transitions
        s, a, r, s2, done = zip(*batch)

        s    = torch.as_tensor(np.array(s),  dtype=torch.float32)             # (B, state_dim)
        a    = torch.as_tensor(a,            dtype=torch.int64).unsqueeze(1)  # (B, 1)
        r    = torch.as_tensor(r,            dtype=torch.float32).unsqueeze(1)# (B, 1)
        s2   = torch.as_tensor(np.array(s2), dtype=torch.float32)             # (B, state_dim)
        done = torch.as_tensor(done,         dtype=torch.float32).unsqueeze(1)# (B, 1)

        q = self.net(s).gather(1, a)

        with torch.no_grad():
            max_q2 = self.target(s2).max(dim=1, keepdim=True).values
            target = r + self.gamma * max_q2 * (1.0 - done)

        loss = nn.functional.mse_loss(q, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss.item())


    def choose(self, state, explore=True):
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.action_dim))
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.net(s)
        return int(q_values.argmax().item())

    def on_episode_end(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay)


class PPO(Agent):
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=3e-4,
                 clip_eps=0.2, gae_lambda=0.95, epochs=10, minibatch=64,
                 entropy_coef=0.0, vf_coef=0.5, rollout_len=2048, seed=0):
        super().__init__()
        self.action_dim   = action_dim
        self.gamma        = gamma
        self.clip_eps     = clip_eps        # PPO clip ratio ε (NOT exploration ε)
        self.gae_lambda   = gae_lambda
        self.epochs       = epochs          # K update passes over each rollout
        self.minibatch    = minibatch
        self.entropy_coef = entropy_coef
        self.vf_coef      = vf_coef
        self.rollout_len  = rollout_len
        self.rng = np.random.default_rng(seed)

        self.actor  = QNetwork(state_dim, action_dim)   # logits over actions
        self.critic = QNetwork(state_dim, 1)            # V(s): single scalar
        self.opt = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)

        self.buffer = []                                # rollout — cleared each iteration

    def state_dict(self):
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}

    def load_state_dict(self, sd):
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])

    def choose(self, state, explore=True):
        s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self.actor(s)
        dist = torch.distributions.Categorical(logits=logits)
        if explore:
            return int(dist.sample().item())              # sample from the policy
        return int(logits.argmax(dim=-1).item())          # greedy, for eval

    def observe(self, s, a, r, s2, done):
       
        s_t = torch.as_tensor(s, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            dist     = torch.distributions.Categorical(logits=self.actor(s_t))
            logp_old = float(dist.log_prob(torch.as_tensor([a], dtype=torch.int64)).item())  # π_old(a|s)
            value    = float(self.critic(s_t).squeeze().item())                              # V(s)
        self.buffer.append((s, a, r, s2, done, logp_old, value))

        if len(self.buffer) < self.rollout_len:
            return                                        # still collecting
        self._update()                                    # rollout full -> train
        self.buffer = []                                  # ...then DISCARD it (on-policy)

    def _compute_gae(self, r, done, values, last_value):
        """Walk the rollout backwards to get per-step advantages + value targets."""
        N = len(r)
        adv = np.zeros(N, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(N)):
            next_v = last_value if t + 1 == N else values[t + 1]
            # (1 - done) zeroes the future term at episode boundaries inside the rollout
            delta = r[t] + self.gamma * next_v * (1.0 - done[t]) - values[t]
            gae   = delta + self.gamma * self.gae_lambda * (1.0 - done[t]) * gae
            adv[t] = gae
        returns = adv + np.asarray(values, dtype=np.float32)   # critic regresses toward this
        return adv, returns

    def compute_clip(self, logp_current, logp_old, A):
        ratio     = torch.exp(logp_current - logp_old)         # log-space -> exp of the DIFF
        unclipped = ratio * A
        clipped   = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * A
        return -torch.min(unclipped, clipped).mean()           # negative: we minimize the loss

    def _update(self):
        s, a, r, s2, done, logp_old, values = zip(*self.buffer)

        # Bootstrap the value of the state AFTER the last one (0 if it was terminal).
        with torch.no_grad():
            last = torch.as_tensor(s2[-1], dtype=torch.float32).unsqueeze(0)
            last_value = 0.0 if done[-1] else float(self.critic(last).squeeze().item())

        adv, returns = self._compute_gae(r, done, values, last_value)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)          # normalize advantages (standard PPO)

        s        = torch.as_tensor(np.array(s), dtype=torch.float32)   # (N, state_dim)
        a        = torch.as_tensor(a,           dtype=torch.int64)     # (N,)
        logp_old = torch.as_tensor(logp_old,    dtype=torch.float32)   # (N,)  — detached constant
        adv      = torch.as_tensor(adv,         dtype=torch.float32)   # (N,)
        returns  = torch.as_tensor(returns,     dtype=torch.float32)   # (N,)

        N = len(self.buffer)
        for _ in range(self.epochs):                           # K passes over THIS rollout
            idx = self.rng.permutation(N)                      # reshuffle every epoch
            for start in range(0, N, self.minibatch):
                mb = idx[start:start + self.minibatch]

                dist = torch.distributions.Categorical(logits=self.actor(s[mb]))
                logp_current = dist.log_prob(a[mb])            # RECOMPUTED each step, carries gradients

                actor_loss  = self.compute_clip(logp_current, logp_old[mb], adv[mb])
                value_pred  = self.critic(s[mb]).squeeze(-1)
                critic_loss = nn.functional.mse_loss(value_pred, returns[mb])
                entropy     = dist.entropy().mean()

                loss = actor_loss + self.vf_coef * critic_loss - self.entropy_coef * entropy
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

    def on_episode_end(self):
        pass    # PPO updates on the rollout boundary (inside observe), not per episode
