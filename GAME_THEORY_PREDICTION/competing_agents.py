

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd

from config import ASIN_COL
from Matrix_Factorization_Clustering.Matrix_Factorization import matrix_factorization
from game import pricing_game



def build_vn(search_term="Headphones", k=1, resolution=1.0):
    path = os.path.join(ROOT, f"data_files/all_feature_data_{search_term}.csv")
    data = pd.read_csv(path, header=[0, 1], low_memory=False)

    print(data.head())
    data = data.drop(columns=data.columns[0])
    filtered_df = data[data[('clean', 'market_id')] == 1]
    vn = matrix_factorization(filtered_df)
    vn.matrix_factorization_tf_idf()
    vn.svd_product_communities(k=k, resolution=resolution)
    return vn, filtered_df


def pick_target(data, random_state=2):
    unique = data.drop_duplicates(subset=[ASIN_COL])
    rng = np.random.default_rng(random_state)
    i = int(rng.integers(len(unique)))
    return unique.iloc[[i]]



def train_solo(target, vn, agent_type="PPO", competitor="promo_cycler",
               episodes=2000, seed=0):
    """Train ONE agent as a solo pricer against a *scripted* (stationary)
    competitor -- the setting where it learns cleanly. Returns the trained
    pricing_game; its ``.agent`` holds the learned policy."""
    g = pricing_game(target, vn, competitor_strategy=competitor, seed=seed)
    print(f"  training solo {agent_type} (seed={seed}) vs '{competitor}' ...")
    g.train(episodes=episodes, type=agent_type)
    return g


# ----------------------------------------------------------------------------
# PHASE 2 -- freeze both agents and play one head-to-head season
# ----------------------------------------------------------------------------
def play_match(env, target, agent_us, agent_them, start_week=1, seed=0):
    """One season, head-to-head. ``agent_us`` prices OUR product; ``agent_them``
    is dropped into the competitor slot. Both act GREEDILY and neither learns
    (``competitor_learning=False`` freezes the opponent; we call choose with
    explore=False). Returns both price paths and per-week profits.

    Note: an agent trained as "us" plugs straight into the competitor role --
    _state() and _competitor_state() are the same [my price, opp price, my
    buy-box, season] layout from each player's own point of view.
    """
    env.competitor_agent = agent_them
    prev_learning = env.competitor_learning
    env.competitor_learning = False               # freeze opponent: greedy, no learning
    state = env.reset(target, start_week=start_week, seed=seed, competitor_strategy="RL")
    done = False
    rec = {"our_price": [], "comp_price": [], "our_profit": [], "comp_profit": [], "week": []}
    try:
        while not done:
            a = agent_us.choose(state, explore=False)      # our greedy action
            state, r, done = env.step(a)                   # competitor moves greedily inside step()
            rec["our_price"].append(env.own_price)
            rec["comp_price"].append(env.comp_price)
            rec["our_profit"].append(r)
            rec["comp_profit"].append(env.last_comp_profit)
            rec["week"].append(env.iso_week)
    finally:
        env.competitor_learning = prev_learning
    rec["our_total"] = float(np.sum(rec["our_profit"]))
    rec["comp_total"] = float(np.sum(rec["comp_profit"]))
    return rec


def plot_match(rec, tag, path=None):
    """Two-panel figure: prices (top) and cumulative profit (bottom) over the
    season. Saves under two_player_game_plts/ unless ``path`` is given."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    weeks = np.arange(1, len(rec["our_price"]) + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    ax1.plot(weeks, rec["our_price"],  lw=2, label="our price")
    ax1.plot(weeks, rec["comp_price"], lw=2, label="competitor price")
    ax1.set_ylabel("price")
    ax1.set_title(f"Head-to-head match (both frozen) — {tag}")
    ax1.legend(loc="best", fontsize=8)

    ax2.plot(weeks, np.cumsum(rec["our_profit"]),  lw=2,
             label=f"our cumulative profit  (${rec['our_total']:,.0f})")
    ax2.plot(weeks, np.cumsum(rec["comp_profit"]), lw=2,
             label=f"competitor cumulative profit  (${rec['comp_total']:,.0f})")
    ax2.set_xlabel("prediction week")
    ax2.set_ylabel("cumulative profit")
    ax2.legend(loc="best", fontsize=8)

    fig.tight_layout()
    if path is None:
        out_dir = os.path.join(HERE, "two_player_game_plts")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"match_{tag}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main(episodes=2000):
    print("building vn ")
    vn, data = build_vn()

    print("getting target ")
    target = pick_target(data)
    asin_col = ("clean", "asin")
    target_asin = target[asin_col].iloc[0] if asin_col in target.columns else None
    print("target ASIN:", target_asin if pd.notna(target_asin) else "unresolved")

    # --- PHASE 1: two agents, trained INDEPENDENTLY as solo pricers ---
    # (Different torch weight inits -> genuinely distinct policies, even though
    #  both are solved against the same scripted competitor.)
    print("\n--- PHASE 1: train two solo pricers (vs scripted competitor) ---")
    game_a = train_solo(target, vn, agent_type="PPO", episodes=episodes, seed=0)
    game_b = train_solo(target, vn, agent_type="PPO", episodes=episodes, seed=1)

    # Both live on the same cluster env (cached per cluster); use it for the match.
    env = game_a.env

    # --- PHASE 2: freeze both and let them compete ---
    print("\n--- PHASE 2: head-to-head match (both frozen, greedy) ---")
    rec = play_match(env, target, agent_us=game_a.agent, agent_them=game_b.agent,
                     start_week=1, seed=0)
    print(f"  our season profit:        ${rec['our_total']:,.0f}")
    print(f"  competitor season profit: ${rec['comp_total']:,.0f}")
    print(f"  final prices -> us: {rec['our_price'][-1]:.2f}   "
          f"competitor: {rec['comp_price'][-1]:.2f}")

    tag = str(target_asin) if pd.notna(target_asin) else f"cluster{game_a.cluster_id}"
    out = plot_match(rec, tag=tag)
    print(f"  saved match plot -> {out}")


if __name__ == "__main__":
    main()
