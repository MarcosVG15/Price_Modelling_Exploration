import os
import io
import contextlib
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_absolute_error

# make the repo root importable (config, helper_methods, FEATURE_PREDICTION)
# when this script is run directly from inside the FEATURE_PREDICTION folder
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from config import ASIN_COL
from FEATURE_PREDICTION.predictor import predictor, load_scale_params
from FEATURE_PREDICTION.sim_analysis_regression_SHAP_training import extract_target_data
from helper_methods.general import find_csv
from Matrix_Factorization_Clustering.Matrix_Factorization import matrix_factorization

'''
Head-to-head comparison of the two price-prediction approaches built on top of
the same product graph/embeddings:

  - "voting"  : predictor.predict_basic() - similarity-weighted average of the
                target's cluster neighbours' actual prices (no training, pure
                graph/embedding proximity - the SVD/community "voting" scheme).
  - "shap"    : predictor.predict_price_parsed_columns() - a HistGradientBoosting
                model trained on the parsed product features, with the price
                reconstructed additively from each feature's SHAP contribution.

Both run on the exact same held-out target products so R2/MAE/MAPE are
directly comparable.

LEAKAGE NOTE: target_data must be excluded from the pool BEFORE vn is fit and
BEFORE the SHAP model is trained (see build_train_pool()). Both underlying
methods otherwise let a target product see itself:
  - predict_basic(): find_cluster() folds target_data into an already-fit vn
    a second time and only excludes the *newly appended* duplicate from the
    neighbour search, not the product's original row already in the graph -
    so a product can vote for itself with ~1.0 similarity.
  - train_price_model(): dedupes training rows by ASIN keeping the *first*
    occurrence, which is that same original row - so the GBM can end up
    trained on the exact product/price it is later asked to predict.
'''


def build_train_pool(all_feature_data, target_data):
    """Remove every row that shares an ASIN with target_data, so vn and the
    GBM are fit with zero overlap against the held-out evaluation set."""
    target_asins = set(target_data[ASIN_COL])
    mask = ~all_feature_data[ASIN_COL].isin(target_asins)
    train_pool = all_feature_data[mask].reset_index(drop=True)
    assert not (set(train_pool[ASIN_COL]) & target_asins), "target ASIN leaked into train_pool"
    return train_pool


def compare_predictions(regressor, target_data, price_center, price_scale,
                        gamma=0.75, feature_types=None, price_params_path=None,
                        random_state=0):
    """Run both predictors over the same target_data and return a tidy
    DataFrame with one row per target product: actual price + both predictions.

    Order matters: the SHAP model is trained first, while regressor.feature_data
    is still the clean target-free pool. find_cluster() (needed for the voting
    method) appends target_data back into regressor.feature_data afterwards -
    running SHAP training after that point would reintroduce leak 2 above.

    random_state drives the GBM's train/test split AND the model itself
    (HistGradientBoostingRegressor), so a Monte Carlo run can vary it per trial."""

    shap_preds = regressor.predict_price_parsed_columns(
        feature_types=feature_types, price_params_path=price_params_path,
        log_price=True, random_state=random_state)

    if not hasattr(regressor, "query_clusters"):
        regressor.find_cluster()

    voting_preds = regressor.predict_basic(
        gamma=gamma, price_center=price_center, price_scale=price_scale)

    actual = target_data[("clean", "price")].to_numpy(dtype=float) * price_scale + price_center
    asins = target_data[ASIN_COL].to_numpy()
    markets = target_data[("clean", "market_id")].to_numpy()

    # both dicts are keyed differently (voting_preds by positional query index,
    # shap_preds by target_data's original row index) but preserve insertion
    # order == target_data row order, so zip by position for alignment.
    rows = []
    for i, ((_, voting_pred), (_, shap_info)) in enumerate(
            zip(voting_preds.items(), shap_preds.items())):
        rows.append({
            "asin":         asins[i],
            "market_id":    markets[i],
            "actual":       actual[i],
            "voting_pred":  voting_pred,
            "shap_pred":    shap_info["predicted"],
        })

    df = pd.DataFrame(rows)
    df["voting_error"] = df["voting_pred"] - df["actual"]
    df["shap_error"] = df["shap_pred"] - df["actual"]
    df["voting_abs_pct_err"] = (df["voting_error"] / df["actual"]).abs() * 100
    df["shap_abs_pct_err"] = (df["shap_error"] / df["actual"]).abs() * 100
    return df


METRIC_COLS = ["R2", "MAE", "RMSE", "MedAE", "MAPE", "sMAPE", "Bias_pct", "Hit20", "Spearman"]


def summarize_metrics(df, verbose=True):
    """Compute an accuracy + robustness + ranking comparison, dropping rows
    where a method couldn't produce a prediction (e.g. voting has no
    same-cluster neighbours).

      R2, MAE       - the originals: fit quality, average $ error.
      RMSE          - like MAE but squares errors first, so it is dragged up
                      by a few big misses (e.g. the seed=8 Voting Network
                      blow-up) instead of averaging them away like MAE does.
      MedAE         - median $ error: the "typical" miss, immune to that
                      same blow-up. MAE >> MedAE signals a fat-tailed error
                      distribution rather than uniformly-so-so predictions.
      MAPE, sMAPE   - MAPE blows up when the actual price is small (a $2
                      miss on a $5 product is "40%"); sMAPE divides by the
                      average of actual & predicted instead, so it stays
                      bounded in [0, 200] and is far less outlier-prone.
      Bias_pct      - signed mean % error: positive = systematically
                      over-priced, negative = under-priced. MAE/RMSE only
                      show error size, not direction.
      Hit20         - % of predictions within +-20% of actual: a practical
                      "close enough to be useful" business readout that
                      R2/MAE don't directly answer.
      Spearman      - rank correlation between predicted and actual price;
                      both methods are fundamentally about relative product
                      positioning, so this asks "does it at least get the
                      ordering right" even when absolute values are off.
    """
    metrics = {}
    for name, pred_col in [("Voting Network", "voting_pred"), ("SHAP (GBM)", "shap_pred")]:
        valid = df[["actual", pred_col]].dropna()
        if valid.empty:
            metrics[name] = {c: np.nan for c in METRIC_COLS}
            metrics[name]["n"] = 0
            continue

        actual, pred = valid["actual"], valid[pred_col]
        err = pred - actual
        abs_err = err.abs()
        pct_err = err / actual * 100

        metrics[name] = {
            "R2":       r2_score(actual, pred),
            "MAE":      mean_absolute_error(actual, pred),
            "RMSE":     float(np.sqrt(np.mean(err ** 2))),
            "MedAE":    float(abs_err.median()),
            "MAPE":     float(pct_err.abs().mean()),
            "sMAPE":    float((abs_err / ((actual.abs() + pred.abs()) / 2)).mean() * 100),
            "Bias_pct": float(pct_err.mean()),
            "Hit20":    float((pct_err.abs() <= 20).mean() * 100),
            "Spearman": float(spearmanr(actual, pred).correlation) if len(valid) > 1 else np.nan,
            "n":        len(valid),
        }

    if verbose:
        cols = ["n"] + METRIC_COLS
        header = f"{'model':>16} | " + " | ".join(f"{c:>9}" for c in cols)
        print(f"\n{header}")
        print("-" * len(header))
        for name, m in metrics.items():
            vals = " | ".join(f"{m[c]:>9.3f}" if c != "n" else f"{m[c]:>9d}" for c in cols)
            print(f"{name:>16} | {vals}")

    return metrics


# ---------------------------------------------------------------------- #
#  Monte Carlo: repeat the whole comparison over many random target
#  samples / GBM random states so the metrics above can be read as a
#  distribution (how much they fluctuate) instead of a single lucky/unlucky
#  draw.
# ---------------------------------------------------------------------- #
def run_trial(all_feature_data, search_term, target_n, seed, price_center, price_scale,
             gamma=0.75, quiet=True):
    """One Monte Carlo trial: resample target_data with `seed`, rebuild a
    leak-free train_pool, refit vn, and train the GBM with `seed` as its
    random state. Returns {"Voting Network": {...}, "SHAP (GBM)": {...}}."""
    buf = io.StringIO()
    ctx = contextlib.redirect_stdout(buf) if quiet else contextlib.nullcontext()
    try:
        with ctx:
            target_data = extract_target_data(all_feature_data, n=target_n, random_state=seed)
            train_pool = build_train_pool(all_feature_data, target_data)

            vn = matrix_factorization(train_pool)
            vn.matrix_factorization_tf_idf()
            vn.svd_product_communities(k=1, resolution=1)

            regressor = predictor(k=1, feature_data=train_pool, target_data=target_data, vn=vn)

            comparison = compare_predictions(
                regressor, target_data, price_center, price_scale, gamma=gamma,
                feature_types=f"data_files/feature_types_{search_term}.json",
                price_params_path=f"data_files/all_feature_data_{search_term}.params.json",
                random_state=seed)

            metrics = summarize_metrics(comparison, verbose=False)
    except Exception:
        print(buf.getvalue())   # surface what happened before the crash
        raise
    return metrics


def monte_carlo(all_feature_data, search_term, price_center, price_scale,
                n_trials=15, target_n=50, base_seed=0, gamma=0.75):
    """Run n_trials independent trials (each with its own target sample +
    GBM random state) and return a long-format DataFrame: one row per
    (trial, model) with that trial's R2/MAE/MAPE."""
    records = []
    for trial in range(n_trials):
        seed = base_seed + trial
        metrics = run_trial(all_feature_data, search_term, target_n, seed,
                            price_center, price_scale, gamma=gamma)
        for model_name, m in metrics.items():
            records.append({"trial": trial, "seed": seed, "model": model_name, **m})

        # keep the per-trial console line short; the full metric set (incl.
        # RMSE/MedAE/sMAPE/Bias/Hit20/Spearman) is in the returned DataFrame
        line = " | ".join(f"{name}: R2={m['R2']:+.3f} MAE=${m['MAE']:.2f} Hit20={m['Hit20']:.0f}%"
                          for name, m in metrics.items())
        print(f"trial {trial + 1:>2}/{n_trials} (seed={seed}) -> {line}")

    return pd.DataFrame.from_records(records)


def summarize_monte_carlo(results):
    """Print mean/std/min/max per model across all trials - the fluctuation
    the individual trial lines above only hint at."""
    agg = results.groupby("model")[METRIC_COLS].agg(["mean", "std", "min", "max"])
    print(f"\nMonte Carlo summary over {results['trial'].nunique()} trials:")
    print(agg.to_string(float_format=lambda v: f"{v:,.3f}"))
    return agg


# the subset actually worth eyeballing as boxplots (Bias_pct/Spearman still
# come out of summarize_monte_carlo() above, just not plotted here to keep
# the grid readable)
PLOT_METRICS = ["R2", "MAE", "RMSE", "MedAE", "sMAPE", "Hit20"]


def plot_monte_carlo(results, out_dir="prediction_comparison_charts",
                     filename="monte_carlo_shap_vs_voting.png"):
    os.makedirs(out_dir, exist_ok=True)
    palette = {"Voting Network": "tab:orange", "SHAP (GBM)": "tab:blue"}

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"Monte Carlo spread across {results['trial'].nunique()} trials "
                f"(varying target sample + GBM random state)", fontsize=14)

    for ax, metric in zip(axes.flat, PLOT_METRICS):
        sns.boxplot(data=results, x="model", y=metric, hue="model", ax=ax,
                   palette=palette, legend=False)
        sns.stripplot(data=results, x="model", y=metric, ax=ax,
                     color="black", alpha=0.5, size=4)
        ax.set_title(metric)
        ax.set_xlabel("")

    plt.tight_layout()
    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved Monte Carlo chart -> {out_path}")


def plot_comparison(df, metrics, out_dir="prediction_comparison_charts", filename="shap_vs_voting.png"):
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("SHAP price model vs. Voting Network price prediction", fontsize=15)

    lo = df[["actual", "voting_pred", "shap_pred"]].min().min()
    hi = df[["actual", "voting_pred", "shap_pred"]].max().max()

    ax = axes[0]
    ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=1, label="perfect")
    ax.scatter(df["actual"], df["voting_pred"], alpha=0.7, label="Voting Network", color="tab:orange")
    ax.scatter(df["actual"], df["shap_pred"], alpha=0.7, label="SHAP (GBM)", color="tab:blue")
    ax.set_xlabel("Actual price ($)")
    ax.set_ylabel("Predicted price ($)")
    ax.set_title("Predicted vs. actual")
    ax.legend()

    ax2 = axes[1]
    sns.histplot(df["voting_abs_pct_err"].dropna(), color="tab:orange", label="Voting Network",
                kde=False, ax=ax2, alpha=0.5, bins=20)
    sns.histplot(df["shap_abs_pct_err"].dropna(), color="tab:blue", label="SHAP (GBM)",
                kde=False, ax=ax2, alpha=0.5, bins=20)
    ax2.set_xlabel("Absolute % error")
    ax2.set_title("Error distribution")
    ax2.legend()

    plt.tight_layout()
    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved comparison chart -> {out_path}")


if __name__ == "__main__":
    search_term = "Headphones"
    N_TRIALS = 15
    TARGET_N = 50
    BASE_SEED = 0

    path = find_csv(search_term)
    all_feature_data = pd.read_csv(path, header=[0, 1], low_memory=False)
    all_feature_data = all_feature_data.drop(columns=all_feature_data.columns[0])

    price_center, price_scale = load_scale_params(f"data_files/all_feature_data_{search_term}.params.json")

    # ---- single run: per-product table + predicted-vs-actual / error plot ----
    target_data = extract_target_data(all_feature_data, n=TARGET_N, random_state=BASE_SEED)
    train_pool = build_train_pool(all_feature_data, target_data)

    vn = matrix_factorization(train_pool)
    vn.matrix_factorization_tf_idf()
    vn.svd_product_communities(k=1, resolution=1)

    regressor = predictor(k=1, feature_data=train_pool, target_data=target_data, vn=vn)

    comparison = compare_predictions(
        regressor, target_data, price_center, price_scale, gamma=0.75,
        feature_types=f"data_files/feature_types_{search_term}.json",
        price_params_path=f"data_files/all_feature_data_{search_term}.params.json",
        random_state=BASE_SEED)

    print(f"\nPer-product comparison (single run, seed={BASE_SEED}):")
    print(comparison[["asin", "market_id", "actual", "voting_pred", "shap_pred"]]
          .to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    single_run_metrics = summarize_metrics(comparison)
    plot_comparison(comparison, single_run_metrics)

    # ---- monte carlo: repeat across many seeds to see how much the above
    #      per-product snapshot (and its metrics) actually fluctuates ----
    results = monte_carlo(all_feature_data, search_term, price_center, price_scale,
                         n_trials=N_TRIALS, target_n=TARGET_N, base_seed=BASE_SEED)

    summarize_monte_carlo(results)
    plot_monte_carlo(results)
