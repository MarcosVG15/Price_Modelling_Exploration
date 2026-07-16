"""
Backtest the demand model against real SP-API history for ONE own-catalog product.

Only a handful of products in the feature CSV actually exist in the SP-API tables
(the folded-in own catalog) -- everything else is scraped competitors with no units
history. This script picks one such product (the best-covered by default, or an ASIN
you pass), then, per Amazon marketplace and per month, compares:

    real units sold        (sales_traffic_daily.units_ordered)
    predicted units        (real sessions  x  model CVR(real price))
    real selling price      (ordered_product_sales / units_ordered)

Why "real sessions x model CVR" and not the model's own volume: the demand model only
learned conversion-vs-price. Traffic volume is exogenous (market_size is a hardcoded
1000, uncalibrated), so we hold sessions at their real value and test the one thing the
model predicts -- the conversion curve. Predicted units are then the fair, apples-to-
apples counterpart of the actual units.

Run:
    python GAME_THEORY_PREDICTION/backtest.py            # auto-pick best-covered own product
    python GAME_THEORY_PREDICTION/backtest.py B07KX7V4N3 # a specific own ASIN

Outputs (under "<repo>/prediction vs real/"):
    <ASIN>_prediction_vs_real.png    per-marketplace panels
    <ASIN>_prediction_vs_real.csv    monthly table, one block of columns per marketplace
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # headless: save to file, never open a window
import matplotlib.pyplot as plt
from sqlalchemy import text

from market_env import MarketEnv, _api_engine
from main import build_vn

ASIN_COL = ("clean", "asin")
OUT_DIR = os.path.join(ROOT, "prediction vs real")

# Amazon EU marketplace IDs -> readable country codes (all eurozone, so no FX needed).
MARKETPLACE_NAMES = {
    "A1PA6795UKMFR9": "DE",
    "A13V1IB3VIYZZH": "FR",
    "APJ6JRA9NG5V4":  "IT",
    "A1805IZSGTT6HS": "ES",
    "A1RKKUPIHCS9HS": "NL",
    "AMEN7PMS3EDWL":  "BE",
}


def _own_asins_in_csv(csv_asins):
    """The ASINs that are both in the feature CSV and in the SP-API history."""
    with _api_engine().connect() as c:
        st = pd.read_sql(text("SELECT DISTINCT asin FROM sales_traffic_daily"), c)
    return sorted(set(csv_asins) & set(st["asin"].astype(str)))


def pick_backtest_target(vn, override_asin=None):
    """Choose an own-catalog ASIN that has real history. Default: the one with the
    most total units sold (the strongest signal to compare against)."""
    csv_asins = set(vn.feature_data[ASIN_COL].dropna().astype(str))
    candidates = _own_asins_in_csv(csv_asins)
    if not candidates:
        raise SystemExit("No CSV product exists in the SP-API history; nothing to backtest.")

    if override_asin is not None:
        if override_asin not in candidates:
            raise SystemExit(
                f"{override_asin} is not an own-catalog product with SP-API history.\n"
                f"Backtestable ASINs: {candidates}")
        return override_asin

    al = "','".join(candidates)
    with _api_engine().connect() as c:
        tot = pd.read_sql(text(
            f"SELECT asin, SUM(units_ordered) units FROM sales_traffic_daily "
            f"WHERE asin IN ('{al}') GROUP BY asin ORDER BY units DESC"), c)
    best = str(tot.iloc[0]["asin"])
    print(f"[backtest] backtestable own ASINs: {candidates}")
    print(f"[backtest] auto-selected '{best}' (most units sold: {int(tot.iloc[0]['units'])})")
    return best


def cluster_of_asin(vn, asin):
    """Positional cluster label of an ASIN already present in the trained vn."""
    asins = vn.feature_data[ASIN_COL].astype(str).to_numpy()
    idx = np.where(asins == asin)[0]
    if idx.size == 0:
        raise SystemExit(f"{asin} not found in the feature CSV.")
    return int(vn.product_labels[idx[0]])


def load_history(asin):
    """Monthly real units / sessions / revenue / price per marketplace for one ASIN."""
    with _api_engine().connect() as c:
        df = pd.read_sql(text("""
            SELECT marketplace_id, data_date, units_ordered, sessions,
                   ordered_product_sales, unit_session_percentage
            FROM sales_traffic_daily
            WHERE asin = :asin
            ORDER BY data_date
        """), c, params={"asin": asin})

    if df.empty:
        raise SystemExit(f"No sales_traffic_daily history for {asin}.")

    df["data_date"] = pd.to_datetime(df["data_date"])
    for col in ["units_ordered", "sessions", "ordered_product_sales"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Aggregate to ISO week per marketplace (Mon-Sun, labelled by the Sunday). Weekly
    # is the model's native cadence -- HORIZON=52 and the seasonal index are per ISO
    # week -- so this is the resolution the demand model actually predicts at.
    weekly = (df.groupby(["marketplace_id", pd.Grouper(key="data_date", freq="W-SUN")])
                .agg(real_units=("units_ordered", "sum"),
                     sessions=("sessions", "sum"),
                     revenue=("ordered_product_sales", "sum"))
                .reset_index())

    # Real selling price = revenue / units, only where something actually sold.
    weekly["real_price"] = np.where(
        weekly["real_units"] > 0,
        weekly["revenue"] / weekly["real_units"].replace(0, np.nan),
        np.nan)
    return weekly


def add_predictions(weekly, env):
    """Predicted units = real sessions x model CVR(price). For weeks with sessions
    but no sale (hence no observed price) we still need a price to score the model,
    so we fall back to the product's own median selling price, then the cluster
    reference price -- the real_price column itself is left blank on those weeks."""
    med_price = weekly["real_price"].median(skipna=True)
    if not np.isfinite(med_price):
        med_price = env.params.get("reference_price", 1.0)

    price_for_model = weekly["real_price"].fillna(med_price)
    weekly["pred_cvr"] = [env.predict_cvr(float(p)) for p in price_for_model]
    weekly["pred_units"] = weekly["sessions"] * weekly["pred_cvr"]
    weekly["real_cvr"] = np.where(weekly["sessions"] > 0,
                                  weekly["real_units"] / weekly["sessions"].replace(0, np.nan),
                                  np.nan)
    return weekly


def plot(weekly, asin, cluster_id):
    mkts = [m for m in MARKETPLACE_NAMES if m in set(weekly["marketplace_id"])]
    # keep any unexpected marketplace ids too, appended after the known ones
    mkts += [m for m in weekly["marketplace_id"].unique() if m not in MARKETPLACE_NAMES]

    n = len(mkts)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.4 * nrows), squeeze=False)

    for i, mkt in enumerate(mkts):
        ax = axes[i // ncols][i % ncols]
        d = weekly[weekly["marketplace_id"] == mkt].sort_values("data_date")
        x = d["data_date"]

        # Left axis: units (real bars vs predicted line)
        ax.bar(x, d["real_units"], width=5, color="#4C78A8", alpha=0.75, label="real units")
        ax.plot(x, d["pred_units"], color="#E45756", marker="o", ms=3, lw=1.6, label="predicted units")
        ax.set_ylabel("units / week")
        ax.set_title(f"{MARKETPLACE_NAMES.get(mkt, mkt)}  ·  real total = {int(d['real_units'].sum())}")
        ax.tick_params(axis="x", rotation=45, labelsize=8)

        # Right axis: real selling price -- only if this marketplace ever sold a
        # unit (otherwise revenue/units is undefined and the axis would auto-scale
        # to a meaningless range).
        h1, l1 = ax.get_legend_handles_labels()
        if d["real_price"].notna().any():
            ax2 = ax.twinx()
            ax2.plot(x, d["real_price"], color="#54A24B", marker="s", ms=4, lw=1.2,
                     ls="--", label="real price")
            ax2.set_ylabel("real price (EUR)", color="#54A24B")
            ax2.tick_params(axis="y", labelcolor="#54A24B")
            h2, l2 = ax2.get_legend_handles_labels()
            h1, l1 = h1 + h2, l1 + l2
        ax.legend(h1, l1, fontsize=8, loc="upper left")

    # blank any unused panels
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(
        f"Prediction vs real — {asin} (cluster {cluster_id})\n"
        f"predicted units = real sessions × model CVR(price)",
        fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(OUT_DIR, exist_ok=True)
    png = os.path.join(OUT_DIR, f"{asin}_prediction_vs_real.png")
    fig.savefig(png, dpi=130)
    plt.close(fig)
    return png


def save_table(weekly, asin):
    """Wide table: one block of columns (real_units / pred_units / real_price) per
    marketplace, indexed by ISO week -- 'a column for each marketplace and the price'."""
    m = weekly.copy()
    m["mkt"] = m["marketplace_id"].map(lambda x: MARKETPLACE_NAMES.get(x, x))
    m["week"] = m["data_date"].dt.strftime("%G-W%V")
    wide = m.pivot_table(
        index="week", columns="mkt",
        values=["real_units", "pred_units", "real_price"], aggfunc="first")
    # flatten ('real_units','DE') -> 'DE_real_units'
    wide.columns = [f"{mkt}_{metric}" for metric, mkt in wide.columns]
    wide = wide.reindex(sorted(wide.columns), axis=1)

    os.makedirs(OUT_DIR, exist_ok=True)
    csv = os.path.join(OUT_DIR, f"{asin}_prediction_vs_real.csv")
    wide.round(3).to_csv(csv)
    return csv


def main():
    override = sys.argv[1].strip() if len(sys.argv) > 1 else None

    print("building vn ...")
    vn, _ = build_vn()

    asin = pick_backtest_target(vn, override)
    cluster_id = cluster_of_asin(vn, asin)
    env = MarketEnv.for_cluster(vn, cluster_id)   # triggers/loads the global CVR model
    print(f"[backtest] ASIN={asin}  cluster={cluster_id}  "
          f"reference_price={env.params.get('reference_price'):.2f}  "
          f"CVR_model={'fitted' if MarketEnv._CVR_MODEL is not None else 'CONSTANT 0.03 fallback'}")

    weekly = add_predictions(load_history(asin), env)

    # console summary per marketplace
    summ = (weekly.groupby("marketplace_id")
                  .agg(real=("real_units", "sum"), pred=("pred_units", "sum"))
                  .rename(index=MARKETPLACE_NAMES))
    print("\n=== totals per marketplace (real vs predicted units) ===")
    print(summ.round(1).to_string())

    png = plot(weekly, asin, cluster_id)
    csv = save_table(weekly, asin)
    print(f"\nsaved: {png}")
    print(f"saved: {csv}")


if __name__ == "__main__":
    main()
