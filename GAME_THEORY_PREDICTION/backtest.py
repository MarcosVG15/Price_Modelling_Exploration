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
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import roc_curve, auc

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
                   ordered_product_sales, unit_session_percentage, buy_box_percentage
            FROM sales_traffic_daily
            WHERE asin = :asin
            ORDER BY data_date
        """), c, params={"asin": asin})

    if df.empty:
        raise SystemExit(f"No sales_traffic_daily history for {asin}.")

    df["data_date"] = pd.to_datetime(df["data_date"])
    for col in ["units_ordered", "sessions", "ordered_product_sales"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    # Buy-box % is a rate, not a count -- keep NaN (don't zero-fill) so the weekly
    # mean averages only days that actually reported a buy-box percentage.
    df["buy_box_percentage"] = pd.to_numeric(df["buy_box_percentage"], errors="coerce")

    # Aggregate to ISO week per marketplace (Mon-Sun, labelled by the Sunday). Weekly
    # is the model's native cadence -- HORIZON=52 and the seasonal index are per ISO
    # week -- so this is the resolution the demand model actually predicts at. Buy-box %
    # is averaged within the week (matching the dashboard's simple mean, just bucketed).
    weekly = (df.groupby(["marketplace_id", pd.Grouper(key="data_date", freq="W-SUN")])
                .agg(real_units=("units_ordered", "sum"),
                     sessions=("sessions", "sum"),
                     revenue=("ordered_product_sales", "sum"),
                     real_buybox=("buy_box_percentage", "mean"))
                .reset_index())

    # Sessions-report proxy price (revenue / units) -- kept only as a fallback. It's a
    # weekly average that barely moves, which is why the price x buy-box plane was flat.
    weekly["real_price_stx"] = np.where(
        weekly["real_units"] > 0,
        weekly["revenue"] / weekly["real_units"].replace(0, np.nan),
        np.nan)

    # REAL realized unit price from order_items (line total / qty), joined to the order's
    # purchase_date. This is the actual price charged -- 2 years deep with genuine week-to-
    # week movement -- so it's the preferred own-price for the plane and both models.
    with _api_engine().connect() as c:
        oi = pd.read_sql(text("""
            SELECT o.marketplace_id, o.purchase_date, oi.item_price, oi.quantity_ordered
            FROM order_items oi
            JOIN orders o ON o.amazon_order_id = oi.amazon_order_id
            WHERE oi.asin = :asin AND oi.item_price IS NOT NULL AND oi.quantity_ordered > 0
        """), c, params={"asin": asin})

    if not oi.empty:
        # purchase_date is timestamptz (UTC); strip tz so its W-SUN weeks align with the
        # naive data_date weeks from sales_traffic_daily (else the merge dtype-clashes).
        oi["purchase_date"] = pd.to_datetime(oi["purchase_date"], utc=True).dt.tz_localize(None)
        oi["item_price"] = pd.to_numeric(oi["item_price"], errors="coerce")
        oi["quantity_ordered"] = pd.to_numeric(oi["quantity_ordered"], errors="coerce")
        opw = (oi.groupby(["marketplace_id", pd.Grouper(key="purchase_date", freq="W-SUN")])
                 .agg(_rev=("item_price", "sum"), _qty=("quantity_ordered", "sum"))
                 .reset_index()
                 .rename(columns={"purchase_date": "data_date"}))
        opw["order_price"] = opw["_rev"] / opw["_qty"].replace(0, np.nan)
        weekly = weekly.merge(opw[["marketplace_id", "data_date", "order_price"]],
                              on=["marketplace_id", "data_date"], how="left")
    else:
        weekly["order_price"] = np.nan

    # Prefer the real order price; fall back to the sessions-report proxy where absent.
    weekly["real_price"] = weekly["order_price"].fillna(weekly["real_price_stx"])
    return weekly


def add_predictions(weekly, env):
   

    med_price = weekly["real_price"].median(skipna=True)
    if not np.isfinite(med_price):
        med_price = env.params.get("reference_price", 1.0)

    price_for_model = weekly["real_price"].fillna(med_price)
  

    weeks = weekly["data_date"].dt.isocalendar().week.astype(int).clip(1, 52)

    # Buy-box share (0-1) first: it feeds BOTH the chained CVR prediction and the
    # displayed pred_buybox, so compute it once against the fitted competitor reference.
    comp_ref = env._competitor_reference()
    bb_share = [env._buybox_prob(float(p), comp_ref, week=int(w))
                for p, w in zip(price_for_model, weeks)]
    weekly["pred_buybox"] = [100.0 * b for b in bb_share]     # x100 to compare with real buy_box_percentage

    weekly["pred_cvr"] = [env.predict_cvr(float(p), b, week=int(w))
                          for p, b, w in zip(price_for_model, bb_share, weeks)]
    weekly["pred_units"] = weekly["sessions"] * weekly["pred_cvr"]
    
    n_draws = 500
    draws = np.array([[env._draw_demand(m) for m in weekly["pred_units"]]
                      for _ in range(n_draws)])                 # (n_draws, n_weeks)
    weekly["pred_units_p90"] = np.percentile(draws, 90, axis=0)
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
        ax.fill_between(x, 0, d["pred_units_p90"], color="#E45756", alpha=0.15,
                        label="NB realised (0–90%)")
        ax.plot(x, d["pred_units"], color="#E45756", marker="o", ms=3, lw=1.6, label="predicted units (expected)")
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


def plot_buybox(weekly, asin, cluster_id):
    """REAL buy-box percentage over time, one line per marketplace -- the weekly
    average of sales_traffic_daily.buy_box_percentage (Amazon's Featured-Offer %),
    bucketed by ISO week and split by marketplace_id. NaN weeks (no reported buy-box)
    are simply gaps in a line."""
    bb = weekly.copy()
    # Normalise to a 0-100 % axis (Amazon reports 0-100; be robust to a 0-1 fraction).
    valid = bb["real_buybox"].dropna()
    if not valid.empty and valid.max() <= 1.5:
        bb["real_buybox"] = bb["real_buybox"] * 100.0

    mkts = [m for m in MARKETPLACE_NAMES if m in set(bb["marketplace_id"])]
    mkts += [m for m in bb["marketplace_id"].unique() if m not in MARKETPLACE_NAMES]

    fig, ax = plt.subplots(figsize=(11, 5))
    for mkt in mkts:
        d = bb[bb["marketplace_id"] == mkt].sort_values("data_date")
        name = MARKETPLACE_NAMES.get(mkt, mkt)
        line, = ax.plot(d["data_date"], d["real_buybox"], marker="o", ms=3, lw=1.8,
                        label=f"{name} real")
        if "pred_buybox" in d:            # model's predicted % (dashed, same colour)
            ax.plot(d["data_date"], d["pred_buybox"], ls="--", lw=1.4,
                    color=line.get_color(), label=f"{name} predicted")
    ax.set_ylim(0, 100)
    ax.set_xlabel("week")
    ax.set_ylabel("buy-box % (weekly avg)")
    ax.set_title(f"Buy-box % — predicted (dashed) vs real (solid) — {asin} (cluster {cluster_id})")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.legend(fontsize=7, loc="best", ncol=2)
    fig.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    png = os.path.join(OUT_DIR, f"{asin}_buybox_evolution.png")
    fig.savefig(png, dpi=130)
    plt.close(fig)
    return png


def plot_buybox_price_plane(weekly, asin, cluster_id):
    """Buy-box vs price as a 2D plane: each marketplace's weekly (price, buy-box %) points,
    connected in time order and coloured by ISO week. Reveals the CO-EVOLUTION -- a path
    that drifts down-and-right means "priced up -> lost the buy-box". Only weeks that sold
    (real price known) place a point. One panel per marketplace with real price movement."""
    bb = weekly.copy()
    valid = bb["real_buybox"].dropna()
    if not valid.empty and valid.max() <= 1.5:
        bb["real_buybox"] = bb["real_buybox"] * 100.0
    bb = bb.dropna(subset=["real_price", "real_buybox"])
    # keep only marketplaces with at least 2 distinct prices (a trajectory to see)
    mkts = [m for m in list(MARKETPLACE_NAMES) + list(bb["marketplace_id"].unique())
            if m in set(bb["marketplace_id"]) and bb.loc[bb["marketplace_id"] == m, "real_price"].nunique() >= 2]
    mkts = list(dict.fromkeys(mkts))
    if not mkts:
        return None

    n = len(mkts); ncols = min(3, n); nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 4.6 * nrows), squeeze=False)
    wk_all = bb["data_date"].dt.isocalendar().week.astype(int)
    norm = plt.Normalize(int(wk_all.min()), int(wk_all.max()))
    sc = None
    for i, mkt in enumerate(mkts):
        ax = axes[i // ncols][i % ncols]
        d = bb[bb["marketplace_id"] == mkt].sort_values("data_date")
        wk = d["data_date"].dt.isocalendar().week.astype(int)
        ax.plot(d["real_price"], d["real_buybox"], color="0.75", lw=0.9, zorder=1)   # time trajectory
        sc = ax.scatter(d["real_price"], d["real_buybox"], c=wk, cmap="viridis",
                        norm=norm, s=45, zorder=2, edgecolor="k", linewidth=0.3)
        ax.set_title(MARKETPLACE_NAMES.get(mkt, mkt))
        ax.set_xlabel("real selling price (EUR)")
        ax.set_ylabel("buy-box %")
        ax.set_ylim(-2, 102)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    if sc is not None:
        fig.colorbar(sc, ax=axes.ravel().tolist(), label="ISO week", shrink=0.7)
    fig.suptitle(f"Buy-box × price plane (colour = week) — {asin} (cluster {cluster_id})", fontsize=13)

    os.makedirs(OUT_DIR, exist_ok=True)
    png = os.path.join(OUT_DIR, f"{asin}_buybox_price_plane.png")
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
    env = MarketEnv.for_cluster(vn, cluster_id, target_asin=asin)   # triggers/loads the global CVR model
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
    bb_png = plot_buybox(weekly, asin, cluster_id)         # buy-box vs week
    plane_png = plot_buybox_price_plane(weekly, asin, cluster_id)  # buy-box x price plane
    csv = save_table(weekly, asin)
    cvr_png = env.plot_cvr_seasonality()          # learned seasonal conversion curve
    print(f"\nsaved: {png}")
    print(f"saved: {bb_png}")
    if plane_png:
        print(f"saved: {plane_png}")
    else:
        print("  (no buy-box x price plane: no marketplace had >=2 distinct real prices)")
    print(f"saved: {csv}")
    print(f"saved: {cvr_png}")


def main2():
    override = sys.argv[1].strip() if len(sys.argv) > 1 else None

    vn, _ = build_vn()
    asin = pick_backtest_target(vn, override)
    cluster_id = cluster_of_asin(vn, asin)
    env = MarketEnv.for_cluster(vn, cluster_id, target_asin=asin)   # triggers/loads the global CVR model
    X_test, Y_test, bbox_model  = MarketEnv.fit_cvr()
    output = bbox_model.predict(X_test)

    mae = mean_absolute_error(Y_test, output)
    r2 = r2_score(Y_test, output)

    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared (R2) Score: {r2:.4f}")


    Y_test_binary = (Y_test > 0.5).astype(int)

    # 2. Now you can run the ROC curve because Y_test_binary only contains 0s and 1s
    fpr, tpr, thresholds = roc_curve(Y_test_binary, output)
    AUC = auc(fpr, tpr)
    print("AUC is : ", AUC)



if __name__ == "__main__":
    # main()
    main2()