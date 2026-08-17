"""
Exploratory plots only -- NOT a model. Purpose: eyeball whether buy-box share visibly
tracks price, est_margin, or own_landed price, per ASIN, before encoding any relationship
as a hand-set rule (see conversation -- train_bbox_two_stage.py's fixed-effects residual
already found ~0 R^2 for price; this is "go look at the raw rows yourself" instead of
trusting one more aggregate number).

Two views per selected ASIN, one figure:
  - top:    normalized time series -- buybox_pct, price, est_margin, own_landed all min-max
            scaled to [0,1] so DIFFERENTLY-SCALED series (buybox_pct in [0,100], price in
            EUR, est_margin ~EUR too) can be compared for co-movement/shape, with each
            series' real range in the legend since the plot itself only shows shape.
  - bottom: buybox_pct (raw, 0-100) vs each of price/est_margin/own_landed directly, with a
            linear trend line + Pearson r annotated -- the actual "is there a visible trend"
            check, per ASIN.

Only ASINs with enough history AND enough of their own price movement are plotted -- an
ASIN that never repriced is a vertical line of points on the scatter panels and tells you
nothing about a price-vs-buybox trend (same eligibility bar train_bbox_two_stage.py's
within-ASIN check used: >=8 priced weeks, price coefficient-of-variation >=2%).

Also saves one pooled overview (all selected ASINs on the same three scatter axes, colored
by ASIN) to check whether any trend is consistent ACROSS products or just idiosyncratic
per-product noise.

Four more diagnostics, added after the per-ASIN price/margin/own_landed scatter came back
flat everywhere -- not more of the same variable, but checking assumptions the first pass
took for granted:

  - plot_quantization(): buybox_pct isn't a smooth percentage -- the panel's own column list
    has no "n_checks" field, but the VALUES themselves reveal it: they cluster at simple
    fractions (66.667%, 83.333%, ...), i.e. k/n for a small n (median inferred n=15). This
    explains why every per-ASIN time series above looks so jagged. Checked separately
    (not plotted, since it came back negative): smoothing buybox_pct with a 4-week rolling
    mean before computing the price correlation does NOT change it -- so the jaggedness and
    the lack of a price trend are two different findings, not the same one wearing a
    disguise.

  - plot_won_vs_lost_profile(): stop looking for a continuous trend (three separate checks
    now -- pooled scatter, fixed-effects residual, this) and instead contrast the two
    DOMINANT groups directly: always-won rows (buybox_pct==100, 45.9% of the panel) vs
    always-lost (==0, 21.7%). If buy-box ownership is more "which kind of listing is this"
    than "what price is it charging right now," group profiles should show it even where a
    scatter plot doesn't.

  - plot_competitor_sliver() / plot_stockout_sliver(): the panel DOES carry the columns you'd
    actually want -- price_vs_lowest, min_competitor_landed, frac_oos, offer counts, the
    buy-box winner's own FBA/Prime/feedback stats -- but coverage is 0.2-1% of rows, all from
    a ~2-week window (matches the known sales_traffic ingestion gap). This is likely the
    "different parameter" your intuition is pointing at: it may not be absent from reality,
    just absent from 99%+ of this specific panel. Plotted honestly at actual n so it's clear
    how thin this evidence is, not held up as a finding.

Run:  python GAME_THEORY_PREDICTION/BBOX/plotting.py
"""
import os
from fractions import Fraction

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PANEL_PATH = os.path.join(HERE, "bbox_feature_panel.csv")
OUT_DIR = os.path.join(HERE, "explore_plots")

VARS = ["price", "est_margin", "own_landed"]
COLORS = {"price": "#4C78A8", "est_margin": "#F58518", "own_landed": "#54A24B"}

# Manual choices, same eligibility bar as train_bbox_two_stage.py's within-ASIN check:
# an ASIN with < 2% price CV never really moved, so its scatter panels would just be a
# vertical line of points -- no trend to see. TOP_N caps how many figures get produced;
# ranked by history length (most priced weeks first) among the eligible ones so the
# richest, most-repriced ASINs are shown first.
MIN_WEEKS = 8
MIN_PRICE_CV = 0.02
TOP_N = 12


def _select_asins(panel, top_n=TOP_N, min_weeks=MIN_WEEKS, min_cv=MIN_PRICE_CV):
    g = panel.groupby("asin")["price"]
    n_weeks = g.count()
    cv = (g.std() / g.mean()).replace([np.inf, -np.inf], np.nan)
    eligible = n_weeks[(n_weeks >= min_weeks) & (cv.reindex(n_weeks.index) >= min_cv)].index
    ranked = n_weeks.loc[eligible].sort_values(ascending=False)
    print(f"[plotting] {len(eligible)} ASINs pass eligibility (>= {min_weeks} priced weeks, "
          f"price CV >= {min_cv*100:.0f}%) out of {panel['asin'].nunique()} total -- "
          f"plotting top {min(top_n, len(ranked))} by history length")
    return ranked.index[:top_n].tolist()


def _normalize(s):
    lo, hi = s.min(), s.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-9:
        return s * 0.0
    return (s - lo) / (hi - lo)


def plot_asin(panel, asin, path):
    rows = panel[panel["asin"] == asin].copy()
    rows["week"] = pd.to_datetime(rows["week"])
    rows = rows.sort_values("week")
    rows["buybox_pct"] = pd.to_numeric(rows["buybox_pct"], errors="coerce")

    fig = plt.figure(figsize=(13, 7), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15])

    ax_ts = fig.add_subplot(gs[0, :])
    bb = rows["buybox_pct"]
    ax_ts.plot(rows["week"], _normalize(bb), color="black", lw=2.2,
               label=f"buybox_pct (range {bb.min():.0f}-{bb.max():.0f})")
    for v in VARS:
        vals = pd.to_numeric(rows[v], errors="coerce")
        if vals.notna().sum() < 2:
            continue
        ax_ts.plot(rows["week"], _normalize(vals), color=COLORS[v], lw=1.4, alpha=0.85,
                   label=f"{v} (range {vals.min():.2g}-{vals.max():.2g})")
    ax_ts.set_title("normalized time series (each series min-max scaled to [0,1] for shape "
                     "comparison -- see legend for real ranges)")
    ax_ts.set_xlabel("week")
    ax_ts.set_ylabel("normalized value")
    ax_ts.legend(loc="upper left", fontsize=8)

    for i, v in enumerate(VARS):
        ax = fig.add_subplot(gs[1, i])
        x = pd.to_numeric(rows[v], errors="coerce")
        y = bb
        mask = x.notna() & y.notna()
        ax.scatter(x[mask], y[mask], s=18, alpha=0.6, color=COLORS[v])
        if mask.sum() >= 3 and x[mask].std() > 1e-9:
            coef = np.polyfit(x[mask], y[mask], 1)
            xs = np.linspace(x[mask].min(), x[mask].max(), 50)
            ax.plot(xs, np.polyval(coef, xs), "--", color="black", lw=1.2)
            r = np.corrcoef(x[mask], y[mask])[0, 1]
            ax.set_title(f"{v}  (r={r:.2f}, n={int(mask.sum())})")
        else:
            ax.set_title(f"{v}  (not enough variation)")
        ax.set_xlabel(v)
        ax.set_ylabel("buybox_pct" if i == 0 else "")

    fig.suptitle(f"ASIN {asin}", fontsize=13, fontweight="bold")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_overview(panel, asins, path):
    rows = panel[panel["asin"].isin(asins)].copy()
    rows["buybox_pct"] = pd.to_numeric(rows["buybox_pct"], errors="coerce")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    cmap = plt.get_cmap("tab20")
    for i, v in enumerate(VARS):
        ax = axes[i]
        for j, asin in enumerate(asins):
            sub = rows[rows["asin"] == asin]
            x = pd.to_numeric(sub[v], errors="coerce")
            y = sub["buybox_pct"]
            mask = x.notna() & y.notna()
            ax.scatter(x[mask], y[mask], s=10, alpha=0.5, color=cmap(j % 20),
                       label=asin if i == 0 else None)
        ax.set_xlabel(v)
        ax.set_ylabel("buybox_pct" if i == 0 else "")
        ax.set_title(v)
    axes[0].legend(fontsize=6, loc="upper right", ncol=2)
    fig.suptitle(f"buybox_pct vs {', '.join(VARS)} -- pooled across {len(asins)} selected "
                 f"ASINs (color = ASIN)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[plotting] saved -> {path}")


def plot_quantization(panel, path):
    """buybox_pct's own values reveal it's k/n for a small n, not a smooth measurement --
    0%/100% rows are consistent with ANY n so they're excluded here (uninformative), but
    every OTHER value's simplest fraction pins down what n must have been."""
    def infer_n(v):
        if pd.isna(v) or v in (0.0, 100.0):
            return np.nan
        return Fraction(v / 100).limit_denominator(30).denominator

    ns = panel["buybox_pct"].apply(infer_n).dropna()
    counts = ns.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(counts.index, counts.to_numpy(), color="#4C78A8")
    ax.set_xlabel("inferred weekly sample size behind buybox_pct (denominator of k/n)")
    ax.set_ylabel("row count")
    ax.set_title(f"buybox_pct is k/n for a SMALL n (median={ns.median():.0f}) -- not a smooth "
                 f"percentage. Rows at exactly 0%/100% excluded (consistent with any n).")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[plotting] saved -> {path}")


def plot_won_vs_lost_profile(panel, path_prefix):
    """Contrast the two dominant groups (always-won vs always-lost) across every other
    available column, instead of hunting for a continuous price trend that three separate
    checks now agree isn't there."""
    numeric_cols = ["price", "own_landed", "est_margin", "image_count", "n_variation_siblings",
                     "fba_fee_per_unit", "commission_per_unit", "return_rate", "sessions"]
    cat_cols = ["has_aplus", "manufacturer", "product_type"]

    won = panel[panel["buybox_pct"] == 100.0]
    lost = panel[panel["buybox_pct"] == 0.0]
    print(f"[plotting] profile comparison: {len(won):,} always-won rows vs {len(lost):,} "
          f"always-lost rows")

    avail = [c for c in numeric_cols if panel[c].notna().mean() > 0.2]
    fig, axes = plt.subplots(1, len(avail), figsize=(2.8 * len(avail), 4.5))
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, avail):
        w = pd.to_numeric(won[c], errors="coerce").dropna()
        l = pd.to_numeric(lost[c], errors="coerce").dropna()
        ax.boxplot([l, w], tick_labels=["LOST\n(bb=0%)", "WON\n(bb=100%)"], showfliers=False)
        ax.set_title(c, fontsize=10)
    fig.suptitle("Always-won vs always-lost rows: distribution of each numeric feature", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    numeric_path = f"{path_prefix}_numeric.png"
    fig.savefig(numeric_path, dpi=120)
    plt.close(fig)
    print(f"[plotting] saved -> {numeric_path}")

    for c in cat_cols:
        if panel[c].notna().mean() < 0.2:
            continue
        sub = panel.dropna(subset=[c, "buybox_pct"])
        top_levels = sub[c].value_counts().head(8).index
        rate = sub[sub[c].isin(top_levels)].groupby(c)["buybox_pct"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(7, 4))
        rate.plot.barh(ax=ax, color="#54A24B")
        ax.set_xlabel("mean buybox_pct")
        ax.set_title(f"mean buybox_pct by {c} (top 8 levels by row count)")
        fig.tight_layout()
        cat_path = f"{path_prefix}_{c}.png"
        fig.savefig(cat_path, dpi=120)
        plt.close(fig)
        print(f"[plotting] saved -> {cat_path}")


def plot_competitor_sliver(panel, path):
    """The panel DOES have the "price relative to whoever's currently cheapest" column --
    price_vs_lowest -- but it's populated for only ~0.3% of rows, all from one recent
    window (matches the known sales_traffic ingestion gap). Plotted at its real n, not
    treated as a finding either way -- too thin to conclude anything from alone."""
    sub = panel.dropna(subset=["price_vs_lowest", "buybox_pct"]).copy()
    if sub.empty:
        print("[plotting] skipped competitor sliver -- no rows with price_vs_lowest")
        return
    weeks = pd.to_datetime(sub["week"])
    r = sub["price_vs_lowest"].corr(sub["buybox_pct"])

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(sub["price_vs_lowest"], sub["buybox_pct"], alpha=0.6, s=22, color="#B279A2")
    if sub["price_vs_lowest"].std() > 1e-9:
        coef = np.polyfit(sub["price_vs_lowest"], sub["buybox_pct"], 1)
        xs = np.linspace(sub["price_vs_lowest"].min(), sub["price_vs_lowest"].max(), 50)
        ax.plot(xs, np.polyval(coef, xs), "--", color="black", lw=1.2)
    ax.set_xlabel("price_vs_lowest (own price relative to cheapest competitor)")
    ax.set_ylabel("buybox_pct")
    ax.set_title(f"the ONE slice with real competitor context -- n={len(sub)}, "
                 f"{sub['asin'].nunique()} ASINs, {weeks.min().date()}..{weeks.max().date()} "
                 f"only (r={r:.3f})", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[plotting] saved -> {path}")


def plot_stockout_sliver(panel, path):
    """Same story as the competitor sliver -- frac_oos exists but covers ~1% of rows."""
    sub = panel.dropna(subset=["frac_oos", "buybox_pct"]).copy()
    if sub.empty:
        print("[plotting] skipped stockout sliver -- no rows with frac_oos")
        return
    r = sub["frac_oos"].corr(sub["buybox_pct"])

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(sub["frac_oos"], sub["buybox_pct"], alpha=0.6, s=22, color="#E45756")
    ax.set_xlabel("frac_oos (fraction of the week out of stock)")
    ax.set_ylabel("buybox_pct")
    ax.set_title(f"stockout sliver -- n={len(sub)}, {sub['asin'].nunique()} ASINs (r={r:.3f})",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[plotting] saved -> {path}")


def main():
    panel = pd.read_csv(PANEL_PATH)
    os.makedirs(OUT_DIR, exist_ok=True)
    asins = _select_asins(panel)

    for asin in asins:
        path = os.path.join(OUT_DIR, f"{asin}.png")
        plot_asin(panel, asin, path)
        print(f"[plotting] saved -> {path}")

    plot_overview(panel, asins, os.path.join(OUT_DIR, "_overview_all_selected_asins.png"))

    plot_quantization(panel, os.path.join(OUT_DIR, "_quantization.png"))
    plot_won_vs_lost_profile(panel, os.path.join(OUT_DIR, "_won_vs_lost"))
    plot_competitor_sliver(panel, os.path.join(OUT_DIR, "_competitor_sliver.png"))
    plot_stockout_sliver(panel, os.path.join(OUT_DIR, "_stockout_sliver.png"))


if __name__ == "__main__":
    main()
