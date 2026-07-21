"""
Assemble the weekly (asin, marketplace) CONVERSION-RATE feature panel for SHAP + a GBT.

Target  = CVR = units_ordered / sessions  (predict conversion, not raw demand -- traffic
          is exogenous and drowns the price signal, so model the price-sensitive part).
Price   = implied own price = ordered_product_sales / units_ordered, straight from
          sales_traffic_daily (~100% covered on any week that sold; not a leak -- units
          cancels, so the ratio is the true avg price, no order_items join needed).

Leakage note: units / sessions / revenue are kept in the output for inspection/weighting,
but they are the target's ingredients -- DO NOT feed them as features to the CVR GBT.

Run:  python GAME_THEORY_PREDICTION/build_feature_panel.py
Out:  GAME_THEORY_PREDICTION/demand_feature_panel.csv
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import pandas as pd
from sqlalchemy import text
from market_env import _api_engine

SQL = """
WITH st AS (   -- spine: CVR target + implied price + Tier-1 traffic, weekly per asin/mkt
    SELECT asin, marketplace_id,
           date_trunc('week', data_date)::date AS week,
           SUM(units_ordered)         AS units,
           SUM(sessions)              AS sessions,
           SUM(ordered_product_sales) AS revenue,
           SUM(browser_sessions)      AS browser_sessions,
           SUM(mobile_app_sessions)   AS mobile_sessions,
           SUM(page_views)            AS page_views,
           AVG(buy_box_percentage)    AS buybox_pct
    FROM sales_traffic_daily
    WHERE sessions > 0
    GROUP BY 1, 2, 3
),
comp AS (      -- competition (offer_summaries_daily; ~2 weeks coverage)
    SELECT asin, marketplace_id,
           date_trunc('week', captured_at::date)::date AS week,
           AVG(lowest_landed_price)   AS lowest_price,
           AVG(buy_box_landed_price)  AS bb_price,
           AVG(total_offer_count)     AS offer_count,
           AVG(competitive_threshold) AS comp_threshold
    FROM offer_summaries_daily GROUP BY 1, 2, 3
),
stock AS (     -- availability (inventory_summaries; ~2 weeks)
    SELECT asin, marketplace_id,
           date_trunc('week', captured_at::date)::date AS week,
           AVG(fulfillable_quantity) AS fulfillable_qty,
           AVG(total_quantity)       AS total_qty
    FROM inventory_summaries GROUP BY 1, 2, 3
),
rank_w AS (    -- sales rank (daily_ranks; ~2 weeks) -- OUTCOME of demand, flagged
    SELECT asin, marketplace_id,
           date_trunc('week', captured_at::date)::date AS week,
           AVG(rank_value) AS rank_value
    FROM daily_ranks GROUP BY 1, 2, 3
),
prod AS (      -- catalog/quality (latest snapshot per asin)
    SELECT DISTINCT ON (asin, marketplace_id)
           asin, marketplace_id, brand, product_type, main_browser_node_id
    FROM product_content ORDER BY asin, marketplace_id, captured_at DESC
),
pics AS (
    SELECT asin, marketplace_id, COUNT(*) AS image_count
    FROM product_pictures GROUP BY 1, 2
),
aplus AS (
    SELECT DISTINCT asin, marketplace_id, TRUE AS has_aplus FROM aplus_contents
),
rev AS (       -- review sentiment (static, over all periods)
    SELECT asin, marketplace_id,
           COUNT(*) FILTER (WHERE sentiment_type ILIKE 'pos%') AS pos_topics,
           COUNT(*) FILTER (WHERE sentiment_type ILIKE 'neg%') AS neg_topics
    FROM review_topic_trends GROUP BY 1, 2
),
repeat_p AS (
    SELECT DISTINCT ON (asin, marketplace_id)
           asin, marketplace_id, repeat_purchase_rate
    FROM repeat_purchase ORDER BY asin, marketplace_id, period_start DESC
),
oi_price AS (  -- realized transaction price from order_items (most granular own price)
    SELECT oi.asin AS asin, o.marketplace_id AS marketplace_id,
           date_trunc('week', o.purchase_date::date)::date AS week,
           SUM(oi.item_price) / NULLIF(SUM(oi.quantity_ordered), 0) AS oi_price
    FROM order_items oi JOIN orders o ON o.amazon_order_id = oi.amazon_order_id
    WHERE oi.item_price IS NOT NULL AND oi.quantity_ordered > 0
    GROUP BY 1, 2, 3
),
sl_price AS (  -- posted listing price from seller_listings (exists even on no-sale weeks)
    SELECT asin, marketplace_id,
           date_trunc('week', captured_at::date)::date AS week,
           AVG(listing_price) AS listing_price
    FROM seller_listings WHERE listing_price IS NOT NULL
    GROUP BY 1, 2, 3
)
SELECT
    st.asin, st.marketplace_id, st.week,
    EXTRACT(WEEK  FROM st.week)::int AS week_of_year,
    EXTRACT(MONTH FROM st.week)::int AS month,
    -- TARGET: conversion rate
    (st.units::float / NULLIF(st.sessions, 0))  AS cvr,
    -- the three own-price sources + the coalesced raw price (ffill'd to `price` in pandas)
    (st.revenue::float / NULLIF(st.units, 0))   AS implied_price,   -- revenue/units (sale weeks)
    oip.oi_price,                                                    -- order_items realized
    slp.listing_price,                                              -- seller_listings posted
    COALESCE(oip.oi_price,
             st.revenue::float / NULLIF(st.units, 0),
             slp.listing_price)                 AS raw_price,       -- best available this week
    -- target ingredients: kept for inspection/weighting, NOT features (leakage for CVR)
    st.units, st.sessions, st.revenue,
    -- traffic composition + visibility
    st.browser_sessions, st.mobile_sessions, st.page_views, st.buybox_pct,
    -- competition (2wk)
    cp.lowest_price, cp.bb_price, cp.offer_count, cp.comp_threshold,
    -- stock (2wk)
    stk.fulfillable_qty, stk.total_qty,
    -- product/quality (static)
    pc.brand, pc.product_type, pc.main_browser_node_id, pics.image_count,
    (ap.has_aplus IS NOT NULL) AS has_aplus, rv.pos_topics, rv.neg_topics, rp.repeat_purchase_rate,
    dr.rank_value                                    -- OUTCOME (consider dropping)
FROM st
LEFT JOIN comp     cp  ON (cp.asin, cp.marketplace_id, cp.week)   = (st.asin, st.marketplace_id, st.week)
LEFT JOIN stock    stk ON (stk.asin, stk.marketplace_id, stk.week) = (st.asin, st.marketplace_id, st.week)
LEFT JOIN rank_w   dr  ON (dr.asin, dr.marketplace_id, dr.week)   = (st.asin, st.marketplace_id, st.week)
LEFT JOIN prod     pc  ON (pc.asin, pc.marketplace_id)            = (st.asin, st.marketplace_id)
LEFT JOIN pics         ON (pics.asin, pics.marketplace_id)        = (st.asin, st.marketplace_id)
LEFT JOIN aplus    ap  ON (ap.asin, ap.marketplace_id)            = (st.asin, st.marketplace_id)
LEFT JOIN rev      rv  ON (rv.asin, rv.marketplace_id)            = (st.asin, st.marketplace_id)
LEFT JOIN repeat_p rp  ON (rp.asin, rp.marketplace_id)            = (st.asin, st.marketplace_id)
LEFT JOIN oi_price oip ON (oip.asin, oip.marketplace_id, oip.week) = (st.asin, st.marketplace_id, st.week)
LEFT JOIN sl_price slp ON (slp.asin, slp.marketplace_id, slp.week) = (st.asin, st.marketplace_id, st.week)
"""


def main():
    with _api_engine().connect() as c:
        df = pd.read_sql(text(SQL), c)

    # Coalesced price -> carry the last known price forward (then back) within each
    # product, so weeks with no direct price inherit the most recent one. Turns the
    # sparse raw_price into a near-complete `price` column without selection bias.
    df = df.sort_values(["asin", "marketplace_id", "week"]).reset_index(drop=True)
    df["price"] = (df.groupby(["asin", "marketplace_id"])["raw_price"]
                     .transform(lambda s: s.ffill().bfill()))
    df["price_vs_lowest"] = df["price"] / df["lowest_price"].replace(0, pd.NA)

    print(f"panel shape: {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"weeks: {df['week'].min()} -> {df['week'].max()}   asins: {df['asin'].nunique()}")
    print("\nprice coverage by source:")
    for col in ["oi_price", "implied_price", "listing_price", "raw_price", "price"]:
        print(f"   {col:<16} {df[col].notna().mean()*100:5.1f}%")
    print("\nTARGET (cvr) and final price:")
    print(df[["cvr", "price"]].describe().round(4).to_string())
    print("\nper-column non-null coverage:")
    cov = (df.notna().mean().sort_values(ascending=False) * 100).round(1)
    for name, pct in cov.items():
        print(f"   {name:<26} {pct:5.1f}%")
    out = os.path.join(HERE, "sp_api_demand_forcast_data_CVR.csv")
    df.to_csv(out, index=False)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
