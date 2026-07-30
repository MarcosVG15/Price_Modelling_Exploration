"""
Assemble the weekly (asin, marketplace) BUY-BOX feature panel for SHAP + a GBT.

Target = buy_box_percentage (our Featured-Offer share), weekly average per asin/marketplace.
Buy-box is a competition-for-the-featured-offer problem, so the features emphasise the OFFER
landscape (our price vs the lowest / buy-box price, offer counts, FBA/Prime, seller feedback),
STOCK (out-of-stock -> no buy-box) and LISTING HEALTH (suppressed/inactive/defect ->
ineligible) -- NOT the traffic features that drive conversion.

Leakage note: units / sessions / revenue are demand OUTCOMES of the buy-box, kept in the
output for reference only -- do NOT feed them as features to the buy-box GBT.

Coverage caveat: buy_box_percentage (target) is 2 years deep, but its real drivers
(competitor prices, offer counts, stock, listing issues, category rank, raw offer dispersion)
are the ~3-week snapshots (daily_ranks, offers_daily included), so those columns are near-empty
over the full panel. Run SHAP on the recent window to see them; the full panel will lean on own
price + fees + sentiment + season + product.

2-years-deep additions: fba_fee_per_unit/commission_per_unit (financial_shipment_events, via
order_items -> real per-unit margin) and review sentiment / repeat-purchase loyalty
(review_topic_trends / repeat_purchase, MONTHLY grain, broadcast across the weeks in a month).

NOT included: vendor_* tables. Several tracked ASINs also show up in vendor_sales/vendor_traffic
on the SAME marketplace_id -- i.e. this catalog appears to sell dual-channel (3P Seller Central
+ 1P Vendor Central) for at least some ASINs. That changes the competitive picture (Amazon
retail could be a buy-box competitor on those ASINs) and needs a deliberate modeling decision,
not a blind join -- left for a follow-up.

Run:  python GAME_THEORY_PREDICTION/BBOX/build_feature_panel_BBox.py
Out:  GAME_THEORY_PREDICTION/BBOX/bbox_feature_panel.csv
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import pandas as pd
from sqlalchemy import text
from market_env import _api_engine

SQL = """
WITH st AS (   -- target: buy-box % + own-price ingredients (weekly per asin/mkt)
    SELECT asin, marketplace_id, date_trunc('week', data_date)::date AS week,
           AVG(buy_box_percentage)    AS buybox_pct,
           SUM(units_ordered)         AS units,
           SUM(sessions)              AS sessions,
           SUM(ordered_product_sales) AS revenue
    FROM sales_traffic_daily
    WHERE buy_box_percentage IS NOT NULL AND sessions > 0
    GROUP BY 1, 2, 3
),
comp AS (   -- the OFFER landscape (offer_summaries_daily; ~2 weeks coverage)
    SELECT asin, marketplace_id, date_trunc('week', captured_at::date)::date AS week,
           AVG(lowest_landed_price)          AS lowest_price,
           AVG(buy_box_landed_price)         AS bb_price,
           AVG(total_offer_count)            AS offer_count,
           AVG(fba_offer_count)              AS fba_offers,
           AVG(mfn_offer_count)              AS mfn_offers,
           AVG(buy_box_eligible_offer_count) AS eligible_offers,
           AVG(competitive_threshold)        AS comp_threshold,
           AVG(CASE WHEN is_buy_box_fba   THEN 1.0 ELSE 0.0 END) AS bb_is_fba,
           AVG(CASE WHEN is_buy_box_prime THEN 1.0 ELSE 0.0 END) AS bb_is_prime,
           AVG(buy_box_seller_feedback_rating) AS bb_feedback_rating,
           AVG(buy_box_seller_feedback_count)  AS bb_feedback_count,
           AVG(buy_box_shipping_amount)        AS bb_shipping,      -- shipping half of the LANDED buy-box price
           AVG(lowest_shipping_amount)         AS lowest_shipping
    FROM offer_summaries_daily GROUP BY 1, 2, 3
),
stock AS (   -- availability / stock depth (inventory_summaries; ~2 weeks)
    SELECT asin, marketplace_id, date_trunc('week', captured_at::date)::date AS week,
           AVG(fulfillable_quantity)    AS units_in_stock,          -- sellable on-hand units
           AVG(total_quantity)          AS total_qty,
           AVG(total_reserved_quantity) AS reserved_qty,            -- units reserved for open orders
           AVG(inbound_working_quantity + inbound_shipped_quantity + inbound_receiving_quantity) AS inbound_qty,
           AVG(CASE WHEN fulfillable_quantity = 0 THEN 1.0 ELSE 0.0 END) AS frac_oos
    FROM inventory_summaries GROUP BY 1, 2, 3
),
issues AS (  -- listing health: suppressed / inactive / defect -> buy-box ineligible
    SELECT asin, marketplace_id, date_trunc('week', captured_at::date)::date AS week,
           MAX(CASE WHEN is_suppressed THEN 1 ELSE 0 END) AS is_suppressed,
           MAX(CASE WHEN is_inactive   THEN 1 ELSE 0 END) AS is_inactive,
           MAX(CASE WHEN is_defect     THEN 1 ELSE 0 END) AS is_defect,
           COUNT(*)                                        AS issue_count
    FROM seller_listing_issues WHERE asin IS NOT NULL GROUP BY 1, 2, 3
),
oi_price AS (  -- own realized price + own shipping charged (order_items)
    SELECT oi.asin AS asin, o.marketplace_id AS marketplace_id,
           date_trunc('week', o.purchase_date::date)::date AS week,
           SUM(oi.item_price)     / NULLIF(SUM(oi.quantity_ordered), 0) AS oi_price,
           SUM(oi.shipping_price) / NULLIF(SUM(oi.quantity_ordered), 0) AS own_shipping
    FROM order_items oi JOIN orders o ON o.amazon_order_id = oi.amazon_order_id
    WHERE oi.item_price IS NOT NULL AND oi.quantity_ordered > 0 GROUP BY 1, 2, 3
),
sl_price AS ( -- own posted listing price (seller_listings)
    SELECT asin, marketplace_id, date_trunc('week', captured_at::date)::date AS week,
           AVG(listing_price) AS listing_price
    FROM seller_listings WHERE listing_price IS NOT NULL GROUP BY 1, 2, 3
),
prod AS (   -- product content (brand / type / manufacturer)
    SELECT DISTINCT ON (asin, marketplace_id) asin, marketplace_id, brand, product_type, manufacturer
    FROM product_content ORDER BY asin, marketplace_id, captured_at DESC
),
pics AS (
    SELECT asin, marketplace_id, COUNT(*) AS image_count FROM product_pictures GROUP BY 1, 2
),
aplus AS (   -- product content richness: has A+ content
    SELECT DISTINCT asin, marketplace_id, TRUE AS has_aplus FROM aplus_contents
),
ret AS (     -- units returned per product (static, all-time)
    SELECT asin, marketplace_id, SUM(quantity_returned) AS returned
    FROM returns WHERE asin IS NOT NULL GROUP BY 1, 2
),
tot_units AS ( -- total units ordered per product (denominator for return rate)
    SELECT asin, marketplace_id, SUM(units_ordered) AS tot_units
    FROM sales_traffic_daily GROUP BY 1, 2
),
deliv AS (   -- avg promised days to delivery (scheduled delivery - purchase date)
    SELECT oi.asin AS asin, o.marketplace_id AS marketplace_id,
           AVG(oi.scheduled_delivery_start_date::date - o.purchase_date::date) AS avg_delivery_days
    FROM order_items oi JOIN orders o ON o.amazon_order_id = oi.amazon_order_id
    WHERE oi.scheduled_delivery_start_date IS NOT NULL AND o.purchase_date IS NOT NULL
    GROUP BY 1, 2
),
defect AS (  -- defective-return units per product (ODR "defect" proxy).
             -- return_dispositions has no asin -> join to returns on return_id.
    SELECT r.asin, r.marketplace_id,
           SUM(CASE WHEN rd.is_defective = 'true' THEN rd.quantity ELSE 0 END) AS defective_units
    FROM return_dispositions rd
    JOIN returns r ON r.return_id = rd.return_id
    GROUP BY 1, 2
),
cat_rank AS (   -- category standing (daily_ranks; ~3 weeks coverage, same window as `comp`)
    SELECT asin, marketplace_id, date_trunc('week', captured_at::date)::date AS week,
           MIN(rank_value) AS best_category_rank   -- best (lowest) rank across all listed categories
    FROM daily_ranks WHERE rank_value IS NOT NULL GROUP BY 1, 2, 3
),
raw_off AS (   -- per-seller offer dispersion (offers_daily; raw companion to `comp`, ~3 weeks)
    SELECT asin, marketplace_id, date_trunc('week', captured_at::date)::date AS week,
           COUNT(DISTINCT seller_id)          AS n_competing_sellers,
           STDDEV(landed_price)               AS price_dispersion,
           MIN(landed_price)                  AS min_competitor_landed,
           MAX(landed_price)                  AS max_competitor_landed,
           AVG(CASE WHEN is_fulfilled_by_amazon THEN 1.0 ELSE 0.0 END) AS frac_offers_fba,
           AVG(seller_feedback_count)         AS avg_competitor_feedback
    FROM offers_daily GROUP BY 1, 2, 3
),
fin AS (   -- real fees (financial_shipment_events; 2 yrs via amazon_order_id -> order_items)
    SELECT oi.asin AS asin, oi.marketplace_id AS marketplace_id,
           date_trunc('week', o.purchase_date::date)::date AS week,
           SUM(f.fba_fee_amount)    / NULLIF(SUM(oi.quantity_ordered), 0) AS fba_fee_per_unit,
           SUM(f.commission_amount) / NULLIF(SUM(oi.quantity_ordered), 0) AS commission_per_unit
    FROM financial_shipment_events f
    JOIN order_items oi ON oi.amazon_order_id = f.amazon_order_id
    JOIN orders o       ON o.amazon_order_id  = oi.amazon_order_id
    GROUP BY 1, 2, 3   -- fba_fee/commission are order-level; 99.5% of orders are single-ASIN so this
                       -- fans out negligibly for the rare multi-ASIN order
),
reviews AS (  -- review topic sentiment (review_topic_trends; MONTHLY grain, 2+ yrs deep)
    SELECT asin, marketplace_id, date_trunc('month', period_start)::date AS month,
           AVG(CASE WHEN sentiment_type = 'positive' THEN asin_occurrence_percentage END) AS pct_positive_topics,
           AVG(CASE WHEN sentiment_type = 'negative' THEN asin_occurrence_percentage END) AS pct_negative_topics
    FROM review_topic_trends GROUP BY 1, 2, 3
),
repeat_buy AS (  -- customer loyalty (repeat_purchase; MONTHLY grain, 2025-06 -> 2026-05).
                 -- repeat_purchase_rate/repeat_customers/new_customers are NULL for every row
                 -- at the source (upstream gap) -- only unique_customers is ever populated.
    SELECT asin, marketplace_id, date_trunc('month', period_start)::date AS month,
           AVG(unique_customers) AS unique_customers
    FROM repeat_purchase GROUP BY 1, 2, 3
),
variation AS (  -- listing structure (product_relationships; static, all VARIATION/CHILD rows)
    SELECT asin, marketplace_id, COUNT(DISTINCT related_asin) AS n_variation_siblings
    FROM product_relationships GROUP BY 1, 2
)
SELECT
    st.asin, st.marketplace_id, st.week,             -- week is an identifier, NOT a feature
    st.buybox_pct,                                   -- TARGET
    -- own price sources -> coalesced raw_price (ffill'd to `price` in pandas)
    (st.revenue::float / NULLIF(st.units, 0)) AS implied_price,
    oip.oi_price, slp.listing_price,
    COALESCE(oip.oi_price, st.revenue::float / NULLIF(st.units, 0), slp.listing_price) AS raw_price,
    oip.own_shipping,                                -- shipping WE charge (own landed-price component)
    -- competition / offer landscape (incl. shipping half of the landed price)
    cp.lowest_price, cp.bb_price, cp.offer_count, cp.fba_offers, cp.mfn_offers,
    cp.eligible_offers, cp.comp_threshold, cp.bb_is_fba, cp.bb_is_prime,
    cp.bb_feedback_rating, cp.bb_feedback_count,
    cp.bb_shipping, cp.lowest_shipping,
    -- availability / stock depth
    stk.units_in_stock, stk.total_qty, stk.reserved_qty, stk.inbound_qty, stk.frac_oos,
    -- listing health
    iss.is_suppressed, iss.is_inactive, iss.is_defect, iss.issue_count,
    -- product content (time-independent attributes) + returns + defects + delivery speed
    pc.brand, pc.product_type, pc.manufacturer, pics.image_count,
    (ap.has_aplus IS NOT NULL)                       AS has_aplus,
    (ret.returned::float / NULLIF(tu.tot_units, 0))  AS return_rate,
    (dfc.defective_units::float / NULLIF(tu.tot_units, 0)) AS defect_return_rate,
    dv.avg_delivery_days,
    -- category standing + raw offer dispersion (complements `comp`'s buy-box-only view)
    rk.best_category_rank,
    ro.n_competing_sellers, ro.price_dispersion, ro.min_competitor_landed, ro.max_competitor_landed,
    ro.frac_offers_fba, ro.avg_competitor_feedback,
    -- real per-unit fees (financial_shipment_events; negative = cost, 2 yrs deep)
    fin.fba_fee_per_unit, fin.commission_per_unit,
    -- review sentiment + repeat-purchase loyalty (monthly grain, broadcast across weeks in month)
    rev.pct_positive_topics, rev.pct_negative_topics,
    rp.unique_customers,
    -- listing structure
    var.n_variation_siblings,
    -- demand OUTCOMES (reference only, NOT features)
    st.units, st.sessions, st.revenue
FROM st
LEFT JOIN comp     cp  ON (cp.asin, cp.marketplace_id, cp.week)   = (st.asin, st.marketplace_id, st.week)
LEFT JOIN stock    stk ON (stk.asin, stk.marketplace_id, stk.week) = (st.asin, st.marketplace_id, st.week)
LEFT JOIN issues   iss ON (iss.asin, iss.marketplace_id, iss.week) = (st.asin, st.marketplace_id, st.week)
LEFT JOIN oi_price oip ON (oip.asin, oip.marketplace_id, oip.week) = (st.asin, st.marketplace_id, st.week)
LEFT JOIN sl_price slp ON (slp.asin, slp.marketplace_id, slp.week) = (st.asin, st.marketplace_id, st.week)
LEFT JOIN prod     pc  ON (pc.asin, pc.marketplace_id)  = (st.asin, st.marketplace_id)
LEFT JOIN pics         ON (pics.asin, pics.marketplace_id) = (st.asin, st.marketplace_id)
LEFT JOIN aplus    ap  ON (ap.asin, ap.marketplace_id)  = (st.asin, st.marketplace_id)
LEFT JOIN ret          ON (ret.asin, ret.marketplace_id) = (st.asin, st.marketplace_id)
LEFT JOIN tot_units tu ON (tu.asin, tu.marketplace_id)  = (st.asin, st.marketplace_id)
LEFT JOIN deliv    dv  ON (dv.asin, dv.marketplace_id)  = (st.asin, st.marketplace_id)
LEFT JOIN defect   dfc ON (dfc.asin, dfc.marketplace_id) = (st.asin, st.marketplace_id)
LEFT JOIN cat_rank rk  ON (rk.asin, rk.marketplace_id, rk.week) = (st.asin, st.marketplace_id, st.week)
LEFT JOIN raw_off  ro  ON (ro.asin, ro.marketplace_id, ro.week) = (st.asin, st.marketplace_id, st.week)
LEFT JOIN fin          ON (fin.asin, fin.marketplace_id, fin.week) = (st.asin, st.marketplace_id, st.week)
LEFT JOIN reviews  rev ON (rev.asin, rev.marketplace_id) = (st.asin, st.marketplace_id)
                       AND rev.month = date_trunc('month', st.week)::date
LEFT JOIN repeat_buy rp ON (rp.asin, rp.marketplace_id) = (st.asin, st.marketplace_id)
                       AND rp.month = date_trunc('month', st.week)::date
LEFT JOIN variation var ON (var.asin, var.marketplace_id) = (st.asin, st.marketplace_id)
"""


def main():
    with _api_engine().connect() as c:
        df = pd.read_sql(text(SQL), c)

    # coalesced own price -> carry forward within product (near-complete `price`)
    df = df.sort_values(["asin", "marketplace_id", "week"]).reset_index(drop=True)
    df["price"] = (df.groupby(["asin", "marketplace_id"])["raw_price"]
                     .transform(lambda s: s.ffill().bfill()))
    # competitiveness ratios (the core buy-box signal): our price vs the field
    df["price_vs_lowest"] = df["price"] / df["lowest_price"].replace(0, pd.NA)
    df["price_vs_bb"]     = df["price"] / df["bb_price"].replace(0, pd.NA)
    # landed price (item + own shipping) vs the buy-box landed price -- Amazon ranks
    # on LANDED cost, so shipping matters as much as the sticker price
    df["own_landed"]   = df["price"] + df["own_shipping"].fillna(0.0)
    df["landed_vs_bb"] = df["own_landed"] / df["bb_price"].replace(0, pd.NA)
    # days of supply = sellable on-hand units / daily sales velocity (weekly units / 7)
    df["days_of_supply"] = df["units_in_stock"] * 7.0 / df["units"].replace(0, pd.NA)
    # real per-unit margin (fees are already negative -> add, don't subtract)
    df["est_margin"] = df["price"] + df["fba_fee_per_unit"].fillna(0.0) + df["commission_per_unit"].fillna(0.0)
    # net review sentiment: share of topics skewing positive vs negative
    df["net_sentiment"] = df["pct_positive_topics"] - df["pct_negative_topics"]

    print(f"panel shape: {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"weeks: {df['week'].min()} -> {df['week'].max()}   asins: {df['asin'].nunique()}")
    print("\nTARGET buy_box_percentage:")
    print(df["buybox_pct"].describe().round(2).to_string())
    print("\nper-column non-null coverage:")
    cov = (df.notna().mean().sort_values(ascending=False) * 100).round(1)
    for name, pct in cov.items():
        print(f"   {name:<22} {pct:5.1f}%")
    out = os.path.join(HERE, "bbox_feature_panel.csv")
    df.to_csv(out, index=False)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
