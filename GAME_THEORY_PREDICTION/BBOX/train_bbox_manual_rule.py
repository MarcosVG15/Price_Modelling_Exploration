"""
Hand-authored buy-box rule -- not fit from data. This is the "pure man-made rule set"
discussed at length in conversation: every data-driven attempt at this panel (the tuned
tree in train_bbox.py, the monotonic GAM, the fixed-effects two-stage residual, the shallow
rule-based classifier) converged on the same finding -- price/margin/own_landed carry ~0
independent marginal signal here, most likely because the actual determinant (buy-box share
relative to WHO you're currently competing against) is a variable this panel has for only
~0.3% of rows. Rather than keep fitting a model to columns that don't contain the answer,
this file states the mechanism as an assumption and lets market_env.py use it directly.

FORM: logistic in the SAME two gap terms market_env.py already computes every step --
  gap     = (own_price - comp_ref) / comp_ref   -- relative to the rival's CURRENT price
  abs_gap = (own_price - reference_price) / reference_price  -- relative to the cluster's
            fixed historical anchor, so joint inflation (both sides drifting up together,
            which a purely relative gap can't see) still gets penalized
plus the three listing-quality terms (FBA / Prime / feedback) already used as fallback
defaults elsewhere in market_env.py -- kept for continuity, not re-derived.

gap_coef=-12.0 / abs_gap_coef=-4.0: reuses the magnitudes market_env.py's OWN previous
fallback formula already used as literal defaults (self.params.get("buybox_gap_coef",
-12.0), etc.) -- not a new number invented for this file. Steepness check: at a 15% price
disadvantage (gap=0.15), sigmoid(-12*0.15) = sigmoid(-1.8) = 0.14 -- consistent with the
observed 46% WON / 22% LOST / 32% MIXED split implying a fairly steep, winner-take-most
mechanism (see the per-leaf histograms in rules_eval/leaf_distributions.png -- EVERY leaf
kept mass at both 0 and 100, never a smooth middle), not a gentle elasticity.

intercept=-1.45: NOT copied from the old default (which was 1.0, tuned for a DIFFERENT
formula that also had a fitted tree's logit added in). Solved here so that price parity
(gap=abs_gap=0) with typical assumptions (own_fba=own_prime=1, feedback=4.5, all matching
market_env.py's own hardcoded/default values) gives exactly P(buy-box)=0.5 -- a fair contest
at parity, matching the "MIXED" band being centered near parity rather than skewed toward
an automatic win:
    0 = intercept + fba_coef*1 + prime_coef*1 + feedback_coef*4.5
    intercept = -(0.5*1 + 0.5*1 + 0.1*4.5) = -1.45

Run:  python GAME_THEORY_PREDICTION/BBOX/train_bbox_manual_rule.py
"""
import os

import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "bbox_manual_rule.joblib")

RULE_PARAMS = {
    "buybox_intercept": -1.45,
    "buybox_gap_coef": -12.0,
    "buybox_abs_gap_coef": -4.0,
    "buybox_fba_coef": 0.5,
    "buybox_prime_coef": 0.5,
    "buybox_feedback_coef": 0.1,
}


def build_and_save():
    bundle = {"type": "manual_rule", "params": RULE_PARAMS}
    joblib.dump(bundle, MODEL_PATH)
    print(f"[bbox-manual-rule] saved -> {MODEL_PATH}")
    print(f"[bbox-manual-rule] params: {RULE_PARAMS}")

    # sanity check: monotonic in gap, and =0.5 at exact parity under the calibration assumptions
    import numpy as np
    p = RULE_PARAMS
    own_fba, own_prime, own_feedback = 1.0, 1.0, 4.5
    for gap in [-0.30, -0.15, -0.05, 0.0, 0.05, 0.15, 0.30]:
        z = (p["buybox_intercept"] + p["buybox_gap_coef"] * gap + p["buybox_abs_gap_coef"] * 0.0
             + p["buybox_fba_coef"] * own_fba + p["buybox_prime_coef"] * own_prime
             + p["buybox_feedback_coef"] * own_feedback)
        prob = 1.0 / (1.0 + np.exp(-z))
        print(f"[bbox-manual-rule] gap={gap:+.2f} (abs_gap=0, typical FBA/Prime/feedback) "
              f"-> P(buy-box)={prob:.3f}")
    return bundle


if __name__ == "__main__":
    build_and_save()
