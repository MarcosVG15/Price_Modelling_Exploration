"""
Shallow, fully-readable decision-tree classifier. Replaces train_bbox_two_stage.py --
not because the two-stage architecture was wrong, but because it's the wrong TOOL for
where this project is right now. Every check run in the conversation (pooled scatter,
fixed-effects residual, manufacturer/product_type segmentation, the quantization
diagnostic) agrees: this panel doesn't contain a fittable price effect worth chasing
further. The next real step is a hand-authored, assumption-driven buy-box rule -- but
before writing that by hand, this file exists to show, in literal if/then text a team can
argue with, which of the columns we DO have actually separate winners from losers, and
where a simple model's mistakes cluster. It's a diagnostic to look at and disagree with,
not a candidate for market_env.py.

TARGET: 3 classes, not a continuous percentage -- LOST (buybox_pct==0), WON (==100), MIXED
(anything else). This mirrors the panel's own shape (46% WON, 22% LOST, 32% MIXED, per the
bbox_distribution.png produced earlier) rather than forcing a threshold-driven, mostly-
boundary target through a regressor that has to work just as hard to get 0 and 100 "close
enough" as it does anywhere else.

FEATURES: intentionally excludes manufacturer/product_type (31 / 58 levels) despite them
mattering in the earlier two-stage tree (R^2 contribution was real) -- a split like
"manufacturer in {Alecto, Lenco, Profoon, ...}" is not a rule a team can read and argue
with, which is the entire point of this file. Kept: price, own_landed, est_margin,
image_count, n_variation_siblings, fba_fee_per_unit, commission_per_unit, return_rate,
has_aplus -- plus frac_oos and price_vs_lowest despite ~0.3-1% coverage, specifically to
see whether the tree finds them useful DESPITE the sparsity (a real signal so strong it
shows up in <1% of rows would itself be worth knowing).

MISSING VALUES: passed straight through as NaN, no imputation -- DecisionTreeClassifier
(sklearn >=1.3) natively routes missing values to whichever branch reduces impurity more,
which is a more honest treatment than median-filling 70-99%-missing columns the way the
tree-based scripts before this one had to.

MAX_DEPTH=5 (manual choice, exposed as a constant, checked empirically before picking it):
depth 3-4 (8-16 leaves) find ZERO leaves where LOST is even the plurality class -- with
min_samples_leaf=200, this simple feature set genuinely cannot isolate a "this listing
tends to lose" region until depth 5 (27 leaves, LOST recall 0.0 -> 0.124). Depth 6 barely
moves past that (38 leaves, LOST recall 0.125) -- diminishing returns, so 5 is the point
past which more depth mostly costs readability without buying detection power. Still large
enough that the leaf-distribution grid (see plot_leaf_distributions()) is the primary way
to actually read this tree -- the flowchart PNG is included but expect it to be dense.

PER-LEAF DISTRIBUTION: a leaf's "class: WON" label is one number thrown away -- it says
nothing about how uniform or split that leaf's ACTUAL buybox_pct values are. Instead of
switching to a different distributional model (checked in conversation: quantile
regression / NGBoost solve a different problem, since buybox_pct's spread is already known
to be mostly small-n Binomial sampling noise, not something worth re-deriving a whole
model to learn), plot_leaf_distributions() reuses this exact tree: route every row to its
leaf via clf.apply(), then histogram the RAW buybox_pct values landing in each leaf. That's
the actual "how likely is a product like this to get X% ownership" answer, for free, from a
model that's already fully readable.

Run:  python GAME_THEORY_PREDICTION/BBOX/train_bbox_rules.py
"""
import os
import textwrap

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PANEL_PATH = os.path.join(HERE, "bbox_feature_panel.csv")
MODEL_PATH = os.path.join(HERE, "bbox_rules_model.joblib")
OUT_DIR = os.path.join(HERE, "rules_eval")

FEATURES = [
    "price", "own_landed", "est_margin", "image_count", "n_variation_siblings",
    "fba_fee_per_unit", "commission_per_unit", "return_rate", "has_aplus",
    "frac_oos", "price_vs_lowest",
]
CLASSES = ["LOST", "MIXED", "WON"]   # alphabetical -- sklearn sorts class labels this way anyway

MAX_DEPTH = 5
MIN_SAMPLES_LEAF = 200   # manual choice: at ~34k train rows, keeps each leaf backed by
                          # enough rows that its majority class/distribution isn't a fluke
TEST_SIZE = 0.4           # same holdout convention as every BBOX script before this one


def _label(buybox_pct):
    y = pd.Series(index=buybox_pct.index, dtype=object)
    y[buybox_pct == 0.0] = "LOST"
    y[buybox_pct == 100.0] = "WON"
    y[(buybox_pct > 0.0) & (buybox_pct < 100.0)] = "MIXED"
    return y


def _encode(panel):
    X = pd.DataFrame(index=panel.index)
    for c in FEATURES:
        if c == "has_aplus":
            X[c] = panel[c].map({True: 1.0, False: 0.0})
        else:
            X[c] = pd.to_numeric(panel[c], errors="coerce")
    return X


def _leaf_rule_text(clf, leaf_id, feature_names):
    """Reconstruct the conjunction of splits leading to `leaf_id` by walking clf.tree_ --
    export_text() already prints this once for the whole tree, but plot_leaf_distributions()
    needs it per-leaf, attached to that leaf's own histogram."""
    tree_ = clf.tree_

    def find_path(node, path):
        if node == leaf_id:
            return path
        if tree_.children_left[node] == -1:
            return None
        found = find_path(tree_.children_left[node], path + [(node, True)])
        if found is not None:
            return found
        return find_path(tree_.children_right[node], path + [(node, False)])

    path = find_path(0, []) or []
    parts = [f"{feature_names[tree_.feature[n]]} {'<=' if went_left else '>'} "
             f"{tree_.threshold[n]:.3g}" for n, went_left in path]
    return " & ".join(parts) if parts else "(root)"


def plot_leaf_distributions(clf, X, buybox_pct, path):
    """For every leaf, histogram the ACTUAL buybox_pct values of the rows routed there --
    the real distribution a "class: WON" label was hiding. Uses the FULL panel (not just
    train or test) since this is a descriptive artifact, not a scored metric -- more data
    makes each leaf's histogram a better empirical picture of what that rule actually means."""
    leaf_ids = clf.apply(X)
    buybox_pct = np.asarray(buybox_pct)
    unique_leaves = sorted(set(leaf_ids))
    n = len(unique_leaves)
    ncols = min(5, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.6 * nrows), squeeze=False)
    for i, leaf in enumerate(unique_leaves):
        ax = axes[i // ncols][i % ncols]
        vals = buybox_pct[leaf_ids == leaf]
        ax.hist(vals, bins=np.linspace(0, 100, 21), color="#4C78A8", edgecolor="black", alpha=0.85)
        rule = textwrap.fill(_leaf_rule_text(clf, leaf, X.columns.tolist()), width=34)
        ax.set_title(f"leaf {leaf}  n={len(vals):,}\nmean={vals.mean():.0f}  "
                     f"median={np.median(vals):.0f}", fontsize=8)
        ax.set_xlabel(rule, fontsize=6.3)
        ax.set_xlim(0, 100)
        ax.set_yticks([])
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle("Distribution of ACTUAL buybox_pct within each leaf (not just the majority "
                 "class label)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[bbox-rules] saved -> {path}")


def fit_and_report():
    panel = pd.read_csv(PANEL_PATH)
    y = _label(panel["buybox_pct"])
    X = _encode(panel)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=0, stratify=y)
    print(f"[bbox-rules] train={len(X_train):,}  test={len(X_test):,}")
    print(f"[bbox-rules] class balance (train): {y_train.value_counts().to_dict()}")

    clf = DecisionTreeClassifier(max_depth=MAX_DEPTH, min_samples_leaf=MIN_SAMPLES_LEAF,
                                  random_state=0)
    clf.fit(X_train, y_train)

    os.makedirs(OUT_DIR, exist_ok=True)

    rules = export_text(clf, feature_names=FEATURES, show_weights=True)
    rules_path = os.path.join(OUT_DIR, "rules.txt")
    with open(rules_path, "w") as f:
        f.write(rules)
    print(f"\n[bbox-rules] the actual rules -- read this with your team:\n{rules}")
    print(f"[bbox-rules] saved -> {rules_path}")

    n_leaves = clf.get_n_leaves()
    fig, ax = plt.subplots(figsize=(max(20, 2.2 * n_leaves), 10 + MAX_DEPTH))
    plot_tree(clf, feature_names=FEATURES, class_names=clf.classes_.tolist(),
              filled=True, rounded=True, fontsize=8, ax=ax)
    fig.tight_layout()
    tree_path = os.path.join(OUT_DIR, "tree.png")
    fig.savefig(tree_path, dpi=140)
    plt.close(fig)
    print(f"[bbox-rules] saved -> {tree_path}  ({n_leaves} leaves -- see leaf_distributions.png "
          f"for the readable version)")

    dist_path = os.path.join(OUT_DIR, "leaf_distributions.png")
    plot_leaf_distributions(clf, X, panel["buybox_pct"], dist_path)

    importances = pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print(f"\n[bbox-rules] feature importances (which columns the tree actually used):")
    print(importances.to_string())

    y_pred = clf.predict(X_test)
    print(f"\n[bbox-rules] holdout classification report:")
    # zero_division=0 silences the warning for LOST -- the tree never predicts it at all
    # (see feature importances/confusion matrix below), which is a real finding to bring
    # to the team, not a metric-computation error to suppress and forget about.
    print(classification_report(y_test, y_pred, labels=CLASSES, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=CLASSES)
    print(f"[bbox-rules] confusion matrix (rows=actual, cols=predicted, order={CLASSES}):")
    print(pd.DataFrame(cm, index=[f"actual_{c}" for c in CLASSES],
                        columns=[f"pred_{c}" for c in CLASSES]).to_string())

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title("holdout confusion matrix -- where the rules go wrong")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    cm_path = os.path.join(OUT_DIR, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=120)
    plt.close(fig)
    print(f"[bbox-rules] saved -> {cm_path}")

    joblib.dump({"model": clf, "features": FEATURES, "classes": clf.classes_.tolist()}, MODEL_PATH)
    print(f"[bbox-rules] saved -> {MODEL_PATH}")
    return clf


if __name__ == "__main__":
    fit_and_report()
