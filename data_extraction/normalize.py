'''
Generic, category-agnostic normalization for the feature matrix.

The data is non-deterministic across product families (headphones today, some-
thing else tomorrow), so nothing here is hard-coded to a product, a feature name,
or a unit. Every decision is made from the *values* of each column:

  - numeric-ish columns  -> pull the real number out of messy strings
                            ("$1,299.00", "20 hrs", "20-24h", "IPX4", "95%")
                            then robust-scale (median / IQR) so wildly different
                            magnitudes become comparable and outliers (common in
                            scraped data) don't blow up the scale.
  - categorical columns  -> canonicalize the string form (unicode, case, spacing,
                            stray punctuation) so "Bluetooth 5.0", "bluetooth 5.0 "
                            and "BLUETOOTH  5.0" collapse to one token.

Design goals:
  * fit/transform, sklearn-style: parameters learned on fit are reused on
    transform, so a normalizer fit on one batch applies identically to the next.
  * robust by default: median/IQR, with graceful fallbacks (MAD -> std -> 1) for
    degenerate columns, so a constant or near-empty column never divides by zero.
  * self-contained type detection: if no external feature_types map is supplied,
    a column is treated as numeric when a high fraction of its non-null values
    parse to a number, categorical otherwise.
  * missing stays missing: NaN is preserved, never silently imputed to 0 (the
    downstream pipeline tracks present/absent explicitly and relies on that).
  * deterministic: sorted iteration and fixed rules, matching the reproducibility
    constraints already honored elsewhere in the pipeline.
'''

import re
import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


# ---- numeric extraction -----------------------------------------------------

# a signed decimal number, optionally with thousands separators (1,299 / 1 299)
_NUMBER_RE = re.compile(r"[-+]?\d{1,3}(?:[, \s]\d{3})+(?:\.\d+)?|[-+]?\d*\.?\d+")
# words/symbols that signal the two numbers form a range, so we take the midpoint
# (second operand may carry its own sign, e.g. "-10 to -5")
_RANGE_HINT_RE = re.compile(r"\d\s*(?:-|–|—|to|/|~|\.\.)\s*[-+]?\d", re.I)


def to_number(value):
    '''Best-effort single float out of an arbitrary cell. NaN if nothing usable.

    Handles currency, units, thousands separators and ranges without knowing
    which unit or currency it is looking at:
        "$1,299.00" -> 1299.0     "20 hrs"  -> 20.0
        "IPX4"      -> 4.0        "20-24 h" -> 22.0 (range midpoint)
        "95%"       -> 95.0       "n/a"     -> nan
    '''
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = unicodedata.normalize("NFKC", str(value))
    is_range = bool(_RANGE_HINT_RE.search(text))
    if is_range:
        # a dash between two digits ("20-24") is a range separator, not a minus
        # sign on the second number; blank it so we don't read 24 as -24
        text = re.sub(r"(?<=\d)\s*[-–—~](?=\d)", " ", text)

    matches = _NUMBER_RE.findall(text)
    if not matches:
        return np.nan

    nums = []
    for m in matches:
        # drop thousands separators (comma / space / nbsp between digit groups)
        cleaned = re.sub(r"[, \s](?=\d{3}\b)", "", m)
        try:
            nums.append(float(cleaned))
        except ValueError:
            continue
    if not nums:
        return np.nan

    # a range like "20-24 hours" -> use the midpoint; otherwise the first number
    if len(nums) >= 2 and is_range:
        return float(np.mean(nums[:2]))
    return nums[0]


def parse_numeric(series):
    '''Vectorized to_number over a Series -> float Series (NaN where unparseable).'''
    return series.map(to_number).astype(float)


# ---- categorical canonicalization -------------------------------------------

_WS_RE = re.compile(r"\s+")
_EDGE_PUNCT_RE = re.compile(r"^[\s\-_.,;:/\\|]+|[\s\-_.,;:/\\|]+$")


def canonicalize(value):
    '''Collapse trivially-different string spellings of the same category.

    Unicode-normalize, casefold, squeeze internal whitespace, strip edge
    punctuation. Deliberately conservative: it does NOT stem, translate, or
    reorder tokens, so distinct real values are never merged.
    '''
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = _WS_RE.sub(" ", text).strip()
    text = _EDGE_PUNCT_RE.sub("", text)
    return text if text else np.nan


def canonicalize_series(series):
    return series.map(canonicalize)


# ---- robust scaling ----------------------------------------------------------

def _robust_params(values, method="robust"):
    '''Center/scale for a 1-D array of finite numbers, with safe fallbacks.

    Returns (center, scale) such that (x - center) / scale is the normalized
    value. `scale` is guaranteed > 0 so the transform never divides by zero.
    '''
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, 1.0

    if method == "zscore":
        center = float(x.mean())
        scale = float(x.std())
    elif method == "minmax":
        lo, hi = float(x.min()), float(x.max())
        return lo, (hi - lo) or 1.0
    else:  # "robust" (default): median / IQR, resistant to outliers
        center = float(np.median(x))
        q75, q25 = np.percentile(x, [75, 25])
        scale = float(q75 - q25)
        if scale == 0:  # degenerate IQR -> fall back to MAD, then std
            mad = float(np.median(np.abs(x - center)))
            scale = mad * 1.4826 if mad > 0 else float(x.std())

    if not np.isfinite(scale) or scale == 0:
        scale = 1.0
    return center, scale


# ---- the normalizer ----------------------------------------------------------

class Normalizer:
    '''Type-adaptive fit/transform over a (multi-index) feature DataFrame.

    Works on both flat columns and the (provenance, feature) two-level columns
    used by the voting pipeline. When columns are two-level, all provenance
    columns of the same feature share one set of scaling parameters, so the same
    physical attribute is normalized consistently regardless of where it was
    scraped from.

    Usage:
        norm = Normalizer(method="robust")
        clean = norm.fit_transform(df)                 # or fit(df); transform(df)
        norm.save("data_files/normalizer_<term>.json") # reuse on the next batch
    '''

    def __init__(self, method="robust", numeric_min_frac=0.90, min_non_null=5,
                 id_uniq_ratio=0.98):
        self.method = method
        self.numeric_min_frac = numeric_min_frac  # share of parseable values to call a column numeric
        self.min_non_null = min_non_null          # below this we don't trust auto-detection -> categorical
        self.id_uniq_ratio = id_uniq_ratio        # near-unique numeric = identifier, left untouched
        self.params_ = {}                         # feature key -> {"type", "center", "scale"}
        self._two_level = None

    # -- helpers ----------------------------------------------------------------

    @staticmethod
    def _feature_key(col):
        # two-level -> the feature (level 1); flat -> the column name
        return str(col[1]) if isinstance(col, tuple) else str(col)

    @staticmethod
    def _nunique(series):
        '''nunique() that tolerates unhashable cells.

        Upstream extraction can leave list-valued cells in a column that still
        reads as numeric (str([1, 2]) -> "[1, 2]" parses to a number), and
        pandas' nunique() raises on unhashable values. Fall back to counting
        unique string forms, which is exact enough for the near-unique
        identifier heuristic this feeds.
        '''
        try:
            return series.nunique()
        except TypeError:
            def _hashable(v):
                try:
                    hash(v)
                    return v
                except TypeError:
                    return repr(v)
            return series.map(_hashable).nunique()

    def _group_columns(self, df):
        '''feature key -> list of dataframe columns contributing to it (sorted).'''
        groups = {}
        for col in df.columns:
            groups.setdefault(self._feature_key(col), []).append(col)
        return {k: groups[k] for k in sorted(groups)}

    def _detect_numeric(self, pooled, feature_types, key):
        if feature_types is not None and key in feature_types:
            return feature_types[key] == "numeric"
        non_null = pooled.dropna()
        if len(non_null) < self.min_non_null:
            return False
        frac = parse_numeric(non_null).notna().mean()
        return frac >= self.numeric_min_frac

    # -- fit / transform --------------------------------------------------------

    def fit(self, df, feature_types=None):
        '''Learn per-feature type and (for numerics) center/scale.

        feature_types: optional {feature_name: "numeric"|"categorical"|...} to
        override auto-detection (e.g. the pipeline's FeatureType verdicts).
        '''
        self._two_level = isinstance(df.columns, pd.MultiIndex)
        self.params_ = {}

        n_rows = len(df)
        for key, cols in self._group_columns(df).items():
            # explicit identifier verdict from the pipeline's typer: pass through
            if feature_types is not None and feature_types.get(key) == "identifier":
                self.params_[key] = {"type": "identifier", "center": None, "scale": None}
                continue

            pooled = pd.Series(df[cols].to_numpy().ravel())
            if self._detect_numeric(pooled, feature_types, key):
                # a near-unique numeric column is a key (id/serial/barcode), not a
                # measurable quantity: scaling it is meaningless, so pass it through.
                # Only when the type was auto-detected, never over an explicit
                # verdict -- so the uniqueness count is skipped entirely (and never
                # touches possibly-unhashable cells) when a verdict is present.
                if key not in (feature_types or {}) and n_rows:
                    uniq_ratio = self._nunique(pooled) / n_rows
                    if uniq_ratio >= self.id_uniq_ratio:
                        self.params_[key] = {"type": "identifier", "center": None, "scale": None}
                        continue
                center, scale = _robust_params(parse_numeric(pooled), self.method)
                self.params_[key] = {"type": "numeric", "center": center, "scale": scale}
            else:
                self.params_[key] = {"type": "categorical", "center": None, "scale": None}
        return self

    def transform(self, df):
        '''Return a normalized copy of df, same shape and column layout.

        Numeric columns become floats scaled by the fitted params; categorical
        columns become canonicalized strings. Unseen features (not in fit) are
        auto-detected on the fly so transform never silently drops a column.
        '''
        if not self.params_:
            raise RuntimeError("Normalizer.transform called before fit")

        out = df.copy()
        for key, cols in self._group_columns(df).items():
            spec = self.params_.get(key)
            if spec is None:  # feature unseen at fit time: detect and scale locally
                pooled = pd.Series(df[cols].to_numpy().ravel())
                if self._detect_numeric(pooled, None, key):
                    c, s = _robust_params(parse_numeric(pooled), self.method)
                    spec = {"type": "numeric", "center": c, "scale": s}
                else:
                    spec = {"type": "categorical"}

            if spec["type"] == "numeric":
                for col in cols:
                    parsed = parse_numeric(df[col])
                    out[col] = (parsed - spec["center"]) / spec["scale"]
            elif spec["type"] == "identifier":
                pass  # keys carry no similarity signal; leave the raw value in place
            else:
                for col in cols:
                    out[col] = canonicalize_series(df[col])
        return out

    def fit_transform(self, df, feature_types=None):
        return self.fit(df, feature_types).transform(df)

    # -- persistence ------------------------------------------------------------

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "method": self.method,
            "numeric_min_frac": self.numeric_min_frac,
            "min_non_null": self.min_non_null,
            "two_level": self._two_level,
            "params": self.params_,
        }, indent=2, sort_keys=True))
        return path

    @classmethod
    def load(cls, path):
        blob = json.loads(Path(path).read_text())
        obj = cls(method=blob["method"],
                  numeric_min_frac=blob["numeric_min_frac"],
                  min_non_null=blob["min_non_null"])
        obj._two_level = blob.get("two_level")
        obj.params_ = blob["params"]
        return obj

    # -- introspection ----------------------------------------------------------

    def summary(self):
        '''DataFrame of what was learned per feature (for a quick sanity look).'''
        rows = [{"feature": k, **v} for k, v in sorted(self.params_.items())]
        return pd.DataFrame(rows)


# ---- standardize step: persist the normalized matrix to disk -----------------

def load_type_cache(path):
    '''{feature: type} out of a FeatureType cache json ({feat: {"type": ...}}).'''
    blob = json.loads(Path(path).read_text())
    return {k: v["type"] for k, v in blob.items()}


def standardize(in_path, out_path=None, feature_types=None, type_cache=None,
                method="robust", save_params=True):
    '''Read the raw feature CSV, normalize it, and write the standardized CSV back.

    This is the persisted normalization step: rather than re-normalizing on every
    run, the cleaned matrix is materialized once to `out_path` and the learned
    parameters to a `.params.json` sidecar, so downstream stages read already-
    standardized data and the transform is exactly reproducible on new batches.

    Column types come from the pipeline's FeatureType verdicts whenever they are
    available, so the standardized matrix is typed IDENTICALLY to how the voting
    stages will read it (a column the pipeline treats as categorical must never
    be numerically scaled here, and vice versa). Auto-detection is only the
    fallback for features with no verdict (e.g. a brand-new product category).

    The two-level (provenance, feature) header is preserved, so both readers in
    the pipeline keep working unchanged:
        - voting_network / feature_cluster : pd.read_csv(header=[0, 1])
        - pre_cluster                      : pd.read_csv(skiprows=[0])

    in_path       : raw all_feature_data_*.csv
    out_path      : defaults to '<in>_standardized.csv' next to the input
    feature_types : {feature: "numeric"|"categorical"|"identifier"} dict
    type_cache    : path to a FeatureType cache json (data_files/feature_types_*.json);
                    ignored when feature_types is given
    Returns (standardized_df, fitted_Normalizer).
    '''
    in_path = Path(in_path)
    if out_path is None:
        out_path = in_path.with_name(f"{in_path.stem}_standardized.csv")
    out_path = Path(out_path)

    if feature_types is None and type_cache is not None:
        feature_types = load_type_cache(type_cache)

    df = pd.read_csv(in_path, header=[0, 1], low_memory=False)
    norm = Normalizer(method=method)
    clean = norm.fit_transform(df, feature_types=feature_types)

    if feature_types:
        missing = [k for k in norm.params_ if k not in feature_types]
        if missing:
            print(f"[standardize] {len(missing)} features had no type verdict; "
                  f"auto-detected (e.g. {missing[:3]})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(out_path, index=False)  # writes both header rows for MultiIndex cols
    if save_params:
        norm.save(out_path.with_suffix(".params.json"))
    return clean, norm
