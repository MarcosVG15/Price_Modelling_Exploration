"""
Generic, category-agnostic standardisation of the merged spec table.

This does NOT assume speakers/audio. It processes *every* spec column in
``data_files/grouped_data.csv`` automatically:

  1. Coalesce ttf_/btf_ + language-duplicate headers into one column per concept.
  2. Type each concept: numeric+unit / dimensions(AxBxC) / boolean / count / categorical / drop.
  3. For numeric columns, auto-detect the physical DIMENSION from whatever unit words
     appear (power, mass, length, time, frequency, voltage, current, resistance,
     charge/mAh, energy, data, sound-dB, temperature, pressure, angle, count) and
     convert to that dimension's canonical base unit. Handles comma-decimals, DE/IT/FR
     unit words, and ranges (-> _min/_max).
  4. Null order-of-magnitude parse errors with a scale-free (log-MAD) outlier rule.
  5. Normalise universal categoricals (boolean, colour) and drop identifier/garbled columns.
  6. Prune columns below a coverage threshold.

New physical units are added by editing DIMENSIONS; new products need no code changes.

Output:
  data_files/clean_features.csv    - one typed column per concept (numeric -> float, etc.)
  data_files/data_dictionary.csv   - concept, source columns, detected type/dimension, coverage
  data_files/standardize_report.txt- per-column decisions (what was parsed / dropped and why)

Run:  python standardize.py
"""
from __future__ import annotations
import re
import unicodedata
import numpy as np
import pandas as pd
from tqdm import tqdm

SRC = "data_files/grouped_data.csv"
OUT = "data_files/clean_features.csv"
DICT = "data_files/data_dictionary.csv"
REPORT = "data_files/standardize_report.txt"

N_BASE_COLS = 18            # cols 0..17 are product_id, price, rank, ... (non-spec)
MIN_COVERAGE = 0.02         # drop concepts filled in < 2% of rows
KEEP_BASE = ["product_id", "keyword_id", "market_id", "price", "rank",
             "average_rating", "number_of_reviews", "buy_box", "discount", "day",
             "title", "asin_ean_id", "page", "delivery_time", "image_url"]

# ---------------------------------------------------------------------------
# European number parsing (comma = decimal separator)
# ---------------------------------------------------------------------------
_NUM_RE = re.compile(r"[+-]?[0-9][0-9\.\,  ]*[0-9]|[0-9]")

def to_float_eu(tok):
    t = str(tok).strip().replace(" ", "").replace(" ", "")
    if not t:
        return None
    if "," in t:                       # comma decimal, dot = thousands
        t = t.replace(".", "").replace(",", ".")
    elif t.count(".") == 1:
        left, right = t.split(".")
        if len(right) == 3 and len(left) <= 3:   # 20.000 -> 20000
            t = left + right
    else:
        t = t.replace(".", "")
    try:
        return float(t)
    except ValueError:
        return None


def _norm(s):
    """lowercase, strip accents, collapse whitespace."""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s.strip().lower())


def find_pairs(text):
    """[(value, unit_word)] for every number in text (unit = first word after it)."""
    s = _norm(text)
    out = []
    for m in _NUM_RE.finditer(s):
        val = to_float_eu(m.group(0))
        if val is None:
            continue
        tail = s[m.end():].strip()
        word = re.match(r"[^\d,;/()]*", tail).group(0).strip()
        unit = word.split()[0] if word else ""
        out.append((val, unit))
    return out


# ---------------------------------------------------------------------------
# Physical-dimension registry:  dimension -> (base_unit, {unit_word: ->base multiplier})
# Unit words are accent-stripped/lowercased to match _norm(). Add freely.
# ---------------------------------------------------------------------------
DIMENSIONS = {
    "power_w": ("W", {"watt": 1, "watts": 1, "w": 1, "kilowatt": 1e3, "kw": 1e3,
                      "milliwatt": 1e-3, "mw": 1e-3}),
    "voltage_v": ("V", {"volt": 1, "volts": 1, "v": 1, "millivolt": 1e-3, "mv": 1e-3,
                        "kilovolt": 1e3, "kv": 1e3}),
    "current_a": ("A", {"ampere": 1, "amperes": 1, "ampere(s)": 1, "amp": 1, "a": 1,
                        "milliampere": 1e-3, "milliamperes": 1e-3, "ma": 1e-3}),
    "resistance_ohm": ("ohm", {"ohm": 1, "ohms": 1, "kiloohm": 1e3, "kohm": 1e3,
                               "megaohm": 1e6}),
    "freq_hz": ("Hz", {"hz": 1, "hertz": 1, "khz": 1e3, "kilohertz": 1e3, "mhz": 1e6,
                       "megahertz": 1e6, "ghz": 1e9, "gigahertz": 1e9,
                       "millihertz": 1e-3, "mikrohertz": 1e-6, "microhertz": 1e-6}),
    "charge_mah": ("mAh", {"mah": 1, "milliamperes-heure": 1, "milliamperora": 1,
                           "milliamperestunden": 1, "milliampere-hours": 1,
                           "milliamperestunde": 1, "ah": 1e3, "amperestunden": 1e3,
                           "amperes-heure": 1e3, "amperora": 1e3, "amperestunde": 1e3}),
    "energy_wh": ("Wh", {"wh": 1, "wattstunden": 1, "wattheures": 1, "wattheure": 1,
                         "wattora": 1, "kwh": 1e3, "kilowattstunden": 1e3}),
    "mass_g": ("g", {"g": 1, "gr": 1, "gramm": 1, "gramme": 1, "grammes": 1, "grammi": 1,
                     "grammo": 1, "kg": 1e3, "kilogramm": 1e3, "kilogramme": 1e3,
                     "kilogrammes": 1e3, "chilogrammi": 1e3, "kilogrammi": 1e3,
                     "chilogrammo": 1e3, "mg": 1e-3, "milligramm": 1e-3,
                     "pound": 453.592, "pounds": 453.592, "livre": 453.592,
                     "livres": 453.592, "libbre": 453.592, "libbra": 453.592,
                     "lb": 453.592, "lbs": 453.592, "once": 28.3495, "oncia": 28.3495,
                     "onces": 28.3495, "oz": 28.3495, "ounce": 28.3495,
                     "ounces": 28.3495, "unze": 28.3495}),
    "length_mm": ("mm", {"mm": 1, "millimetre": 1, "millimetres": 1, "millimeter": 1,
                         "millimetri": 1, "millimetro": 1, "cm": 10, "centimetre": 10,
                         "centimetres": 10, "zentimeter": 10, "centimetri": 10,
                         "centimetro": 10, "m": 1e3, "metre": 1e3, "metres": 1e3,
                         "meter": 1e3, "metri": 1e3, "metro": 1e3, "km": 1e6,
                         "inch": 25.4, "inches": 25.4, "pouce": 25.4, "pouces": 25.4,
                         "zoll": 25.4, "pollici": 25.4, "pollice": 25.4, "foot": 304.8,
                         "feet": 304.8, "pied": 304.8, "pieds": 304.8, "piede": 304.8,
                         "piedi": 304.8, "fuss": 304.8, "micron": 1e-3,
                         "mikrometer": 1e-3}),
    "time_h": ("h", {"hour": 1, "hours": 1, "heure": 1, "heures": 1, "ora": 1, "ore": 1,
                     "stunde": 1, "stunden": 1, "std": 1, "h": 1, "minute": 1/60,
                     "minutes": 1/60, "minuti": 1/60, "minuto": 1/60, "min": 1/60,
                     "minuten": 1/60, "second": 1/3600, "seconds": 1/3600,
                     "secondes": 1/3600, "secondi": 1/3600, "sekunden": 1/3600,
                     "day": 24, "days": 24, "jour": 24, "jours": 24, "giorno": 24,
                     "giorni": 24, "tag": 24, "tage": 24, "week": 168, "weeks": 168,
                     "month": 730, "months": 730, "mois": 730, "mese": 730, "mesi": 730,
                     "monat": 730, "monate": 730, "year": 8760, "years": 8760, "an": 8760,
                     "ans": 8760, "anno": 8760, "anni": 8760, "jahr": 8760,
                     "jahre": 8760}),
    "data_gb": ("GB", {"gb": 1, "go": 1, "gigabyte": 1, "mb": 1e-3, "mo": 1e-3,
                       "megabyte": 1e-3, "tb": 1e3, "to": 1e3, "terabyte": 1e3,
                       "kb": 1e-6, "ko": 1e-6}),
    "sound_db": ("dB", {"db": 1, "decibel": 1, "decibels": 1, "dezibel": 1}),
    "temp_c": ("degC", {"degc": 1, "c": 1, "celsius": 1, "grad": 1, "degre": 1,
                        "degres": 1, "gradi": 1}),
    "pressure_bar": ("bar", {"bar": 1, "mbar": 1e-3, "pa": 1e-5, "kpa": 1e-2,
                             "psi": 0.0689476}),
    "angle_deg": ("deg", {"deg": 1, "degree": 1, "degrees": 1, "grado": 1, "gradi": 1}),
    "count": ("count", {"count": 1, "counts": 1, "conteggio": 1, "unite": 1, "unites": 1,
                        "unita": 1, "unit": 1, "units": 1, "stuck": 1, "stucke": 1,
                        "anzahl": 1, "pezzi": 1, "pezzo": 1, "piece": 1, "pieces": 1,
                        "stk": 1, "total": 1, "": 1}),
}

# Unit words that flag a column as NOT a measurement (rating/junk/battery-chemistry text)
BAD_UNIT_WORDS = {"sur", "su", "von", "etoiles", "stelle", "sterne", "stars", "modificateur",
                  "modificatore", "lithium-ion", "lithium-polymere", "ioni", "aaa", "aa",
                  "lithium", "li-ion", "unknown"}


def classify_dimension(unit_counts):
    """Given Counter of unit words in a column, return (dimension, match_fraction)."""
    total = sum(unit_counts.values())
    if total == 0:
        return None, 0.0
    best, best_frac = None, 0.0
    for dim, (_base, lex) in DIMENSIONS.items():
        matched = sum(c for u, c in unit_counts.items() if u in lex)
        frac = matched / total
        if frac > best_frac:
            best, best_frac = dim, frac
    return best, best_frac


def convert_cell(text, dim):
    """Return list of values (base unit) parsed from a cell for dimension *dim*."""
    _base, lex = DIMENSIONS[dim]
    vals = []
    for v, u in find_pairs(text):
        if u in lex:
            vals.append(v * lex[u])
        elif u == "" and "" in lex:      # count-type: bare number
            vals.append(v)
    return vals


# ---------------------------------------------------------------------------
# Dimension-triple detector  (AxBxC  ->  depth/width/height)
# ---------------------------------------------------------------------------
_TRIPLE_RE = re.compile(
    r"([0-9][0-9\.\,]*)\s*[a-z]?\s*x\s*([0-9][0-9\.\,]*)\s*[a-z]?\s*x\s*([0-9][0-9\.\,]*)",
    re.IGNORECASE)

def parse_triple_mm(text):
    m = _TRIPLE_RE.search(_norm(text))
    if not m:
        return None
    nums = [to_float_eu(m.group(i)) for i in (1, 2, 3)]
    tail = _norm(text)[m.end():]
    uw = re.match(r"[^\d]*", tail).group(0).strip().split()
    mult = DIMENSIONS["length_mm"][1].get(uw[0], 10) if uw else 10
    return [n * mult if n is not None else None for n in nums]


# ---------------------------------------------------------------------------
# Universal categoricals
# ---------------------------------------------------------------------------
COLOR = {
    "black": ["noir", "schwarz", "black", "nero", "nera", "negro"],
    "white": ["blanc", "weiss", "white", "bianco", "bianca", "blanco"],
    "blue": ["bleu", "blau", "blue", "blu", "azul", "azzurro"],
    "red": ["rouge", "rot", "red", "rosso", "rojo"],
    "green": ["vert", "grun", "green", "verde"],
    "grey": ["gris", "grau", "grey", "gray", "grigio", "anthrazit"],
    "brown": ["marron", "braun", "brown", "marrone", "bois", "holz", "wood", "legno"],
    "pink": ["rose", "rosa", "pink", "rosato"],
    "silver": ["argent", "silber", "silver", "argento", "plata"],
    "gold": ["gold", "oro", "dore", "dorato"],
    "beige": ["beige", "creme", "cream"],
    "orange": ["orange", "arancione", "arancio"],
    "yellow": ["jaune", "gelb", "yellow", "giallo"],
    "purple": ["violet", "lila", "purple", "viola", "violett"],
}
TRUE_WORDS = {"true", "vrai", "vraie", "vrais", "vraies", "richtig", "echt", "wahr",
              "oui", "ja", "si", "yes", "wasserfest"}
FALSE_WORDS = {"false", "faux", "fausse", "fausses", "falsch", "non", "nein", "no",
               "nicht wasserfest"}

def map_color(v):
    low = _norm(v)
    for canon, syns in COLOR.items():
        if any(re.search(r"\b" + re.escape(s), low) for s in syns):
            return canon
    return None

def map_bool(v):
    low = _norm(v)
    if re.search(r"ipx?\s*\d", low):
        return True
    if low in TRUE_WORDS:
        return True
    if low in FALSE_WORDS:
        return False
    return None


# ---------------------------------------------------------------------------
# Concept coalescing:  normalise headers, merge ttf_/btf_ + language duplicates
# ---------------------------------------------------------------------------
HEADER_SYNONYMS = {   # frequent leftover native headers -> canonical english concept
    "marque": "brand", "marchio": "brand", "marke": "brand", "manufacturer": "brand",
    "producer": "brand", "hersteller": "brand", "colore": "color", "couleur": "color",
    "farbe": "color", "materiale": "material", "matiere": "material",
    "modele": "model name", "modello": "model name", "model number": "model name",
}

def concept_key(col):
    name = re.sub(r"^(ttf|btf)_", "", col)
    name = _norm(name)
    name = name.replace("_", " ")
    name = re.sub(r"\([^)]*\)", "", name)      # drop "(in watts)"-style suffixes
    name = name.rstrip(".:").strip()
    name = re.sub(r"\s+", " ", name)
    name = HEADER_SYNONYMS.get(name, name)
    return name


def coalesce_concepts(df, spec_cols):
    """Return {concept -> coalesced Series}, merging duplicate-header columns by bfill."""
    groups = {}
    for c in spec_cols:
        groups.setdefault(concept_key(c), []).append(c)
    concepts = {}
    sources = {}
    for concept, cols in groups.items():
        # order by coverage so the best-filled column leads the bfill
        cols = sorted(cols, key=lambda c: df[c].notna().mean(), reverse=True)
        concepts[concept] = df[cols].bfill(axis=1).iloc[:, 0]
        sources[concept] = cols
    return concepts, sources


# ---------------------------------------------------------------------------
# Conservative parse-error nulling.  Only removes values a full order of magnitude
# beyond the 1st/99th percentiles, so genuine spread and multi-modality (e.g. 4 vs
# 10000 ohm) survive, while concatenation errors like "7020000 Hz" are dropped.
# ---------------------------------------------------------------------------
def null_extreme(s):
    v = s.dropna()
    if len(v) < 30:
        return s, 0
    lo, hi = v.quantile(0.01) / 10, v.quantile(0.99) * 10
    bad = v.index[(v < lo) | (v > hi)]
    out = s.copy()
    out.loc[bad] = np.nan
    return out, len(bad)


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------
def typed_column(concept, raw, report):
    """Return dict {out_name: Series} for one coalesced concept, or {} to drop."""
    from collections import Counter
    s = raw.dropna().astype(str)
    n = len(s)
    if n == 0:
        return {}
    frac_num = s.str.match(r"^\s*[+-]?[0-9]").mean()

    # 1) dimension triples (A x B x C)
    # Use .search (not str.contains) so _TRIPLE_RE's capture groups don't trigger a warning.
    if s.apply(lambda v: _TRIPLE_RE.search(v) is not None).mean() > 0.4:
        parsed = raw.apply(lambda x: parse_triple_mm(x) if pd.notna(x) else None)
        report.append(f"[dims ] {concept}: A x B x C -> depth/width/height (mm)")
        return {f"{concept}__depth_mm": parsed.apply(lambda t: t[0] if t else np.nan),
                f"{concept}__width_mm": parsed.apply(lambda t: t[1] if t else np.nan),
                f"{concept}__height_mm": parsed.apply(lambda t: t[2] if t else np.nan)}

    # 2) numeric + unit
    if frac_num >= 0.5:
        units = Counter()
        for v in s:
            for _val, u in find_pairs(v):
                units[u] += 1
        bad = sum(c for u, c in units.items() if u in BAD_UNIT_WORDS)
        if bad / max(sum(units.values()), 1) > 0.3:
            report.append(f"[DROP ] {concept}: rating/junk units {units.most_common(3)}")
            return {}
        dim, frac = classify_dimension(units)
        if dim and frac >= 0.5:
            base, _ = DIMENSIONS[dim]
            per_cell = raw.apply(lambda x: convert_cell(x, dim) if pd.notna(x) else [])
            is_range = per_cell.apply(lambda l: len(l) >= 2 and max(l) > min(l))
            if is_range.mean() > 0.05:
                lo, _ = null_extreme(per_cell.apply(lambda l: min(l) if l else np.nan))
                hi, _ = null_extreme(per_cell.apply(lambda l: max(l) if l else np.nan))
                report.append(f"[num  ] {concept}: {dim} range (base {base}, unit-match {frac:.0%})")
                return {f"{concept}__{dim}_min": lo, f"{concept}__{dim}_max": hi}
            col, nout = null_extreme(per_cell.apply(lambda l: l[0] if l else np.nan))
            report.append(f"[num  ] {concept}: {dim} (base {base}, unit-match {frac:.0%}, {nout} outliers nulled)")
            return {f"{concept}__{dim}": col}
        report.append(f"[text ] {concept}: numeric but no clear dimension {units.most_common(3)}")
        # fall through to categorical handling

    # 3) boolean
    bmap = raw.apply(map_bool)
    if bmap.notna().mean() >= 0.6 * raw.notna().mean() and raw.notna().mean() > 0:
        report.append(f"[bool ] {concept}")
        return {f"{concept}__bool": bmap}

    # 4) colour concept
    if "color" in concept or "colour" in concept:
        report.append(f"[color] {concept}")
        return {f"{concept}__color": raw.apply(lambda v: map_color(v) if pd.notna(v) else np.nan)}

    # 5) identifier drop:  ASIN/EAN/GTIN-style codes, high cardinality, or long free text
    asin_frac = s.str.match(r"(?i)^b0[a-z0-9]{8}$").mean()
    ean_frac = s.str.match(r"^\d{8,14}$").mean()
    card = s.nunique() / n
    avglen = s.str.len().mean()
    if asin_frac > 0.3 or ean_frac > 0.5 or card > 0.9 or avglen > 120:
        why = ("asin" if asin_frac > 0.3 else "ean" if ean_frac > 0.5 else
               f"cardinality {card:.2f}" if card > 0.9 else f"avglen {avglen:.0f}")
        report.append(f"[DROP ] {concept}: id/free-text ({why})")
        return {}

    # 6) generic categorical -> normalised string (lowercase, accent-free, trimmed)
    report.append(f"[cat  ] {concept}: normalised string (cardinality {card:.2f})")
    return {concept: raw.apply(lambda v: _norm(v) if pd.notna(v) else np.nan)}


def build(df):
    spec_cols = list(df.columns[N_BASE_COLS:])
    concepts, sources = coalesce_concepts(df, spec_cols)
    report = [f"{len(spec_cols)} spec columns -> {len(concepts)} concepts after coalescing", ""]
    print(f"      {len(spec_cols)} spec cols -> {len(concepts)} concepts to standardize")

    columns = {b: df[b] for b in KEEP_BASE if b in df.columns}

    dict_rows = []
    ordered = sorted(concepts.items(), key=lambda kv: -kv[1].notna().mean())
    for concept, raw in tqdm(ordered, total=len(ordered), desc="  standardizing concepts"):
        if raw.notna().mean() < MIN_COVERAGE:
            continue
        produced = typed_column(concept, raw, report)
        for name, series in produced.items():
            cov = series.notna().mean()
            if cov < MIN_COVERAGE:
                continue
            columns[name] = series
            dict_rows.append({
                "feature": name, "concept": concept,
                "source_columns": "; ".join(sources[concept]),
                "type": name.split("__")[-1] if "__" in name else "categorical",
                "coverage": round(cov, 3),
            })
    out = pd.concat(columns, axis=1)
    ddict = pd.DataFrame(dict_rows).sort_values("coverage", ascending=False)
    return out, ddict, report


def main(df=None):
    # Accept a DataFrame directly; fall back to reading SRC only for standalone runs.
    if df is None:
        print(f"Reading {SRC} ...")
        df = pd.read_csv(SRC, low_memory=False)
    print(f"  {df.shape[0]} rows x {df.shape[1]} cols")

    out, ddict, report = build(df)

    # Returns dataframes instead of writing CSVs.
    numeric = ddict[ddict["type"].str.contains("_", na=False) | ddict["type"].isin(["count"])]
    print(f"Numeric features: {len(numeric)} | categorical/bool/color: {len(ddict) - len(numeric)}\n")
    with pd.option_context("display.max_rows", None, "display.width", 160,
                           "display.max_colwidth", 45):
        print(ddict.head(60).to_string(index=False))

    return out, ddict


if __name__ == "__main__":
    main()
