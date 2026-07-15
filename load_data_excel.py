import os
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, bindparam
from sqlalchemy.orm import sessionmaker

load_dotenv()

SCRAPE_DATABASE_URL = os.getenv('SCRAPE_DATABASE_URL')
API_DATABASE_URL = os.getenv('API_DATABASE_URL')


from data_extraction.helper_methods.textual_feature_extraction import textual_feature_extractors
from data_extraction.paragraph_data_extraction import paragraph_extraction
from data_extraction.clean_system import clean_system
from data_extraction.normalize import Normalizer, load_type_cache
from sqlalchemy import create_engine, text


SEGMENTATION_COLUMNS = {
    1: '"Segmentation_level_1"',
    2: '"Segmentation_level_2"',
    3: '"Segmentation_level_3"',
}

def extract_keyword_ids(level, value):
    if level not in SEGMENTATION_COLUMNS:
        raise ValueError(f"level must be one of {list(SEGMENTATION_COLUMNS)}")

    column = SEGMENTATION_COLUMNS[level]          # safe: comes from the whitelist

    engine = create_engine(SCRAPE_DATABASE_URL)
    query = text(f"""
        SELECT id
        FROM key_words
        WHERE {column} ILIKE :value
    """)   # ILIKE = case-insensitive; {column} = whitelisted; :value = bind param (injection-safe)

    with engine.connect() as conn:
        # %value% = "contains", so 'speaker' matches 'Speakers'
        df = pd.read_sql(query, conn, params={"value": f"%{value}%"})

    return df["id"].tolist()



def extract_feature_para(paragraphs):
    # Build a per-paragraph feature table: one row per product, columns are a 2-level
    # MultiIndex (paragraph_N, feature_name) so we KEEP TRACK of which paragraph each
    # feature came from. Feeds straight into merge_tables (already MultiIndexed).
    tfe = textual_feature_extractors()
    feature_names = list(tfe.STYLISTIC_FEATURES)

    base_cols = ['product_id', 'keyword_id', 'market_id']
    para_cols = [c for c in paragraphs.columns if c not in base_cols]   # paragraph_1, paragraph_2, ...

    records = []
    index = []
    for _, row in tqdm(paragraphs.iterrows(), total=len(paragraphs),
                       desc="Extracting Feature Vector"):
        record = {}
        for para_col in para_cols:
            text = row[para_col]
            if not isinstance(text, str) or not text.strip():
                continue                                   # skip empty / NaN paragraph slots
            vector = tfe.extract_stylistic_vector(text)
            # (paragraph, feature) key = which paragraph this feature belongs to
            for fname, val in zip(feature_names, vector):
                record[(para_col, fname)] = val

        if not record:                                     # product had no usable paragraphs
            continue
        records.append(record)
        index.append(row['product_id'])

    out = pd.DataFrame(records, index=pd.Index(index, name='product_id'))

    # Order entities: 'title' first, then paragraph_1, paragraph_2, ... (missing -> NaN).
    def entity_order(name):
        if name == 'title':
            return (0, 0)                       # title always leads
        if name.startswith('paragraph_'):
            return (1, int(name.split('_')[-1]))  # then paragraphs in numeric order
        return (2, 0)

    present = sorted({c[0] for c in out.columns}, key=entity_order)
    out = out.reindex(columns=pd.MultiIndex.from_product([present, feature_names]))
    return out



def merge_tables(*tables, key="product_id"):

    frames = []
    for tbl in tables:
        if tbl is None:                     # e.g. image features -> not developed yet
            continue

        label, df = tbl if isinstance(tbl, tuple) else (None, tbl)
        if df is None or df.empty:
            continue

        df = df.copy()

        # Move product_id to the index so every table aligns on it.
        if key in df.columns:
            df = df.set_index(key)

        # Build / keep the (entity, feature) 2-level column header.
        if isinstance(df.columns, pd.MultiIndex):
            pass                            # already (entity, feature) e.g. parra_1/..
        else:
            if label is None:
                raise ValueError("A flat table needs a label: pass (label, df).")
            df.columns = pd.MultiIndex.from_product([[label], df.columns])

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    # Inner join: keep only product_ids present in EVERY table, so every row is
    # fully populated (no half-NaN rows from tables built with different filters).
    merged = pd.concat(frames, axis=1, join="inner").sort_index()

    return merged.reset_index()


def attach_real_identifiers(data, id_col=("clean", "asin_ean_id")):

    if id_col not in data.columns:
        print(f"[attach_real_identifiers] {id_col} not in frame; skipping")
        return data

    ids = pd.to_numeric(data[id_col], errors="coerce").astype("Int64")
    uniq = [int(i) for i in ids.dropna().unique()]
    if not uniq:
        print("[attach_real_identifiers] no numeric ids to resolve; skipping")
        return data

    engine = create_engine(SCRAPE_DATABASE_URL)
    query = text("SELECT id, asin, ean FROM asin_ean WHERE id IN :ids").bindparams(
        bindparam("ids", expanding=True))
    with engine.connect() as conn:
        lut = pd.read_sql(query, conn, params={"ids": uniq}).set_index("id")

    data[("clean", "asin")] = ids.map(lut["asin"])
    data[("clean", "ean")]  = ids.map(lut["ean"])

    matched = data[("clean", "asin")].notna().sum()
    print(f"[attach_real_identifiers] resolved {matched}/{len(data)} rows to a real ASIN "
          f"({data[('clean','ean')].notna().sum()} with an EAN)")
    return data


# ============================================================================
#  SP-API fold-in
#  ---------------------------------------------------------------------------
#  Add own-catalog (SP-API) products to the scraped feature CSV WITHOUT
#  re-scraping. The browse node is a recursive tree, so a product filed several
#  levels below the category still belongs to it. The flow:
#     search_term + granularity
#       -> best-matching Amazon browse node per marketplace   (translate->embed->cosine)
#       -> that node PLUS its entire subtree (all descendants) = the relevant set
#       -> every SP-API product whose main_browser_node_id sits in that set
#       -> build feature rows (stylistic from title/bullets, specs from attributes)
#       -> normalize with the CSV's own params sidecar and append.
#
#  Clustering is presence-based (matrix_factorization_tf_idf builds a
#  notna() & !=0 mask), so what matters most is that a folded-in product
#  populates the SAME columns a scraped product of its type would.
# ============================================================================

import json as _json
from collections import defaultdict

# SP-API marketplace_id -> the key_words / node language (for translation sanity only).
MKT_LANG = {
    "A1PA6795UKMFR9": "de", "APJ6JRA9NG5V4": "it", "A13V1IB3VIYZZH": "fr",
    "A1RKKUPIHCS9HS": "es", "A1805IZSGTT6HS": "nl", "AMEN7PMS3EDWL": "nl",
}


def _api_engine():
    return create_engine(API_DATABASE_URL, connect_args={"connect_timeout": 30})


_EMBED_CACHE = {}
def _embed(text_in):
    """nomic-embed-text vector via ollama (same backend as clean_system)."""
    import ollama
    key = str(text_in).strip().lower()
    if not key:
        return None
    if key in _EMBED_CACHE:
        return _EMBED_CACHE[key]
    vec = np.asarray(ollama.embeddings(model="nomic-embed-text", prompt=key)["embedding"], dtype=float)
    _EMBED_CACHE[key] = vec
    return vec


_TRANS_CACHE = {}
def _to_english(text_in):
    """Best-effort English translation (node names / spec keys are localized)."""
    from deep_translator import GoogleTranslator
    t = str(text_in).strip()
    if not t:
        return t
    if t in _TRANS_CACHE:
        return _TRANS_CACHE[t]
    try:
        out = GoogleTranslator(source="auto", target="en").translate(t) or t
    except Exception:
        out = t
    _TRANS_CACHE[t] = out
    return out


def _cos(a, b):
    if a is None or b is None:
        return -1.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / denom)


def load_node_trees():
    with _api_engine().connect() as c:
        return pd.read_sql(text(
            "SELECT browse_node_id, marketplace_id, name, parent_browse_node_id, level "
            "FROM node_trees"), c)


def map_search_term_to_browse_nodes(search_term, granularity, nt=None, sim_threshold=0.50):
    """Map (search_term, granularity) -> best browse node per marketplace.

    Matching is semantic: translate every node name to English, embed it, and
    cosine-compare to the embedded search term. `granularity` mirrors the scrape's
    segmentation depth (1..3): the finest granularity (3) keeps the matched node;
    coarser granularities climb toward the root so the returned subtree is broader
    (climb = 3 - granularity parents).

    Returns {marketplace_id: {"node_id","name","match_name","score"}}.
    """
    if nt is None:
        nt = load_node_trees()
    nt = nt.copy()
    nt["bid"] = nt["browse_node_id"].astype(str)
    nt["pid"] = nt["parent_browse_node_id"].astype(str)

    q = _embed(_to_english(search_term))
    nt["score"] = [_cos(q, _embed(_to_english(nm))) for nm in nt["name"]]

    parent_of = dict(zip(nt["bid"], nt["pid"]))
    name_of = dict(zip(nt["bid"], nt["name"]))
    climb = max(0, 3 - int(granularity))

    out = {}
    for mkt, sub in nt.groupby("marketplace_id"):
        best = sub.sort_values("score", ascending=False).iloc[0]
        if best["score"] < sim_threshold:
            continue
        node = best["bid"]
        for _ in range(climb):
            node = parent_of.get(node, node)
        out[mkt] = {"node_id": node, "name": name_of.get(node, best["name"]),
                    "match_name": best["name"], "score": float(best["score"])}
    return out


def collect_subtree(nt, marketplace, root_id):
    """root_id + every descendant browse_node_id within one marketplace (recursive)."""
    sub = nt[nt["marketplace_id"] == marketplace]
    kids = defaultdict(list)
    for _, r in sub.iterrows():
        kids[str(r["parent_browse_node_id"])].append(str(r["browse_node_id"]))
    seen, stack = set(), [str(root_id)]
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        stack.extend(kids.get(n, []))
    return seen


def fetch_products_under_nodes(node_ids_by_mkt):
    """SP-API product_content rows whose main_browser_node_id is in the subtree,
    with each product's own listing price (for the ('clean','price') column)."""
    frames = []
    with _api_engine().connect() as c:
        for mkt, ids in node_ids_by_mkt.items():
            if not ids:
                continue
            df = pd.read_sql(text("""
                SELECT pc.asin, pc.marketplace_id, pc.item_name, pc.brand, pc.manufacturer,
                       pc.product_type, pc.main_browser_node_id, pc.attributes_json,
                       sl.listing_price
                FROM product_content pc
                LEFT JOIN LATERAL (
                    SELECT listing_price FROM seller_listings s
                    WHERE s.asin = pc.asin AND s.marketplace_id = pc.marketplace_id
                      AND s.listing_price IS NOT NULL
                    ORDER BY s.captured_at DESC LIMIT 1
                ) sl ON true
                WHERE pc.marketplace_id = :mkt
                  AND pc.main_browser_node_id::text = ANY(:ids)
            """), c, params={"mkt": mkt, "ids": list(ids)})
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(subset=["asin", "marketplace_id"]).reset_index(drop=True)


def _flatten_attributes(attrs):
    """attributes_json -> {attribute_name: scalar/str}. Joins multi-valued
    attributes (bullets, included components); skips nested dimension objects."""
    if isinstance(attrs, str):
        try:
            attrs = _json.loads(attrs)
        except Exception:
            return {}
    if not isinstance(attrs, dict):
        return {}
    flat = {}
    for k, v in attrs.items():
        if not isinstance(v, list) or not v:
            continue
        vals = []
        for item in v:
            if isinstance(item, dict):
                if "value" in item and not isinstance(item["value"], (dict, list)):
                    vals.append(item["value"])
            elif not isinstance(item, (dict, list)):
                vals.append(item)
        if vals:
            flat[k] = " | ".join(str(x) for x in vals) if len(vals) > 1 else vals[0]
    return flat


def _bullets(attrs_raw):
    """Ordered bullet_point strings from a raw attributes_json blob."""
    attrs = attrs_raw
    if isinstance(attrs, str):
        try:
            attrs = _json.loads(attrs)
        except Exception:
            return []
    if not isinstance(attrs, dict):
        return []
    return [b.get("value") for b in attrs.get("bullet_point", [])
            if isinstance(b, dict) and b.get("value")]


def _ean_from_attrs(attrs):
    v = attrs.get("externally_assigned_product_identifier")
    return str(v).split(" | ")[0] if v else np.nan


def _embed_clean_columns(clean_spec_cols, cache_path=None):
    """Map each clean spec column feature-name -> embedding of its English name.
    Cached to JSON so the ~hundreds of translations/embeddings run once."""
    cache = {}
    if cache_path and os.path.exists(cache_path):
        cache = {k: np.asarray(v, float) for k, v in _json.loads(Path(cache_path).read_text()).items()}
    out = {}
    for col in clean_spec_cols:
        feat = str(col[1])
        if feat in cache:
            out[feat] = cache[feat]
            continue
        # strip the ttf_/btf_ provenance prefix, de-snake, translate, embed
        bare = feat.split("_", 1)[1] if "_" in feat else feat
        emb = _embed(_to_english(bare.replace("_", " ")))
        if emb is not None:
            out[feat] = emb
    if cache_path:
        Path(cache_path).write_text(_json.dumps({k: v.tolist() for k, v in out.items()}))
    return out


def _match_attribute_to_column(attr_name, clean_col_embeddings, sim_threshold=0.62):
    """Best clean spec column for an SP-API attribute name (embedding cosine)."""
    emb = _embed(_to_english(str(attr_name).replace("_", " ")))
    if emb is None:
        return None
    best, best_s = None, sim_threshold
    for feat, cemb in clean_col_embeddings.items():
        s = _cos(emb, cemb)
        if s >= best_s:
            best, best_s = feat, s
    return best


def build_sp_api_feature_frame(products, template_columns, keyword_id=None,
                               clean_col_embeddings=None, tfe=None,
                               attr_sim_threshold=0.62):
    """Turn SP-API products into feature rows matching the scraped-CSV schema.

    Stylistic block: item_name -> ('title', *STYLISTIC_FEATURES); each bullet_point
    (and the description) -> ('paragraph_i', *). Clean block: each attribute is
    embedding-matched to an existing ttf_/btf_ spec column so the presence pattern
    lines up; price/asin/ean/keyword_id fill the base identity columns. Everything
    is reindexed to `template_columns` (unmapped -> NaN, exactly as the scraper leaves
    absent specs)."""
    from data_extraction.helper_methods.textual_feature_extraction import textual_feature_extractors
    if tfe is None:
        tfe = textual_feature_extractors()
    feats = list(tfe.STYLISTIC_FEATURES)

    clean_spec_cols = [c for c in template_columns
                       if c[0] == "clean" and (str(c[1]).startswith("ttf_") or str(c[1]).startswith("btf_"))]
    if clean_col_embeddings is None:
        clean_col_embeddings = _embed_clean_columns(clean_spec_cols)

    rows = []
    for _, p in products.iterrows():
        attrs = _flatten_attributes(p["attributes_json"])
        rec = {}

        # ---- stylistic: title + bullets (+ description) ----
        title = p.get("item_name") or attrs.get("item_name") or ""
        for f, val in zip(feats, tfe.extract_stylistic_vector(str(title))):
            rec[("title", f)] = val
        paras = _bullets(p["attributes_json"])
        if attrs.get("product_description"):
            paras = paras + [attrs["product_description"]]
        for i, para in enumerate(paras[:14], start=1):
            for f, val in zip(feats, tfe.extract_stylistic_vector(str(para))):
                rec[(f"paragraph_{i}", f)] = val

        # ---- clean specs via embedding match ----
        for aname, aval in attrs.items():
            if aname in ("bullet_point", "product_description", "item_name",
                         "externally_assigned_product_identifier"):
                continue
            col = _match_attribute_to_column(aname, clean_col_embeddings, attr_sim_threshold)
            if col is not None and ("clean", col) not in rec:
                rec[("clean", col)] = aval

        # ---- base identity columns ----
        rec[("clean", "asin")] = p.get("asin")
        rec[("clean", "ean")] = _ean_from_attrs(attrs)
        if p.get("listing_price") is not None and not pd.isna(p.get("listing_price")):
            rec[("clean", "price")] = float(p["listing_price"])
        if keyword_id is not None:
            rec[("clean", "keyword_id")] = keyword_id

        rows.append(rec)

    frame = pd.DataFrame(rows)
    return frame.reindex(columns=pd.MultiIndex.from_tuples(template_columns))


def add_sp_api_products_to_csv(search_term, granularity, csv_path=None,
                               node_sim_threshold=0.50, attr_sim_threshold=0.62,
                               dry_run=False):
    """End-to-end: map -> subtree -> select SP-API products -> append to the CSV.

    dry_run=True stops after selection and returns a summary (nodes chosen, subtree
    sizes, product counts) without touching the file — use it to sanity-check the
    node mapping before committing rows.
    """
    csv_path = csv_path or f"data_files/all_feature_data_{search_term}.csv"
    base = pd.read_csv(csv_path, header=[0, 1], low_memory=False)
    template_columns = list(base.columns)

    nt = load_node_trees()
    roots = map_search_term_to_browse_nodes(search_term, granularity, nt=nt,
                                            sim_threshold=node_sim_threshold)
    node_ids_by_mkt = {mkt: collect_subtree(nt, mkt, info["node_id"])
                       for mkt, info in roots.items()}
    products = fetch_products_under_nodes(node_ids_by_mkt)

    summary = {
        "search_term": search_term, "granularity": granularity,
        "roots": roots,
        "subtree_sizes": {m: len(s) for m, s in node_ids_by_mkt.items()},
        "n_products": int(len(products)),
    }
    if dry_run or products.empty:
        return summary, products

    keyword_ids = extract_keyword_ids(granularity, search_term)
    keyword_id = keyword_ids[0] if keyword_ids else None

    new_rows = build_sp_api_feature_frame(products, template_columns,
                                          keyword_id=keyword_id,
                                          attr_sim_threshold=attr_sim_threshold)

    # Normalize the new rows with the SAME params the saved CSV was built with, so
    # the appended values live on the same scale (the sidecar is the one market_env
    # already reads: all_feature_data_<term>.params.json).
    params_path = Path(csv_path).with_suffix("").as_posix() + ".params.json"
    if os.path.exists(params_path):
        new_rows = Normalizer.load(params_path).transform(new_rows)
    else:
        print(f"[fold_in] no params sidecar at {params_path}; appending un-normalized")

    combined = pd.concat([base, new_rows], ignore_index=True)
    combined.to_csv(csv_path, index=False)
    summary["appended"] = int(len(new_rows))
    summary["csv_rows"] = int(len(combined))
    return summary, products


def main():


    #  i need this to load from the database
    #  this will be then run on a two week basis in order to be able to always have up to date files in the actual systen


    granularity = 2
    search_term = 'Headphones'

    keyword_ids = extract_keyword_ids(granularity, search_term)
    paragraphs = paragraph_extraction(keyword_ids).extract_all_data()
    print("paragraphs shape:", paragraphs.shape)

    # pre process the paragraphs such that i will have feature vectors instead of just paragrphs
    para_data = extract_feature_para(paragraphs)
    print(para_data.head())


    grouped, (clean, data_dict) = clean_system(keyword_ids).run()
    print("grouped shape:", grouped.shape)

    # reduce the vectors such that i dont have any duplicate between it and paragraphs data


    #  Include image features. 
    ''' 
    
    This will be section where I will do some feature extraction on the images. 
    
    '''


    # # para_data already carries per-paragraph (paragraph_N, feature) MultiIndex columns.
    # # grouped is flat -> label it 'clean'. image features not built yet -> None (skipped).
    merged_data = merge_tables(para_data, ("clean", grouped), None)

    # Normalize right before saving so the file always holds clean, standardized
    # data. Types come from the pipeline's feature-type cache when it exists
    # (keeps this file consistent with how the voting stages read it); on a brand
    # new search term the cache isn't there yet and types are auto-detected.
    type_cache = f"data_files/feature_types_{search_term}.json"
    feature_types = load_type_cache(type_cache) if os.path.exists(type_cache) else None
    merged_data = Normalizer().fit_transform(merged_data, feature_types=feature_types)

    # Resolve internal asin_ean_id -> real ASIN/EAN so the saved file can join to
    # the SP-API DB. Done AFTER normalize so the identifier strings stay intact.
    merged_data = attach_real_identifiers(merged_data)

    merged_data.to_csv(f"data_files/all_feature_data_{search_term}.csv", index=False)

    # Fold in own-catalog SP-API products under the same segment (browse-node
    # subtree), so they compete in the same feature space without re-scraping.
    # Needs a GPU: build_sp_api_feature_frame() constructs textual_feature_extractors,
    # whose zero-shot classifier is pinned to device=0.
    summary, _ = add_sp_api_products_to_csv(search_term, granularity)
    print(f"[sp_api fold-in] matched nodes: { {m: r['name'] for m, r in summary['roots'].items()} }")
    print(f"[sp_api fold-in] subtree sizes: {summary['subtree_sizes']}")
    print(f"[sp_api fold-in] appended {summary.get('appended', 0)} SP-API products "
          f"({summary['n_products']} found) -> csv now {summary.get('csv_rows', '?')} rows")


if __name__ == "__main__":
    main()