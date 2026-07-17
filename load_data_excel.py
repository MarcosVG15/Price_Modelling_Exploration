import os
from pathlib import Path
from datetime import date, timedelta
import pandas as pd
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


def main(lookback_days=45):
    """
    lookback_days: how far back (in days) to pull scraped listings from. A narrow
    window captures a more homogeneous slice of the catalog (fewer distinct
    sellers/product types), which raises feature-presence density -- TF-IDF then
    suppresses most of those near-ubiquitous features at once, embeddings collapse
    together, and svd_product_communities' Louvain step slows way down chewing
    through a near-flat modularity landscape. Widen this (e.g. 90) if that happens.
    """

    #  i need this to load from the database
    #  this will be then run on a two week basis in order to be able to always have up to date files in the actual systen


    granularity = 1
    search_term = 'Audio'
    time_frame = date.today() - timedelta(days=lookback_days)

    keyword_ids = extract_keyword_ids(granularity, search_term)
    paragraphs = paragraph_extraction(keyword_ids, time_frame=time_frame).extract_all_data()
    print("paragraphs shape:", paragraphs.shape)

    # pre process the paragraphs such that i will have feature vectors instead of just paragrphs
    para_data = extract_feature_para(paragraphs)
    print(para_data.head())


    grouped, (clean, data_dict) = clean_system(keyword_ids, time_frame=time_frame).run()
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


if __name__ == "__main__":
    import sys
    # python load_data_excel.py [lookback_days]
    # No arg -> fall through to main()'s own default (single source of truth),
    # so the signature default can't be silently overridden here.
    if len(sys.argv) > 1:
        main(lookback_days=int(sys.argv[1]))
    else:
        main()