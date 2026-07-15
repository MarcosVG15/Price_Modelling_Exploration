import os

import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import date, timedelta
from sqlalchemy import create_engine, text

# Default look-back window (mirrors paragraph_extraction) so a run is never unbounded.
DEFAULT_LOOKBACK_DAYS = 30

# Number of leading "base" columns in final_df (product_id .. btf_id); everything after
# these is a technical-spec column that gets translated / embedded / clustered.
DISPLACEMENT = 17

# Words that overpower the embedding similarity; stripped before clustering, and can be
# re-attached to the title afterwards.
NOISE_WORDS = ["type", "level"]

# Cosine-distance ceiling for two spec columns to be considered the same concept.
STRICT_THRESHOLD = 0.025


class clean_system :

    def __init__(self , search_term_id, time_frame = None):
        self.search_term_id_arr = search_term_id
        self.time_frame = time_frame
        self.engine = create_engine(os.getenv('SCRAPE_DATABASE_URL'))

        # Filled in during column clustering.
        self.ttf_btf_indicator = []
        self.noise_words_indicator = []
        self._nlp = None

    # ------------------------------------------------------------------ helpers
    @property
    def nlp(self):
        # Load the spaCy model once, lazily (keeps the class importable without spaCy).
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
        return self._nlp

    def check_string(self, input_str, check_str):
        for x in range(len(check_str) - len(input_str) + 1):
            if check_str[x: x + len(input_str)] == input_str:
                return True
        return False

    @staticmethod
    def extract_properties(data_list, text_to_remove="\u200E"):
        # TTF/BTF `data` is a list of {"key": ..., "value": ...}; flatten to {key: value}.
        if not isinstance(data_list, list):
            return {}
        props = {}
        for item in data_list:
            if isinstance(item, dict) and 'key' in item and 'value' in item:
                k = str(item['key']).replace(text_to_remove, "")
                v = str(item['value']).replace(text_to_remove, "")
                props[k] = v
        return props

    def get_canonical_name(self, text):
        doc = self.nlp(text.lower())
        # Noun chunks keep context, e.g. "connectivity technology".
        chunks = [chunk.text for chunk in doc.noun_chunks]
        if chunks:
            return " ".join(chunks)
        return text.lower()

    def clean_text(self, text):
        if "_" in text:
            parts = text.split("_", 1)          # split at the first underscore
            clean = parts[1]
            self.ttf_btf_indicator.append([parts[0], clean])
        else:
            clean = text
            self.ttf_btf_indicator.append([-1, clean])

        clean = clean.replace("_", " ").strip()

        # Remove noise words that would otherwise dominate the similarity.
        words = clean.split()
        filtered = [w for w in words if w.lower() not in NOISE_WORDS]
        return " ".join(filtered)

    def get_embedding(self, text):
        import ollama
        embed = ollama.embeddings(model="nomic-embed-text", prompt=text)  # english only
        return np.array(embed['embedding'])

    @staticmethod
    def average_vec(members):
        matrix = np.array([m['vector'] for m in members])
        return np.mean(matrix, axis=0)

    # ------------------------------------------------------------------ 1. pull + clean
    def extract_all_data(self):
        # Nothing to look up if the caller passed an empty id list.
        if not self.search_term_id_arr:
            return pd.DataFrame()

        # Fall back to "last 3 weeks" when no time_frame is given.
        time_frame = self.time_frame
        if time_frame is None:
            time_frame = date.today() - timedelta(days=DEFAULT_LOOKBACK_DAYS)

        # Postgres lowercases unquoted aliases, so TTF_data -> ttf_data, etc.
        query_join = text("""
            SELECT p.id as product_id ,
                   p.keyword_id ,
                   p.title ,
                   p.asin_ean_id ,
                   p.discount ,
                   p.price ,
                   p.page,
                   p.rank ,
                   p.average_rating ,
                   p.number_of_reviews,
                   p.buy_box ,
                   p.day as day ,
                   p.delivery_time,
                   p.market_id,
                   p.image_url,
                   tcf.id as TTF_id,
                   tcf.data as TTF_data ,
                   btf.id as BTF_id ,
                   btf.data as BTF_data,
                   gtf.data as GTF_data
            FROM products as p
            JOIN general_information_files as gtf ON p.general_information_id = gtf.id
            JOIN top_technical_files as tcf ON p.top_technical_sheet_id = tcf.id
            JOIN bottom_technical_files as btf ON p.bottom_technical_sheet_id = btf.id
            WHERE p.keyword_id = ANY(:ids)
              AND p.day >= :time_frame
              AND p.market_id < 7
              AND p.title != 'error'
              AND p.delivery_time >= 0
              AND p.image_url != ''
            ORDER BY p.keyword_id
        """)
        params = {"ids": self.search_term_id_arr, "time_frame": time_frame}

        print(f"[1/4] Querying database (day >= {time_frame}) ...")
        with self.engine.connect() as conn:
            general_data = pd.read_sql(query_join, conn, params=params)
        print(f"      loaded {len(general_data)} product rows")

        print("[2/4] Flattening technical spec sheets ...")
        return self.clean(general_data)

    def clean(self, general_data):
        # Flatten the top / bottom technical spec sheets into their own prefixed columns.
        # If a spec key exists it fills that column, otherwise the row is NaN there.
        clean_dicts_ttf = general_data['ttf_data'].apply(self.extract_properties)
        specs_ttf = pd.DataFrame(clean_dicts_ttf.tolist(), index=general_data.index).add_prefix('ttf_')

        clean_dicts_btf = general_data['btf_data'].apply(self.extract_properties)
        specs_btf = pd.DataFrame(clean_dicts_btf.tolist(), index=general_data.index).add_prefix('btf_')

        # Keep the general-info sheet column as-is.
        specs_gtf = pd.DataFrame(general_data['gtf_data'])

        # Drop the raw json columns and join the flattened specs back on.
        final_df = general_data.drop(['ttf_data', 'btf_data', 'gtf_data'], axis=1) \
                               .join(specs_gtf) \
                               .join(specs_ttf) \
                               .join(specs_btf)

        return final_df

    # ------------------------------------------------------------------ 2. cluster spec cols
    def cluster_columns(self, final_df):
        from deep_translator import GoogleTranslator

        column_names = final_df.columns[DISPLACEMENT:]
        metadata = []
        embeddings_list = []

        print("[3/4] Clustering spec columns ...")
        for i, original_title in tqdm(enumerate(column_names), total=len(column_names),
                                      desc="  translating + embedding"):
            translated = GoogleTranslator(source="auto", target='en').translate(original_title)
            clean_translated = self.clean_text(translated)
            subject = self.get_canonical_name(translated)
            vector = self.get_embedding(subject)

            embeddings_list.append(vector)
            metadata.append({
                'original_index': i,
                'original_traduction': translated,
                'translated_name': clean_translated,
                'original_title': original_title,
                'vector': vector,
            })

        # Greedy clustering: attach each column to a matching cluster, else start a new one.
        clusters_dict = {}
        for i, item in tqdm(enumerate(metadata), total=len(metadata),
                            desc="  grouping"):
            vec = embeddings_list[i]
            subject = self.get_canonical_name(item['translated_name'].lower())
            assigned = False

            for cluster_id, members in clusters_dict.items():
                leader_vec = self.average_vec(members)
                leader_subject = self.get_canonical_name(members[0]['translated_name'].lower())
                dist = 1 - (np.dot(vec, leader_vec) / (np.linalg.norm(vec) * np.linalg.norm(leader_vec)))

                if dist < STRICT_THRESHOLD and subject == leader_subject:
                    clusters_dict[cluster_id].append(item)
                    assigned = True
                    break

            if not assigned:
                item['vector'] = vec
                clusters_dict[len(clusters_dict)] = [item]

        return clusters_dict

    def merge_clusters(self, final_df, clusters_dict):
        # Base (non-spec) columns pass through untouched.
        final_base_columns = final_df.columns[:DISPLACEMENT].tolist()
        new_df = final_df[final_base_columns].copy()

        # Merge every clustered column group into a single column (first non-null wins).
        for cluster_id, items in tqdm(clusters_dict.items(), total=len(clusters_dict),
                                      desc="  merging clusters"):
            cluster_column_names = [item['original_title'] for item in items]
            # Canonical name: strip + lowercase so case variants of the same concept
            # (e.g. "Frequency response" vs "frequency response") collapse to one column.
            new_col_name = items[0]['original_traduction'].strip().lower()

            if len(cluster_column_names) > 1:
                merged_series = final_df[cluster_column_names].bfill(axis=1).iloc[:, 0]
            else:
                merged_series = final_df[cluster_column_names[0]]

            if new_col_name in new_df.columns:
                # Another cluster already produced this name -> combine, don't overwrite.
                new_df[new_col_name] = new_df[new_col_name].fillna(merged_series)
            else:
                new_df[new_col_name] = merged_series

        return new_df

    def save_groups(self, clusters_dict, path="data_files/cluster_groups_review.xlsx"):
        # Review-only export of the discovered clusters (one row per member column).
        # This file is NOT read back anywhere in the pipeline.
        rows = []
        for cluster_id, items in clusters_dict.items():
            for item in items:
                rows.append({
                    "group_id": cluster_id,
                    "group_size": len(items),
                    "original_title": item["original_title"],
                    "translated_name": item["translated_name"],
                    "english_translation": item["original_traduction"],
                })
        pd.DataFrame(rows).to_excel(path, index=False)
        return path

    # ------------------------------------------------------------------ 3. orchestrate
    def run(self, run_standardizer=True):
        final_df = self.extract_all_data()
        clusters_dict = self.cluster_columns(final_df)

        # # Save the discovered groups to Excel for manual review (not printed, not reused).
        # groups_path = self.save_groups(clusters_dict)
        # print(f"      saved {len(clusters_dict)} groups for review -> {groups_path}")

        grouped_data = self.merge_clusters(final_df, clusters_dict)

        print(f"      columns: {len(final_df.columns)} -> {len(grouped_data.columns)}")

        # grouped_data is kept in memory only (intentionally not written to disk).
        if run_standardizer:
            return grouped_data, self.run_standardizer(grouped_data)
        return grouped_data

    def run_standardizer(self, grouped_data):
        # Deterministic standardiser, run in-memory via build() so nothing is persisted
        # (no grouped_data.csv read, no data_dictionary.csv written).
        print("[4/4] Standardizing features (in memory) ...")
        try:
            from data_extraction import standardize
        except ImportError:
            import standardize
        clean, data_dict, report = standardize.build(grouped_data)
        return clean, data_dict
