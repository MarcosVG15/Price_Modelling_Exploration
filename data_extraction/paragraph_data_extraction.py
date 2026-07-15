import os
import json
import pandas as pd
from datetime import date, timedelta
from sqlalchemy import create_engine, text

# Default look-back window. Matches the ~3-week refresh cadence and guards against
# accidentally pulling the full history (a single keyword_id is ~100k+ rows).
DEFAULT_LOOKBACK_DAYS = 30


class paragraph_extraction :

    def __init__(self , search_term_id, time_frame = None):
        self.search_term_id_arr = search_term_id
        self.time_frame = time_frame

    def extract_all_data(self):
        if not self.search_term_id_arr:
            return pd.DataFrame()

        time_frame = self.time_frame
        if time_frame is None:
            time_frame = date.today() - timedelta(days=DEFAULT_LOOKBACK_DAYS)

        engine = create_engine(os.getenv('SCRAPE_DATABASE_URL'))

        sql = """
            SELECT p.id           AS product_id,
                   p.title        AS title,
                   p.keyword_id   AS keyword_id,
                   p.market_id    AS market_id,
                   g.data         AS paragraphs
            FROM products p
            JOIN general_information_files g
              ON g.id = p.general_information_id
            WHERE p.keyword_id = ANY(:ids)
              AND p.day >= :time_frame
        """
        params = {"ids": self.search_term_id_arr, "time_frame": time_frame}

        with engine.connect() as conn:
            df = pd.read_sql(text(sql), conn, params=params)

        return self.segment(df)

    def segment(self, df):

        def to_paragraphs(data):
            if isinstance(data, str):        # json column usually arrives already parsed
                data = json.loads(data)
            if not isinstance(data, list):   # None / {} / scalar -> no paragraphs
                return []
            return [item.get('about_item') for item in data if isinstance(item, dict)]

        para_lists = df['paragraphs'].apply(to_paragraphs)

        # Drop items with no paragraphs (keeps output clean for the NLP step).
        keep = para_lists.map(len) > 0
        df = df[keep].reset_index(drop=True)
        para_lists = para_lists[keep].reset_index(drop=True)

        # Uneven lengths pad with NaN automatically.
        para_df = pd.DataFrame(para_lists.tolist())
        para_df.columns = [f'paragraph_{i + 1}' for i in range(para_df.shape[1])]

        return pd.concat([df[['product_id', 'keyword_id', 'market_id', 'title' ]], para_df], axis=1)
