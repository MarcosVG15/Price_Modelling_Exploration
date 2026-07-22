import numpy as np
import pandas as pd 



'''  

 for each of the csv generate a vn cluster network and add it to a dict such that you can reference based on the segmenation level 


for this you will need to extract all of the search terms for a specific segmentation level that were used to load the data
if the csv is not found then it should be genereted. 

then we will find in each of the csv the commaxx products that we hae extracted in the bbox feature panel
    - to each of these we will append the keyword and the cluster id that it contains
    WHY ? = such that I can extract the competitor data - this means that we can accurately trace the buybox


We will do a shap analysis on the features to understand which are important to the buybox prediction
    - this will include all of the ones that are in the data panel as well as the data that is extracted from the competitor in the cluster id
        - so we have in fact so many more features to understand the buybox
    

Storing 

The model will be stored in a joblib file for easy access - later this model will be updated every 3 weeks or so - I dont fuking know



TESTING 

for the commax product we will for each period stored in the csv we will predict the buybox percetange vs the actual one and see how much they differ in size




'''


class create_bbox_predictor : 

    #  this method needs to know the location of the path that contains all of the csv that will be used to train the cvr predictor
    def __init__(self , cvs_folder_path , seg_level):
        self.folder_path = cvs_folder_path
        self.seg_level = seg_level

        self.aggregate_data = 

    def load_data(self , key_word ) :



    # def extract_feature_importance_BBox(cls):
                
    #     if not os.path.exists(BBOX_path):
    #         raise FileNotFoundError(
    #             f"buy-box feature panel not found: {BBOX_path}\n"
    #             f"Run `python GAME_THEORY_PREDICTION/BBOX/build_feature_panel_BBox.py` first.")
    #     shap_data = pd.read_csv(BBOX_path)

    #     # drop identifiers, the target, demand OUTCOMES, and the duplicate raw
    #     # price sources (keep the coalesced+ffilled `price`).
    #     drop_columns = [
    #         "asin", "marketplace_id", "week", "buybox_pct",
    #         "units", "sessions", "revenue",
    #         "implied_price", "oi_price", "listing_price", "raw_price",
    #         "rank_value", "main_browser_node_id",
    #     ]

    #     Y = (shap_data["buybox_pct"] / 100.0).clip(0.0, 1.0)   # buy-box share in [0, 1]
    #     X = shap_data.drop(columns=[c for c in drop_columns if c in shap_data.columns])

    #     # near-empty numeric cols (the ~2-week competition/stock snapshots over a
    #     # 2-yr panel) break HistGBR's binning -> drop anything under 2% coverage.
    #     sparse = [c for c in X.columns if X[c].notna().mean() < 0.02]
    #     X = X.drop(columns=sparse)

    #     # one-hot the low-card string/bool categoricals so SHAP never sees a str.
    #     cat_cols = [c for c in ["brand", "product_type", "manufacturer", "has_aplus"]
    #                 if c in X.columns]
    #     X = pd.get_dummies(X, columns=cat_cols, drop_first=True).astype(float)
    #     print(f"[shap-bbox] {X.shape[1]} features; dropped {len(sparse)} near-empty cols: {sparse}")

    #     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

    #     model = HistGradientBoostingRegressor(max_iter=200, random_state=0)
    #     model.fit(X_train, Y_train)
    #     cls._BUYBOX_SHAP_MODEL = {'model': model , 'features': X}            
    

    # def fit_buybox(cls, top_k=15):
        
    #     if cls._BUYBOX_SHAP_MODEL is None:
    #         cls.extract_feature_importance_BBox()
    #     model = cls._BUYBOX_SHAP_MODEL["model"]
    #     feats = cls._BUYBOX_SHAP_MODEL["features"]

    #     shap_vals = shap.TreeExplainer(model).shap_values(feats.iloc[:2000])
    #     importance = (pd.Series(abs(shap_vals).mean(axis=0), index=feats.columns)
    #                     .sort_values(ascending=False))
    #     top_features = importance.index.tolist()[:top_k]

    #     cats = ["brand", "product_type", "manufacturer", "has_aplus"]
    #     raw_keep = []
    #     for f in top_features:
    #         parent = next((c for c in cats if f == c or f.startswith(c + "_")), f)
    #         raw_keep.append(parent)
    #     raw_keep = list(dict.fromkeys(raw_keep))          # dedupe, preserve order
    #     print(f"[bbox] top {top_k} one-hot features -> {len(raw_keep)} raw cols: {raw_keep}")

    #     # build X/Y from the panel using ONLY those raw columns; keep categoricals NATIVE
    #     bbox_data = pd.read_csv(BBOX_path)
    #     Y = (bbox_data["buybox_pct"] / 100.0).clip(0.0, 1.0)
    #     X = bbox_data[[c for c in raw_keep if c in bbox_data.columns]].copy()
    #     cat_in_X = [c for c in cats if c in X.columns]
    #     for c in cat_in_X:
    #         X[c] = X[c].astype("category")
    #     cat_mask = [c in cat_in_X for c in X.columns]

    #     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=0)
    #     best_tuned = HistGradientBoostingRegressor(
    #         max_iter=200, learning_rate=0.08, max_leaf_nodes=31,
    #         min_samples_leaf=10, l2_regularization=0.5,
    #         categorical_features=cat_mask, random_state=0,
    #     )
    #     best_tuned.fit(X_train, Y_train)
    #     cls._BUYBOX_MODEL = best_tuned
    #     cls._BUYBOX_FEATURES = list(X.columns)
    #     cls._BUYBOX_CAT_COLS = cat_in_X
    #     cls._BUYBOX_CAT_LEVELS = {c: list(X[c].cat.categories) for c in cat_in_X}
    #     print(f"[bbox] refit on {X.shape[1]} raw cols -> holdout R^2 = {best_tuned.score(X_test, Y_test):.3f}")
    #     return X_test, Y_test, cls._BUYBOX_MODEL
    

    #  @classmethod
    #     def _buybox_panel(cls):
    #         if cls._BUYBOX_PANEL_DF is None:
    #             cls._BUYBOX_PANEL_DF = pd.read_csv(BBOX_path)
    #         return cls._BUYBOX_PANEL_DF
    
    #     @classmethod
    #     def _snapshot_buybox_features(cls, asin):
    #         # Per-product snapshot of the buy-box panel: median for numerics, mode for
    #         # categoricals. price/own_landed/own_shipping are overwritten at predict time.
    #         if cls._BUYBOX_FEATURES is None:
    #             return None
    
    #         panel = cls._buybox_panel()
    #         rows = panel[panel["asin"].astype(str) == str(asin)] if "asin" in panel.columns else panel.iloc[0:0]
    
    #         cat_cols = cls._BUYBOX_CAT_COLS or []
    #         snap = {}
    #         for col in cls._BUYBOX_FEATURES:
    #             if rows.empty or col not in panel.columns:
    #                 snap[col] = np.nan
    #             elif col in cat_cols:
    #                 m = rows[col].mode(dropna=True)
    #                 snap[col] = m.iloc[0] if not m.empty else np.nan
    #             else:
    #                 snap[col] = float(pd.to_numeric(rows[col], errors="coerce").median())
    #         return snap
    