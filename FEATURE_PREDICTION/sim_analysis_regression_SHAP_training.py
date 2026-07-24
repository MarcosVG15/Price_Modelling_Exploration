import os
import random
import numpy as np
import pandas as pd
import networkx as nx


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error


import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm

# make the repo root importable (config, helper_methods, FEATURE_PREDICTION)
# when this script is run directly from inside the FEATURE_PREDICTION folder
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ASIN_COL
from FEATURE_PREDICTION.predictor import predictor , load_scale_params
from helper_methods.general import find_csv
from helper_methods.calculations import calculate_unified_score
from Matrix_Factorization_Clustering.Matrix_Factorization import matrix_factorization 
from Matrix_Factorization_Clustering.effective_tests import presence_auc , global_fidelity , modularity_k,conductance , LCC_computer

''' 

What I understand is that I need to find a way to extract the parameters for the weighted sum of the product features based on their impact on price. 

What I will do is train a weighted sum formula that uses the weight/ proximity in the graph network of the nodes to weigh the general impact of that specific data point and
the individiual weights of the features such that I get the most accurate yet fitted model. Since each segementation level has different features I will have to compute the weights
for the different countries which means that this will be again one of those background processes that will happen periodically to make sure that the system is reliable

I will use ElasticNet regression to not overfit combined with Corss Validation to make sure that the system is accurate


IDEA : we can also use the SVD data estimation to predict the price . risky frisky!!!!
     : we can use shap instead of this regression model. 

Things that I need to take into account  :
    1. If there is a value that was never seen before - weight of 1
    2.

'''


def extract_target_data(all_data, n, random_state=None):
  
    unique = all_data.drop_duplicates(subset=[ASIN_COL])
    n = min(n, len(unique))
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(unique), size=n, replace=False)
    return unique.iloc[idx]



def export_clusters_csv(vn, all_data, search_term, out_dir="data_files"):
   
    labels = vn.product_labels
    if len(labels) != len(all_data):
        raise ValueError(f"labels ({len(labels)}) != rows ({len(all_data)}); "
                         "run svd_product_communities() on this same data first")

    cluster_col = pd.DataFrame({("cluster", "community"): labels}, index=all_data.index)
    out = (pd.concat([cluster_col, all_data], axis=1)
             .sort_values(("cluster", "community"))
             .reset_index(drop=True))

    out_path = os.path.join(out_dir, f"clusters_{search_term}.csv")
    out.to_csv(out_path, index=False)
    print(f"wrote {len(out)} rows x {out.shape[1]} cols -> {out_path} "
          f"({out[('cluster','community')].nunique()} clusters)")
    return out_path




def predict_models(vn, all_data, search_term, test_size=0.3, random_state=0, n_show=8):
    """Train several regressors on the SVD product vectors to predict price,
    evaluate on a held-out set, and print predicted vs actual price (in dollars)
    per model. Dedupes by product id so the same product can't leak train->test."""
    TARGET_COL = ("clean", "price")

    # keep only the first listing of each product (no duplicate leakage)
    first = ~all_data[ASIN_COL].duplicated()
    pos = np.where(first.to_numpy())[0]

    X = vn.X_transformed[pos]                                # SVD embeddings (unique products)
    y = all_data[TARGET_COL].to_numpy(dtype=float)[pos]      # normalized price

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    center, scale = load_scale_params(f"data_files/all_feature_data_{search_term}.params.json")
    actual_d = y_test * scale + center

    models = {
        "ElasticNetCV": ElasticNetCV(cv=5, random_state=random_state),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1),
        "SVR(rbf)":     make_pipeline(StandardScaler(), SVR(C=10.0, gamma="scale")),
    }

    trained, preds_d = {}, {}
    print(f"\n{'model':>14} | {'test R2':>8} | {'MAE ($)':>10}")
    print("-" * 40)
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        trained[name] = model
        preds_d[name] = y_pred * scale + center
        print(f"{name:>14} | {r2_score(y_test, y_pred):>8.3f} | "
              f"{mean_absolute_error(actual_d, preds_d[name]):>10.2f}")

    # side-by-side predicted vs actual (dollars) for a few test products
    print(f"\n{'actual':>10} | " + " | ".join(f"{n:>14}" for n in models))
    print("-" * (13 + 17 * len(models)))
    for i in range(min(n_show, len(y_test))):
        row = f"${actual_d[i]:>8.2f} | " + " | ".join(f"${preds_d[n][i]:>12.2f}" for n in models)
        print(row)

    return trained

     

def find_k_res(vn):
    
    results = {}
    resolutions = [round(r, 1) for r in np.arange(0.5, 1.5, 0.1)] 

    for i in tqdm(range(5,30, 5)) :
        row_data = []
        for j in resolutions:

            # cluster FIRST so product_graph / product_labels reflect THIS (k, res)
            vn.svd_product_communities(k=i , resolution= j)
            graph = vn.product_graph
            labels = vn.product_labels

            dp = modularity_k( graph , labels )
            cd = conductance( graph , labels )

            row_data.append([dp , cd])
        
        results[i] = np.array(row_data)
    
   
    scores = calculate_unified_score(results, vn, resolutions)
    k = max(scores, key=scores.get)

    #  for now leave res = 1
    res = 1.0 
    return k , res 



def basic_predict(regressor):
   

    center, scale = load_scale_params(f"data_files/all_feature_data_{search_term}.params.json")
    prices = regressor.predict_basic(gamma=0.750, price_center=center, price_scale=scale)  # dict {row: dollars}

    # actual prices of the sampled target products, de-normalized to dollars.
    # target_data rows line up (in order) with the prediction keys.
    actual = target_data[("clean", "price")].to_numpy(dtype=float) * scale + center

    print(f"{'row':>6} | {'predicted':>10} | {'actual':>10}")
    for (row_idx, pred), act in zip(prices.items(), actual):
        print(f"{row_idx:>6} | ${pred:>9.2f} | ${act:>9.2f}")



if __name__ == "__main__" :
    search_term = "Headphones"
    path = find_csv(search_term)


    all_feature_data = pd.read_csv(path, header=[0, 1], low_memory=False)
    all_feature_data  = all_feature_data.drop(columns=all_feature_data.columns[0])

    vn  = matrix_factorization(all_feature_data)
    populated_data = vn.matrix_factorization_tf_idf()

    k  = 1
    res = 1
    vn.svd_product_communities(k=k, resolution=res)

    # dump all feature data + their cluster to a CSV for inspection
    # export_clusters_csv(vn, all_feature_data, search_term)



    target_data = extract_target_data(all_feature_data, n=50)
    regressor  = predictor(k , feature_data=all_feature_data , target_data= target_data , vn= vn)
    query_clusters = regressor.find_cluster()


    # regressor.train_price_model(
    #     feature_types=f"data_files/feature_types_{search_term}.json",
    #     price_params_path=f"data_files/all_feature_data_{search_term}.params.json")

    price_preds = regressor.predict_price_parsed_columns(
        feature_types=f"data_files/feature_types_{search_term}.json",
        price_params_path=f"data_files/all_feature_data_{search_term}.params.json",
        log_price=True)


    actual = target_data[("clean", "price")].to_numpy(dtype=float) \
        * regressor.price_scale + regressor.price_center
    asins   = target_data[ASIN_COL].to_numpy()             # same order as price_preds / actual
    markets = target_data[("clean", "market_id")].to_numpy()
    print("\nSHAP price reconstruction (target products):")


    for (row, info), act, asin, market in zip(price_preds.items(), actual, asins, markets):
        unit = info.get("contrib_unit", "$")   # "%" in log mode, "$" otherwise
        print(f"\nrow {row} | asin {asin} | market {market}: predicted ${info['predicted']:.2f} "
              f"| actual ${act:.2f} | base ${info['base']:.2f}")
        for feat, contrib in info["top_features"]:
            suffix = "%" if unit == "%" else ""
            prefix = "" if unit == "%" else "$"
            print(f"     {feat[:44]:<44} {prefix}{contrib:+.2f}{suffix}")


