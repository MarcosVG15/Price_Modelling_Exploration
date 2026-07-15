from pathlib import Path
from config import FOLDER_PATH


def find_csv(search_term):
    folder_path = Path(FOLDER_PATH)
    # match the source data file specifically (all_feature_data_<term>.csv), NOT
    # derived artifacts like clusters_<term>.csv that also contain the term.
    pattern = f"all_feature_data*{search_term}*.csv"

    files_found = sorted(folder_path.glob(pattern))
    if files_found:
        return files_found[0]
    else:
        print(f"No Match found for {search_term}")
        print(f"You should run/check load_data_excel.py")
