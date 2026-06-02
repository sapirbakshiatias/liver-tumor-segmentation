"""
טעינת נתונים מה-CSV.
"""
import numpy as np
import pandas as pd

CSV_PATH = r"C:\Users\ronin\PycharmProjects\PFinalproject\Cropped_Data\all_series_radiomics.csv"


def load_data():
    df        = pd.read_csv(CSV_PATH)
    feat_cols = [c for c in df.columns
                 if c not in ("series", "patient", "group", "label")]
    groups       = df["group"].tolist()
    series_names = df["series"].tolist()
    y_all        = df["label"].values.astype(int)
    X_raw        = df[feat_cols].values.astype(float)
    return df, feat_cols, groups, series_names, y_all, X_raw
