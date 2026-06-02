"""Helper: generate spatial heatmaps only (after main report ran)."""
import sys
sys.path.insert(0, r"C:\Users\ronin\PycharmProjects\PFinalproject")

import os, pandas as pd
from train_all_series_report import plot_spatial_heatmaps, OUT_DIR

CSV_PATH = os.path.join(r"C:\Users\ronin\PycharmProjects\PFinalproject",
                        "Cropped_Data", "all_series_radiomics.csv")
df = pd.read_csv(CSV_PATH)

all_top_feats = [
    "fo_kurtosis", "fo_p10", "fo_mean", "sagittal_glcm_dissimilarity",
    "sagittal_glcm_contrast", "fo_median", "fo_p90",
]

print(f"Generating spatial heatmaps for {len(all_top_feats)} features x {len(df)} series...")
plot_spatial_heatmaps(df, all_top_feats)
print(f"\nDone. Saved to: {OUT_DIR}")
