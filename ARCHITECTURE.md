# Code Map — Liver Cancer Detection via CT

## What This Project Does

Automatically classifies liver CT scans as **Cancer** or **Healthy** using machine learning.  
**5 patients** | **28 scan series** | **41 radiomics features** | **Patient-level LOOCV**  
Best model: **VaRFS + KNN + rank-based weighting** — Acc=100%, AUC=1.0

---

## Pipeline Flow

```
DICOM folders (raw CT data)
        │
        ▼
   [ data/ ]
   read_dicom.py              Read DICOM → build 3D volumes → save as NIfTI
        │
        ▼
   liver_segmentation.py      TotalSegmentator → liver mask (label=5)
   segment_all_series.py      Full pipeline: DICOM → NIfTI → mask → preview images
        │
        ▼
   crop_liver.py              Crop to liver bounding box + Clip HU [-100,400]
   crop_all_series.py         Same crop for all series
        │
        ▼
   extract_all_series_features.py   41 features × 28 series → CSV
   extract_2d_slice_features.py     16 features × 795 slices → CSV
        │
        ▼
   [ pipeline/ ]              ML logic (modular)
   select_features.py         VaRFS / ANOVA → top-5 features
        │
        ▼
   loocv.py                   Leave-One-Patient-Out CV
        │
        ▼
   [ training/ ]
   train_all_series_report.py      8 ML models × 2 selectors × 2 stages + visualizations
   _run_attention_all_series.py    Gated Attention MLP
   train_2d_slices.py              ML on 2D slices (795 slices)
   train_2d_cnn.py                 CNN: SimpleCNN + ResNet18 + SE-Attn + Grad-CAM
        │
        ▼
   [ analysis/ ]
   _full_results_table.py     Full results table for all models
   _all_metrics.py            Acc/Prec/Recall/Spec/F1/AUC per combination
   _confusion_matrices.py     Confusion matrix per combination
        │
        ▼
   [ results/ ]               Output images and plots
```

---

## Folder Structure

```
PFinalproject/
│
├── data/                               Stage 1+2: Raw data processing
│   ├── read_dicom.py                   Read DICOM, build 3D volumes, save NIfTI
│   ├── liver_segmentation.py           TotalSegmentator → liver masks
│   ├── segment_all_series.py           Full pipeline for all series per patient
│   ├── crop_liver.py                   Crop liver ROI + HU clip (primary series)
│   ├── crop_all_series.py              Crop liver ROI + HU clip (all series)
│   ├── extract_all_series_features.py  Extract 41 radiomics features per series
│   └── extract_2d_slice_features.py    Extract 16 features per axial slice
│
├── training/                           Stage 3: Model training
│   ├── train_all_series_report.py  ←── MAIN: 8 ML models, LOOCV, visualizations
│   ├── _run_attention_all_series.py    Gated Attention MLP
│   ├── train_2d_slices.py              ML on 795 2D slices
│   └── train_2d_cnn.py                 CNN: SimpleCNN, ResNet18, SE-Attention, Grad-CAM
│
├── analysis/                           Stage 4: Results analysis
│   ├── _full_results_table.py          All models, Stage 1 (simple avg) vs Stage 2 (weighted)
│   ├── _all_metrics.py                 Acc/Prec/Recall/Spec/F1/AUC for every combination
│   ├── _confusion_matrices.py          Confusion matrix for every combination
│   ├── _analyze_vt.py                  Why Patient_VT is misclassified
│   ├── _debug_rf_vs_knn.py             Per-series breakdown: RF vs KNN on Patient_VT
│   ├── _compare_weighting.py           Rank-based vs volume-based series weighting
│   ├── _tune_rf.py                     Random Forest hyperparameter search
│   ├── _inspect_gray_areas.py          Diagnose gray areas in spatial heatmaps
│   └── _check_per_patient.py           Per-patient prediction debug
│
├── viz/                                Visualizations
│   ├── visualize_all_series_2d.py      2D preview of all series per patient
│   ├── visualize_all_series_3d.py      3D visualization
│   ├── visualize_dicom_raw.py          Raw DICOM viewer
│   ├── visualize_liver_gif.py          Slice-by-slice GIF of liver
│   └── _run_spatial_heatmaps.py        Spatial feature heatmaps on CT slices
│
├── pipeline/                           Modular ML logic (imported by training/)
│   ├── load_data.py                    Load radiomics CSV
│   ├── clean_features.py               Remove NaN, constants, clip outliers
│   ├── compute_icc.py                  Intraclass Correlation Coefficient
│   ├── select_features.py              VaRFS and ANOVA feature selection
│   ├── augment.py                      SMOTE / RandomOverSampler
│   ├── metrics.py                      Accuracy, Recall, Specificity, F1, AUC
│   ├── series_weight.py                1/rank weighting per series
│   ├── loocv.py                        Leave-One-Patient-Out Cross Validation
│   ├── models/
│   │   ├── knn.py                      K-Nearest Neighbors (k=3)
│   │   ├── random_forest.py            Random Forest (100 trees)
│   │   ├── svm.py                      SVM (RBF kernel)
│   │   ├── logistic_regression.py      Logistic Regression
│   │   ├── naive_bayes.py              Gaussian Naive Bayes
│   │   ├── decision_tree.py            Decision Tree (max_depth=3)
│   │   ├── gradient_boost.py           Gradient Boosting
│   │   └── mlp.py                      MLP (32→16, early stopping)
│   └── visualizations/
│       ├── heatmap_static.py           Seaborn Z-normalized heatmap
│       ├── heatmap_interactive.py      Plotly interactive HTML heatmap
│       ├── feature_importance.py       F-score vs ICC bar chart
│       └── spatial_heatmap.py          Local feature overlay on CT slices
│
├── results/                            Output files (images, HTML)
│   ├── heatmap_varfs.png
│   ├── heatmap_anova.png
│   ├── heatmap_interactive.html
│   ├── feature_importance.png
│   ├── spatial_fo_kurtosis.png
│   ├── spatial_fo_p10.png
│   ├── spatial_fo_mean.png
│   ├── spatial_sagittal_glcm_dissimilarity.png
│   ├── spatial_sagittal_glcm_contrast.png
│   ├── gradcam_resnet18.png
│   ├── gray_area_inspection.png
│   ├── per_series_vt_ga.png
│   ├── pca_tsne_feature_space.png
│   └── attention_weights.png
│
├── Cropped_Data/                       Processed data (not in git)
│   ├── All_Series_CT/                  Cropped CT volumes (.nii.gz)
│   ├── All_Series_Masks/               Liver masks (.nii.gz)
│   ├── all_series_radiomics.csv        41 features × 28 series
│   └── slice_radiomics_2d.csv          16 features × 795 slices
│
├── README.md                           Project overview and quickstart
├── ARCHITECTURE.md                     This file — code navigation map
└── project_report.md                   Full academic report (Hebrew)
```

---

## Selected Features

### VaRFS (F-score × ICC) — Used by the winning model
| # | Feature | Clinical meaning |
|---|---|---|
| 1 | `fo_kurtosis` | Shape of HU distribution — tumors create extreme voxel values |
| 2 | `fo_p10` | 10th percentile — necrotic/fatty areas in tumor |
| 3 | `fo_mean` | Mean HU — cancerous liver is darker on CT |
| 4 | `sagittal_glcm_dissimilarity` | Texture non-uniformity — tumor creates a "patchy" appearance |
| 5 | `sagittal_glcm_contrast` | Texture contrast — sharp boundaries between regions |

### ANOVA (F-score only)
| # | Feature | F-score |
|---|---|---|
| 1 | `fo_median` | 20.32 |
| 2 | `fo_p10` | 18.25 |
| 3 | `fo_mean` | 17.37 |
| 4 | `fo_p90` | 14.22 |
| 5 | `fo_kurtosis` | 8.68 |

> **Key difference:** VaRFS includes `sagittal_glcm_dissimilarity` (ICC=0.573, reproducible across scans).  
> ANOVA misses it because its F-score alone is low. This one feature is what allowed KNN to reach 100%.

---

## Final Results

| Approach | Model | Acc | Recall | Spec | AUC | F1 |
|---|---|---|---|---|---|---|
| **3D VaRFS + 1/rank** | **KNN** | **100%** | **100%** | **100%** | **1.000** | **1.000** |
| 3D VaRFS + 1/rank | Attention | 80% | 67% | 100% | 1.000 | 0.800 |
| 3D VaRFS + 1/rank | RF / SVM / others | 80% | 100% | 50% | ~0.85 | 0.857 |
| 3D simple average | All models | 80% | 100% | 0% | — | 0.857 |
| 2D Slice | RF / SVM | 80% | 100% | 50% | 0.833 | 0.857 |
| 2D CNN | All architectures | 60% | 100% | 0% | ~0.1 | 0.750 |

**Why KNN won:** When all 3 nearest neighbors of VT_s01 in 5D feature space are healthy → P(Cancer)=0.000 (hard zero).  
**Why CNN failed:** 638:157 cancer:healthy slice imbalance — network learns to always predict Cancer.

---

## Key Parameters

| Parameter | Value | Location |
|---|---|---|
| Number of selected features | 5 | `pipeline/select_features.py` |
| k in KNN | 3 | `pipeline/models/knn.py` |
| HU clip range | [-100, 400] | `data/crop_all_series.py` |
| Outlier clipping | ±3 SD | `pipeline/clean_features.py` |
| 2D slice stride | 3 | `data/extract_2d_slice_features.py` |
| Min liver pixels per slice | 200 | `data/extract_2d_slice_features.py` |
| Series weighting | 1/rank | `pipeline/series_weight.py` |
| CNN training epochs | 20 | `training/train_2d_cnn.py` |

---

## Run Order

```bash
# Stage 1 — Data processing
.venv\Scripts\python.exe data\segment_all_series.py
.venv\Scripts\python.exe data\crop_all_series.py
.venv\Scripts\python.exe data\extract_all_series_features.py
.venv\Scripts\python.exe data\extract_2d_slice_features.py

# Stage 2 — Model training
.venv\Scripts\python.exe training\train_all_series_report.py
.venv\Scripts\python.exe training\_run_attention_all_series.py
.venv\Scripts\python.exe training\train_2d_slices.py
.venv\Scripts\python.exe training\train_2d_cnn.py   # Warning: ~90 min on CPU

# Stage 3 — Results
.venv\Scripts\python.exe analysis\_full_results_table.py
.venv\Scripts\python.exe analysis\_all_metrics.py
.venv\Scripts\python.exe analysis\_confusion_matrices.py
```
