# Liver Cancer Detection from CT Scans

Automatic classification of liver CT scans as **Cancer** or **Healthy** using radiomics features and machine learning — no radiologist required.

## Dataset

| Patient | Label | Series |
|---|---|---|
| Patient_1 | Cancer | 7 series (before + after treatment) |
| Patient_2 | Cancer | 6 series (before + after treatment) |
| Patient_KB | Cancer | 8 series (before + with cancer) |
| Patient_GA | Healthy | 3 series |
| Patient_VT | Healthy | 5 series |

**Total:** 5 patients · 28 CT series · 795 axial slices

---

## Best Result

**VaRFS + KNN + rank-based weighting → 100% Accuracy, 100% Sensitivity, 100% Specificity, AUC=1.0**

| Model | Acc | Recall | Spec | AUC | F1 |
|---|---|---|---|---|---|
| **VaRFS + KNN** | **100%** | **100%** | **100%** | **1.000** | **1.000** |
| VaRFS + Attention | 80% | 67% | 100% | 1.000 | 0.800 |
| VaRFS + RF / SVM | 80% | 100% | 50% | ~0.85 | 0.857 |
| 2D Slice + RF / SVM | 80% | 100% | 50% | 0.833 | 0.857 |
| 2D CNN (all variants) | 60% | 100% | 0% | ~0.1 | 0.750 |

---

## Method Overview

```
DICOM (raw CT)
    → liver segmentation (TotalSegmentator)
    → crop liver region + HU clip [-100, 400]
    → extract 41 radiomics features per series
    → VaRFS feature selection (top-5 by F-score x ICC)
    → patient-level LOOCV (4 train, 1 test)
    → rank-weighted prediction (s01 weight=1.0, s02=0.5, ...)
```

### Feature Groups (41 total)
- **First-order** (13): mean, median, std, kurtosis, skewness, p10, p90, IQR, range, entropy, energy, variance, uniformity
- **Gradient** (3): mean/std/p90 Sobel magnitude
- **GLCM texture** (18): contrast, correlation, energy, homogeneity, dissimilarity, ASM — on 3 orthogonal planes (axial, sagittal, coronal)
- **Shape** (7): volume, extents, elongation, flatness, sphericity

### Top-5 VaRFS Features
| Feature | Clinical meaning |
|---|---|
| `fo_kurtosis` | Heavy tails in HU distribution — tumor creates extreme voxel values |
| `fo_p10` | Low HU regions — necrotic / fatty areas inside the tumor |
| `fo_mean` | Mean HU — cancerous liver appears darker on CT |
| `sagittal_glcm_dissimilarity` | Texture non-uniformity (sagittal view) |
| `sagittal_glcm_contrast` | Sharp contrast between neighboring voxels |

### Key Design Decisions
- **Patient-level LOOCV**: all series of one patient = test set — no data leakage across patients
- **ICC in VaRFS**: features must be reproducible across repeated scans of the same patient
- **Rank-based weighting**: `weight = 1/rank` — primary series (s01) dominates; secondary contrast phases get less weight
- **SMOTE**: synthetic oversampling to handle 3:2 cancer/healthy imbalance at series level

---

## Project Structure

```
PFinalproject/
├── data/           Raw data processing: DICOM → NIfTI → crop → features
├── training/       Model training: ML, Attention, 2D slices, CNN + Grad-CAM
├── analysis/       Results, diagnostics, confusion matrices
├── viz/            Visualization scripts
├── pipeline/       Reusable ML modules
│   ├── models/         One file per classifier (KNN, RF, SVM, ...)
│   └── visualizations/ One file per plot type
└── results/        Output images and plots
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full code map with file descriptions.

---

## Quickstart

```bash
# 1. Install dependencies
python -m venv .venv
.venv\Scripts\pip install pydicom SimpleITK nibabel numpy pandas scipy
.venv\Scripts\pip install scikit-learn imbalanced-learn seaborn plotly
.venv\Scripts\pip install torch torchvision totalsegmentator

# 2. Process data  (run once — data already exists if cloning this repo)
.venv\Scripts\python.exe data\segment_all_series.py
.venv\Scripts\python.exe data\crop_all_series.py
.venv\Scripts\python.exe data\extract_all_series_features.py

# 3. Train and evaluate
.venv\Scripts\python.exe training\train_all_series_report.py

# 4. View full results
.venv\Scripts\python.exe analysis\_full_results_table.py
.venv\Scripts\python.exe analysis\_all_metrics.py
.venv\Scripts\python.exe analysis\_confusion_matrices.py
```

---

## Output Files

| File | Description |
|---|---|
| `results/heatmap_varfs.png` | Z-normalized heatmap — VaRFS top-5 features |
| `results/heatmap_anova.png` | Z-normalized heatmap — ANOVA top-5 features |
| `results/heatmap_interactive.html` | Interactive Plotly heatmap |
| `results/feature_importance.png` | F-score vs ICC bar chart |
| `results/spatial_*.png` | Local feature value overlaid on CT slices |
| `results/gradcam_resnet18.png` | Grad-CAM attention maps (ResNet18) |
| `results/pca_tsne_feature_space.png` | PCA and t-SNE projection of feature space |
| `results/per_series_vt_ga.png` | Per-series cancer probability for VT and GA |
| `results/attention_weights.png` | Learned attention weights per feature |
