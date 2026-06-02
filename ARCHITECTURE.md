# מפת התמצאות בקוד — זיהוי סרטן כבד באמצעות CT

## מה הפרויקט עושה

סיווג אוטומטי של סריקות CT של כבד כ-**סרטני** או **בריא** תוך שימוש בלמידת מכונה.  
**5 מטופלים** | **28 סדרות סריקה** | **41 פיצ'רים** | **LOOCV**  
מודל מנצח: **VaRFS + KNN + 1/rank weighting** — Acc=100%, AUC=1.0

---

## זרימת העבודה (Pipeline)

```
תיקיות DICOM (עברית)
        │
        ▼
   [ data/ ]
   read_dicom.py            קריאת DICOM → נפחים 3D → NIfTI
        │
        ▼
   liver_segmentation.py    TotalSegmentator → מסכת כבד (label=5)
   segment_all_series.py    pipeline מלא לכל הסדרות
        │
        ▼
   crop_liver.py            חיתוך לאזור הכבד + Clip HU [-100,400]
   crop_all_series.py       אותו crop לכל הסדרות
        │
        ▼
   extract_all_series_features.py   41 פיצ'ר × 28 סדרות → CSV
   extract_2d_slice_features.py     16 פיצ'ר × 795 פרוסות → CSV
        │
        ▼
   [ pipeline/ ]            לוגיקת ה-ML
   select_features.py       VaRFS / ANOVA → 5 פיצ'רים
        │
        ▼
   loocv.py                 LOOCV — 4 מטופלים train, 1 test
        │
        ▼
   [ training/ ]
   train_all_series_report.py     8 מודלי ML × 2 שיטות × 2 stages
   _run_attention_all_series.py   Gated Attention MLP
   train_2d_slices.py             ML על פרוסות 2D
   train_2d_cnn.py                CNN: SimpleCNN + ResNet18 + Grad-CAM
        │
        ▼
   [ analysis/ ]
   _full_results_table.py   טבלת תוצאות מלאה
   _all_metrics.py          Acc/Prec/Recall/Spec/F1/AUC לכל שילוב
   _confusion_matrices.py   Confusion Matrix לכל שילוב
        │
        ▼
   [ results/ ]             גרפים ותמונות פלט
```

---

## מבנה התיקיות

```
PFinalproject/
│
├── data/                          שלב 1+2: עיבוד נתונים גולמיים
│   ├── read_dicom.py              קורא DICOM, בונה נפחי 3D, שומר NIfTI
│   ├── liver_segmentation.py      TotalSegmentator → מסכת כבד
│   ├── segment_all_series.py      pipeline מלא: DICOM → NIfTI → מסכה → תמונות
│   ├── crop_liver.py              חיתוך כבד + Clip HU (סדרה ראשית)
│   ├── crop_all_series.py         חיתוך כבד + Clip HU (כל הסדרות)
│   ├── extract_all_series_features.py  41 פיצ'ר לכל סדרה
│   └── extract_2d_slice_features.py    16 פיצ'ר לכל פרוסה אקסיאלית
│
├── training/                      שלב 3: אימון מודלים
│   ├── train_all_series_report.py ← המרכזי: 8 מודלי ML, LOOCV, ויזואליזציות
│   ├── _run_attention_all_series.py  Gated Attention MLP
│   ├── train_2d_slices.py         ML על 795 פרוסות 2D
│   └── train_2d_cnn.py            CNN: SimpleCNN, ResNet18, SE-Attn, Grad-CAM
│
├── analysis/                      שלב 4: ניתוח תוצאות
│   ├── _full_results_table.py     טבלת תוצאות כל המודלים (Stage 1 + 2)
│   ├── _all_metrics.py            Acc/Prec/Recall/Spec/F1/AUC לכל שילוב
│   ├── _confusion_matrices.py     Confusion Matrix לכל שילוב
│   ├── _analyze_vt.py             ניתוח למה Patient_VT מסווג שגוי
│   ├── _debug_rf_vs_knn.py        השוואה RF vs KNN לפי סדרה
│   ├── _compare_weighting.py      rank-based vs volume-based weighting
│   ├── _tune_rf.py                חיפוש hyperparameters ל-RF
│   ├── _inspect_gray_areas.py     בדיקת חורים אפורים במפות חום
│   └── _check_per_patient.py      בדיקת ניבוי לפי מטופל
│
├── viz/                           ויזואליזציות
│   ├── visualize_all_series_2d.py תמונות 2D של כל הסדרות
│   ├── visualize_all_series_3d.py ויזואליזציה 3D
│   ├── visualize_dicom_raw.py     תצוגת DICOM גולמי
│   ├── visualize_liver_gif.py     GIF slice-by-slice של הכבד
│   └── _run_spatial_heatmaps.py   מפות חום מרחביות
│
├── pipeline/                      לוגיקה מחולקת (ייבוא ע"י training/)
│   ├── load_data.py               טעינת CSV
│   ├── clean_features.py          ניקוי NaN, קבועים, outliers
│   ├── compute_icc.py             ICC — עקביות בין-סריקות
│   ├── select_features.py         VaRFS ו-ANOVA
│   ├── augment.py                 SMOTE / RandomOverSampler
│   ├── metrics.py                 Acc, Recall, Spec, F1, AUC
│   ├── series_weight.py           משקל 1/rank לכל סדרה
│   ├── loocv.py                   Leave-One-Patient-Out CV
│   ├── models/
│   │   ├── knn.py                 K-Nearest Neighbors (k=3)
│   │   ├── random_forest.py       Random Forest (100 עצים)
│   │   ├── svm.py                 SVM (RBF kernel)
│   │   ├── logistic_regression.py Logistic Regression
│   │   ├── naive_bayes.py         Gaussian Naive Bayes
│   │   ├── decision_tree.py       Decision Tree (max_depth=3)
│   │   ├── gradient_boost.py      Gradient Boosting
│   │   └── mlp.py                 MLP (32→16, early stopping)
│   └── visualizations/
│       ├── heatmap_static.py      Seaborn heatmap
│       ├── heatmap_interactive.py Plotly HTML
│       ├── feature_importance.py  גרף F-score vs ICC
│       └── spatial_heatmap.py     מפות חום על תמונות CT
│
├── results/                       פלטים (תמונות, גרפים)
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
├── Cropped_Data/                  נתונים מעובדים (לא ב-git)
│   ├── All_Series_CT/             נפחי CT חתוכים (.nii.gz)
│   ├── All_Series_Masks/          מסכות כבד (.nii.gz)
│   ├── all_series_radiomics.csv   41 פיצ'ר × 28 סדרות
│   └── slice_radiomics_2d.csv     16 פיצ'ר × 795 פרוסות
│
└── project_report.md              דוח מפורט לפרויקט
```

---

## פיצ'רים שנבחרו

### VaRFS (F-score × ICC) — המודל המנצח
| # | פיצ'ר | משמעות |
|---|---|---|
| 1 | `fo_kurtosis` | חדות ההתפלגות — גידול יוצר ערכי HU קיצוניים |
| 2 | `fo_p10` | אחוזון 10 — אזורי נמק בגידול |
| 3 | `fo_mean` | ממוצע HU — כבד סרטני כהה יותר |
| 4 | `sagittal_glcm_dissimilarity` | אי-אחידות טקסטורה — גידול "מנומר" |
| 5 | `sagittal_glcm_contrast` | ניגודיות טקסטורה — גבולות חדים |

### ANOVA (F-score בלבד)
| # | פיצ'ר |
|---|---|
| 1 | `fo_median` |
| 2 | `fo_p10` |
| 3 | `fo_mean` |
| 4 | `fo_p90` |
| 5 | `fo_kurtosis` |

---

## תוצאות סופיות

| גישה | מודל | Acc | Recall | Spec | AUC | F1 |
|---|---|---|---|---|---|---|
| **3D VaRFS + 1/rank** | **KNN** | **100%** | **100%** | **100%** | **1.000** | **1.000** |
| 3D VaRFS + 1/rank | Attention | 80% | 67% | 100% | 1.000 | 0.800 |
| 3D VaRFS + 1/rank | RF / SVM / ... | 80% | 100% | 50% | ~0.85 | 0.857 |
| 2D Slice | RF / SVM | 80% | 100% | 50% | 0.833 | 0.857 |
| 2D CNN | כל הארכיטקטורות | 60% | 100% | 0% | ~0.1 | 0.750 |

**הסיבה ש-KNN ניצח:** כשכל 3 שכנים של VT_s01 בחלל 5D הם בריאים → P(Cancer)=0.000 (קוטבי לחלוטין).  
**הסיבה ש-CNN נכשל:** יחס 638:157 (סרטן:בריא) ברמת הפרוסות — מודל לומד לנחש תמיד סרטן.

---

## סדר הרצה

```bash
# שלב 1 — נתונים
.venv\Scripts\python.exe data\segment_all_series.py
.venv\Scripts\python.exe data\crop_all_series.py
.venv\Scripts\python.exe data\extract_all_series_features.py
.venv\Scripts\python.exe data\extract_2d_slice_features.py

# שלב 2 — אימון
.venv\Scripts\python.exe training\train_all_series_report.py
.venv\Scripts\python.exe training\_run_attention_all_series.py
.venv\Scripts\python.exe training\train_2d_slices.py
.venv\Scripts\python.exe training\train_2d_cnn.py   # אזהרה: ~90 דקות על CPU

# שלב 3 — תוצאות
.venv\Scripts\python.exe analysis\_full_results_table.py
.venv\Scripts\python.exe analysis\_all_metrics.py
.venv\Scripts\python.exe analysis\_confusion_matrices.py
```

---

## הגדרות מפתח

| פרמטר | ערך | מיקום |
|---|---|---|
| מספר פיצ'רים נבחרים | 5 | `pipeline/select_features.py` |
| k ב-KNN | 3 | `pipeline/models/knn.py` |
| Clip HU | [-100, 400] | `data/crop_all_series.py` |
| Outlier threshold | ±3 SD | `pipeline/clean_features.py` |
| Slice stride (2D) | 3 | `data/extract_2d_slice_features.py` |
| Min liver pixels | 200 | `data/extract_2d_slice_features.py` |
| Series weight | 1/rank | `pipeline/series_weight.py` |
| CNN epochs | 20 | `training/train_2d_cnn.py` |
