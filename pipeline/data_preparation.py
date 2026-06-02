"""
Data preparation: loading, cleaning, feature selection, LOOCV utilities.

כל מה שקורה לפני הרצת המודלים:
  1. טעינת ה-CSV
  2. ניקוי פיצ'רים (NaN, קבועים, outliers)
  3. חישוב ICC (עקביות בין-סריקות)
  4. בחירת פיצ'רים: VaRFS (F-score x ICC) ו-ANOVA (F-score בלבד)
  5. SMOTE / augmentation לטיפול באי-שיווי מעמד
  6. LOOCV — Leave-One-Patient-Out Cross Validation
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE, RandomOverSampler

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR = r"C:\Users\ronin\PycharmProjects\PFinalproject"
CSV_PATH = os.path.join(BASE_DIR, "Cropped_Data", "all_series_radiomics.csv")
CT_DIR   = os.path.join(BASE_DIR, "Cropped_Data", "All_Series_CT")
MASK_DIR = os.path.join(BASE_DIR, "Cropped_Data", "All_Series_Masks")
OUT_DIR  = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────

N_FEATURES   = 5      # כמה פיצ'רים לבחור
CV_THRESHOLD = 100.0  # סף Coefficient of Variation לסינון פיצ'רים
WINDOW       = 16     # גודל חלון sliding לmaps מרחביים
STRIDE       = 4

# זוגות ICC: אותו מטופל, שתי סריקות שונות (לפני ואחרי)
ICC_PAIR_KEYS = [
    ("Patient_1_Before",  "Patient_1_After"),
    ("Patient_2_Before",  "Patient_2_After"),
    ("Patient_KB_Before", "Patient_KB_Cancer"),
]


# ══════════════════════════════════════════════════════════════════════════════
# שלב 1: ניקוי פיצ'רים
# ══════════════════════════════════════════════════════════════════════════════

def clean_features(X, feature_names):
    """
    מסיר פיצ'רים בעייתיים ומגביל outliers.

    שלבים:
      1. מסיר עמודות עם NaN / Inf
      2. מסיר עמודות קבועות (סטיית תקן = 0)
      3. מחתך outliers מעבר ל-3 סטיות תקן
    """
    # 1. הסרת NaN/Inf
    bad = np.any(~np.isfinite(X), axis=0)
    X   = X[:, ~bad]
    names = [f for f, b in zip(feature_names, bad) if not b]

    # 2. הסרת פיצ'רים קבועים
    std   = X.std(axis=0)
    const = std == 0
    X     = X[:, ~const]
    names = [f for f, c in zip(names, const) if not c]

    # 3. Clipping ±3 סטיות תקן
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    X  = np.clip(X, mu - 3*sd, mu + 3*sd)

    return X, names


def cv_filter(X, feature_names):
    """
    מחזיר מסיכה בוליאנית: True = הפיצ'ר עובר את סף ה-CV.
    CV = std/mean * 100. פיצ'ר עם CV גבוה מדי = לא יציב.
    """
    mean = np.abs(X.mean(axis=0))
    std  = X.std(axis=0)
    cv   = np.where(mean > 1e-10, std / mean * 100, np.inf)
    return cv <= CV_THRESHOLD


# ══════════════════════════════════════════════════════════════════════════════
# שלב 2: ICC — Intraclass Correlation Coefficient
# ══════════════════════════════════════════════════════════════════════════════

def compute_icc(df, fnames):
    """
    מחשב ICC לכל פיצ'ר.

    ICC קרוב ל-1 = הפיצ'ר עקבי בין שתי סריקות של אותו מטופל.
    ICC קרוב ל-0 = הפיצ'ר משתנה בין סריקות — לא אמין לאבחון.

    משתמשים בזוגות: Patient_X_Before vs Patient_X_After.
    """
    B_rows, A_rows = [], []
    for b_key, a_key in ICC_PAIR_KEYS:
        b = df[df["patient"] == b_key]
        a = df[df["patient"] == a_key]
        if b.empty or a.empty:
            continue
        B_rows.append(b[fnames].mean().values)
        A_rows.append(a[fnames].mean().values)

    if len(B_rows) < 2:
        return np.ones(len(fnames))

    B = np.array(B_rows)
    A = np.array(A_rows)
    n, k = B.shape[0], 2
    vals  = np.stack([B, A], axis=1)
    subj  = vals.mean(axis=1)
    grand = vals.mean(axis=(0, 1))
    MSB   = k * np.sum((subj - grand)**2, axis=0) / (n - 1)
    MSW   = np.sum((vals - subj[:, None, :])**2, axis=(0, 1)) / (n*(k-1))
    denom = MSB + (k - 1)*MSW
    icc   = np.where(denom > 0, (MSB - MSW) / denom, 0.0)
    return np.clip(icc, 0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# שלב 3: בחירת פיצ'רים
# ══════════════════════════════════════════════════════════════════════════════

def select_varfs(X, y, fnames, icc_vals, cv_keep, k=N_FEATURES):
    """
    VaRFS: בוחר פיצ'רים לפי F-score מנורמל × ICC.

    מאפשר לפיצ'ר עם ICC גבוה לנצח פיצ'ר עם F-score גבוה יותר אבל לא עקבי.
    זה הסיבה ש-sagittal_glcm_dissimilarity נבחר למרות F-score נמוך.

    מחזיר: אינדקסים של הפיצ'רים הנבחרים + ציונים לדיבוג
    """
    f_scores, _ = f_classif(X, y)
    f_scores     = np.nan_to_num(f_scores)
    f_norm       = f_scores / (f_scores.max() + 1e-10)
    score        = f_norm * icc_vals
    score[~cv_keep] = 0.0
    top = np.argsort(score)[::-1][:k]
    return top, f_scores, icc_vals, score


def select_anova(X, y, fnames, cv_keep, k=N_FEATURES):
    """
    ANOVA: בוחר לפי F-score בלבד (ללא ICC).

    פשוטה יותר — בוחרת את הפיצ'רים שהכי מבדילים בין קבוצות.
    מחסרון: לא מתחשבת בעקביות הפיצ'ר בין סריקות.

    מחזיר: אינדקסים של הפיצ'רים הנבחרים
    """
    f_scores, _ = f_classif(X, y)
    f_scores     = np.nan_to_num(f_scores)
    f2           = f_scores.copy()
    f2[~cv_keep] = 0.0
    top = np.argsort(f2)[::-1][:k]
    return top, f_scores


# ══════════════════════════════════════════════════════════════════════════════
# שלב 4: Augmentation — טיפול באי-שיווי מעמד
# ══════════════════════════════════════════════════════════════════════════════

def augment(X, y):
    """
    יוצר דוגמאות סינתטיות של הקבוצה הקטנה (בריאים) לאזן את ה-dataset.

    SMOTE: יוצר נקודות חדשות לאורך הקו בין שכנים של הקבוצה הקטנה.
    RandomOverSampler: כפל פשוט כשיש רק דוגמה אחת.
    """
    n_min = (y == 0).sum()
    if n_min >= 2:
        return SMOTE(k_neighbors=1, random_state=42).fit_resample(X, y)
    if n_min == 1:
        return RandomOverSampler(random_state=42).fit_resample(X, y)
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
# שלב 5: מדדי ביצוע
# ══════════════════════════════════════════════════════════════════════════════

def metrics(y_true, y_pred, y_prob=None):
    """
    מחשב: Accuracy, Sensitivity (Recall), Specificity, AUC, F1.

    TP = Cancer מסווג נכון    | FP = Healthy מסווג כסרטן (ביופסיה מיותרת)
    TN = Healthy מסווג נכון   | FN = Cancer מסווג כבריא (פספוס מסוכן)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    acc  = (tp + tn) / len(y_true)
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan  # Recall
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    f1   = (2*prec*sens / (prec+sens)
            if not (np.isnan(prec) or np.isnan(sens)) and (prec+sens) > 0
            else np.nan)
    auc = np.nan
    if y_prob is not None and len(np.unique(y_true)) == 2:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            pass

    return acc, sens, spec, auc, f1


# ══════════════════════════════════════════════════════════════════════════════
# שלב 6: LOOCV — Leave-One-Patient-Out
# ══════════════════════════════════════════════════════════════════════════════

def _series_weight(series_name):
    """
    מחזיר משקל לכל סדרת סריקה לפי מיקומה.

    s01 (הסדרה הראשית — הגדולה ביותר) מקבל משקל 1.0.
    סדרות משניות (שלבי קונטרסט) מקבלות פחות: s02=0.5, s05=0.2 וכו'.

    הסיבה: בread_dicom.py מיינו לפי מספר slices יורד,
    כך s01 תמיד הסריקה המייצגת ביותר.
    """
    m    = re.search(r"_s(\d+)$", series_name)
    rank = int(m.group(1)) if m else 99
    return 1.0 / rank


def run_patient_loocv(X, y, groups, feat_idx, clf_name,
                      series_names=None, weighted=True):
    """
    Leave-One-Patient-Out CV: בכל fold מטופל אחד = test, השאר = train.

    כל הסדרות של המטופל הנבחן מוצאות מהאימון (אין data leakage).
    ניבוי מטופל = ממוצע משוקלל של הסתברויות הסדרות שלו.

    weighted=True  → משקל 1/rank (s01 חשוב פי 2 מ-s02)
    weighted=False → ממוצע פשוט (לפני הפתרון)
    """
    from pipeline.models import make_clf

    unique_patients = list(dict.fromkeys(groups))
    val_true, val_pred, val_prob = [], [], []
    fold_log = []

    for patient in unique_patients:
        test_mask  = np.array([g == patient for g in groups])
        train_mask = ~test_mask

        Xf_tr = X[train_mask][:, feat_idx]
        Xf_te = X[test_mask][:, feat_idx]
        y_tr  = y[train_mask]
        true_label = int(np.round(y[test_mask].mean()))

        scaler    = StandardScaler()
        Xf_tr_s   = scaler.fit_transform(Xf_tr)
        Xf_te_s   = scaler.transform(Xf_te)
        Xf_aug, y_aug = augment(Xf_tr_s, y_tr)

        if len(np.unique(y_aug)) < 2:
            only = int(y_aug[0])
            val_true.append(true_label)
            val_pred.append(only)
            val_prob.append(float(only))
            fold_log.append((patient, true_label, only, float(only)))
            continue

        clf = make_clf(clf_name)
        clf.fit(Xf_aug, y_aug)
        proba = clf.predict_proba(Xf_te_s)
        probs = proba[:, 1] if proba.shape[1] == 2 else np.full(len(Xf_te_s), 0.5)

        if weighted and series_names is not None:
            test_series = [series_names[i] for i, m in enumerate(test_mask) if m]
            weights     = np.array([_series_weight(s) for s in test_series])
            avg_prob    = float(np.average(probs, weights=weights))
        else:
            avg_prob = float(probs.mean())

        pred = int(avg_prob >= 0.5)
        val_true.append(true_label)
        val_pred.append(pred)
        val_prob.append(avg_prob)
        fold_log.append((patient, true_label, pred, avg_prob))

    acc, sens, spec, auc, f1 = metrics(val_true, val_pred, val_prob)
    return acc, sens, spec, auc, f1, fold_log


# ══════════════════════════════════════════════════════════════════════════════
# עזר: טעינת נתונים
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    """
    טוען את ה-CSV ומחזיר df, feat_cols, groups, series_names, y_all, X_raw.
    """
    df        = pd.read_csv(CSV_PATH)
    feat_cols = [c for c in df.columns
                 if c not in ("series", "patient", "group", "label")]
    groups       = df["group"].tolist()
    series_names = df["series"].tolist()
    y_all        = df["label"].values.astype(int)
    X_raw        = df[feat_cols].values.astype(float)
    return df, feat_cols, groups, series_names, y_all, X_raw


# ══════════════════════════════════════════════════════════════════════════════
# עזר: מפות מרחביות (Spatial Heatmaps)
# ══════════════════════════════════════════════════════════════════════════════

def sliding_feature_map(ct_sl, mask_sl, feature_name, win=WINDOW, stride=STRIDE):
    """
    מחשב ערך פיצ'ר מקומי בחלון הזזה מעל פרוסת הכבד.
    משמש ליצירת מפות חום מרחביות.
    """
    h, w = ct_sl.shape
    out  = np.full((h, w), np.nan)

    for y0 in range(0, h - win + 1, stride):
        for x0 in range(0, w - win + 1, stride):
            patch = ct_sl[y0:y0+win, x0:x0+win]
            pmask = mask_sl[y0:y0+win, x0:x0+win]
            if pmask.sum() < (win // 2):
                continue
            pix = patch[pmask > 0]

            if "range" in feature_name:
                val = float(pix.max() - pix.min())
            elif "entropy" in feature_name:
                hist, _ = np.histogram(pix, bins=16, density=True)
                h_nz = hist[hist > 0]
                val = float(-np.sum(h_nz * np.log2(h_nz + 1e-10)))
            elif "p10" in feature_name:
                val = float(np.percentile(pix, 10))
            elif "homogeneity" in feature_name:
                if pix.max() == pix.min():
                    val = 1.0
                else:
                    norm   = np.clip(((pix - pix.min()) / (pix.max() - pix.min()) * 31), 0, 31).astype(np.uint8)
                    norm2d = norm.reshape(1, -1)
                    g      = graycomatrix(norm2d, [1], [0], levels=32, symmetric=True, normed=True)
                    val    = float(graycoprops(g, "homogeneity").mean())
            elif "dissimilarity" in feature_name:
                if pix.max() == pix.min():
                    val = 0.0
                else:
                    norm   = np.clip(((pix - pix.min()) / (pix.max() - pix.min()) * 31), 0, 31).astype(np.uint8)
                    norm2d = norm.reshape(1, -1)
                    g      = graycomatrix(norm2d, [1], [0], levels=32, symmetric=True, normed=True)
                    val    = float(graycoprops(g, "dissimilarity").mean())
            else:
                val = float(pix.mean())

            region = out[y0:y0+win, x0:x0+win]
            out[y0:y0+win, x0:x0+win] = np.where(np.isnan(region), val, region)

    return out


def load_best_axial_slice(series_stem):
    """
    טוען את הפרוסה האקסיאלית עם הכי הרבה כבד מסריקה נתונה.
    """
    ct_path   = os.path.join(CT_DIR,   f"cropped_{series_stem}.nii.gz")
    mask_path = os.path.join(MASK_DIR, f"cropped_{series_stem}_mask.nii.gz")
    if not os.path.exists(ct_path) or not os.path.exists(mask_path):
        return None, None
    ct   = nib.load(ct_path).get_fdata().astype(float)
    mask = nib.load(mask_path).get_fdata().astype(np.uint8)
    if ct.ndim != 3 or mask.ndim != 3:
        return None, None
    iz     = int(np.argmax(mask.sum(axis=(0, 1))))
    ct_sl  = ct[:, :, iz].T
    msk_sl = mask[:, :, iz].T
    return ct_sl, msk_sl
