"""
חישוב ICC — Intraclass Correlation Coefficient.
מודד עקביות של פיצ'ר בין שתי סריקות של אותו מטופל.
ICC קרוב ל-1 = פיצ'ר אמין. ICC קרוב ל-0 = פיצ'ר רועש.
"""
import numpy as np

ICC_PAIR_KEYS = [
    ("Patient_1_Before",  "Patient_1_After"),
    ("Patient_2_Before",  "Patient_2_After"),
    ("Patient_KB_Before", "Patient_KB_Cancer"),
]


def compute_icc(df, fnames):
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
    n, k  = B.shape[0], 2
    vals  = np.stack([B, A], axis=1)
    subj  = vals.mean(axis=1)
    grand = vals.mean(axis=(0, 1))
    MSB   = k * np.sum((subj - grand)**2, axis=0) / (n - 1)
    MSW   = np.sum((vals - subj[:, None, :])**2, axis=(0, 1)) / (n*(k-1))
    denom = MSB + (k - 1)*MSW
    icc   = np.where(denom > 0, (MSB - MSW) / denom, 0.0)
    return np.clip(icc, 0.0, 1.0)
