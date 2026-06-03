"""
Intraclass Correlation Coefficient (ICC).

Measures how reproducible a feature is across two scans of the same patient.
ICC close to 1 = reliable feature. ICC close to 0 = noisy, scan-dependent.

ICC pairs: Patient_X_Before vs Patient_X_After (same liver, different scan date).
"""
import numpy as np

ICC_PAIR_KEYS = [
    ("Patient_1_Before",  "Patient_1_After"),
    ("Patient_2_Before",  "Patient_2_After"),
    ("Patient_KB_Before", "Patient_KB_Cancer"),
]


def compute_icc(df, fnames):
    """Compute ICC for each feature using the before/after scan pairs."""
    B_rows, A_rows = [], []
    for b_key, a_key in ICC_PAIR_KEYS:
        b = df[df["patient"] == b_key]
        a = df[df["patient"] == a_key]
        if b.empty or a.empty:
            continue
        # Average across all series for each patient-phase
        B_rows.append(b[fnames].mean().values)
        A_rows.append(a[fnames].mean().values)

    if len(B_rows) < 2:
        return np.ones(len(fnames))  # not enough pairs — assume full reliability

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
