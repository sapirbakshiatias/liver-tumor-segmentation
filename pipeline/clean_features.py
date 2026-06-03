"""
Feature cleaning: remove NaN columns, constant columns, and clip outliers.
"""
import numpy as np

CV_THRESHOLD = 100.0  # max allowed coefficient of variation (%)


def clean_features(X, feature_names):
    """
    Remove bad features and clip extreme values.

    Steps:
      1. Drop columns with NaN or Inf
      2. Drop constant columns (std == 0)
      3. Clip values beyond ±3 standard deviations
    """
    # 1. Remove NaN / Inf columns
    bad   = np.any(~np.isfinite(X), axis=0)
    X     = X[:, ~bad]
    names = [f for f, b in zip(feature_names, bad) if not b]

    # 2. Remove constant columns
    std   = X.std(axis=0)
    const = std == 0
    X     = X[:, ~const]
    names = [f for f, c in zip(names, const) if not c]

    # 3. Clip outliers at ±3 SD
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    X  = np.clip(X, mu - 3*sd, mu + 3*sd)

    return X, names


def cv_filter(X, feature_names):
    """
    Return a boolean mask: True = feature passes the CV threshold.
    CV = std/mean * 100. High CV means the feature is unstable across scans.
    """
    mean = np.abs(X.mean(axis=0))
    std  = X.std(axis=0)
    cv   = np.where(mean > 1e-10, std / mean * 100, np.inf)
    return cv <= CV_THRESHOLD
