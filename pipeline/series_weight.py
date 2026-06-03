"""
Rank-based series weighting.

In read_dicom.py, series are sorted by number of slices descending,
so s01 is always the largest (most representative) scan.
Secondary series (s02, s03...) are typically contrast phases that
can look cancer-like even in healthy patients.

Weighting by 1/rank ensures s01 dominates the patient prediction:
  s01 -> 1.000  (primary CT, most slices)
  s02 -> 0.500
  s03 -> 0.333
  s05 -> 0.200
  s06 -> 0.167
"""
import re


def series_weight(series_name):
    """Return 1/rank for a series name ending in _sNN."""
    m    = re.search(r"_s(\d+)$", series_name)
    rank = int(m.group(1)) if m else 99
    return 1.0 / rank


# Alias for backward compatibility with existing scripts
_series_weight = series_weight
