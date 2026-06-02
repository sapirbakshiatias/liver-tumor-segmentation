"""
משקל לפי דרגת הסדרה (rank-based weighting).

s01 (הסדרה הראשית — הכי הרבה slices) → משקל 1.0
s02 → 0.5 | s03 → 0.33 | s05 → 0.2 | s06 → 0.167
"""
import re


def series_weight(series_name):
    m    = re.search(r"_s(\d+)$", series_name)
    rank = int(m.group(1)) if m else 99
    return 1.0 / rank


# alias לתאימות אחורה עם קוד קיים
_series_weight = series_weight
