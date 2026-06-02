"""
Random Forest classifier.

100 עצי החלטה. class_weight='balanced' מפצה על אי-שיווי מעמד (3 סרטן : 2 בריא).
נכשל ב-Patient_VT כי כל עץ מחשב הסתברות חלקית — גם כשהכבד בריא מקבל 0.41.
"""
from sklearn.ensemble import RandomForestClassifier

NAME = "RandomForest"

def make_clf():
    return RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
    )
