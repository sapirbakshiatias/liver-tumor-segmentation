"""
Gaussian Naive Bayes.

מניח שכל פיצ'ר מתפלג נורמלית בתוך כל קבוצה ושהפיצ'רים בלתי תלויים.
פשוט מאוד, לא דורש hyperparameters. AUC=0.5 — מצביע על מגבלת ההנחה.
"""
from sklearn.naive_bayes import GaussianNB

NAME = "NaiveBayes"

def make_clf():
    return GaussianNB()
