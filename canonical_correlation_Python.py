# প্রয়োজনীয় লাইব্রেরি ইম্পোর্ট করুন
import pandas as pd
import numpy as np
from statsmodels.multivariate.cancorr import CanCorr

# R কোডের মতো ডামি ডেটা তৈরি করুন
np.random.seed(42)
mm_data = np.random.randn(600, 8)
columns = ["Control", "Concept", "Motivation", "Read", "Write", "Math", "Science", "Sex"]
mm = pd.DataFrame(mm_data, columns=columns)

# ডেটা দুটি সেটে ভাগ করুন
psych = mm[["Control", "Concept", "Motivation"]]
acad = mm[["Read", "Write", "Math", "Science", "Sex"]]

print("Data Head:")
print(mm.head())
print("------------------------------")

# CCA মডেল তৈরি করুন
cca_model = CanCorr(endog=acad, exog=psych)

# Canonical correlations দেখুন
print("Canonical Correlations:")
print(cca_model.cancorr)
print("------------------------------")

# Canonical coefficients দেখুন
print("Canonical Coefficients (Psych - Set 1):")
print(cca_model.x_cancoef)
print("\nCanonical Coefficients (Acad - Set 2):")
print(cca_model.y_cancoef)
print("------------------------------")

# Significance tests (Wilks, Hotelling, Pillai, Roy)
# আপনার দেওয়া তালিকা অনুযায়ী সঠিক মেথডটি হলো corr_test()
mv_tests = cca_model.corr_test()
print("Multivariate Significance Tests:")
print(mv_tests)