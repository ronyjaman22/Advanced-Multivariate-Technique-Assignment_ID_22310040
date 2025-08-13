import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from factor_analyzer import FactorAnalyzer
import statsmodels.api as sm

# ----------------------------------------------------
# Iris ডেটাসেটের জন্য ফ্যাক্টর অ্যানালাইসিস
# ----------------------------------------------------

# Iris ডেটাসেট লোড করুন
iris = load_iris()
iris_data = pd.DataFrame(iris.data, columns=iris.feature_names)
print("### Iris Dataset Head ###")
print(iris_data.head())
print("\n" + "="*40 + "\n")


# ডেটা স্কেল করুন
scaler = StandardScaler()
iris_data_scaled = scaler.fit_transform(iris_data)

# আইগেনভ্যালু গণনা করে ফ্যাক্টরের সংখ্যা নির্ধারণ (Kaiser Criterion)
covariance_matrix = np.cov(iris_data_scaled.T)
eigen_values, _ = np.linalg.eig(covariance_matrix)
n_factors_iris = sum(eigen_values > 1)
print(f"Number of factors to retain for Iris dataset: {n_factors_iris}")

# ফ্যাক্টর অ্যানালাইসিস চালান
fa_iris = FactorAnalyzer(n_factors=n_factors_iris, rotation="varimax")
fa_iris.fit(iris_data_scaled)

# ফ্যাক্টর লোডিং দেখুন
print("\n### Iris Factor Loadings ###")
print(pd.DataFrame(fa_iris.loadings_, index=iris_data.columns))

# ফলাফলের সারসংক্ষেপ দেখুন (ভ্যারিয়েন্স)
iris_variance = fa_iris.get_factor_variance()
print("\n### Iris Factor Variance ###")
print(pd.DataFrame(iris_variance, index=['SS Loadings', 'Proportion Var', 'Cumulative Var']))

print("\n" + "="*40 + "\n")

# ----------------------------------------------------
# mtcars ডেটাসেটের জন্য ফ্যাক্টর অ্যানালাইসিস
# ----------------------------------------------------

# mtcars ডেটাসেট লোড করুন
mtcars_data = sm.datasets.get_rdataset("mtcars", "datasets", cache=True).data
mtcars_df = pd.DataFrame(mtcars_data)
print("### mtcars Dataset Head ###")
print(mtcars_df.head())
print("\n" + "="*40 + "\n")


# ফ্যাক্টর অ্যানালাইসিস চালান (ফ্যাক্টর সংখ্যা ৩ ধরে)
fa_mtcars = FactorAnalyzer(n_factors=3, rotation="varimax")
fa_mtcars.fit(mtcars_df)

# লোডিং প্রিন্ট করুন
print("### mtcars Factor Loadings ###")
print(pd.DataFrame(fa_mtcars.loadings_, index=mtcars_df.columns))

# ইউনিকনেস (Uniqueness) প্রিন্ট করুন
print("\n### mtcars Uniquenesses ###")
print(pd.DataFrame(fa_mtcars.get_uniquenesses(), index=mtcars_df.columns, columns=['uniqueness']))