# --- Full Corrected Python Code for Correspondence Analysis ---

# Step 1: Install or upgrade the necessary libraries.
# Run this command in your terminal to ensure you have recent versions:
# pip install --upgrade prince pandas statsmodels matplotlib seaborn

import pandas as pd
import prince
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load Data ---
# Equivalent to R's data("USArrests")
try:
    arrests_data = sm.datasets.get_rdataset("USArrests", "datasets")
    USArrests = arrests_data.data
    print("USArrests Dataset Head:")
    print(USArrests.head())
    print("-" * 30)
except Exception as e:
    print(f"Failed to load dataset. Error: {e}")
    exit()

# --- Perform Correspondence Analysis ---
# Equivalent to FactoMineR's CA() or ca's ca()
ca = prince.CA(n_components=2, random_state=42)
ca = ca.fit(USArrests)


# --- Extract and Visualize Eigenvalues (Scree Plot) ---
# Equivalent to get_eigenvalue() and fviz_eig()
print("Eigenvalues and Explained Inertia:")
print(ca.eigenvalues_summary)
print("-" * 30)

# CORRECTED CODE: Create the scree plot from the summary table for compatibility
# This is a more robust way that avoids the AttributeError
explained_variance = ca.eigenvalues_summary['% of variance']

plt.figure(figsize=(8, 5))
plt.bar(
    explained_variance.index,
    explained_variance.values,
    color='steelblue',
    tick_label=[f'Dimension {i+1}' for i in range(len(explained_variance))]
)
plt.title('Scree Plot')
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Dimensions')
plt.show()


# --- Extract Row and Column Profiles (Coordinates) ---
# Equivalent to get_ca_row() and get_ca_col()
row_coords = ca.row_coordinates(USArrests)
col_coords = ca.column_coordinates(USArrests)

print("Row Coordinates (Top 5):")
print(row_coords.head())
print("\nColumn Coordinates:")
print(col_coords)
print("-" * 30)


# --- Visualize Row and Column Contributions ---
# A custom visualization inspired by fviz_ca_row() and fviz_ca_col()

# Row Contributions Plot
fig, ax = plt.subplots(figsize=(10, 8))
ca.row_contributions_.sort_values(by=0, ascending=True)[0].plot.barh(ax=ax)
ax.set_title("Row Contributions to Dimension 1")
ax.set_xlabel("Contribution (%)")
plt.show()

# Column Contributions Plot
fig, ax = plt.subplots(figsize=(8, 5))
ca.column_contributions_.sort_values(by=0, ascending=True)[0].plot.barh(ax=ax)
ax.set_title("Column Contributions to Dimension 1")
ax.set_xlabel("Contribution (%)")
plt.show()


# --- Create a Biplot ---
# Equivalent to fviz_ca_biplot()
ax_biplot = ca.plot(
    X=USArrests,
    show_row_labels=True,
    show_col_labels=True
)
ax_biplot.set_title("Correspondence Analysis Biplot of USArrests")
ax_biplot.figure.set_size_inches(12, 10)
plt.show()