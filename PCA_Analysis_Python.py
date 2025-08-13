# Step 1: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

# Step 2: Create the protein dataset
data = {
    'Country': ['Albania', 'Austria', 'Belgium', 'Bulgaria', 'Czechoslovakia', 
                'Denmark', 'E.Germany', 'Finland', 'France', 'Greece', 
                'Hungary', 'Ireland', 'Italy', 'Netherlands', 'Norway', 
                'Poland', 'Portugal', 'Romania', 'Spain', 'Sweden', 
                'Switzerland', 'UK', 'USSR', 'W.Germany', 'Yugoslavia'],
    'RedMeat': [10.1, 8.9, 13.5, 7.8, 9.7, 10.6, 8.4, 9.5, 18.0, 10.2, 
                5.3, 13.9, 9.0, 9.5, 9.4, 6.9, 6.2, 6.2, 7.1, 9.9, 
                13.1, 17.4, 9.3, 11.4, 4.4],
    'WhiteMeat': [1.4, 14.0, 9.3, 6.0, 11.4, 10.8, 11.6, 4.9, 9.9, 3.0, 
                  12.4, 10.0, 5.1, 13.6, 4.7, 10.2, 3.7, 6.3, 3.4, 7.8, 
                  10.1, 5.7, 4.6, 12.5, 5.0],
    'Eggs': [0.5, 4.3, 4.1, 1.6, 2.8, 3.7, 3.7, 2.7, 3.3, 2.8, 
             2.9, 4.7, 2.9, 3.6, 2.7, 2.7, 1.1, 1.5, 3.1, 3.5, 
             3.1, 4.7, 2.1, 4.1, 1.2],
    'Milk': [8.9, 19.9, 17.5, 8.3, 12.5, 25.0, 11.1, 33.7, 19.5, 17.6, 
             9.7, 25.8, 13.7, 23.4, 23.3, 19.3, 4.9, 11.1, 8.6, 24.7, 
             23.8, 20.6, 16.6, 18.8, 9.5],
    'Fish': [0.2, 2.1, 4.5, 1.2, 2.0, 9.9, 5.4, 5.8, 5.7, 5.9, 
             0.3, 2.2, 3.4, 2.5, 9.7, 3.0, 14.2, 1.0, 7.0, 7.5, 
             2.3, 4.3, 3.0, 3.4, 0.6],
    'Cereals': [42.3, 28.0, 26.6, 56.7, 34.3, 21.9, 24.6, 26.3, 28.1, 41.7, 
                41.4, 24.0, 36.8, 22.4, 23.0, 36.1, 27.0, 49.6, 29.2, 22.1, 
                25.6, 24.3, 43.6, 18.6, 55.9],
    'Starchy': [0.6, 3.6, 5.7, 1.1, 5.0, 4.8, 6.5, 5.1, 4.8, 2.2, 
                5.1, 6.2, 2.1, 4.2, 4.6, 5.9, 5.9, 3.1, 5.7, 3.8, 
                2.8, 4.7, 6.4, 5.2, 3.0],
    'Nuts': [5.5, 1.3, 2.1, 3.7, 1.1, 0.7, 0.8, 1.0, 2.4, 7.8, 
             0.4, 1.6, 4.3, 1.8, 1.6, 2.0, 4.7, 5.3, 3.7, 1.3, 
             2.4, 3.4, 3.4, 1.5, 5.7],
    'Fr.Veg': [1.7, 4.3, 4.0, 4.2, 4.0, 2.4, 3.6, 1.4, 6.5, 6.5, 
               3.0, 2.9, 6.7, 3.7, 2.7, 6.6, 7.9, 2.8, 7.2, 4.3, 
               4.9, 3.3, 2.9, 3.8, 3.2]
}

protein_df = pd.DataFrame(data)


# Set country as index
protein_df.set_index('Country', inplace=True)

# Check for missing values
print("Missing values count:")
print(protein_df.isnull().sum())

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(protein_df)

# Step 4: Apply PCA
pca = PCA()
pca.fit(scaled_data)

# Get transformed data
X_pca = pca.transform(scaled_data)

# Step 5: Visualizations
# 5.1 Scree plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
         np.cumsum(pca.explained_variance_ratio_), 
         'o-', linewidth=2)
plt.title('Scree Plot - Cumulative Explained Variance')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.savefig('scree_plot.png', dpi=300)
plt.show()

# 5.2 Biplot of variables
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

plt.figure(figsize=(12, 8))
plt.scatter(loadings[:, 0], loadings[:, 1])
for i, feature in enumerate(protein_df.columns):
    plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], color='r', alpha=0.5)
    plt.text(loadings[i, 0]*1.15, loadings[i, 1]*1.15, feature, 
             color='darkred', fontsize=10)
    
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('Biplot of Variables')
plt.grid(True)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.savefig('biplot_variables.png', dpi=300)
plt.show()

# 5.3 Cos² plot (Quality of representation)
cos2 = loadings**2
cos2_df = pd.DataFrame(cos2, 
                      columns=[f'PC{i+1}' for i in range(cos2.shape[1])],
                      index=protein_df.columns)

# Plot cos2 for first two components
plt.figure(figsize=(12, 6))
cos2_df[['PC1', 'PC2']].plot(kind='bar', stacked=True)
plt.title('Variable Representation Quality (cos²)')
plt.ylabel('cos² Value')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('cos2_plot.png', dpi=300)
plt.show()

# 5.4 Combined biplot with cos2 coloring
total_cos2 = cos2_df['PC1'] + cos2_df['PC2']

plt.figure(figsize=(12, 10))
scatter = plt.scatter(loadings[:, 0], loadings[:, 1], 
                     c=total_cos2, cmap='viridis', 
                     s=total_cos2*300, alpha=0.7)

# Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Total cos² (PC1 + PC2)')

# Add variable names
for i, feature in enumerate(protein_df.columns):
    plt.text(loadings[i, 0]*1.1, loadings[i, 1]*1.1, feature, 
             fontsize=10, ha='center', va='center')
    
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('Biplot with cos² Coloring')
plt.grid(True)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('combined_biplot.png', dpi=300)
plt.show()

# 5.5 Additional: PCA Scores Plot
plt.figure(figsize=(12, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1])
for i, country in enumerate(protein_df.index):
    plt.text(X_pca[i, 0], X_pca[i, 1], country, 
             fontsize=9, ha='center', va='center')
    
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('PCA Scores Plot - Countries')
plt.grid(True)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('pca_scores.png', dpi=300)
plt.show()