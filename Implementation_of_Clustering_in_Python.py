import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generate synthetic data with 3 distinct clusters
n_samples = 300
centers = [[2, 2], [8, 8], [2, 8]]
cluster_std = 1.1
X, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std, random_state=42)

# Add three outliers
outliers = np.array([[6, 0], [0, 5], [9, 3]])
X = np.concatenate([X, outliers])

# Visualize the generated data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.title('Generated Synthetic Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)
plt.show()


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Assuming 'X' is the synthetic data from the previous step

# Apply K-Means clustering
# Note: K-Means is sensitive to outliers, which might be pulled into clusters
# or form their own. Let's try with k=3 and k=4.
kmeans_k3 = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_k3 = kmeans_k3.fit_predict(X)
centroids_k3 = kmeans_k3.cluster_centers_

# Visualize the K-Means result with k=3
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels_k3, s=50, cmap='viridis')
plt.scatter(centroids_k3[:, 0], centroids_k3[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title('K-Means Clustering Result (k=3)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)
plt.show()







from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
import numpy as np

# Assuming 'X' is the synthetic data

# --- Elbow Method using Inertia ---
inertia_values = []
k_range = range(2, 7)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(k_range, inertia_values, marker='o')
plt.title('Elbow Diagram using Inertia')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.grid(True)

# --- Elbow Method using Davies-Bouldin Index ---
db_scores = []
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    db_scores.append(davies_bouldin_score(X, labels))

plt.subplot(1, 2, 2)
plt.plot(k_range, db_scores, marker='o')
plt.title('Davies-Bouldin Index')
plt.xlabel('Number of clusters (k)')
plt.ylabel('DB Index (lower is better)')
plt.grid(True)

plt.tight_layout()
plt.show()



from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
import numpy as np

# Assuming 'X' is the synthetic data

# --- Elbow Method using Inertia ---
inertia_values = []
k_range = range(2, 7)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(k_range, inertia_values, marker='o')
plt.title('Elbow Diagram using Inertia')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.grid(True)

# --- Elbow Method using Davies-Bouldin Index ---
db_scores = []
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    db_scores.append(davies_bouldin_score(X, labels))

plt.subplot(1, 2, 2)
plt.plot(k_range, db_scores, marker='o')
plt.title('Davies-Bouldin Index')
plt.xlabel('Number of clusters (k)')
plt.ylabel('DB Index (lower is better)')
plt.grid(True)

plt.tight_layout()
plt.show()


from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np

# Assuming 'X' is the synthetic data

# Fit a Gaussian Mixture Model
# We can specify a range of components, and it will choose the best one via BIC.
gmm = GaussianMixture(n_components=4, random_state=42, covariance_type='full')
labels_gmm = gmm.fit_predict(X)

# Visualize the GMM clustering result
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels_gmm, s=50, cmap='viridis')
plt.title('Gaussian Mixture Model (GMM) Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)
plt.show()




from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np

# Assuming 'X' is the synthetic data

# Apply DBSCAN
# Note: Parameter tuning is crucial for DBSCAN.
# Let's try values that might separate the main groups and isolate outliers.
dbscan = DBSCAN(eps=0.8, min_samples=5)
labels_dbscan = dbscan.fit_predict(X) # Noise points are labeled -1

# Visualize the DBSCAN result
plt.figure(figsize=(8, 6))
# Use a mask to plot noise points differently
noise_mask = labels_dbscan == -1
plt.scatter(X[~noise_mask, 0], X[~noise_mask, 1], c=labels_dbscan[~noise_mask], s=50, cmap='viridis')
plt.scatter(X[noise_mask, 0], X[noise_mask, 1], c='red', s=50, marker='x', label='Outliers')
plt.title('DBSCAN Clustering Result')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()



from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import numpy as np

# Assuming 'X' is the synthetic data

# Perform hierarchical clustering using Ward's linkage method
# Ward's method is often a good default choice.
linked = linkage(X, method='ward')

# Plot the dendrogram
plt.figure(figsize=(12, 8))
dendrogram(linked,
            orientation='top',
            distance_sort='descending',
            show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram (Ward Linkage)')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.grid(axis='y')
plt.show()



import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons, make_circles
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

# Create a dataset similar to the 'bananas' example
n_samples = 1000
# Combine two moon shapes and a circle to create complex structures
noisy_moons = make_moons(n_samples=n_samples, noise=.05)
noisy_circles = make_circles(n_samples=n_samples, factor=.5, noise=.05)

# Join the datasets together
X, y = make_moons(n_samples=n_samples, noise=0.08, random_state=42)
X_circ, y_circ = make_circles(n_samples=n_samples//2, noise=0.08, factor=0.5, random_state=42)
X = np.vstack([X, X_circ + 2.5]) # Place the circle away from the moons

# Scale the data for better performance with clustering algorithms
X = StandardScaler().fit_transform(X)

# --- Apply K-Means (as shown in the lecture notes, this should fail) ---
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
y_kmeans = kmeans.fit_predict(X)


# --- Apply DBSCAN (this should succeed) ---
dbscan = DBSCAN(eps=0.2)
y_dbscan = dbscan.fit_predict(X)


# --- Plot the results ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot K-Means result
ax1.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=10, cmap='viridis')
ax1.set_title('K-Means Result (Fails on complex shapes)')
ax1.grid(True)

# Plot DBSCAN result
ax2.scatter(X[:, 0], X[:, 1], c=y_dbscan, s=10, cmap='viridis')
ax2.set_title('DBSCAN Result (Correctly identifies clusters)')
ax2.grid(True)

plt.suptitle("Comparison on 'Bananas'-like Dataset")
plt.show()



import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import SpectralClustering

# Function to generate spiral data
def generate_spirals(n_points, noise=.5):
    """Generate two intertwined spirals."""
    n = np.sqrt(np.random.rand(n_points, 1)) * 780 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_points, 1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_points, 1) * noise
    return (np.vstack((np.hstack((d1x,d1y)), np.hstack((-d1x,-d1y)))),
            np.hstack((np.zeros(n_points), np.ones(n_points))))

# Generate the data
X_spiral, y_spiral = generate_spirals(300)

# --- Apply Spectral Clustering ---
# n_clusters=2 because there are two spirals
spectral = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=42)
y_spectral = spectral.fit_predict(X_spiral)

# --- Plot the results ---
plt.figure(figsize=(8, 6))
plt.scatter(X_spiral[:, 0], X_spiral[:, 1], c=y_spectral, s=20, cmap='viridis')
plt.title('Spectral Clustering on Spirals Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)
plt.show()

