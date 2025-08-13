import pandas as pd
import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt

# [cite_start]Data for U.S. Cities from the lecture notes [cite: 172]
data = [
    [0, 587, 1212, 701, 1936, 604, 748, 2139, 2182, 543],
    [587, 0, 920, 940, 1745, 1188, 713, 1858, 1737, 597],
    [1212, 920, 0, 879, 831, 1726, 1631, 949, 1021, 1494],
    [701, 940, 879, 0, 1374, 968, 1420, 1645, 1891, 1220],
    [1936, 1745, 831, 1374, 0, 2339, 2451, 347, 959, 2300],
    [604, 1188, 1726, 968, 2339, 0, 1092, 2594, 2734, 923],
    [748, 713, 1631, 1420, 2451, 1092, 0, 2571, 2408, 205],
    [2139, 1858, 949, 1645, 347, 2594, 2571, 0, 678, 2442],
    [2182, 1737, 1021, 1891, 959, 2734, 2408, 678, 0, 2329],
    [543, 597, 1494, 1220, 2300, 923, 205, 2442, 2329, 0]
]
city_names = ["Atlanta", "Chicago", "Denver", "Houston", "Los Angeles", "Miami", "New York", "San Francisco", "Seattle", "Washington D.C."]

# Initialize and fit Metric MDS
mds = MDS(n_components=2, metric=True, dissimilarity='precomputed', random_state=42)
pos = mds.fit_transform(data)

# --- CORRECTION STEP ---
# Flip the horizontal axis to match the R plot's orientation from page 18
pos[:, 0] = -pos[:, 0]

# Plot the results
plt.figure(figsize=(10, 8))
plt.scatter(pos[:, 0], pos[:, 1], s=0) 
for i, city in enumerate(city_names):
    plt.text(pos[i, 0], pos[i, 1], city, ha='center', va='center', fontsize=10)

plt.title('Metric MDS of U.S. City Distances (Matched to PDF)')
plt.xlabel('MDS Dimension 1')
plt.ylabel('MDS Dimension 2')
plt.grid(True)
plt.show()





import pandas as pd
import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt

# [cite_start]Dissimilarity data for WWII politicians from the lecture notes [cite: 302]
data_politicians = [
    [0, 5, 11, 15, 8, 17, 5, 10, 16, 17, 12, 16], [5, 0, 14, 16, 13, 18, 3, 11, 18, 18, 14, 17],
    [11, 14, 0, 7, 11, 11, 12, 5, 16, 8, 10, 18], [15, 16, 7, 0, 16, 16, 14, 8, 17, 6, 7, 12],
    [8, 13, 11, 16, 0, 15, 13, 11, 12, 14, 16, 12], [17, 18, 11, 16, 15, 0, 16, 12, 16, 12, 9, 13],
    [5, 3, 12, 14, 13, 16, 0, 9, 17, 16, 10, 12], [10, 11, 5, 8, 11, 12, 9, 0, 13, 9, 11, 7],
    [16, 18, 16, 17, 12, 16, 17, 13, 0, 12, 17, 10], [17, 18, 8, 6, 14, 12, 16, 9, 12, 0, 9, 11],
    [12, 14, 10, 7, 16, 9, 10, 11, 17, 9, 0, 15], [16, 17, 18, 12, 12, 13, 12, 7, 10, 11, 15, 0]
]
politician_names = ["Hitler", "Mussolini", "Churchill", "Eisenhower", "Stalin", "Attlee", "Franco", "De Gaulle", "Mao Tse", "Truman", "Chamberlain", "Tito"]

# Initialize and fit Non-Metric MDS
mds_nonmetric = MDS(n_components=2, metric=False, dissimilarity='precomputed', random_state=42)
pos_nonmetric = mds_nonmetric.fit_transform(data_politicians)

# --- CORRECTION STEP ---
# Flip the vertical axis to match the R plot's orientation from page 22
pos_nonmetric[:, 1] = -pos_nonmetric[:, 1]

# Plot the results
plt.figure(figsize=(10, 8))
plt.scatter(pos_nonmetric[:, 0], pos_nonmetric[:, 1], s=0)
for i, name in enumerate(politician_names):
    plt.text(pos_nonmetric[i, 0], pos_nonmetric[i, 1], name, ha='center', va='center', fontsize=10)

plt.title('Non-Metric MDS of WWII Politicians (Matched to PDF)')
plt.xlabel('MDS Dimension 1')
plt.ylabel('MDS Dimension 2')
plt.grid(True)
plt.show()