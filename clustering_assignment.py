
"""
Clustering Assignment - Theoretical and Practical Questions
-----------------------------------------------------------
This script contains answers to both theoretical and practical questions from the clustering assignment.
"""

# ---------------------------
# Theoretical Questions
# ---------------------------

"""
1. What is clustering in machine learning?
   - Clustering is an unsupervised learning technique used to group similar data points into clusters
     based on feature similarity. It helps identify inherent patterns in data without predefined labels.

2. Difference between supervised and unsupervised learning.
   - Supervised learning uses labeled data for training, while unsupervised learning finds hidden patterns in unlabeled data.

3. Name three popular clustering algorithms.
   - K-Means, DBSCAN, Agglomerative Hierarchical Clustering.

4. What is the role of the 'k' parameter in K-Means?
   - 'k' represents the number of clusters to divide the data into.

5. What is inertia in K-Means?
   - Inertia is the sum of squared distances of samples to their closest cluster center, used to evaluate the compactness of clusters.

6. What is a dendrogram?
   - A dendrogram is a tree-like diagram that records the sequences of merges or splits in hierarchical clustering.

7. What is the silhouette score?
   - The silhouette score measures how similar an object is to its own cluster compared to other clusters. Ranges from -1 to 1.

8. Difference between K-Means and DBSCAN.
   - K-Means requires specifying the number of clusters and works best with spherical clusters. DBSCAN does not need the number of clusters and can find arbitrarily shaped clusters and noise.

9. What is linkage in hierarchical clustering?
   - Linkage refers to the criterion used to determine the distance between clusters: single, complete, average, etc.

10. When to use DBSCAN over K-Means?
    - When the data has noise and clusters of arbitrary shape, DBSCAN is preferred.
"""

# ---------------------------
# Practical & Coding Questions
# ---------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles, load_iris, load_wine, load_digits, load_breast_cancer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage

# 1. make_blobs + KMeans
X, y = make_blobs(n_samples=300, centers=4, random_state=42)
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)

# 2. Iris + Agglomerative
iris = load_iris()
X_iris = iris.data
agg = AgglomerativeClustering(n_clusters=3)
labels = agg.fit_predict(X_iris)
print("First 10 labels (Agglomerative on Iris):", labels[:10])

# 3. make_moons + DBSCAN
X_moons, _ = make_moons(n_samples=300, noise=0.1)
dbscan = DBSCAN(eps=0.2, min_samples=5).fit(X_moons)

# 4. Wine + KMeans after StandardScaler
wine = load_wine()
X_wine_scaled = StandardScaler().fit_transform(wine.data)
kmeans_wine = KMeans(n_clusters=3, random_state=42).fit(X_wine_scaled)
print("Cluster sizes (Wine + KMeans):", np.bincount(kmeans_wine.labels_))

# 5. make_circles + DBSCAN
X_circles, _ = make_circles(n_samples=300, factor=0.5, noise=0.05)
dbscan_circles = DBSCAN(eps=0.2, min_samples=5).fit(X_circles)

# 6. Breast Cancer + KMeans after MinMaxScaler
cancer = load_breast_cancer()
X_cancer_scaled = MinMaxScaler().fit_transform(cancer.data)
kmeans_cancer = KMeans(n_clusters=2, random_state=42).fit(X_cancer_scaled)
print("Cluster centroids (Breast Cancer + KMeans):\n", kmeans_cancer.cluster_centers_)

# 7. make_blobs with varying std + DBSCAN
X_blobs, _ = make_blobs(n_samples=300, centers=3, cluster_std=[1.0, 2.5, 0.5], random_state=42)
dbscan_blobs = DBSCAN(eps=1.5, min_samples=5).fit(X_blobs)

# 8. Digits + PCA + KMeans
digits = load_digits()
X_digits_pca = PCA(n_components=2).fit_transform(digits.data)
kmeans_digits = KMeans(n_clusters=10, random_state=42).fit(X_digits_pca)

# 9. Silhouette scores for k=2 to 5
X_test, _ = make_blobs(n_samples=300, centers=4, random_state=42)
scores = []
for k in range(2, 6):
    labels = KMeans(n_clusters=k, random_state=42).fit_predict(X_test)
    scores.append(silhouette_score(X_test, labels))
print("Silhouette Scores for k=2 to 5:", scores)

# 10. Dendrogram for Iris
linked = linkage(iris.data, method='average')
plt.figure(figsize=(10, 5))
dendrogram(linked, labels=iris.target)
plt.title("Hierarchical Clustering Dendrogram")
plt.show()
