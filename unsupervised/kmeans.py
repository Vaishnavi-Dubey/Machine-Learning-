"""
K-Means Clustering from Scratch (NumPy).
An iterative unsupervised learning algorithm that partitions data into 'K' clusters.

Steps:
1. Initialize centroids randomly.
2. Assign each data point to the nearest centroid.
3. Update centroids by calculating the mean of all points assigned to them.
4. Repeat until convergence.
"""

import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None
        self.clusters = [[] for _ in range(self.k)]

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # Initialize centroids randomly from data points
        random_sample_idxs = np.random.choice(self.n_samples, self.k, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        for _ in range(self.max_iters):
            # Assign samples to closest centroids (create clusters)
            self.clusters = self._create_clusters(self.centroids)
            
            # Cache old centroids for convergence check
            centroids_old = self.centroids
            
            # Calculate new centroids from clusters
            self.centroids = self._get_centroids(self.clusters)
            
            # Check for convergence
            if self._is_converged(centroids_old, self.centroids):
                break

        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels

    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.k)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        distances = [np.sqrt(np.sum((sample - c)**2)) for c in centroids]
        return np.argmin(distances)

    def _get_centroids(self, clusters):
        centroids = np.zeros((self.k, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids_new):
        distances = [np.sqrt(np.sum((centroids_old[i] - centroids_new[i])**2)) for i in range(self.k)]
        return sum(distances) == 0

if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    X, y = make_blobs(centers=3, n_samples=300, n_features=2, shuffle=True, random_state=42)

    k = KMeans(k=3, max_iters=150)
    y_pred = k.predict(X)

    plt.scatter(X[:,0], X[:,1], c=y_pred, cmap='viridis')
    plt.title("K-Means Clustering from Scratch")
    print("Clusters identified. Plotting results...")
